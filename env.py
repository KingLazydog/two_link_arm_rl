"""
env.py —— 「让机械臂末端够到指定点」的强化学习环境

不依赖 gymnasium，接口刻意与 gym 对齐
（reset -> (obs, info)；step -> (obs, reward, terminated, truncated, info)），
将来换成 gym.Env 或 real robot 都不用改训练代码。

为什么动作设计成"关节角速度"而不是"力矩"：
真实工业机器人的伺服模式之一就是速度模式；而纯力矩控制（二阶动力学）
会让 RL 的学习难度陡增。先做位控，是工程上最常见的取舍。
"""

import collections

import numpy as np

from arm import L1, L2, fk, wrap
from dynamics import DEFAULT_PARAMS, rk4_step

MAX_W = 3.0        # 关节最大角速度 rad/s
DT = 0.05          # 一步的时间长度
TOL = 0.05         # 末端进入这个半径算够到了（5 cm，和真实机器人的重复定位精度量级相当）
MAX_STEPS = 200
TAU_MAX = 60.0     # 关节力矩上限 N*m（重力矩峰值约 43 N*m，留三成余量）
MAX_DTH = 6.0      # 关节角速度上限 rad/s，对应驱动器的过流保护


class TwoLinkReachEnv:
    """
    观测 8 维，动作 2 维连续。

    joint_limits: ((lo1, hi1), (lo2, hi2))，单位弧度；None 表示不限制。
    加限位之后，纯几何的雅可比伺服会明显变差 —— 它不知道自己撞墙了，
    还会继续要求往越界方向转。这就是"专家失效"的场景，
    也是 RL 真正有用武之地的地方。

    action_delay: 动作要过多少步才真正生效（仿真步，1 步 = DT 秒）。
    真实的串口 / CAN 通信加上伺服响应，几个十毫秒的延迟是常态。
    雅可比伺服的控制律假设"算完立刻执行"，一旦有延迟就会系统性超调、
    在目标附近来回抖 —— 而这是任何基于当前误差的控制器都无法消除的。

    deadzone: 关节静摩擦死区，0~1，作用在"归一化动作"上。
        |a| < deadzone 时关节纹丝不动（推力顶不过静摩擦）；
        |a| >= deadzone 时实际速度 = sign(a)·(|a| - d)/(1 - d)，
        保证 |a| = 1 时仍是满速（所以它是"丢掉小信号"，不是"整体变慢"）。

        这是真实舵机 / 谐波减速器最典型的一种非线性：
        控制律里没有它，模型里通常也不写它。
        雅可比伺服在误差小时算出的关节速度本来就小，正好整段落进死区，
        于是它会在离目标几厘米的地方永远停住 —— 推不动，也走不完。
        注意观测里**不告诉** agent 有死区，它只能从"推了没反应"里自己学。

    reward_mode: 奖励的构造方式，两种。
        "dist"（默认）：每步 -0.5·d - 0.01，只回答"离目标还有多远"。
        "progress"  ：每步 prog_scale·(d_prev - d_now) - 0.01，回答"这一步走对没有"。

        两者的信息量差很多。dist 模式下，只要没够到，每一步的奖励都差不多，
         advantage 几乎只能靠"成没成功"这一个稀疏事件来区分；
        progress 模式把同一个目标拆成了每一步都能拿到的即时反馈。
        而且 progress 恰好是 potential-based shaping 的标准形式
        （Φ(s) = -d(s)，F = γΦ(s') - Φ(s)），**理论上不改变最优策略**，
        只改变学习信号的密度 —— 所以拿它做对照非常干净。
    """

    def __init__(self, max_w=MAX_W, dt=DT, tol=TOL, max_steps=MAX_STEPS,
                 seed=None, joint_limits=None, action_delay=0, deadzone=0.0,
                 reward_mode="dist", prog_scale=10.0, obs_noise=0.0,
                 control_mode="velocity", tau_max=TAU_MAX, substeps=20,
                 max_dth=MAX_DTH, vel_lag=0.0, obs_vel=False,
                 dyn_params=DEFAULT_PARAMS):
        self.max_w = max_w
        self.dt = dt
        self.tol = tol
        self.max_steps = max_steps
        self.joint_limits = joint_limits
        self.action_delay = int(action_delay)
        self.deadzone = float(deadzone)
        self.reward_mode = reward_mode
        self.prog_scale = prog_scale
        self.obs_noise = float(obs_noise)
        self.control_mode = control_mode
        self.tau_max = float(tau_max)
        self.substeps = int(substeps)
        self.max_dth = float(max_dth)
        self.vel_lag = float(vel_lag)
        self.obs_vel = bool(obs_vel)
        self.dyn_params = dyn_params
        self.rng = np.random.default_rng(seed)
        self.th = None
        self.dth = np.zeros(2)
        self._v_act = np.zeros(2)
        self.target = None
        self.steps = 0
        self.blocked = 0          # 被限位挡住的次数，诊断用
        self.stalled = 0          # 被死区吃掉的步数，诊断用
        self.saturated = 0        # 动作用满的时刻数，判"是不是被力矩上限卡死"
        self._aq = None

    def _apply_deadzone(self, a):
        """关节静摩擦：小命令推不动，大命令按超出部分线性输出。"""
        if self.deadzone <= 0.0:
            return a
        d = self.deadzone
        mag = np.abs(a)
        out = np.zeros_like(a)
        hit = mag >= d
        out[hit] = np.sign(a[hit]) * (mag[hit] - d) / (1.0 - d)
        self.stalled += int(np.sum(~hit))
        return out

    # ---- 采样 ----
    def _sample_pose(self):
        return self.rng.uniform(-np.pi, np.pi, size=2)

    def _sample_target(self):
        """在工作空间环内取点，避免采样到够不到的位置。"""
        r = self.rng.uniform(abs(L1 - L2) + 0.15, L1 + L2 - 0.15)
        a = self.rng.uniform(-np.pi, np.pi)
        return np.array([r * np.cos(a), r * np.sin(a)])

    # ---- 观测 ----
    def _obs(self):
        _, p = fk(*self.th)
        dx = self.target[0] - p[0]
        dy = self.target[1] - p[1]
        if self.obs_noise > 0.0:
            dx += self.rng.normal(0.0, self.obs_noise)
            dy += self.rng.normal(0.0, self.obs_noise)
        base = [
            np.cos(self.th[0]), np.sin(self.th[0]),
            np.cos(self.th[1]), np.sin(self.th[1]),
            dx, dy,
            self.th[0] / np.pi, self.th[1] / np.pi,
        ]
        if self.obs_vel:
            base += [self.dth[0] / self.max_dth, self.dth[1] / self.max_dth]
        return np.array(base, dtype=np.float32)

    def dist(self):
        _, p = fk(*self.th)
        return float(np.linalg.norm(self.target - p))

    # ---- 主循环 ----
    def reset(self, pose=None, target=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.th = self._sample_pose() if pose is None else np.array(pose, dtype=float)
        self.target = self._sample_target() if target is None else np.array(target, dtype=float)
        if self.joint_limits is not None:
            self.th = np.clip(self.th, self._lo, self._hi)   # 初始姿态也要合规
        self.dth = np.zeros(2)          # 力矩模式下从静止开始
        self._v_act = np.zeros(2)       # 速度环的实际输出，带宽有限时滞后于指令
        self.steps = 0
        self.blocked = 0
        self.stalled = 0
        self.saturated = 0
        self._aq = collections.deque(
            [np.zeros(2)] * (self.action_delay + 1),
            maxlen=self.action_delay + 1)
        return self._obs(), {"dist": self.dist()}

    @property
    def _lo(self):
        return np.array([self.joint_limits[0][0], self.joint_limits[1][0]])

    @property
    def _hi(self):
        return np.array([self.joint_limits[0][1], self.joint_limits[1][1]])

    def _apply_joint_limits(self, nxt):
        """越界就夹住。力矩模式下还要吸收动能，否则下一拍会继续往里顶。"""
        if self.joint_limits is None:
            return nxt
        clipped = np.clip(nxt, self._lo, self._hi)
        hit = np.abs(clipped - nxt) > 1e-9
        if np.any(hit):
            self.blocked += 1      # 想转但被限位挡住
            if self.control_mode == "torque":
                self.dth = np.where(hit, 0.0, self.dth)
        return clipped

    def _step_velocity(self, v_cmd):
        """
        速度模式：假设伺服内环能完美跟踪速度指令。
        vel_lag 给这个假设松绑 —— 一阶惯性环节，时间常数对应速度环带宽
        （带宽 f Hz 时 tau = 1/(2*pi*f)）。真实伺服速度环常是 100~500 Hz，
        老旧驱动器或加了减速器柔性的场合会掉到个位数 Hz。
        """
        if self.vel_lag > 0.0:
            alpha = 1.0 - np.exp(-self.dt / self.vel_lag)
            self._v_act += (v_cmd - self._v_act) * alpha
            v = self._v_act
        else:
            self._v_act = v_cmd
            v = v_cmd
        self.th = self._apply_joint_limits(wrap(self.th + v * self.dt))

    def _step_torque(self, tau):
        """力矩模式：给定力矩，用刚体动力学积分一步（零阶保持 + RK4 子步）。"""
        h = self.dt / self.substeps
        th, dth = self.th.copy(), self.dth.copy()
        for _ in range(self.substeps):
            th, dth = rk4_step(th, dth, tau, h, self.dyn_params)
            dth = np.clip(dth, -self.max_dth, self.max_dth)
        self.th, self.dth = wrap(th), dth
        self.th = self._apply_joint_limits(self.th)

    def step(self, action):
        """
        action: 长度 2 的数组，每个分量在 [-1, 1]，代表期望关节角速度。
        注意返回的 terminated / truncated 是分开的 ——
        terminated = 真够到了（成功），truncated = 超时（没结局）。
        """
        d_prev = self.dist()            # progress 奖励要用"这一步之前多远"
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.action_delay > 0:
            self._aq.append(a)
            a = self._aq.popleft()      # 真正执行的是若干步之前下发的那条
        a = self._apply_deadzone(a)     # 命令先过一遍关节静摩擦
        if np.any(np.abs(a) > 0.999):
            self.saturated += 1         # 动作顶到上限：判"是不是力矩不够"

        if self.control_mode == "torque":
            self._step_torque(a * self.tau_max)
        else:
            self._step_velocity(a * self.max_w)
        self.steps += 1

        d = self.dist()
        if d <= self.tol:                       # 成功终止
            return self._obs(), 10.0 - d, True, False, {"dist": d}
        trunc = self.steps >= self.max_steps    # 仅仅是时间到了
        if self.reward_mode == "progress":
            reward = (d_prev - d) * self.prog_scale - 0.01
        else:
            reward = -d * 0.5 - 0.01            # 稠密：离得越近扣分越少
        return self._obs(), reward, False, trunc, {"dist": d}
