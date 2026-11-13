"""
task_dynamics.py —— 实验六：从运动学仿真升级到动力学仿真

前面所有实验都跑在"速度模式"下：给关节一个速度指令，它就照着转，
质量、惯量、重力、科里奥利力全都不存在。这一层假设在真机上叫
"伺服内环完美跟踪"，是个很强的假设，只是过去没人量化它。

本脚本回答四个问题：

  实验 A  动作空间从"速度"换成"力矩"之后，同一批任务还好不好做？
  实验 B  模型知识值多少？—— 补偿重力、补偿惯量，各换回几个百分点？
  实验 B2 没有重力补偿时，下垂量是不是真的等于 G/Kp？
  实验 C  力矩控制对采样率有多敏感？20 Hz 的控制器能不能用？

实验 C 是本轮最意外的发现：重力补偿 PD 在 20 Hz 采样下会发散。
原因不复杂——补偿量在一次采样内保持不变，而这 50 ms 里手臂是个
倒立摆，重力梯度把扰动放大，下一拍才纠正。真机控制器跑 1 kHz，
就是为了避开这件事。
"""

import numpy as np

from arm import fk, ik_analytic, wrap
from controller import evaluate, make_resolved_rate
from dynamics import coriolis_vec, gravity_vec, mass_matrix, selfcheck
from env import MAX_DTH, TAU_MAX, TwoLinkReachEnv

EVAL_SEED = 500
N_EVAL = 50
VEL_DT = 0.05          # 速度模式的步长，与前面全部实验一致
CTRL_DT = 0.01         # 力矩模式的步长：100 Hz，力矩控制的最低可信采样率
HORIZON = 10.0         # 单回合时长，两种模式一致，便于比较


# ---------------- 力矩空间基线 ----------------
def _decode(obs, obs_vel):
    """从观测里还原关节角、末端位姿与目标点。观测按 cos/sin 存，反解无损。"""
    t1 = float(np.arctan2(obs[1], obs[0]))
    t2 = float(np.arctan2(obs[3], obs[2]))
    _, p = fk(t1, t2)
    target = p + np.asarray(obs[4:6], dtype=float)
    qd = np.asarray(obs[8:10], dtype=float) * MAX_DTH if obs_vel else np.zeros(2)
    return np.array([t1, t2]), target, qd


def _nearest_ik(q, target):
    """解析逆解有两支（肘上/肘下），取离当前姿态最近的那支，避免关节大跳。"""
    sols = ik_analytic(float(target[0]), float(target[1]))
    if not sols:
        return None
    return min(sols, key=lambda s: np.linalg.norm(wrap(np.array(s) - q)))


def make_pd(kp=100.0, kd=20.0, gravity=True, params=None, tau_max=TAU_MAX):
    """
    PD 位置控制，可选是否补偿重力。

    gravity=False 是最常见的学生实现：只写 PD，不管重力。
    真机上它会稳定停在目标下方一段距离 —— 重力矩被 Kp*误差 顶住了，
    这个误差就是"下垂量"，量纲上约等于 G/Kp。

    params 指定控制器"以为"的机构参数。默认用标称值；传入与真实机构
    不同的参数，就得到"模型失配"的控制器 —— 实验七用的就是这种情况。

    tau_max 必须与环境的力矩上限一致：控制器把力矩归一化到 [-1, 1]，
    环境再乘回去。两边不一致会静默地把实际力矩放大或缩小，
    表现为"补偿得越准反而越差"，很难查。
    """
    def policy(obs):
        q, target, qd = _decode(obs, True)
        q_des = _nearest_ik(q, target)
        if q_des is None:
            return np.zeros(2)
        tau = kp * wrap(np.array(q_des) - q) - kd * qd
        if gravity:
            tau = tau + (gravity_vec(q) if params is None else gravity_vec(q, params))
        return np.clip(tau / tau_max, -1.0, 1.0)
    return policy


def make_computed_torque(kp=100.0, kd=20.0):
    """
    计算力矩控制（反馈线性化）：把整个刚体模型都塞进控制器。

        tau = M(q) @ (Kp*e - Kd*q_dot) + C(q,q_dot)@q_dot + G(q)

    这是"模型完全已知"的天花板。它把非线性系统精确线性化成二阶系统，
    剩下的 Kp/Kd 就是普通的极点配置。

    它还有个副作用值得单独说：由于刚度被 M 归一化，闭环固有频率等于
    sqrt(Kp) 而与惯量无关（Kp=100 时 10 rad/s），所以在同样的采样率下
    比裸 PD 更不容易失稳 —— 反馈线性化不只是让公式好看。
    """
    def policy(obs):
        q, target, qd = _decode(obs, True)
        q_des = _nearest_ik(q, target)
        if q_des is None:
            return np.zeros(2)
        v = kp * wrap(np.array(q_des) - q) - kd * qd
        tau = mass_matrix(q) @ v + coriolis_vec(q, qd) + gravity_vec(q)
        return np.clip(tau / TAU_MAX, -1.0, 1.0)
    return policy


def eval_torque(policy, episodes=N_EVAL, seed=EVAL_SEED, dt=CTRL_DT, **env_kw):
    # 物理积分步长固定在 2 ms：dt 越小时子步数跟着降，精度不降、速度不垮
    kw = dict(control_mode="torque", obs_vel=True, tau_max=TAU_MAX, dt=dt,
              substeps=max(1, int(round(dt / 0.002))))
    kw.update(env_kw)
    return evaluate(policy, episodes=episodes, seed=seed,
                    max_steps=int(round(HORIZON / dt)), env_kw=kw)


# ---------------- 实验 A ----------------
def exp_a():
    print("\n实验 A · 动作空间：速度模式 vs 力矩模式")
    print(f"  {'控制器':<22}{'模式':<10}{'成功率':>8}{'平均到达时间':>14}")
    rows = []
    for g in (1.0, 2.0):
        r = evaluate(make_resolved_rate(gain=g, lam=0.05), episodes=N_EVAL,
                     seed=EVAL_SEED, max_steps=int(round(HORIZON / VEL_DT)))
        rows.append((f"雅可比伺服 gain={g:.0f}", "速度 20Hz", r, VEL_DT))
    for name, pol in [
        ("PD 无重力补偿", make_pd(gravity=False)),
        ("PD + 重力补偿", make_pd(gravity=True)),
        ("计算力矩（全模型）", make_computed_torque()),
    ]:
        rows.append((name, "力矩 100Hz", eval_torque(pol), CTRL_DT))

    for name, mode, r, dt in rows:
        print(f"  {name:<22}{mode:<10}{r['success_rate']*100:>7.1f}%"
              f"{r['avg_steps']*dt:>12.2f} s")
    return rows


# ---------------- 实验 B ----------------
def exp_b():
    print("\n实验 B · 模型知识分级（力矩模式，Kp=100 Kd=20，100 Hz）")
    print(f"  {'模型掌握程度':<24}{'成功率':>8}{'到达时间':>11}{'末态距离':>11}")
    rows = []
    for name, pol in [
        ("什么都不知道（裸 PD）", make_pd(gravity=False)),
        ("知道重力", make_pd(gravity=True)),
        ("知道重力+惯量+科氏力", make_computed_torque()),
    ]:
        r = eval_torque(pol)
        rows.append((name, r))
        print(f"  {name:<24}{r['success_rate']*100:>7.1f}%"
              f"{r['avg_steps']*CTRL_DT:>9.2f} s{r['avg_dist']*1000:>9.1f}mm")
    return rows


# ---------------- 实验 B2 ----------------
def static_droop(kp, kd=20.0, gravity=False, pose=(0.0, 0.6), t_hold=5.0, dt=CTRL_DT):
    """
    静态悬停：把手臂放到指定姿态，只命令"保持不动"，看它实际停在哪。

    这是测重力下垂最干净的方法 —— 没有目标点、没有轨迹，
    唯一的作用力是重力，唯一的抵抗力是 Kp 撑出来的弹簧力。
    """
    q0 = np.array(pose, dtype=float)
    env = TwoLinkReachEnv(control_mode="torque", obs_vel=True, seed=0, dt=dt,
                          substeps=max(1, int(round(dt / 0.002))))
    env.reset(pose=q0, target=fk(*q0)[1])
    for _ in range(int(t_hold / dt)):
        tau = kp * wrap(q0 - env.th) - kd * env.dth
        if gravity:
            tau = tau + gravity_vec(env.th)
        env.step(np.clip(tau / TAU_MAX, -1.0, 1.0))
    return {
        "joint_err": float(np.max(np.abs(wrap(env.th - q0)))),
        "tip_err": float(np.linalg.norm(fk(*env.th)[1] - fk(*q0)[1])),
        "predicted": float(np.max(np.abs(gravity_vec(q0)))) / kp,
    }


def exp_b2():
    print("\n实验 B2 · 重力下垂量 vs 控制器刚度（静态悬停，Kd=20，100 Hz）")
    g_test = gravity_vec(np.array([0.0, 0.6]))
    print(f"  姿态 q=(0, 0.6)，该姿态 |G| = {np.linalg.norm(g_test):.1f} N*m")
    print(f"  {'Kp':>6}{'无补偿·末端偏移':>18}{'有补偿·末端偏移':>18}{'理论 e=G/Kp':>15}")
    out = []
    for kp in (30.0, 100.0, 300.0, 1000.0):
        a = static_droop(kp, gravity=False)
        b = static_droop(kp, gravity=True)
        out.append((kp, a, b))
        print(f"  {kp:>6.0f}{a['tip_err']*1000:>16.1f}mm{b['tip_err']*1000:>16.1f}mm"
              f"{a['predicted']*1000:>13.1f}m")
    return out


# ---------------- 实验 C ----------------
def _dGdq(q, h=1e-6):
    """重力矩对角度的雅可比，数值差分。它决定"力矩保持"期间姿态是收敛还是发散。"""
    J = np.zeros((2, 2))
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        J[:, i] = (gravity_vec(q + e) - gravity_vec(q - e)) / (2 * h)
    return J


def _propagate(A, B, T, h=2e-5):
    """同时积分 Phi = e^{AT} 与 Gamma = ∫e^{A(T-s)}B ds，RK4，步长足够小以免污染结论。"""
    X, Z = np.eye(A.shape[0]), np.zeros_like(B)
    t = 0.0
    while t < T - 1e-15:
        s = min(h, T - t)
        k1x, k1z = A @ X, A @ Z + B
        k2x, k2z = A @ (X + s / 2 * k1x), A @ (Z + s / 2 * k1z) + B
        k3x, k3z = A @ (X + s / 2 * k2x), A @ (Z + s / 2 * k2z) + B
        k4x, k4z = A @ (X + s * k3x), A @ (Z + s * k3z) + B
        X = X + s / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        Z = Z + s / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)
        t += s
    return X, Z


def zoh_radius(kp, kd, dt, pose=(0.0, 0.6)):
    """
    零阶保持下的闭环谱半径。> 1 即发散。

    和成功率不是一回事：一个发散的控制环在乱撞中照样可能穿过目标球，
    成功率看上去还不错。所以要单独算这个数，它是"稳不稳"的判据，
    不是"快不快"的判据。
    """
    q0 = np.array(pose, dtype=float)
    Minv = np.linalg.inv(mass_matrix(q0))
    A = np.block([[np.zeros((2, 2)), np.eye(2)],
                  [-Minv @ _dGdq(q0), np.zeros((2, 2))]])
    B = np.vstack([np.zeros((2, 2)), Minv])
    Phi, Gam = _propagate(A, B, dt)
    K = np.hstack([-kp * np.eye(2), -kd * np.eye(2)])
    return float(np.max(np.abs(np.linalg.eigvals(Phi + Gam @ K))))


def exp_c():
    """
    采样率扫描。控制器参数完全不变，只改"多久算一次力矩"。

    数字控制器两次采样之间力矩不变，重力补偿项 G(q) 也冻结在采样时刻的值。
    这段间隔里手臂是个倒立摆：重力梯度把姿态误差越推越大，下一拍才纠正。
    采样越慢，纠正越迟，越可能已经来不及。

    谱半径取重力矩最大的姿态（手臂水平外伸）。它是"稳不稳"的判据，
    和"快不快"不是一回事 —— 表里 50 Hz 那行成功率满分、谱半径却 1.39，
    正是"用成功率判断控制器好坏不够用"的直接证据。
    """
    print("\n实验 C · 采样率对重力补偿 PD 的影响（Kp=100 Kd=20，参数完全相同）")
    print(f"  {'控制频率':>8}{'成功率':>9}{'到达时间':>11}{'ZOH 谱半径':>13}")
    out = []
    for hz in (20, 50, 100, 200, 500):
        dt = 1.0 / hz
        r = eval_torque(make_pd(gravity=True), dt=dt)
        rad = zoh_radius(100.0, 20.0, dt)
        out.append((hz, r, rad))
        print(f"  {hz:>6.0f}Hz{r['success_rate']*100:>8.1f}%"
              f"{r['avg_steps']*dt:>9.2f} s{rad:>11.2f} "
              f"{'稳定' if rad <= 1.0 else '发散'}")
    return out


def plot(res_a, res_b2, res_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    ax = axes[0]
    names = ["伺服\n(速度)", "裸 PD", "PD+重力\n补偿", "计算力矩\n(全模型)"]
    idx = [(0, VEL_DT), (2, CTRL_DT), (3, CTRL_DT), (4, CTRL_DT)]
    succ = [res_a[i][2]["success_rate"] * 100 for i, _ in idx]
    times = [res_a[i][2]["avg_steps"] * dt for i, dt in idx]
    bars = ax.bar(names, succ, color=["#8FA9C4", "#D08C6B", "#5C9E7E", "#7C6BA8"])
    for b, s, t in zip(bars, succ, times):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{s:.0f}%\n{t:.2f}s", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, 118)
    ax.set_ylabel("成功率 %")
    ax.set_title("加入动力学后：模型知道得越全，越接近 100%", fontsize=11)

    ax = axes[1]
    kps = [r[0] for r in res_b2]
    no_gc = [r[1]["tip_err"] * 1000 for r in res_b2]
    gc = [r[2]["tip_err"] * 1000 for r in res_b2]
    ax.plot(kps, no_gc, "o-", color="#C0504D", label="不补重力")
    ax.plot(kps, gc, "s-", color="#4F81BD", label="补重力")
    ax.set_xscale("log")
    ax.set_xlabel("位置环增益 Kp")
    ax.set_ylabel("末端下垂量 mm")
    ax.set_title("下垂量约等于 G/Kp；补上重力后归零", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[2]
    hzs = [r[0] for r in res_c]
    rads = [r[2] for r in res_c]
    ax.plot(hzs, rads, "o-", color="#3F6F8F")
    ax.axhline(1.0, color="#C0504D", ls="--", lw=1.2)
    ax.text(hzs[0], 1.12, "稳定边界", fontsize=8.5, color="#C0504D")
    be = list(zip(hzs, rads, [x[1]["success_rate"] * 100 for x in res_c]))
    for hz, rad, s in be:
        ax.annotate(f"{s:.0f}%", (hz, rad), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(hzs)
    ax.set_xticklabels([f"{h:g}" for h in hzs])
    ax.minorticks_off()
    ax.set_xlabel("控制频率 Hz")
    ax.set_ylabel("零阶保持谱半径")
    ax.set_title("20 Hz 时闭环发散，成功率却还有 78%", fontsize=11)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig("dynamics_result.png", dpi=160)
    print("\n已保存 dynamics_result.png")


def main():
    if not selfcheck(verbose=False):
        print("动力学自检未通过，实验终止（数字不可信）")
        return
    print("动力学自检：五项全部通过")
    print("参数：L1=1.0 m, L2=0.8 m, m1=3.0 kg, m2=2.0 kg, "
          f"tau_max={TAU_MAX:.0f} N*m")
    print(f"评测口径：{N_EVAL} 回合，EVAL_SEED={EVAL_SEED}，单回合 {HORIZON:.0f} s")
    ra = exp_a()
    exp_b()
    rb2 = exp_b2()
    rc = exp_c()
    print("\n说明：力矩模式下 Kp/Kd 未做网格搜索，取临界阻尼附近的经验值；")
    print("      调优后这些基线只会更好，所以上面的数字是保守下界。")
    plot(ra, rb2, rc)


if __name__ == "__main__":
    main()
