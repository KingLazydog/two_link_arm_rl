"""
controller.py —— 三个确定性 baseline，给 RL 当参照物

强化学习最怕的一件事是：不知道"这个分数到底是好是坏"。
所以先把三条尺子摆在桌上：

    zero    完全不动         -> 地板，任何算法都不该比它差
    random  随机推           -> 第二地板，常比"不动"还差（方向错比不动更有害）
    rr_pinv 雅可比伺服       -> 工程解法，用它当"已经有伺服了还要 RL 吗"的对照
    rr_dls  阻尼雅可比伺服   -> 奇异点附近还能用，这是伺服的实用版本

RL 的目标不是打过它们，而是：在不知道动力学、只看得到 (观测, 奖励) 的前提
下，靠试错逼近甚至超过 rr_dls 的表现。
"""

import numpy as np

from arm import dls_step, fk, jacobian
from env import MAX_STEPS, MAX_W, TwoLinkReachEnv


# ---------------- 策略工厂 ----------------
def make_zero():
    """什么都不做。"""
    def policy(obs):
        return np.zeros(2)
    return policy


def make_random(seed=0):
    """均匀随机推，最常见的"看似有动作"的噪声基组。"""
    rng = np.random.default_rng(seed)

    def policy(obs):
        return rng.uniform(-1, 1, size=2)
    return policy


def make_resolved_rate(gain=1.5, lam=0.05, max_w=MAX_W):
    """
    雅可比伺服（resolved-rate motion control）：
        q_dot = J^+ (p_target - p_now)
    把"末端还差多少"翻译成"关节该以多快速度转"。
    lam > 0 就是 DLS 阻尼版 —— 奇异点附近不会爆。

    gain 必须调，不是一个随便填的常数。实测（50 回合，delay=0）：

        gain=1.0 -> 98.0% / 67.2 步    最大但偏慢
        gain=1.5 -> 94.0% / 53.4 步    默认值
        gain=2.0 -> 94.0% / 44.0 步    最快
        gain=4.0 -> 84.0% / 49.1 步    ← 最初随手取的，一直在自己震荡
        gain=8.0 -> 82.5% / 47.4 步

    gain 太大时控制器会过冲、在目标附近来回抖，步数不降反升；
    太小时（0.3）根本来不及走完。这一列数据是"调参不是可选项"的直接证据。
    """
    def policy(obs):
        t1 = np.arctan2(obs[1], obs[0])
        t2 = np.arctan2(obs[3], obs[2])
        err = np.asarray(obs[4:6], dtype=float)
        J = jacobian(t1, t2)
        dq = dls_step(J, err, lam=lam) * gain
        return np.clip(dq / max_w, -1.0, 1.0)
    return policy


# ---------------- 回合运行 ----------------
def run_episode(policy, pose=None, target=None, seed=0, record=True,
                max_steps=MAX_STEPS, env_kw=None):
    """
    跑一整回合，返回 dict：
        steps 跑了多步；success 是否够到；dist 最终距离；path 末端轨迹
    """
    env = TwoLinkReachEnv(max_steps=max_steps, seed=seed, **(env_kw or {}))
    obs, _ = env.reset(pose=pose, target=target)
    path, dists = [], []
    steps, success = 0, False

    for _ in range(max_steps):
        if record:
            path.append(fk(*env.th)[1].copy())
            dists.append(env.dist())
        obs, rew, term, trunc, info = env.step(policy(obs))
        steps += 1
        if term:
            success = True
            break
        if trunc:
            break

    if record:
        path.append(fk(*env.th)[1].copy())
        dists.append(env.dist())
    return {
        "steps": steps,
        "success": success,
        "dist": env.dist(),
        "path": np.array(path),
        "dists": np.array(dists),
        "joints": env.th.copy(),
    }


def evaluate(policy, episodes=50, max_steps=MAX_STEPS, seed=0, env_kw=None):
    """在随机初始姿态 + 随机目标上评测，返回成功率与平均步数。"""
    succ, steps, dists = [], [], []
    for i in range(episodes):
        r = run_episode(policy, seed=seed + i, max_steps=max_steps, env_kw=env_kw)
        succ.append(1.0 if r["success"] else 0.0)
        steps.append(r["steps"])
        dists.append(r["dist"])
    return {
        "success_rate": float(np.mean(succ)),
        "avg_steps": float(np.mean(steps)),
        "avg_dist": float(np.mean(dists)),
    }
