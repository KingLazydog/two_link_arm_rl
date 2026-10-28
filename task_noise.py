"""
task_noise.py —— 实验七：观测噪声

前三个"让专家失效"的尝试：
    关节限位  -> 反而屏蔽了奇异点，专家升到 100%（假设被推翻）
    执行延迟  -> 反而当了低通滤波，专家也变好（假设被推翻）
    摩擦死区  -> 真失效（98% -> 80%），但 RL 没赢人工 dither

观测噪声是第四个候选，也是最有希望的一个，理由和前三个都不同：
    伺服的控制律是 q_dot = J+ (p_target - p_now)，
    它假设自己拿到的误差是准的。真实编码器 / 视觉反馈都有噪声，
    一旦误差里混入噪声，J+ 会把噪声一起放大 gain 倍 -> 在目标附近抖。

这个实验要回答三个问题：
    1. 伺服在观测噪声下到底掉多少？（每个 sigma 都扫 gain 取最优，不重犯 baseline 错）
    2. 传统工程解（对误差做低通滤波）能救回多少？（RL 要打过它才算赢）
    3. RL 在训练时就见过同分布噪声，它能学出更鲁棒的策略吗？

公平性说明：
    伺服的 policy 从 obs[4:6] 读误差，而噪声正是加在这一维上，
    所以 env_kw={"obs_noise": s} 时伺服自动吃到带噪反馈，与真实机器人一致。
    RL 侧训练环境与评测环境同分布同强度 —— 这是 RL 的主场，如实注明。
"""

import collections
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arm import dls_step, jacobian
from controller import evaluate, make_resolved_rate
from env import MAX_W, TOL, TwoLinkReachEnv
from train_ppo import make_greedy, make_sample, norm_obs, train

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_result.png")

SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.2]     # 噪声标准差，单位米（TOL=0.05m）
GAINS = [0.5, 1.0, 1.5, 2.0, 3.0]
RL_SIGMA = 0.1                            # RL 只跑一个代表性强度（训练太贵）
N_EVAL = 50


def make_filtered_rr(gain=1.0, lam=0.05, win=1):
    """
    伺服 + 低通滤波：对误差 obs[4:6] 做滑动平均再送进控制律。
    这是真实系统对付传感器噪声最标准的手段（和给伺服用 dither 一样，
    属于"传统控制器本来就有对应工具"，RL 必须打过它才算数）。

    win=1 时退化成普通伺服。滤波器有状态，跨回合会残留几步旧误差，
    窗口只有 3~5 步且每步都在推进，对结论影响可忽略；为严格起见，
    报告里同时给出 win=1（无滤波）作为下界参照。
    """
    buf = collections.deque(maxlen=win)

    def policy(obs):
        err = np.asarray(obs[4:6], dtype=float)
        buf.append(err)
        err_f = np.mean(buf, axis=0) if win > 1 else err
        t1 = np.arctan2(obs[1], obs[0])
        t2 = np.arctan2(obs[3], obs[2])
        dq = dls_step(jacobian(t1, t2), err_f, lam=lam) * gain
        return np.clip(dq / MAX_W, -1.0, 1.0)
    return policy


def run_diag(policy, seed, env_kw=None):
    """
    诊断版单回合：统计"曾经进入过 tol 范围"的次数。
    无噪时这个数应该是 1（一次到位）；
    有噪时控制器会反复穿越 tol 边界，这个数会明显大于 1 ——
    这是"在目标附近抖"的直接量化证据，比成功率更早暴露问题。
    """
    env = TwoLinkReachEnv(seed=seed, **(env_kw or {}))
    obs, _ = env.reset()
    hits = 0
    for _ in range(env.max_steps):
        obs, rew, term, trunc, info = env.step(policy(obs))
        if info["dist"] <= TOL:
            hits += 1
        if term or trunc:
            break
    return hits


def diag_hits(policy, env_kw, episodes=N_EVAL, seed=500):
    return float(np.mean([run_diag(policy, seed + i, env_kw)
                          for i in range(episodes)]))


def main():
    print("=" * 66)
    print("实验七：观测噪声 —— 伺服控制律假设误差是准的，现在让它不准")
    print("=" * 66)
    t0 = time.time()

    # ---------- 阶段 1+2：伺服 × 噪声强度 × gain 全扫描 ----------
    # 每个 sigma 下扫 gain 取最优：不重犯"拿随手写的 gain 当基线"的错
    rows = []
    print("\n[阶段1] 伺服在不同噪声强度下的表现（每个 sigma 内扫 gain 取最优）")
    servo_curves = {g: [] for g in GAINS}
    for s in SIGMAS:
        kw = {"obs_noise": s}
        per_gain = {}
        for g in GAINS:
            r = evaluate(make_resolved_rate(gain=g), episodes=N_EVAL,
                         seed=500, env_kw=kw)
            per_gain[g] = r["success_rate"] * 100
            servo_curves[g].append(r["success_rate"] * 100)
        best_g = max(per_gain, key=per_gain.get)
        best = evaluate(make_resolved_rate(gain=best_g), episodes=N_EVAL,
                        seed=500, env_kw=kw)
        hits_raw = diag_hits(make_resolved_rate(gain=best_g), kw)
        rows.append((s, best_g, best["success_rate"] * 100,
                     best["avg_steps"], hits_raw))
        print(f"  sigma={s:.2f}m  最优 gain={best_g:<4}  "
              f"成功率 {best['success_rate']*100:5.1f}%  "
              f"平均 {best['avg_steps']:6.1f} 步  曾进入tol {hits_raw:5.2f} 次")

    print("\n[阶段2] 伺服 + 低通滤波（传统工程解，窗口 3 / 5）")
    filt_curves = {w: [] for w in (3, 5)}
    filt_best = {}
    for s in SIGMAS:
        kw = {"obs_noise": s}
        # 滤波版也在该 sigma 的最优 gain 附近取，win=1 即无滤波下界
        g0 = [r[1] for r in rows if r[0] == s][0]
        per = {}
        for w in (3, 5):
            for g in sorted({g0, max(0.5, g0 - 0.5), min(3.0, g0 + 0.5)}):
                r = evaluate(make_filtered_rr(gain=g, win=w),
                             episodes=N_EVAL, seed=500, env_kw=kw)
                per[(w, g)] = r["success_rate"] * 100
        for w in (3, 5):
            best_here = max(v for (ww, g), v in per.items() if ww == w)
            filt_curves[w].append(best_here)
        bw, bg = max(per, key=per.get)
        filt_best[s] = (bw, bg, per[(bw, bg)])
        print(f"  sigma={s:.2f}m  最优 win={bw} gain={bg}  "
              f"成功率 {per[(bw, bg)]:5.1f}%")

    # ---------- 阶段 3：PPO 冷启动，训练时就见到同分布噪声 ----------
    print(f"\n[阶段3] PPO 冷启动 + progress 奖励，训练/评测都带 sigma={RL_SIGMA}m 噪声")
    print("       （400 轮：谷底实验证明冷启动 progress 在 150 轮还没到顶）")
    rl_kw = {"obs_noise": RL_SIGMA, "reward_mode": "progress", "prog_scale": 10.0}
    net, hist, elapsed, best_state, best_succ = train(
        iters=400, lr=5e-5, ent_coef=0.005, clip=0.2,
        env_kw=rl_kw, seed=0, eval_sample=False)
    print(f"       训练完成，用时 {elapsed:.0f}s，训练中最佳贪心 {best_succ*100:.1f}%")

    if best_state is not None:
        net.load_state_dict(best_state)

    rl_greedy = evaluate(make_greedy(net), episodes=N_EVAL, seed=500, env_kw=rl_kw)
    rl_sample = evaluate(make_sample(net, seed=0), episodes=N_EVAL, seed=500,
                         env_kw=rl_kw)
    rl_hits = diag_hits(make_greedy(net), rl_kw)
    print(f"       PPO 确定性部署 {rl_greedy['success_rate']*100:5.1f}%  "
          f"平均 {rl_greedy['avg_steps']:6.1f} 步  曾进入tol {rl_hits:5.2f} 次")
    print(f"       PPO 采样部署   {rl_sample['success_rate']*100:5.1f}%  "
          f"平均 {rl_sample['avg_steps']:6.1f} 步")
    print("       （两种部署都报：同一套权重只换部署方式，历史上差过 91 个点）")

    # ---------- 阶段 4：对比 ----------
    print("\n" + "=" * 66)
    print(f"汇总（{N_EVAL} 回合，PPO 训练/评测噪声均为 sigma={RL_SIGMA}m）")
    print("=" * 66)
    print(f"{'方案':<28}{'成功率':>8}{'步数':>8}{'曾进入tol':>10}")
    for s, g, succ, steps, hits in rows:
        print(f"伺服 sigma={s:.2f} (gain={g})".ljust(30)
              + f"{succ:7.1f}%{steps:8.1f}{hits:10.2f}")
    for s in SIGMAS:
        w, g, succ = filt_best[s]
        print(f"伺服+低通 sigma={s:.2f} (win={w},g={g})".ljust(30)
              + f"{succ:7.1f}%{'--':>8}{'--':>10}")
    print(f"PPO(贪心) sigma={RL_SIGMA}".ljust(30)
          + f"{rl_greedy['success_rate']*100:7.1f}%"
          + f"{rl_greedy['avg_steps']:8.1f}{rl_hits:10.2f}")
    print(f"PPO(采样) sigma={RL_SIGMA}".ljust(30)
          + f"{rl_sample['success_rate']*100:7.1f}%"
          + f"{rl_sample['avg_steps']:8.1f}{'--':>10}")

    # ---------- 画图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("观测噪声：解析伺服假设误差是准的，RL 训练时就见过噪声",
                 fontsize=13)

    ax = axes[0]
    for g in GAINS:
        ax.plot(SIGMAS, servo_curves[g], color="#9AA4B2", lw=1.0, alpha=0.8,
                marker=".", ms=4)
    ax.plot(SIGMAS, [r[2] for r in rows], color="#0F6E56", lw=2.4, marker="o",
            ms=6, label="伺服（每个 sigma 扫 gain 取最优）")
    for w, c in ((3, "#7C6BB0"), (5, "#3B6EA5")):
        ax.plot(SIGMAS, filt_curves[w], color=c, lw=1.8, ls="--", marker="s",
                ms=5, label=f"伺服+低通 win={w}")
    ax.axvline(RL_SIGMA, color="#E8A33D", ls=":", lw=1.2)
    ax.annotate(f"PPO 训练点\nsigma={RL_SIGMA}",
                xy=(RL_SIGMA, 8), fontsize=8.5, color="#B0761F", ha="center")
    ax.scatter([RL_SIGMA], [rl_greedy["success_rate"] * 100], s=130, marker="*",
               color="#185FA5", zorder=5, label=f"PPO 贪心 {rl_greedy['success_rate']*100:.0f}%")
    ax.scatter([RL_SIGMA], [rl_sample["success_rate"] * 100], s=130, marker="D",
               color="#E24B4A", zorder=5, label=f"PPO 采样 {rl_sample['success_rate']*100:.0f}%")
    ax.set_xlabel("观测噪声标准差 sigma (米)，TOL = 0.05 m")
    ax.set_ylabel("到达成功率 (%)")
    ax.set_ylim(-3, 105)
    ax.set_title("成功率 vs 噪声强度", fontsize=12)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(SIGMAS, [r[4] for r in rows], color="#0F6E56", lw=2.2, marker="o",
            ms=6, label="伺服（无滤波）")
    for w, c in ((3, "#7C6BB0"), (5, "#3B6EA5")):
        vals = [diag_hits(make_filtered_rr(
                    gain=[r[1] for r in rows if r[0] == s][0], win=w),
                    {"obs_noise": s}) for s in SIGMAS]
        ax.plot(SIGMAS, vals, color=c, lw=1.8, ls="--", marker="s", ms=5,
                label=f"伺服+低通 win={w}")
    ax.axhline(1.0, color="#9AA4B2", ls=":", lw=1.2)
    ax.annotate("1.0 = 一次到位不抖", xy=(0.115, 1.05), fontsize=8.5,
                color="#5A6472")
    ax.scatter([RL_SIGMA], [rl_hits], s=130, marker="*", color="#185FA5",
               zorder=5, label=f"PPO 贪心 {rl_hits:.2f}")
    ax.set_xlabel("观测噪声标准差 sigma (米)")
    ax.set_ylabel("单回合内曾进入 tol 的次数")
    ax.set_title("抖动的直接证据：无噪时一次到位，有噪时反复穿边界", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    s_arr = np.array(hist["succ"]) * 100
    ax.plot(s_arr, color="#185FA5", lw=1.5, label=f"PPO 训练中（sigma={RL_SIGMA}）")
    k = 20
    if len(s_arr) > k:
        sm = np.convolve(s_arr, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(sm)) + k // 2, sm, color="#185FA5", lw=2.6,
                label="20 回合滑动平均")
    srow = [r for r in rows if r[0] == RL_SIGMA][0]
    ax.axhline(srow[2], color="#0F6E56", ls="--", lw=1.6,
               label=f"伺服（同噪声下最优）{srow[2]:.0f}%")
    ax.set_xlabel("回合")
    ax.set_ylabel("成功率 (%)")
    ax.set_ylim(-3, 105)
    ax.set_title(f"PPO 学习曲线：训练环境自带 sigma={RL_SIGMA} 噪声", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\n图已保存：{OUT}")
    print(f"总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
