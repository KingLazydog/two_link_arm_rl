"""
task_reward.py —— 冷启动 PPO 失败，到底是"任务太难"还是"奖励太笨"？

之前把冷启动 PPO 只有 2% 归因于任务性质：
    要从零学出"关节角 -> 末端方向"的映射、成功样本稀疏、奇异区梯度病态。

但仔细看奖励函数，还有第三种解释：

    dist 模式：r = -0.5·d - 0.01        "你现在离目标有多远"
    progress  ：r = 10·(d_prev - d) - 0.01   "你这一步走对没有"

在 dist 模式下，只要没够到目标，每一步的奖励几乎一样（都是差不多的负数）。
advantage 只能靠"最后成没成功"这一个稀疏事件来区分 —— 那 PPO 学到的其实是噪声。
progress 把同一个目标拆成了每一步都能立刻拿到的反馈，信号密度完全不同。

而且 progress 恰好是 potential-based shaping 的标准形式：
    Φ(s) = -d(s)，  F(s,s') = γ·Φ(s') - Φ(s)
Ng 等人 1999 年证明过：**这种形式的奖励塑形不改变最优策略**，
只是把学习信号摊到了每一步。所以拿它做对照非常干净 ——
如果它能救起冷启动，"任务性质不匹配"这个结论就得改。

三个实验，同一个环境、同一套超参、同一个 seed，只改奖励：
    A 冷启动 PPO + dist        （复现 2%）
    B 冷启动 PPO + progress    （看能不能救起来）
    C BC 预热 + progress 微调  （和 BC + dist 的 78% 比）
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bc_ppo import pretrain_critic
from controller import evaluate, make_resolved_rate
from task_sim2real import collect_demos, train_bc
from train_ppo import make_greedy, train

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reward_result.png")

DIST = {"reward_mode": "dist"}
PROG = {"reward_mode": "progress"}
N_EVAL = 50
EVAL_SEED = 500


def main():
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)

    print("=" * 70)
    print("奖励改造实验：dist（-0.5d） vs progress（10·Δd）")
    print("  同一环境、同一超参、同一 seed，唯一变量是奖励函数")
    print("=" * 70)

    # ---- 基线：伺服 ----
    servo = evaluate(make_resolved_rate(gain=1.0, lam=0.05),
                     episodes=N_EVAL, seed=EVAL_SEED, env_kw=DIST)
    print(f"\n[0] 伺服（gain=1.0）成功率 {servo['success_rate']*100:.1f}%  "
          f"平均 {servo['avg_steps']:.1f} 步")

    # ---- A: 冷启动 + dist ----
    print("\n[A] 冷启动 PPO + dist 奖励（复现旧结论）")
    netA, histA, elA, bestA, bestA_s = train(iters=300, env_kw=DIST)
    if bestA is not None:
        netA.load_state_dict(bestA)
    rA = evaluate(make_greedy(netA), episodes=N_EVAL, seed=EVAL_SEED, env_kw=DIST)
    print(f"  -> 成功率 {rA['success_rate']*100:.1f}%  平均 {rA['avg_steps']:.1f} 步  "
          f"用时 {elA:.0f}s")

    # ---- B: 冷启动 + progress ----
    print("\n[B] 冷启动 PPO + progress 奖励（只换奖励）")
    netB, histB, elB, bestB, bestB_s = train(iters=300, env_kw=PROG)
    if bestB is not None:
        netB.load_state_dict(bestB)
    rB = evaluate(make_greedy(netB), episodes=N_EVAL, seed=EVAL_SEED, env_kw=DIST)
    print(f"  -> 成功率 {rB['success_rate']*100:.1f}%  平均 {rB['avg_steps']:.1f} 步  "
          f"用时 {elB:.0f}s")

    # ---- C: BC 预热 + progress 微调 ----
    print("\n[C] BC 预热 + progress 奖励微调")
    X, Y, demo_succ = collect_demos(n_traj=300, env_kw=None, gain=1.0)
    print(f"  {len(X)} 条演示样本（专家成功率 {demo_succ*100:.0f}%）")
    bc_net = train_bc(X, Y)
    bc_eval = evaluate(make_greedy(bc_net), episodes=N_EVAL, seed=EVAL_SEED, env_kw=DIST)
    print(f"  BC 基线 {bc_eval['success_rate']*100:.1f}%  平均 {bc_eval['avg_steps']:.1f} 步")

    with torch.no_grad():
        bc_net.log_std.fill_(0.0)
    bc_net = pretrain_critic(bc_net, env_kw=PROG)
    netC, histC, elC, bestC, bestC_s = train(
        iters=150, lr=5e-5, ent_coef=0.005,
        init_weights=bc_net.state_dict(), env_kw=PROG)
    if bestC is not None:
        netC.load_state_dict(bestC)
    rC = evaluate(make_greedy(netC), episodes=N_EVAL, seed=EVAL_SEED, env_kw=DIST)
    print(f"  -> 成功率 {rC['success_rate']*100:.1f}%  平均 {rC['avg_steps']:.1f} 步  "
          f"用时 {elC:.0f}s")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("结论（50 个随机姿态，seed=500）")
    print("=" * 70)
    table = [
        ("PPO 冷启动 + dist 奖励", rA["success_rate"] * 100, rA["avg_steps"]),
        ("PPO 冷启动 + progress 奖励", rB["success_rate"] * 100, rB["avg_steps"]),
        ("BC 基线", bc_eval["success_rate"] * 100, bc_eval["avg_steps"]),
        ("BC + dist 微调（旧）", 78.0, 59.5),
        ("BC + progress 微调", rC["success_rate"] * 100, rC["avg_steps"]),
        ("伺服（上限）", servo["success_rate"] * 100, servo["avg_steps"]),
    ]
    for name, s, st in table:
        print(f"  {name:26s} {s:5.1f}%   平均 {st:5.1f} 步")

    # ---- 出图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9))
    fig.suptitle("只换奖励函数：冷启动 PPO 从「学不会」到「学得会」", fontsize=13)

    ax = axes[0]
    ax.plot(np.array(histA["succ"]) * 100, color="#E24B4A", lw=1.2, alpha=0.85,
            label=f"dist 奖励（{rA['success_rate']*100:.0f}%）")
    ax.plot(np.array(histB["succ"]) * 100, color="#185FA5", lw=1.2, alpha=0.85,
            label=f"progress 奖励（{rB['success_rate']*100:.0f}%）")
    ax.axhline(servo["success_rate"] * 100, color="#0F6E56", ls="--", lw=1.4,
               label=f"伺服 {servo['success_rate']*100:.0f}%")
    ax.set_xlabel("训练回合")
    ax.set_ylabel("到达成功率 (%)  [滚动 20 回合]")
    ax.set_ylim(-3, 105)
    ax.set_title("冷启动：同一环境同一超参，只改奖励", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    names = ["冷启动\ndist", "冷启动\nprogress", "BC", "BC+dist\n微调", "BC+progress\n微调", "伺服"]
    vals = [t[1] for t in table]
    colors = ["#E24B4A", "#185FA5", "#EF9F27", "#BA7FD0", "#0F6E56", "#9AA5B1"]
    bars = ax.bar(names, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("成功率 (%)")
    ax.set_title("奖励函数值多少个百分点", fontsize=12)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\n图已保存：{OUT}")
    print(f"总用时 {time.time()-t0:.0f} 秒")

    return rA, rB, rC


if __name__ == "__main__":
    main()
