"""
task_anomaly.py —— 验证实验六里那个没解释干净的反常

反常：同一个 progress 奖励，
    - 从随机初始化开始训        -> 86.0%
    - 从 BC 权重出发微调        -> 50.0%（比 BC 基线 76% 还低 26 个点）

我在 README 里给的假设是「风格切换没走完」：
    BC 学到的是伺服的"平滑风格"（小步、稳），
    progress 奖励的最优解是"每步把距离压到最低"的激进风格（大步、快），
    PPO 的 clip=0.2 限制单步更新幅度，150 轮不足以完成这次切换，
    策略卡在两种风格中间 —— 两边都不像。

假设要能被证伪才有价值。所以设计两个直接放开约束的对照组：

    B 加长：clip=0.2，iters 150 -> 400   （给它足够时间走完切换）
    C 放宽：clip=0.4，iters 150          （允许每步跨得更大）

如果假设成立：B 或 C 至少有一个应该明显回升，向 86% 靠拢。
如果两个都不动：假设被排除，说明根因在别处（那就得换方向查，不能再猜）。

顺带记录三个可观测的"风格指标"：平均步数、平均命令幅度、熵的走向。
"""

import copy
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bc_ppo import pretrain_critic
from controller import evaluate
from task_sim2real import action_stats, collect_demos, train_bc
from train_ppo import make_greedy, train

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "anomaly_result.png")

PROG = {"reward_mode": "progress"}
N_EVAL = 50
EVAL_SEED = 500

# 实验六已测得的三个数，这里当基线引用，不重复跑
BASE_BC = 76.0        # BC 基线
BASE_FT = 50.0        # BC + progress 微调（clip=0.2, 150 轮）
BASE_SCRATCH = 86.0   # 从零开始 + progress


def run(tag, clip, iters, base_net, bc_succ, lr=5e-5, ent_coef=0.005, seed=0):
    """跑一次「BC 预热 + progress 微调」，返回 (成功率, 步数, hist, 幅度, 停滞率)。"""
    print(f"\n--- {tag} ---")
    net = copy.deepcopy(base_net)

    with torch.no_grad():
        net.log_std.fill_(0.0)
    net = pretrain_critic(net, env_kw=PROG)

    net2, hist, el, best, best_succ = train(
        iters=iters, lr=lr, ent_coef=ent_coef, clip=clip,
        init_weights=net.state_dict(), env_kw=PROG, seed=seed)
    if best is not None:
        net2.load_state_dict(best)
    r = evaluate(make_greedy(net2), episodes=N_EVAL, seed=EVAL_SEED, env_kw=PROG)
    mag, stall = action_stats(make_greedy(net2), env_kw=PROG)
    print(f"  -> 微调后 {r['success_rate']*100:.1f}%  "
          f"平均 {r['avg_steps']:.1f} 步  命令幅度 {mag:.3f}  "
          f"用时 {el:.0f}s  （最佳贪心快照 {best_succ*100:.0f}%）")
    return {
        "tag": tag, "succ": r["success_rate"] * 100, "steps": r["avg_steps"],
        "hist": hist, "mag": mag, "stall": stall, "bc": bc_succ,
    }


def main():
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)

    print("=" * 70)
    print("验证：BC + progress 微调的退化，是不是「风格切换没走完」")
    print(f"  基线（实验六已测）：从零 {BASE_SCRATCH}%  /  BC {BASE_BC}%  /  "
          f"BC 微调 {BASE_FT}%")
    print("=" * 70)

    # BC 只训一次，两个对照组共用同一个起点
    X, Y, demo_succ = collect_demos(n_traj=300, env_kw=None, gain=1.0, seed=0)
    print(f"\n演示 {len(X)} 条（专家成功率 {demo_succ*100:.0f}%）")
    base_net = train_bc(X, Y, seed=0)
    bc_eval = evaluate(make_greedy(base_net), episodes=N_EVAL, seed=EVAL_SEED, env_kw=PROG)
    bc_succ = bc_eval["success_rate"] * 100
    bc_mag, _ = action_stats(make_greedy(base_net), env_kw=PROG)
    print(f"BC 基线 {bc_succ:.1f}%  平均 {bc_eval['avg_steps']:.1f} 步  "
          f"命令幅度 {bc_mag:.3f}")

    # B：加长训练，给足时间完成切换
    B = run("B 加长 clip=0.2 / 400轮", clip=0.2, iters=400,
            base_net=base_net, bc_succ=bc_succ)
    # C：放宽 clip，允许单步跨更大
    C = run("C 放宽 clip=0.4 / 150轮", clip=0.4, iters=150,
            base_net=base_net, bc_succ=bc_succ)

    print("\n" + "=" * 70)
    print("验证结果")
    print("=" * 70)
    rows = [
        ("从零开始（冷启动）", BASE_SCRATCH, 52.0, "—"),
        ("BC 基线", B["bc"], bc_eval["avg_steps"], f"{bc_mag:.3f}"),
        ("BC 微调 clip=0.2 / 150 轮（原）", BASE_FT, 121.8, "—"),
        (B["tag"], B["succ"], B["steps"], f"{B['mag']:.3f}"),
        (C["tag"], C["succ"], C["steps"], f"{C['mag']:.3f}"),
    ]
    print(f"  {'配置':34s} {'成功率':>7s} {'步数':>7s} {'命令幅度':>8s}")
    for n, s, st, m in rows:
        print(f"  {n:34s} {s:6.1f}% {st:7.1f} {m:>8s}")

    best_alt = max(B["succ"], C["succ"])
    if best_alt >= BASE_BC - 3:
        verdict = ("假设成立：放开约束后回升到 BC 基线附近，说明原来确实卡在切换途中。"
                   if best_alt < BASE_SCRATCH - 5 else
                   "假设成立且更进一步：放开约束后逼近从零训练的水平。")
    elif best_alt > BASE_FT + 8:
        verdict = "部分成立：有回升但没回到 BC 基线，风格冲突只是原因之一。"
    else:
        verdict = ("假设被排除：加长和放宽都没让它回升，"
                   "「风格切换没走完」不是主因，得换方向查。")
    print(f"\n  判定：{verdict}")

    # ---- 出图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9))
    fig.suptitle("BC + progress 为什么退化到 50%：放开更新约束能不能救回来",
                 fontsize=13)

    ax = axes[0]
    for res, color in ((B, "#185FA5"), (C, "#854F0B")):
        s = np.array(res["hist"]["succ"]) * 100
        ax.plot(s, color=color, lw=1.2, alpha=0.85,
                label=f"{res['tag']}（{res['succ']:.0f}%）")
    ax.axhline(BASE_BC, color="#EF9F27", ls="--", lw=1.4, label=f"BC 基线 {BASE_BC:.0f}%")
    ax.axhline(BASE_FT, color="#E24B4A", ls=":", lw=1.5, label=f"原配置 {BASE_FT:.0f}%")
    ax.axhline(BASE_SCRATCH, color="#0F6E56", ls="-.", lw=1.4,
               label=f"从零训练 {BASE_SCRATCH:.0f}%")
    ax.set_xlabel("微调回合")
    ax.set_ylabel("到达成功率 (%)  [滚动 20 回合]")
    ax.set_ylim(-3, 105)
    ax.set_title("放开约束后，曲线有没有往上走", fontsize=12)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    names = ["从零\n(冷启动)", "BC", "BC微调\n0.2/150", "BC微调\n0.2/400", "BC微调\n0.4/150"]
    vals = [BASE_SCRATCH, BASE_BC, BASE_FT, B["succ"], C["succ"]]
    colors = ["#0F6E56", "#EF9F27", "#E24B4A", "#185FA5", "#854F0B"]
    bars = ax.bar(names, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("成功率 (%)")
    ax.set_title("放开约束能救回多少", fontsize=12)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\n图已保存：{OUT}")
    print(f"总用时 {time.time()-t0:.0f} 秒")

    return B, C, verdict


if __name__ == "__main__":
    main()
