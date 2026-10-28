"""
sweep.py —— 增益 × 延迟的二维扫描

这个脚本是为了一次翻车写的。

我原本的假设是"给动作加延迟会让伺服失效"，结果恰恰相反：延迟让成功率变高了。
原因不在延迟，在增益 —— `gain=4.0` 太大，控制器一直在过冲、在目标附近来回抖，
而延迟起了低通滤波的作用，恰好把震荡压住。

于是把两个变量一起扫一遍。这张表说明两件事：

1. **调参不是可选项。** gain 从 4.0 调到 1.0，成功率从 84% 直接到 98%。
   我拿 84% 当了一整晚的"专家上限"，所有对比结论因此失真。
2. **单变量思维会骗人。** 只改延迟、不改增益，会得出"延迟有用"的错误结论；
   把两个变量放在一起看，才发现延迟的作用只是补偿了另一个参数的错。
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controller import evaluate, make_resolved_rate

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep.png")

GAINS = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
DELAYS = [0, 1, 2, 3, 4, 6]
EPISODES = 40
SEED = 500


def main():
    succ = np.zeros((len(GAINS), len(DELAYS)))
    steps = np.zeros_like(succ)

    print("扫描 gain × delay（每个格子 " + str(EPISODES) + " 回合）")
    for i, g in enumerate(GAINS):
        for j, d in enumerate(DELAYS):
            r = evaluate(make_resolved_rate(gain=g, lam=0.05),
                         episodes=EPISODES, seed=SEED,
                         env_kw={"action_delay": d})
            succ[i, j] = r["success_rate"] * 100
            steps[i, j] = r["avg_steps"]
        print(f"  gain={g:<4} " + "  ".join(f"{v:5.1f}%" for v in succ[i]))

    # 成功率优先，步数少次之 —— 否则 gain=0.5/delay=1 那种"100% 但走 126 步"的会占便宜
    score = succ * 1000.0 - steps
    bi, bj = np.unravel_index(np.argmax(score), score.shape)
    print(f"\n最优点：gain={GAINS[bi]}  delay={DELAYS[bj]}  "
          f"成功率 {succ[bi, bj]:.1f}%  平均 {steps[bi, bj]:.1f} 步")
    i4 = GAINS.index(4.0)
    print(f"  gain=4.0 无延迟时只有 {succ[i4, 0]:.1f}% —— 这就是被我当了一整晚的专家基准")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle("增益 × 延迟扫描：一个没调好的基准，会让所有对比失真", fontsize=13)

    # 左：热力图
    ax = axes[0]
    im = ax.imshow(succ, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(DELAYS)))
    ax.set_xticklabels(DELAYS)
    ax.set_yticks(range(len(GAINS)))
    ax.set_yticklabels([f"{g}" for g in GAINS])
    ax.set_xlabel("动作延迟（仿真步，1 步 = 0.05 s）")
    ax.set_ylabel("伺服增益 gain")
    for i in range(len(GAINS)):
        for j in range(len(DELAYS)):
            v = succ[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 55 else "#2C2C2A")
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor="#E24B4A", lw=2.2))
    ax.set_title("到达成功率（%）：红框是最优点", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02)

    # 右：沿 gain 的曲线
    ax = axes[1]
    for d, c, ls in [(0, "#185FA5", "-"), (3, "#EF9F27", "--")]:
        j = DELAYS.index(d)
        ax.plot(GAINS, succ[:, j], ls, color=c, lw=2, marker="o", ms=5,
                label=f"delay={d}")
    ax.set_xscale("log")
    ax.set_xticks(GAINS)
    ax.set_xticklabels([str(g) for g in GAINS], fontsize=8.5)
    ax.set_xlabel("伺服增益 gain（对数轴）")
    ax.set_ylabel("到达成功率 (%)")
    ax.set_ylim(0, 105)
    ax.axvline(4.0, color="#E24B4A", ls=":", lw=1.5)
    ax.annotate("我最初随手取的 4.0", xy=(4.0, 30), xytext=(1.6, 32),
                fontsize=8.5, color="#A32D2D")
    ax.annotate("调到 1.0 就是 98%", xy=(1.0, 98), xytext=(0.34, 72),
                fontsize=8.5, color="#0F6E56")
    ax.set_title("延迟只在增益偏高时救场，低增益处没有影响", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\n图已保存：{OUT}")


if __name__ == "__main__":
    main()
