"""
bc_ppo.py —— 模仿学习预热 + 强化学习微调

为什么要有这个文件：
    纯 PPO 冷启动在这个任务上失败（成功率 2%，见 README 的失败分析）。
    原因是它要从零学出"关节角 -> 末端方向"的映射，等于在神经网络里重新发明雅可比，
    而且成功样本极稀疏，几乎拿不到正向信号。

    换个思路：先让专家（雅可比伺服）演示一遍，用行为克隆把策略网络"教会走路"，
    再让 PPO 在这个基础上微调。

    这就是真实机器人 RL 的标准做法 —— 也说明"模仿学习"和"强化学习"
    不是两条对立路线，而是一条流水线的前后两段。
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from controller import evaluate, make_resolved_rate
from env import MAX_STEPS, TwoLinkReachEnv
from train_ppo import ActorCritic, make_greedy, norm_obs, train

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "bc_policy.pt")
OUT = os.path.join(HERE, "bc_ppo_result.png")


# ---------------- 1. 专家演示 ----------------
def collect_demos(n_traj=300, seed=0):
    """用阻尼伺服跑 n_traj 条轨迹，把每一步的 (观测, 动作) 都存下来。"""
    expert = make_resolved_rate(gain=4.0, lam=0.05)
    env = TwoLinkReachEnv(seed=seed)
    X, Y, succ = [], [], 0
    for _ in range(n_traj):
        obs, _ = env.reset()
        for _ in range(MAX_STEPS):
            a = expert(obs)
            X.append(norm_obs(obs))
            Y.append(a)
            obs, r, term, trunc, info = env.step(a)
            if term:
                succ += 1
                break
            if trunc:
                break
    return (np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32),
            succ / n_traj)


# ---------------- 2. 行为克隆 ----------------
def train_bc(X, Y, epochs=150, lr=3e-4, batch=128, seed=0, verbose=True):
    """
    监督学习：给定专家看到的观测，预测专家会做的动作。
    用高斯负对数似然而不是 MSE —— 这样 log_std 也会被学出来，
    后面 PPO 微调时能直接沿用同一套网络，不用改结构。
    """
    torch.manual_seed(seed)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    S = torch.from_numpy(X)
    A = torch.from_numpy(Y)
    n = len(X)

    for e in range(epochs):
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            mu, _ = net(S[idx])
            mu = torch.tanh(mu)
            std = net._std().expand_as(mu)
            dist = torch.distributions.Normal(mu, std)
            loss = -dist.log_prob(A[idx]).sum(-1).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            cnt += 1
        if verbose and (e % 30 == 0 or e == epochs - 1):
            print(f"  BC 轮 {e:3d}  负对数似然 {tot/cnt:.3f}  "
                  f"std={net._std().mean().item():.3f}")
    return net


def pretrain_critic(net, rounds=4, traj_per_round=40, epochs=120,
                    lr=1e-3, gamma=0.99, env_kw=None):
    """
    BC 只训了 actor，critic 还是随机初始化。
    直接开始 PPO 的话，前几十轮的 advantage 全是噪声 ——
    critic 说"这个状态能活很久"，实际不能，梯度就把好策略往沟里带。

    所以先用 BC 策略跑一遍，把 critic 回归到真实的折扣回报上。
    这一步叫策略评估（policy evaluation），是 actor-critic 的标准预处理。
    """
    policy = make_greedy(net)
    opt = torch.optim.Adam(net.critic.parameters(), lr=lr)
    env = TwoLinkReachEnv(seed=1234, **(env_kw or {}))

    for rd in range(rounds):
        S, G = [], []
        for _ in range(traj_per_round):
            obs, _ = env.reset()
            ss, rr = [], []
            for _ in range(MAX_STEPS):
                a = policy(obs)
                ss.append(norm_obs(obs))
                obs, r, term, trunc, info = env.step(a)
                rr.append(r)
                if term or trunc:
                    break
            g, rets = 0.0, []
            for r in reversed(rr):
                g = r + gamma * g
                rets.append(g)
            rets.reverse()
            S.extend(ss)
            G.extend(rets)

        S = torch.from_numpy(np.asarray(S, dtype=np.float32))
        G = torch.from_numpy(np.asarray(G, dtype=np.float32))
        n = len(S)
        last = 0.0
        for e in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                v = net.critic(S[idx]).squeeze(-1)
                loss = torch.nn.functional.mse_loss(v, G[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            last = loss.item()
        print(f"  critic 评估第 {rd+1} 轮：{n} 样本，MSE {last:.3f}")
    return net


def main():
    t0 = time.time()
    print("=" * 66)
    print("模仿学习预热 + PPO 微调")
    print("=" * 66)

    # ---- 演示数据 ----
    print("\n[1] 收集专家演示（阻尼伺服，gain=4.0, lambda=0.05）")
    X, Y, demo_succ = collect_demos(n_traj=300)
    print(f"  {len(X)} 条 (观测, 动作) 样本，专家成功率 {demo_succ*100:.0f}%")

    # ---- 行为克隆 ----
    print("\n[2] 行为克隆")
    bc_net = train_bc(X, Y)
    bc_eval = evaluate(make_greedy(bc_net), episodes=50, seed=500)
    print(f"  BC 策略成功率 {bc_eval['success_rate']*100:.1f}%，"
          f"平均 {bc_eval['avg_steps']:.1f} 步")

    torch.save(bc_net.state_dict(), CKPT)
    print(f"  权重已存：{CKPT}")

    # BC 会把 std 一路压到下界（负对数似然天然偏爱小方差），
    # 那样微调时几乎没有探索空间。这里手动把 std 拨回 0.35，
    # 均值不动，只把"犹豫度"还给策略。
    with torch.no_grad():
        bc_net.log_std.fill_(0.0)          # sigmoid(0)=0.5 -> std = 0.35
    print(f"  微调前把 std 从下界拨回 {bc_net._std().mean().item():.2f}（保留探索）")

    print("\n[3] 预训练 critic（BC 从没训过 value 网络，直接用会让 advantage 全是噪声）")
    bc_net = pretrain_critic(bc_net)

    # ---- PPO 微调 ----
    print("\n[4] PPO 微调（从 BC 权重出发，不是从随机出发）")
    net, hist, el, best_state, best_succ = train(
        iters=150, lr=5e-5, ent_coef=0.005,
        init_weights=bc_net.state_dict())
    if best_state is not None:
        net.load_state_dict(best_state)
    rl_eval = evaluate(make_greedy(net), episodes=50, seed=500)
    print(f"\n  微调后成功率 {rl_eval['success_rate']*100:.1f}%，"
          f"平均 {rl_eval['avg_steps']:.1f} 步  用时 {el:.0f}s")

    # ---- 对照 ----
    servo = evaluate(make_resolved_rate(gain=4.0, lam=0.05), episodes=50, seed=500)
    print("\n[5] 四方对照（同一批 50 个随机姿态）")
    rows = [("PPO 冷启动", 2.0, 196.2),
            ("BC 模仿", bc_eval["success_rate"] * 100, bc_eval["avg_steps"]),
            ("BC + PPO 微调", rl_eval["success_rate"] * 100, rl_eval["avg_steps"]),
            ("伺服（专家）", servo["success_rate"] * 100, servo["avg_steps"])]
    for name, s, st in rows:
        print(f"  {name:14s} 成功率 {s:5.1f}%   平均 {st:6.1f} 步")

    # ---- 出图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
    fig.suptitle("模仿学习预热 + 强化学习微调：把 2% 拉回可用区间", fontsize=13)

    ax = axes[0]
    s = np.array(hist["succ"]) * 100
    ax.plot(s, color="#185FA5", lw=1.5, label="PPO 微调（滚动 20 回合均值）")
    ax.axhline(bc_eval["success_rate"] * 100, color="#854F0B", ls="--", lw=1.5,
               label=f"BC 起点 {bc_eval['success_rate']*100:.0f}%")
    ax.axhline(servo["success_rate"] * 100, color="#0F6E56", ls=":", lw=1.5,
               label=f"专家（伺服）{servo['success_rate']*100:.0f}%")
    ax.set_ylim(-3, 105)
    ax.set_xlabel("微调回合")
    ax.set_ylabel("到达成功率 (%)")
    ax.set_title("微调曲线：起点就是 BC，不用再从零爬", fontsize=12)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["#E24B4A", "#EF9F27", "#185FA5", "#0F6E56"]
    bars = ax.bar(names, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("到达成功率 (%)")
    ax.set_title("冷启动 vs 模仿 vs 微调：差别就是这条流水线", fontsize=12)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\n图已保存：{OUT}")
    print(f"总用时 {time.time()-t0:.0f} 秒")


if __name__ == "__main__":
    main()
