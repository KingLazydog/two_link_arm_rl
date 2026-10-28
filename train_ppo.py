"""
train_ppo.py —— 用 PPO 训练一个"不知道运动学方程"的机械臂控制器

这个文件回答一个问题：
    我们已经有了解析 IK 和雅可比伺服（100% 成功率），为什么还要 RL？

答案不在这个任务上，而在这个任务之外：
    ik_analytic 需要知道杆长、需要知道结构是二连杆、需要推导闭式解。
    真实机器人有 6~7 个关节、有冗余、有奇异、有摩擦、有负载，
    没有闭式解，也没有准确模型。
    PPO 只接收 (观测, 奖励)，从头试错，不需要任何一条运动学公式。

训练完你会看到：它能接近伺服，但打不过伺服 —— 这是诚实的结论，
也是"什么时候该用 RL、什么时候不该"的判断依据。
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arm import fk
from controller import evaluate, make_resolved_rate
from env import MAX_STEPS, TwoLinkReachEnv

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_result.png")
SCALE = np.array([2.0, 2.0], dtype=np.float32)   # 只缩放 obs 里的 dx/dy


def norm_obs(o):
    o = np.asarray(o, dtype=np.float32).copy()
    o[4:6] /= SCALE
    return o


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(),
                         nn.Linear(h, h), nn.Tanh(),
                         nn.Linear(h, o))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=8, act_dim=2, h=64):
        super().__init__()
        self.actor = mlp(obs_dim, h, act_dim)
        self.critic = mlp(obs_dim, h, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))

    def _std(self):
        """
        标准差用 sigmoid 软映射到 [0.10, 0.60]。
        不能用 clamp —— clamp 在边界处梯度为零，log_std 一旦被推到边界
        就永久卡死，策略会退化成几乎确定性、再也探索不起来。
        （第一版就是死在这上面：熵恒等于 -1.767 一动不动。）
        """
        lo, hi = 0.10, 0.60
        return lo + (hi - lo) * torch.sigmoid(self.log_std)

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def dist(self, x):
        mu, v = self.forward(x)
        mu = torch.tanh(mu)                       # 均值压到动作范围内
        std = self._std().expand_as(mu)
        return torch.distributions.Normal(mu, std), v


def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """广义优势估计。dones[i] 表示这一步之后轨迹结束（终止或截断）。"""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in reversed(range(n)):
        next_v = 0.0 if t == n - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + np.asarray(values, dtype=np.float32)


def train(iters=350, horizon=2048, epochs=10, batch=64,
          gamma=0.99, lam=0.95, clip=0.2, lr=3e-4, vf_coef=0.5,
          ent_coef=0.01, seed=0, init_weights=None, env_kw=None,
          eval_sample=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = TwoLinkReachEnv(seed=seed, **(env_kw or {}))
    net = ActorCritic()
    if init_weights is not None:
        net.load_state_dict(init_weights)
        print("  已加载预热权重（BC），从学到的策略继续，而不是从随机开始")
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    hist = {"succ": [], "steps": [], "loss_p": [], "entropy": []}
    best_state, best_succ = None, -1.0
    obs, _ = env.reset()
    ep_ret, ep_len, ep_ok = 0.0, 0, []
    t0 = time.time()

    for it in range(iters):
        buf = {"s": [], "a": [], "lp": [], "r": [], "d": [], "v": []}
        ep_done = 0
        while ep_done < horizon:
            st = torch.from_numpy(norm_obs(obs)).unsqueeze(0)
            with torch.no_grad():
                dist, v = net.dist(st)
                a = dist.sample().clamp(-1.0, 1.0)      # 先裁剪
                lp = dist.log_prob(a).sum(-1)           # 再算 log_prob，两者必须一致
            a_env = a.numpy()[0]
            obs2, r, term, trunc, info = env.step(a_env)

            buf["s"].append(norm_obs(obs))
            buf["a"].append(a.numpy()[0])
            buf["lp"].append(lp.item())
            buf["r"].append(r)
            buf["v"].append(v.item())
            buf["d"].append(1.0 if (term or trunc) else 0.0)

            ep_ret += r
            ep_len += 1
            obs = obs2
            ep_done += 1

            if term or trunc:
                ep_ok.append(1.0 if term else 0.0)
                hist["succ"].append(np.mean(ep_ok[-20:]))
                hist["steps"].append(ep_len)
                obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

        adv, ret = gae(buf["r"], buf["v"], buf["d"], gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        S = torch.from_numpy(np.array(buf["s"], dtype=np.float32))
        A = torch.from_numpy(np.array(buf["a"], dtype=np.float32))
        LP = torch.from_numpy(np.array(buf["lp"], dtype=np.float32))
        ADV = torch.from_numpy(adv)
        RET = torch.from_numpy(ret)

        n = len(buf["r"])
        ent_acc, lp_acc = [], []
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                mu, v = net(S[idx])
                mu = torch.tanh(mu)
                std = net._std().expand_as(mu)
                dist = torch.distributions.Normal(mu, std)
                lp = dist.log_prob(A[idx]).sum(-1)
                ratio = torch.exp(lp - LP[idx])
                a1 = ratio * ADV[idx]
                a2 = torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[idx]
                loss_p = -torch.min(a1, a2).mean()
                loss_v = F.mse_loss(v.squeeze(-1), RET[idx])
                ent = dist.entropy().sum(-1).mean()
                loss = loss_p + vf_coef * loss_v - ent_coef * ent
                opt.zero_grad()
                loss.backward()                       # 丢掉 graph 不必要，ok
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                ent_acc.append(ent.item())
                lp_acc.append(loss_p.item())

        hist["loss_p"].append(np.mean(lp_acc))
        hist["entropy"].append(np.mean(ent_acc))

        # 记录训练过程中最好的一次（RL 曲线会震荡，最后一轮未必最好）
        if it % 10 == 0 or it == iters - 1:
            probe = make_sample(net, seed=900) if eval_sample else make_greedy(net)
            quick = evaluate(probe, episodes=20, seed=900, env_kw=env_kw)
            if quick["success_rate"] > best_succ:
                best_succ = quick["success_rate"]
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"  轮 {it:3d} 近20回合成功率 {hist['succ'][-1]*100:5.1f}%  "
                  f"贪心评测 {quick['success_rate']*100:5.1f}%  "
                  f"平均步数 {np.mean(hist['steps'][-20:]):5.1f}  "
                  f"熵 {hist['entropy'][-1]:.3f}  用时 {time.time()-t0:5.0f}s")

    return net, hist, time.time() - t0, best_state, best_succ


def make_greedy(net):
    """确定性部署：直接输出高斯均值。工程上最常规的部署方式。"""
    def policy(obs):
        with torch.no_grad():
            dist, _ = net.dist(torch.from_numpy(norm_obs(obs)).unsqueeze(0))
        return dist.mean.numpy()[0]
    return policy


def make_sample(net, seed=0):
    """
    随机部署：按策略分布采样，而不是取均值。

    为什么要有这个：训练时 PPO 就是靠采样探索的，而部署时通常改用均值。
    在「有静摩擦死区」的任务上，这两种部署方式的表现可以差到 0% 和 90% ——
    因为取均值会把命令拉回死区以内（推不动），采样噪声则时不时会顶过阈值。
    所以评测必须两种都报，只报一边会得出完全相反的结论。
    """
    rng = np.random.default_rng(seed)

    def policy(obs):
        with torch.no_grad():
            dist, _ = net.dist(torch.from_numpy(norm_obs(obs)).unsqueeze(0))
            mu = dist.mean.numpy()[0]
            std = dist.stddev.numpy()[0]
        return np.clip(mu + rng.normal(0.0, 1.0, size=mu.shape) * std, -1.0, 1.0)
    return policy


def main():
    print("=" * 62)
    print("PPO 训练开始：策略网络只看得到 (观测, 奖励)，不知道杆长、不知道雅可比")
    print("=" * 62)
    net, hist, elapsed, _, _ = train()
    print(f"\n训练结束，用时 {elapsed:.0f} 秒，参数量 "
          f"{sum(p.numel() for p in net.parameters())}")

    greedy = make_greedy(net)
    rl = evaluate(greedy, episodes=50, seed=500)
    servo = evaluate(make_resolved_rate(gain=4.0, lam=0.05), episodes=50, seed=500)
    rand = evaluate(lambda o: np.random.uniform(-1, 1, 2), episodes=50, seed=500)

    print("\n[评测] 50 个随机初始姿态 + 随机目标")
    print(f"  随机推        成功率 {rand['success_rate']*100:5.1f}%   平均 {rand['avg_steps']:6.1f} 步")
    print(f"  PPO 学出来的   成功率 {rl['success_rate']*100:5.1f}%   平均 {rl['avg_steps']:6.1f} 步")
    print(f"  雅可比伺服     成功率 {servo['success_rate']*100:5.1f}%   平均 {servo['avg_steps']:6.1f} 步")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    fig.suptitle("PPO 控制二连杆机械臂够目标：从 0 成功率爬到能用的水平", fontsize=13)

    ax = axes[0]
    s = np.array(hist["succ"]) * 100
    ax.plot(s, color="#185FA5", lw=1.6, label="PPO（每回合滚动 20 次均值）")
    ax.axhline(servo["success_rate"] * 100, color="#0F6E56", ls="--", lw=1.5,
               label=f"雅可比伺服 {servo['success_rate']*100:.0f}%")
    ax.axhline(rand["success_rate"] * 100, color="#E24B4A", ls=":", lw=1.5,
               label=f"随机推 {rand['success_rate']*100:.0f}%")
    ax.set_xlabel("回合")
    ax.set_ylabel("到达成功率 (%)")
    ax.set_ylim(-3, 105)
    ax.set_title("学习曲线：没人告诉它任何运动学公式", fontsize=12)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(hist["steps"], color="#EF9F27", lw=1.0, alpha=0.55, label="每回合步数")
    k = 20
    smooth = np.convolve(hist["steps"], np.ones(k) / k, mode="valid")
    ax.plot(np.arange(len(smooth)) + k // 2, smooth, color="#185FA5", lw=2,
            label="20 回合滑动平均")
    ax.axhline(servo["avg_steps"], color="#0F6E56", ls="--", lw=1.5,
               label=f"伺服 {servo['avg_steps']:.0f} 步")
    ax.set_xlabel("回合")
    ax.set_ylabel("到达所需步数")
    ax.set_title("步数：越学越果断，但比解析解还是慢", fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\n图已保存：{OUT}")


if __name__ == "__main__":
    main()
