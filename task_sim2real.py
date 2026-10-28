"""
task_sim2real.py —— 让专家真正失效：关节摩擦死区 + Sim2Real 迁移

前面两个"让专家失效"的尝试都失败了：
    - 加关节限位 -> 反而屏蔽了奇异点，专家从 84% 涨到 100%
    - 加执行延迟 -> 反而当了低通滤波，压住了震荡，专家也变好

这两个失败有一个共同点：它们都是"能被副作用意外补偿"的扰动。
这次换一个不会被补偿的东西 —— 关节静摩擦死区。

    命令太小 -> 关节纹丝不动（推力顶不过静摩擦）
    命令够大 -> 按超出部分线性输出

雅可比伺服在误差小时算出的关节速度本来就小，
正好整段落进死区，于是它会在离目标几厘米的地方永远停住：
推不动，也走不完。这不是震荡，不是超调，是**结构性推不动**。

实验设计刻意做成 Sim2Real 的标准形态：

    1. 在【理想模型】里（deadzone=0）用伺服采演示 -> 行为克隆 -> BC 策略
       这对应"在仿真器里训好、准备上真机"的经典处境
    2. 把 BC 直接部署到【真实环境】（deadzone=0.25）-> 看 gap 有多大
    3. 同一个真实环境里，把伺服的 gain 从 0.5 扫到 128，看它自己能救回多少
    4. 在真实环境里用 PPO 微调 BC -> 看 RL 能不能从交互中学出死区补偿

第 3 步是上次翻车换来的规矩：基准必须先扫超参校准，不许拿随手填的参数当上限。

另外这一版专门加了"公平对照"：
    RL 部署时是**采样**动作的（带随机性），而传统控制器是确定性的。
    只比"RL 采样 vs 伺服确定"不公平 —— 所以给伺服和 BC 也加上同样强度的
    随机抖动（控制工程里叫 dither，本来就是突破静摩擦的标准手段），
    三方都允许抖，再看谁更好。
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
from train_ppo import ActorCritic, make_greedy, make_sample, norm_obs, train
from bc_ppo import pretrain_critic

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sim2real_result.png")
CKPT = os.path.join(HERE, "sim2real_policy.pt")

DEADZONE = 0.25          # 真实环境的静摩擦死区
DITHER = 0.30            # 给传统控制器的抖动强度
SIM = {}                 # 理想模型：没有死区
REAL = {"deadzone": DEADZONE}
N_EVAL = 50
EVAL_SEED = 500


# ---------------- 1. 在理想模型里采演示 ----------------
def collect_demos(n_traj=300, seed=0, env_kw=None, gain=1.0):
    expert = make_resolved_rate(gain=gain, lam=0.05)
    env = TwoLinkReachEnv(seed=seed, **(env_kw or {}))
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


def train_bc(X, Y, epochs=150, lr=3e-4, batch=128, seed=0):
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
        if e % 50 == 0 or e == epochs - 1:
            print(f"  BC 轮 {e:3d}  负对数似然 {tot/cnt:.3f}")
    return net


# ---------------- 2. 工具 ----------------
def add_dither(policy, sigma=DITHER, seed=0):
    """给任意确定性策略加随机抖动（dither）—— 突破静摩擦的标准工程手段。"""
    rng = np.random.default_rng(seed)

    def wrapped(obs):
        a = np.asarray(policy(obs), dtype=float)
        return np.clip(a + rng.normal(0.0, sigma, size=a.shape), -1.0, 1.0)
    return wrapped


def eval_avg(factory, reps=3, env_kw=None, episodes=N_EVAL, seed=EVAL_SEED):
    """
    带随机性的策略要多跑几遍取平均，否则一次结果可能只是运气。
    factory(i) 返回一个策略，i 是重复编号。
    """
    rs = [evaluate(factory(i), episodes=episodes, seed=seed, env_kw=env_kw)
          for i in range(reps)]
    return {
        "success_rate": float(np.mean([r["success_rate"] for r in rs])),
        "avg_steps": float(np.mean([r["avg_steps"] for r in rs])),
        "avg_dist": float(np.mean([r["avg_dist"] for r in rs])),
    }


def action_stats(policy, env_kw=None, episodes=30, seed=700):
    """统计命令幅度与"被死区吃掉的步数占比" —— 死区补偿的直接证据。"""
    env = TwoLinkReachEnv(seed=seed, **(env_kw or {}))
    mags, stalls = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        ms = []
        for _ in range(MAX_STEPS):
            a = np.clip(policy(obs), -1.0, 1.0)
            ms.append(np.abs(a).mean())
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        mags.append(np.mean(ms))
        # stalled 记的是"关节-步"数（两个关节各算一次），所以要除以 2·steps
        stalls.append(env.stalled / (2.0 * max(env.steps, 1)))
    return float(np.mean(mags)), float(np.mean(stalls))


def main():
    t0 = time.time()
    print("=" * 72)
    print("Sim2Real：在理想模型里学，部署到有摩擦死区的真实环境")
    print(f"  理想模型 deadzone=0.00    真实环境 deadzone={DEADZONE:.2f}")
    print("=" * 72)

    # ---- [1] 伺服在理想模型里 ----
    print("\n[1] 伺服在【理想模型】里的水平（这就是仿真的成绩单）")
    servo_sim = evaluate(make_resolved_rate(gain=1.0, lam=0.05),
                         episodes=N_EVAL, seed=EVAL_SEED, env_kw=SIM)
    print(f"  伺服（仿真）成功率 {servo_sim['success_rate']*100:5.1f}%  "
          f"平均 {servo_sim['avg_steps']:5.1f} 步  残距 {servo_sim['avg_dist']*100:.1f} cm")

    # ---- [2] 伺服在真实环境里扫 gain ----
    print("\n[2] 伺服在【真实环境】里：先把 gain 扫到最好（基准必须校准）")
    gains = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    rows = []
    for g in gains:
        r = evaluate(make_resolved_rate(gain=g, lam=0.05),
                     episodes=N_EVAL, seed=EVAL_SEED, env_kw=REAL)
        rows.append((g, r["success_rate"] * 100, r["avg_steps"], r["avg_dist"]))
        print(f"  gain={g:6.1f}  成功率 {r['success_rate']*100:5.1f}%  "
              f"平均 {r['avg_steps']:5.1f} 步  残距 {r['avg_dist']*100:.1f} cm")
    best_row = max(rows, key=lambda x: (x[1], -x[2]))
    best_g = best_row[0]
    print(f"  -> 伺服在真机最好 {best_row[1]:.1f}%（gain={best_g}），"
          f"比仿真里的 {servo_sim['success_rate']*100:.0f}% 掉了 "
          f"{servo_sim['success_rate']*100-best_row[1]:.0f} 个点")

    servo_best = make_resolved_rate(gain=best_g, lam=0.05)
    servo_dith = eval_avg(lambda i: add_dither(servo_best, DITHER, seed=i), env_kw=REAL)
    print(f"  再给它加 dither 抖动（sigma={DITHER}）："
          f"{servo_dith['success_rate']*100:.1f}%  <-- 让传统方法也用上同样的工具")

    # ---- [3] 演示 + BC ----
    print("\n[3] 在【理想模型】里采演示并做行为克隆（仿真里训练）")
    X, Y, demo_succ = collect_demos(n_traj=300, env_kw=SIM, gain=1.0)
    print(f"  {len(X)} 条演示样本，专家在仿真里成功率 {demo_succ*100:.0f}%")
    bc_net = train_bc(X, Y)

    bc_sim = evaluate(make_greedy(bc_net), episodes=N_EVAL, seed=EVAL_SEED, env_kw=SIM)
    bc_real = evaluate(make_greedy(bc_net), episodes=N_EVAL, seed=EVAL_SEED, env_kw=REAL)
    bc_dith = eval_avg(lambda i: add_dither(make_greedy(bc_net), DITHER, seed=i),
                       env_kw=REAL)
    print(f"  BC 在仿真里   {bc_sim['success_rate']*100:5.1f}%  （仿真自测，看着还行）")
    print(f"  BC 直接上真机 {bc_real['success_rate']*100:5.1f}%  "
          f"残距 {bc_real['avg_dist']*100:.1f} cm  "
          f"<-- Sim2Real gap {(bc_sim['success_rate']-bc_real['success_rate'])*100:.0f} 个点")
    print(f"  BC + dither   {bc_dith['success_rate']*100:5.1f}%")

    # ---- [4] 死区诊断 ----
    print("\n[4] 死区诊断：命令幅度 vs 死区阈值")
    m_s, s_s = action_stats(servo_best, env_kw=REAL)
    m_b, s_b = action_stats(make_greedy(bc_net), env_kw=REAL)
    print(f"  伺服(gain={best_g})  平均命令幅度 {m_s:.3f}   被死区吃掉 {s_s*100:5.1f}% 的步")
    print(f"  BC            平均命令幅度 {m_b:.3f}   被死区吃掉 {s_b*100:5.1f}% 的步")
    print(f"  死区阈值 = {DEADZONE}：单关节命令幅度低于它，这一步就完全不动")

    # ---- [5] PPO 在真实环境微调 ----
    print("\n[5] 在【真实环境】里用 PPO 微调（真机上的交互）")
    with torch.no_grad():
        bc_net.log_std.fill_(0.0)
    bc_net = pretrain_critic(bc_net, env_kw=REAL)
    net, hist, el, best_state, best_succ = train(
        iters=200, lr=5e-5, ent_coef=0.005,
        init_weights=bc_net.state_dict(), env_kw=REAL, eval_sample=True)
    if best_state is not None:
        net.load_state_dict(best_state)
    torch.save(net.state_dict(), CKPT)

    rl_sample = eval_avg(lambda i: make_sample(net, seed=i), env_kw=REAL)
    rl_greedy = evaluate(make_greedy(net), episodes=N_EVAL, seed=EVAL_SEED, env_kw=REAL)
    print(f"\n  微调后·采样部署 {rl_sample['success_rate']*100:5.1f}%  "
          f"平均 {rl_sample['avg_steps']:5.1f} 步  残距 {rl_sample['avg_dist']*100:.1f} cm")
    print(f"  微调后·确定性部署 {rl_greedy['success_rate']*100:5.1f}%  "
          f"平均 {rl_greedy['avg_steps']:5.1f} 步  残距 {rl_greedy['avg_dist']*100:.1f} cm")
    print(f"  用时 {el:.0f}s")

    m_r, s_r = action_stats(make_greedy(net), env_kw=REAL)
    m_rs, s_rs = action_stats(make_sample(net, seed=0), env_kw=REAL)
    print(f"  确定性：命令幅度 {m_r:.3f}，被吃掉 {s_r*100:.1f}%")
    print(f"  采样时：命令幅度 {m_rs:.3f}，被吃掉 {s_rs*100:.1f}%")

    # ---- 总结 ----
    print("\n" + "=" * 72)
    print("结论（同一批 50 个随机初始姿态 + 随机目标，真机环境）")
    print("=" * 72)
    table = [
        ("伺服（仿真里的成绩）", servo_sim["success_rate"] * 100, servo_sim["avg_steps"], "—"),
        (f"伺服（真机，gain={best_g:g}）", best_row[1], best_row[2], f"{best_row[3]*100:.1f} cm"),
        ("伺服 + dither", servo_dith["success_rate"] * 100, servo_dith["avg_steps"],
         f"{servo_dith['avg_dist']*100:.1f} cm"),
        ("BC（仿真训好直接上真机）", bc_real["success_rate"] * 100, bc_real["avg_steps"],
         f"{bc_real['avg_dist']*100:.1f} cm"),
        ("BC + dither", bc_dith["success_rate"] * 100, bc_dith["avg_steps"],
         f"{bc_dith['avg_dist']*100:.1f} cm"),
        ("BC + 真机微调·确定部署", rl_greedy["success_rate"] * 100, rl_greedy["avg_steps"],
         f"{rl_greedy['avg_dist']*100:.1f} cm"),
        ("BC + 真机微调·采样部署", rl_sample["success_rate"] * 100, rl_sample["avg_steps"],
         f"{rl_sample['avg_dist']*100:.1f} cm"),
    ]
    print(f"  {'':30s} {'成功率':>7s} {'平均步数':>8s} {'残距':>9s}")
    for name, s, st, d in table:
        print(f"  {name:30s} {s:6.1f}% {st:8.1f} {d:>9s}")

    # ---- 出图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.1))
    fig.suptitle(
        f"关节摩擦死区 = {DEADZONE}：解析控制器结构性失效，RL 在真机上把 Sim2Real gap 补回来",
        fontsize=13)

    ax = axes[0]
    gs = [r[0] for r in rows]
    ss = [r[1] for r in rows]
    ax.plot(range(len(gs)), ss, "o-", color="#185FA5", lw=1.8, ms=6)
    ax.axhline(servo_sim["success_rate"] * 100, color="#0F6E56", ls="--", lw=1.4,
               label=f"伺服在仿真里 {servo_sim['success_rate']*100:.0f}%")
    ax.axhline(servo_dith["success_rate"] * 100, color="#854F0B", ls="-.", lw=1.4,
               label=f"伺服 + dither {servo_dith['success_rate']*100:.0f}%")
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([f"{g:g}" for g in gs], fontsize=8.5)
    ax.set_xlabel("伺服增益 gain（已扫到最好）")
    ax.set_ylabel("真机成功率 (%)")
    ax.set_ylim(-3, 105)
    ax.set_title("先校准基准：光调 gain 救不回死区", fontsize=11.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    s = np.array(hist["succ"]) * 100
    ax.plot(s, color="#185FA5", lw=1.4, label="真机微调（训练中采样）")
    ax.axhline(bc_real["success_rate"] * 100, color="#E24B4A", ls="--", lw=1.4,
               label=f"BC 上真机 {bc_real['success_rate']*100:.0f}%")
    ax.axhline(rl_sample["success_rate"] * 100, color="#0F6E56", ls=":", lw=1.6,
               label=f"微调后采样部署 {rl_sample['success_rate']*100:.0f}%")
    ax.set_ylim(-3, 105)
    ax.set_xlabel("真机微调回合")
    ax.set_ylabel("到达成功率 (%)")
    ax.set_title("真机微调：从 0% 爬回来", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[2]
    names = ["伺服\n(仿真)", f"伺服\n(真机{gain_label(best_g)})", "伺服\n+dither",
             "BC\n(上真机)", "BC\n+dither", "微调\n确定部署", "微调\n采样部署"]
    vals = [t[1] for t in table]
    colors = ["#9AA5B1", "#8D8D8D", "#854F0B", "#E24B4A", "#EF9F27",
              "#BA7FD0", "#185FA5"]
    bars = ax.bar(names, vals, color=colors, width=0.66)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", fontsize=9.5)
    ax.set_ylim(0, 118)
    ax.set_ylabel("成功率 (%)")
    ax.set_title("公平对照：三方都允许抖，再看谁更好", fontsize=11.5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\n图已保存：{OUT}")
    print(f"策略权重：{CKPT}")
    print(f"总用时 {time.time()-t0:.0f} 秒")


def gain_label(g):
    return f"g{g:g}"


if __name__ == "__main__":
    main()
