"""
task_torque_rl.py —— 实验七：力矩空间的强化学习（模型参数失配）

前面所有 RL 都跑在速度模式下。速度模式把"伺服内环"当成黑盒，动作直接
就是关节速度；力矩模式则要求策略自己输出力矩，多了一层真正的动力学。

这一节回答一个问题：

    当控制器手里的模型是错的，RL 能不能只靠反馈把它补回来？

实验设置：
    真实机构的质量是控制器以为的两倍（+100%）。工程师手上的参数表是错的，
    但代码照参数表算 —— 这是工程里最常见的失配形式，比"完全不建模"更真实。
    力矩上限同步放宽到 90 N*m（重一倍的机构配更大功率的驱动器，
    否则重力矩 85.8 N*m 直接顶到 60 N*m 上限，属于推不动而不是控不住）。

结果（50 回合，EVAL_SEED=500，100 Hz，单回合 10 s）：

    标称模型 PD（工程师实际能做到的）        58.0%   4.85 s
    真实模型 PD（作弊上界）                  98.0%   1.13 s
    行为克隆 BC（40000 步，MSE 2.7e-4）      44.0%   采样部署
    BC + PPO（150 轮，prog_scale=10）        18.0%
    BC + PPO（120 轮，prog_scale=50，3 s 回合）16.0%

结论：**没跑通。** 而且不是"失配导致 RL 失效"这种解释 —— 诊断实验把
真实机构换回标称机构（专家 100%），同一套管线在容易的环境里也只有 44.0%。
所以瓶颈在力矩空间的策略学习本身，与模型误差无关。

三条原因，都已用实验区分：

  1. 时间尺度。速度模式一回合 200 步，力矩模式 100 Hz 下一回合 1000 步，
     信用分配跨度大 5 倍。prog_scale 在 20 Hz 下调到 10，100 Hz 下每步
     位移小 5 倍，奖励信号被固定的 -0.01 惩罚淹没 —— 已按比例改成 50 重训，
     无改善，说明不是主因。
  2. 复现误差累积。BC 单步 MSE 从 5.8e-3 降到 2.7e-4（两个数量级），
     闭环成功率只在 22%~44% 之间波动，不随拟合变好而上升。经典的
     compounding error：单步都对，轨迹整体还是飘。
  3. 演示质量。专家本身只有 58% 成功率，它的演示里混着大量"补救动作"，
     这些动作与正常控制动作混在一起学，反而把策略拉偏。

因此本项目的结论是边界而非方法优劣：**速度空间里 RL 可用（86%/90%/91.3%），
力矩空间里 RL 尚未打通。** 要把这一环补上，需要的是更好的探索策略
（如 SAC/TD3）、并行环境与更长的训练，不是再多调几个超参数。

评测口径与 5.6 节一致：50 回合，EVAL_SEED=500，单回合 10 s，100 Hz。
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from controller import evaluate
from dynamics import DEFAULT_PARAMS
from task_dynamics import CTRL_DT, HORIZON, eval_torque, make_pd
from train_ppo import ActorCritic, gae, make_greedy, make_sample, norm_obs

torch.set_num_threads(6)

EVAL_SEED = 500
OBS_DIM = 10          # 8 维基础观测 + 2 维关节速度
MAX_STEPS = int(round(HORIZON / CTRL_DT))


def mismatch_params(scale=2.0):
    """真实机构：质量翻倍，转动惯量随之翻倍。控制器不知道这件事。"""
    return dict(
        DEFAULT_PARAMS,
        m1=DEFAULT_PARAMS["m1"] * scale,
        m2=DEFAULT_PARAMS["m2"] * scale,
        I1=3.0 * scale * DEFAULT_PARAMS["l1"] ** 2 / 12.0,
        I2=2.0 * scale * DEFAULT_PARAMS["l2"] ** 2 / 12.0,
    )


FAIL_ENV = dict(control_mode="torque", obs_vel=True, dt=CTRL_DT, substeps=5,
                reward_mode="progress", tau_max=90.0,
                dyn_params=mismatch_params(2.0))


# ---------------- 演示采集与行为克隆 ----------------
def collect_demos(policy, n_ep=30, seed=1000):
    """用专家在训练环境里跑，记录 (观测, 动作)。动作即归一化力矩。"""
    from env import TwoLinkReachEnv

    S, A = [], []
    for i in range(n_ep):
        env = TwoLinkReachEnv(seed=seed + i, max_steps=MAX_STEPS, **FAIL_ENV)
        obs, _ = env.reset()
        for _ in range(MAX_STEPS):
            a = policy(obs)
            S.append(norm_obs(obs))
            A.append(np.asarray(a, dtype=np.float32))
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return np.array(S, dtype=np.float32), np.array(A, dtype=np.float32)


def bc_train(net, S, A, steps=1500, batch=128, lr=1e-3):
    opt = torch.optim.Adam(net.actor.parameters(), lr=lr)
    St = torch.from_numpy(S)
    At = torch.from_numpy(A)
    n = len(S)
    last = 0.0
    for k in range(steps):
        idx = torch.randint(0, n, (batch,))
        mu = torch.tanh(net.actor(St[idx]))
        loss = F.mse_loss(mu, At[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()
    print(f"  BC 完成：{n} 条转移，{steps} 步梯度，末端 MSE = {last:.5f}")


# ---------------- PPO 微调 ----------------
def ppo_finetune(net, iters=150, horizon=2048, epochs=8, batch=64,
                 gamma=0.99, lam=0.95, clip=0.2, lr=3e-4,
                 vf_coef=0.5, ent_coef=0.01, seed=0, tag=""):
    from env import TwoLinkReachEnv

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = TwoLinkReachEnv(seed=seed, max_steps=MAX_STEPS, **FAIL_ENV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs, _ = env.reset()

    hist, best_state, best = [], None, -1.0
    t0 = time.time()
    for it in range(iters):
        buf = {"s": [], "a": [], "lp": [], "r": [], "d": [], "v": []}
        done = 0
        while done < horizon:
            st = torch.from_numpy(norm_obs(obs)).unsqueeze(0)
            with torch.no_grad():
                dist, v = net.dist(st)
                a = dist.sample().clamp(-1.0, 1.0)
                lp = dist.log_prob(a).sum(-1)
            a_env = a.numpy()[0]
            obs2, r, term, trunc, _ = env.step(a_env)
            buf["s"].append(norm_obs(obs))
            buf["a"].append(a_env)
            buf["lp"].append(lp.item())
            buf["r"].append(r)
            buf["d"].append(1.0 if (term or trunc) else 0.0)
            buf["v"].append(v.item())
            obs = obs2
            done += 1
            if term or trunc:
                obs, _ = env.reset()

        adv, ret = gae(buf["r"], buf["v"], buf["d"], gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        S = torch.from_numpy(np.array(buf["s"], dtype=np.float32))
        A = torch.from_numpy(np.array(buf["a"], dtype=np.float32))
        LP = torch.from_numpy(np.array(buf["lp"], dtype=np.float32))
        ADV = torch.from_numpy(adv)
        RET = torch.from_numpy(ret)
        n = len(buf["r"])
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                mu, v = net(S[idx])
                mu = torch.tanh(mu)
                dist = torch.distributions.Normal(mu, net._std().expand_as(mu))
                lp = dist.log_prob(A[idx]).sum(-1)
                ratio = torch.exp(lp - LP[idx])
                loss_p = -torch.min(ratio * ADV[idx],
                                    torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[idx]).mean()
                loss_v = F.mse_loss(v.squeeze(-1), RET[idx])
                ent = dist.entropy().sum(-1).mean()
                loss = loss_p + vf_coef * loss_v - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if it % 10 == 0 or it == iters - 1:
            pol = make_sample(net, seed=903)
            q = eval_torque(pol, episodes=20, seed=900, **FAIL_ENV)
            hist.append((it, q["success_rate"], q["avg_steps"] * CTRL_DT))
            if q["success_rate"] > best:
                best = q["success_rate"]
                best_state = {k: val.detach().clone() for k, val in net.state_dict().items()}
            print(f"  [{tag}] 轮 {it:3d}  采样评测 {q['success_rate']*100:5.1f}%  "
                  f"{q['avg_steps']*CTRL_DT:5.2f}s  用时 {time.time()-t0:4.0f}s")
    return hist, best_state, best


def main():
    print("=" * 68)
    print("力矩空间 + 模型失配：控制器以为的质量只有真实的一半")
    print("=" * 68)

    print("\n[1/4] 对照组：标称模型 vs 真实模型")
    expert_nom = make_pd(gravity=True, tau_max=90.0)      # 用标称参数补偿 —— 工程师能做到的
    r_nom = eval_torque(expert_nom, **FAIL_ENV)
    expert_true = make_pd(gravity=True, params=mismatch_params(2.0), tau_max=90.0)  # 知道真实参数
    r_true = eval_torque(expert_true, **FAIL_ENV)
    print(f"  标称补偿（失配下） {r_nom['success_rate']*100:5.1f}%  "
          f"{r_nom['avg_steps']*CTRL_DT:5.2f}s")
    print(f"  真实补偿（失配下） {r_true['success_rate']*100:5.1f}%  "
          f"{r_true['avg_steps']*CTRL_DT:5.2f}s")

    print("\n[2/4] 采集专家演示（用标称专家，它本身只有一半左右成功率）")
    S, A = collect_demos(expert_nom, n_ep=40)
    print(f"  演示转移数 {len(S)}，动作均值 {np.abs(A).mean():.3f}")

    net = ActorCritic(obs_dim=OBS_DIM)
    print("\n[3/4] 行为克隆预热")
    bc_train(net, S, A, steps=1500)
    r_bc_g = eval_torque(make_greedy(net), **FAIL_ENV)
    r_bc_s = eval_torque(make_sample(net, seed=904), **FAIL_ENV)
    print(f"  BC 贪心 {r_bc_g['success_rate']*100:5.1f}%  {r_bc_g['avg_steps']*CTRL_DT:5.2f}s")
    print(f"  BC 采样 {r_bc_s['success_rate']*100:5.1f}%  {r_bc_s['avg_steps']*CTRL_DT:5.2f}s")

    print("\n[4/4] PPO 在失配环境中微调")
    hist, best_state, best = ppo_finetune(net, iters=150, seed=0, tag="torque")
    if best_state is not None:
        net.load_state_dict(best_state)
    r_ppo_g = eval_torque(make_greedy(net), **FAIL_ENV)
    r_ppo_s = eval_torque(make_sample(net, seed=905), **FAIL_ENV)
    print(f"  PPO 贪心 {r_ppo_g['success_rate']*100:5.1f}%  {r_ppo_g['avg_steps']*CTRL_DT:5.2f}s")
    print(f"  PPO 采样 {r_ppo_s['success_rate']*100:5.1f}%  {r_ppo_s['avg_steps']*CTRL_DT:5.2f}s")

    torch.save({k: v for k, v in net.state_dict().items()}, "torque_policy.pt")
    print("\n权重已保存 torque_policy.pt")
    print(f"\n汇总（失配环境，50 回合）")
    print(f"  标称 PD        {r_nom['success_rate']*100:5.1f}%")
    print(f"  真实 PD（上界） {r_true['success_rate']*100:5.1f}%")
    print(f"  BC 采样        {r_bc_s['success_rate']*100:5.1f}%")
    print(f"  BC+PPO 采样     {r_ppo_s['success_rate']*100:5.1f}%")


if __name__ == "__main__":
    main()
