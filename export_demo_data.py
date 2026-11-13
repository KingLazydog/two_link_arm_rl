"""export_demo_data.py —— 导出网页演示所需的数据

产出 webdemo/demo_data.json：
  - 两个策略的 MLP 权重（BC 与 死区微调后的 PPO）
  - 50 个固定的评测回合（初始姿态 + 目标点），与 controller.evaluate(seed=0, episodes=50) 同分布
  - 用 numpy 复算的参考成功率，用来和网页端 JS 的结果互相印证

用法：
    python export_demo_data.py
"""

import json
import os

import numpy as np
import torch

from env import TwoLinkReachEnv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "webdemo", "demo_data.json")

L1, L2, MAXW, DT, TOL, MAXSTEPS = 1.0, 0.8, 3.0, 0.05, 0.05, 200


# ---------------- 权重 ----------------
def load_policy(path):
    ck = torch.load(path, map_location="cpu")
    out = {}
    for k in ["actor.0.weight", "actor.0.bias", "actor.2.weight",
              "actor.2.bias", "actor.4.weight", "actor.4.bias", "log_std"]:
        out[k] = ck[k].detach().numpy().astype(np.float64)
    return {
        "W1": out["actor.0.weight"].tolist(),
        "b1": out["actor.0.bias"].tolist(),
        "W2": out["actor.2.weight"].tolist(),
        "b2": out["actor.2.bias"].tolist(),
        "W3": out["actor.4.weight"].tolist(),
        "b3": out["actor.4.bias"].tolist(),
        "log_std": out["log_std"].tolist(),
    }


# ---------------- numpy 版前向（与 JS 一一对应）----------------
def net_mu(obs, P):
    x = np.array(obs, dtype=np.float64)
    x[4] /= 2.0
    x[5] /= 2.0
    h = np.tanh(np.array(P["W1"]) @ x + np.array(P["b1"]))
    h = np.tanh(np.array(P["W2"]) @ h + np.array(P["b2"]))
    return np.tanh(np.array(P["W3"]) @ h + np.array(P["b3"]))


def net_std(P):
    ls = np.array(P["log_std"], dtype=np.float64)
    return 0.10 + 0.50 / (1.0 + np.exp(-ls))


def make_net_policy(P, sample, rng):
    def pol(obs):
        mu = net_mu(obs, P)
        if not sample:
            return np.clip(mu, -1, 1)
        return np.clip(mu + rng.normal(size=2) * net_std(P), -1, 1)
    return pol


def make_servo_policy(gain, lam):
    from arm import dls_step, jacobian

    def pol(obs):
        t1 = np.arctan2(obs[1], obs[0])
        t2 = np.arctan2(obs[3], obs[2])
        J = jacobian(t1, t2)
        dq = dls_step(J, np.array(obs[4:6]), lam=lam) * gain
        return np.clip(dq / MAXW, -1, 1)
    return pol


# ---------------- 评测（与 JS 的 evaluate 同构）----------------
def eval_policy(pol, episodes, deadzone=0.0, delay=0, noise=0.0):
    ok, st = 0, 0
    for ep in episodes:
        env = TwoLinkReachEnv(deadzone=deadzone, action_delay=delay,
                              obs_noise=noise, seed=0)
        obs, _ = env.reset(pose=ep["th"], target=ep["target"])
        steps, success = 0, False
        for _ in range(MAXSTEPS):
            obs, r, term, trunc, info = env.step(pol(obs))
            steps += 1
            if term:
                success = True
                break
            if trunc:
                break
        ok += 1 if success else 0
        st += steps
    return ok / len(episodes), st / len(episodes)


def main():
    # 与 task_sim2real.py 一致：N_EVAL=50, EVAL_SEED=500
    # （controller.evaluate(seed=500, episodes=50) 就是 README 里所有数字的口径）
    EVAL_SEED, N_EVAL = 500, 50
    episodes = []
    for i in range(N_EVAL):
        env = TwoLinkReachEnv(seed=EVAL_SEED + i)
        env.reset()
        episodes.append({
            "th": [float(env.th[0]), float(env.th[1])],
            "target": [float(env.target[0]), float(env.target[1])],
        })

    P_bc = load_policy(os.path.join(HERE, "bc_policy.pt"))
    P_ppo = load_policy(os.path.join(HERE, "sim2real_policy.pt"))

    rng = np.random.default_rng(20261113)
    ref = {}
    ref["servo_gain1.0_dz0"] = eval_policy(make_servo_policy(1.0, 0.05), episodes)
    ref["servo_gain1.0_dz0.25"] = eval_policy(make_servo_policy(1.0, 0.05), episodes, deadzone=0.25)
    ref["servo_gain8.0_dz0.25"] = eval_policy(make_servo_policy(8.0, 0.05), episodes, deadzone=0.25)
    ref["ppo_dz0"] = eval_policy(make_net_policy(P_ppo, True, rng), episodes)
    ref["ppo_dz0.25"] = eval_policy(make_net_policy(P_ppo, True, rng), episodes, deadzone=0.25)
    ref["ppo_greedy_dz0.25"] = eval_policy(make_net_policy(P_ppo, False, rng), episodes, deadzone=0.25)
    ref["bc_greedy_dz0"] = eval_policy(make_net_policy(P_bc, False, rng), episodes)
    ref["bc_greedy_dz0.25"] = eval_policy(make_net_policy(P_bc, False, rng), episodes, deadzone=0.25)

    data = {
        "meta": {
            "L1": L1, "L2": L2, "maxW": MAXW, "dt": DT,
            "tol": TOL, "maxSteps": MAXSTEPS, "episodes": len(episodes),
            "note": "50 个评测回合与 python controller.evaluate(seed=0, episodes=50) 同分布"
        },
        "episodes": episodes,
        "policyBC": P_bc,
        "policyPPO": P_ppo,
        "reference": {k: {"rate": v[0], "avgSteps": v[1]} for k, v in ref.items()},
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print("已写出", OUT, "大小 %.1f KB" % (os.path.getsize(OUT) / 1024))
    print("\n参考成功率（50 回合）：")
    for k, v in data["reference"].items():
        print("  %-24s %.1f%%   %.1f 步" % (k, v["rate"] * 100, v["avgSteps"]))


if __name__ == "__main__":
    main()
