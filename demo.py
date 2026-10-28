"""
demo.py —— 一键跑完整性gutan demonstration

    python demo.py

做三件事：
  1. 运动学自检（五项，全对才会继续）
  2. 四个基线策略在同一初始姿态下的到达过程
  3. 扫初始 theta2，看奇异点如何把不带阻尼的伺服打崩

输出：终端表格 + demo.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arm import selfcheck, fk, jacobian, dls_step
from controller import (make_random, make_resolved_rate, make_zero,
                        run_episode, evaluate)
from env import TwoLinkReachEnv

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.png")

POSE0 = (0.35, 0.75)        # 统一的初始姿态
TARGET0 = (1.0, 0.85)       # 统一的目标点

POLICIES = [
    ("不动", make_zero(), "#888780"),
    ("随机推", make_random(seed=7), "#E24B4A"),
    ("伺服（纯伪逆）", make_resolved_rate(gain=1.5, lam=0.0), "#EF9F27"),
    ("伺服（阻尼 DLS）", make_resolved_rate(gain=1.5, lam=0.05), "#185FA5"),
]


def main():
    print("=" * 62)
    print("二连杆机械臂到达任务 —— 项目自检与基线演示")
    print("=" * 62)

    # ---------- 1. 运动学自检 ----------
    print("\n[1] 运动学自检")
    if not selfcheck(verbose=False):
        selfcheck(verbose=True)
        raise SystemExit("自检未通过，先修 arm.py")
    selfcheck(verbose=True)

    # ---------- 2. 四条轨迹 ----------
    print("\n[2] 同一初始姿态，四个基线 —— 目标 (1.00, 0.85)")
    results = []
    for name, pol, color in POLICIES:
        r = run_episode(pol, pose=POSE0, target=TARGET0, seed=3)
        results.append((name, r, color))
        tag = "到达" if r["success"] else "未到达"
        print(f"  {name:16s} {tag}  用了 {r['steps']:3d} 步  "
              f"最终距离 {r['dist']:.4f}")

    # ---------- 3. 奇异点扫描 ----------
    print("\n[3] 初始关节角对伺服的影响（目标放在边界 1.70~1.78，"
          "逼着臂接近伸直）")
    print("    每个角度跑 30 个随机目标，看还能不能稳定够到")
    thetas = np.radians(np.arange(8, 178, 10))
    sweep = {"伺服（纯伪逆）": make_resolved_rate(gain=1.5, lam=0.0),
             "伺服（阻尼 DLS）": make_resolved_rate(gain=1.5, lam=0.05)}
    env0 = TwoLinkReachEnv(seed=0)
    sweep_res = {k: [] for k in sweep}
    sweep_steps = {k: [] for k in sweep}
    for t2 in thetas:
        for name, mk in sweep.items():
            ok_list, st_list = [], []
            for i in range(30):
                rng = np.random.default_rng(100 + i)
                r = rng.uniform(1.70, 1.78)
                a = rng.uniform(-np.pi, np.pi)
                obs, _ = env0.reset(pose=(0.0, float(t2)),
                                    target=[r * np.cos(a), r * np.sin(a)])
                ok, steps = False, 0
                for _ in range(200):
                    obs, rw, term, trunc, info = env0.step(mk(obs))
                    steps += 1
                    if term:
                        ok = True
                        break
                    if trunc:
                        break
                ok_list.append(1.0 if ok else 0.0)
                st_list.append(steps)
            sweep_res[name].append(np.mean(ok_list))
            sweep_steps[name].append(np.mean(st_list))
    for name in sweep:
        sweep_res[name] = np.array(sweep_res[name])
        sweep_steps[name] = np.array(sweep_steps[name])
    worst_i = int(np.argmin(sweep_res["伺服（纯伪逆）"]))
    print(f"  纯伪逆在 theta2={np.degrees(thetas[worst_i]):.0f}° 附近最差，"
          f"成功率 {sweep_res['伺服（纯伪逆）'][worst_i]*100:.0f}%、"
          f"平均 {sweep_steps['伺服（纯伪逆）'][worst_i]:.0f} 步；"
          f"同位置 DLS 为 {sweep_res['伺服（阻尼 DLS）'][worst_i]*100:.0f}%、"
          f"平均 {sweep_steps['伺服（阻尼 DLS）'][worst_i]:.0f} 步")

    # ---------- 4. 为什么必须加阻尼：关节速度需求 ----------
    print("\n[4] 末端沿径向往外走 1 m/s，关节要转多快")
    print(r"    （det J = L1·L2·sin(theta2)，theta2 -> 0 或 180 即奇异）")
    scan = np.radians([0.5, 2, 5, 10, 30, 45, 90, 135, 179.5])
    need_p, need_d = [], []
    for t2 in scan:
        _, p = fk(0.5, t2)
        v = p / np.linalg.norm(p)          # 沿当前位置径向往外，单位速度
        J = jacobian(0.5, t2)
        need_p.append(np.linalg.norm(np.linalg.pinv(J) @ v))
        need_d.append(np.linalg.norm(dls_step(J, v, lam=0.05)))
    need_p, need_d = np.array(need_p), np.array(need_d)
    for i, t2 in enumerate(scan):
        print(f"    theta2={np.degrees(t2):6.1f}°  伪逆 {need_p[i]:9.2f} rad/s"
              f"    DLS {need_d[i]:6.2f} rad/s")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    fig.subplots_adjust(wspace=0.26, top=0.86, bottom=0.14)
    fig.suptitle("二连杆机械臂到达任务：运动学自检 + 四个基线 + 奇异点扫描",
                 fontsize=14)

    # 左：轨迹
    ax = axes[0]
    ax.plot([0], [0], "s", color="#2C2C2A", ms=11, zorder=5)
    for name, r, color in results:
        p = r["path"]
        ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=2.2,
                label=f"{name}（{r['steps']} 步）", alpha=0.9)
        ax.plot(p[0, 0], p[0, 1], "o", color=color, ms=6)
    ax.plot(TARGET0[0], TARGET0[1], "*", color="#639922", ms=22,
            markeredgecolor="#173404", markeredgewidth=0.8, zorder=6)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(1.8 * np.cos(th), 1.8 * np.sin(th), "--", color="#B4B2A9", lw=1,
            label="工作空间外边界 1.8")
    ax.plot(0.2 * np.cos(th), 0.2 * np.sin(th), "--", color="#D3D1C7", lw=1,
            label="内边界 0.2（够不到自己）")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("末端轨迹：伺服 14 步直达，随机靠乱飘才撞进去", fontsize=12)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)

    # 中：距离下降
    ax = axes[1]
    for name, r, color in results:
        t = np.arange(len(r["dists"])) * 0.05
        ax.plot(t, r["dists"], "-", color=color, lw=2.2, label=name)
    ax.axhline(0.02, color="#444441", ls=":", lw=1)
    ax.annotate("够到阈值 0.02", xy=(0.15, 0.035), fontsize=8.5,
                color="#444441")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("到目标的距离 (m)")
    ax.set_title("距离下降：伺服是单调的，随机在乱飘", fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    # 右：为什么必须加阻尼
    ax = axes[2]
    dd = np.degrees(scan)
    ax.semilogy(dd, need_p, "o-", color="#EF9F27", lw=2, ms=5,
                label="纯伪逆：奇异点处要求无穷大转速")
    ax.semilogy(dd, need_d, "s-", color="#185FA5", lw=2, ms=5,
                label=r"DLS 阻尼 $\lambda$=0.05：压在 $1/\lambda$=20 以下")
    ax.set_xlim(-5, 185)
    ax.set_xlabel(r"关节角 $\theta_2$ (度)")
    ax.set_ylabel(r"所需关节角速度 $\|\dot\theta\|$ (rad/s，对数轴)")
    ax.set_title(r"为什么必须加阻尼：$\theta_2\to 0^\circ/180^\circ$ 时 "
                 "电机要疯转", fontsize=12)
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(alpha=0.25, which="both")
    ax.grid(alpha=0.5, which="major")

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\n图已保存：{OUT}")
    print("=" * 62)


if __name__ == "__main__":
    main()
