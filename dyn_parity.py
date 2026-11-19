"""
dyn_parity.py —— 与浏览器实现（webdemo/dyn_core.js）逐步对拍

为什么必须对拍：这个页面上要展示"36%→100%""采样率下限 100 Hz"这类结论。
如果浏览器里的积分方式、限幅位置、归一化口径与 Python 有一处不同，
数字就会漂到另一个结论上去，页面就成了误导。

对拍内容：固定姿态、固定目标、固定控制器，跑 300 步，逐步记录
(theta1, theta2, dtheta1, dtheta2)，由 Node 侧导出同名 CSV 后逐点比对。
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dynamics import gravity_vec
from env import TwoLinkReachEnv
from arm import wrap

HERE = Path(__file__).resolve().parent
OUT = HERE / "webdemo" / "parity_py.csv"

CASES = [
    # (控制器, kp, kd, gravity, 起始姿态, 目标, 步数)
    ("pd_nograv", 100.0, 20.0, False, [0.4, -0.7], [0.9, 0.5], 300),
    ("pd_grav",   100.0, 20.0, True,  [0.4, -0.7], [0.9, 0.5], 300),
    ("pd_soft",    25.0,  8.0, True,  [-1.2, 2.1], [-0.4, -0.9], 400),
]


def run_case(kp, kd, gravity, th0, target, n):
    env = TwoLinkReachEnv(control_mode="torque", dt=0.01, substeps=5,
                          tau_max=60.0, obs_vel=True)
    env.reset(pose=np.array(th0, float), target=np.array(target, float))

    rows = [[env.th[0], env.th[1], env.dth[0], env.dth[1]]]
    for _ in range(n):
        q = np.array(env.th, float)
        qd = np.array(env.dth, float)
        from task_dynamics import _nearest_ik
        q_des = _nearest_ik(q, np.array(target, float))
        tau = kp * wrap(q_des - q) - kd * qd
        if gravity:
            tau = tau + gravity_vec(q)
        a = np.clip(tau / 60.0, -1.0, 1.0)
        env.step(a)
        rows.append([env.th[0], env.th[1], env.dth[0], env.dth[1]])
    return rows


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "step", "t1", "t2", "d1", "d2"])
        for name, kp, kd, grav, th0, tgt, n in CASES:
            for i, r in enumerate(run_case(kp, kd, grav, th0, tgt, n)):
                w.writerow([name, i] + [f"{v:.12f}" for v in r])
    print(f"[py] 写出 {OUT}")

    # 调 Node 跑同一组用例
    node = r"C:\Users\huaji\.workbuddy\binaries\node\versions\22.22.2-2\node.exe"
    r = subprocess.run([node, str(HERE / "webdemo" / "dyn_parity.js")],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)

    py = {}
    with OUT.open() as f:
        for row in csv.DictReader(f):
            py.setdefault(row["case"], []).append(
                [float(row["t1"]), float(row["t2"]), float(row["d1"]), float(row["d2"])])
    js = json.loads((HERE / "webdemo" / "parity_js.json").read_text())

    print("\n逐步对拍（最大绝对误差）")
    print(f"  {'用例':<12}{'步数':>6}{'theta':>14}{'dtheta':>14}{'判定':>8}")
    ok = True
    for name, *_ in CASES:
        a = np.array(py[name])
        b = np.array(js[name])
        if a.shape != b.shape:
            print(f"  {name:<12} 形状不一致 {a.shape} vs {b.shape}")
            ok = False
            continue
        err_th = float(np.max(np.abs(a[:, :2] - b[:, :2])))
        err_d = float(np.max(np.abs(a[:, 2:] - b[:, 2:])))
        good = err_th < 1e-9 and err_d < 1e-9
        ok &= good
        print(f"  {name:<12}{len(a):>6}{err_th:>14.2e}{err_d:>14.2e}{'OK' if good else 'XX':>8}")

    print("\n结论：" + ("Python 与浏览器实现逐步一致" if ok else "存在偏差，需排查"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
