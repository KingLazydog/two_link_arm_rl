"""
dynamics.py —— 二连杆刚体动力学（拉格朗日法）

arm.py 回答"末端在哪"，本模块回答"要多大力矩才能让它动"。
速度模式下 theta <- theta + a*dt 只是一句加法，质量、惯量、重力、
科里奥利力全部被跳过。补上这一层，动作空间才有可能从
"关节速度"换成"关节力矩"。

标准形式：

    M(q) @ qdd + C(q, qd) @ qd + G(q) = tau

约定与 arm.py 完全一致：角度自 x 轴量起、逆时针为正、y 轴朝上。
因此重力势能写成 sin(theta)，重力力矩写成 cos(theta)。

参数默认值取一组物理上说得通的组合：铝合金管臂 + 关节，
杆长沿用 arm.py 的 L1=1.0 m、L2=0.8 m。
"""

import numpy as np

from arm import L1, L2

GRAV = 9.81

DEFAULT_PARAMS = dict(
    m1=3.0, m2=2.0,                    # 两节杆质量 kg
    l1=L1, l2=L2,                      # 杆长 m
    lc1=0.5 * L1, lc2=0.5 * L2,        # 质心到各自关节的距离 m
    I1=3.0 * L1 ** 2 / 12.0,           # 绕质心转动惯量 kg*m^2（均质细杆 m*l^2/12）
    I2=2.0 * L2 ** 2 / 12.0,
)


def coeffs(p=DEFAULT_PARAMS):
    """
    把惯量项收成三个常数，M 就只剩一个随 theta2 变化的余弦项。

    a1 = I1 + I2 + m1*lc1^2 + m2*(l1^2 + lc2^2)
    a2 = m2*l1*lc2            耦合项，肘部弯曲时改变两节的耦合强度
    a3 = I2 + m2*lc2^2        第二节自身的等效惯量
    """
    a1 = (p["I1"] + p["I2"] + p["m1"] * p["lc1"] ** 2
          + p["m2"] * (p["l1"] ** 2 + p["lc2"] ** 2))
    a2 = p["m2"] * p["l1"] * p["lc2"]
    a3 = p["I2"] + p["m2"] * p["lc2"] ** 2
    return a1, a2, a3


def mass_matrix(th, p=DEFAULT_PARAMS):
    """
    惯性矩阵 M(q)。对称正定，且随 theta2 变化：
    手臂伸直（theta2 -> 0）时绕基座的惯量最大，完全折叠时最小。
    """
    a1, a2, a3 = coeffs(p)
    c2 = np.cos(th[1])
    return np.array([
        [a1 + 2.0 * a2 * c2, a3 + a2 * c2],
        [a3 + a2 * c2,       a3],
    ])


def coriolis_vec(th, dth, p=DEFAULT_PARAMS):
    """科里奥利力与离心力项 C(q, qd) @ qd，写成向量，省掉显式构造 C。"""
    _, a2, _ = coeffs(p)
    s2 = np.sin(th[1])
    return np.array([
        -a2 * s2 * (2.0 * dth[0] * dth[1] + dth[1] ** 2),
        a2 * s2 * dth[0] ** 2,
    ])


def gravity_vec(th, p=DEFAULT_PARAMS):
    """重力力矩 G(q)。theta1 越接近 0（手臂水平外伸）需要的保持力矩越大。"""
    c1 = np.cos(th[0])
    c12 = np.cos(th[0] + th[1])
    g1 = (p["m1"] * p["lc1"] + p["m2"] * p["l1"]) * GRAV * c1 \
        + p["m2"] * p["lc2"] * GRAV * c12
    g2 = p["m2"] * p["lc2"] * GRAV * c12
    return np.array([g1, g2])


def inverse_dynamics(th, dth, ddth, p=DEFAULT_PARAMS):
    """逆动力学：给定运动求所需力矩。自检、前馈补偿、力矩限幅都靠它。"""
    return (mass_matrix(th, p) @ ddth
            + coriolis_vec(th, dth, p)
            + gravity_vec(th, p))


def forward_dynamics(th, dth, tau, p=DEFAULT_PARAMS):
    """
    正动力学：给定力矩求角加速度。仿真里每步都要调它，所以用 2x2 显式求逆，
    不走 np.linalg.solve —— 后者有个几十微秒的固定开销，乘上子步数就很可观。
    """
    a1, a2, a3 = coeffs(p)
    c2, s2 = np.cos(th[1]), np.sin(th[1])
    m11 = a1 + 2.0 * a2 * c2
    m12 = a3 + a2 * c2
    m22 = a3
    rhs0 = tau[0] + a2 * s2 * (2.0 * dth[0] * dth[1] + dth[1] ** 2) \
        - (p["m1"] * p["lc1"] + p["m2"] * p["l1"]) * GRAV * np.cos(th[0]) \
        - p["m2"] * p["lc2"] * GRAV * np.cos(th[0] + th[1])
    rhs1 = tau[1] - a2 * s2 * dth[0] ** 2 \
        - p["m2"] * p["lc2"] * GRAV * np.cos(th[0] + th[1])
    det = m11 * m22 - m12 * m12
    return np.array([(m22 * rhs0 - m12 * rhs1) / det,
                     (m11 * rhs1 - m12 * rhs0) / det])


def kinetic_energy(th, dth, p=DEFAULT_PARAMS):
    return 0.5 * float(dth @ mass_matrix(th, p) @ dth)


def potential_energy(th, p=DEFAULT_PARAMS):
    """以基座高度为零势能面；y 坐标 = l*sin(theta)。"""
    y1 = p["lc1"] * np.sin(th[0])
    y2 = p["l1"] * np.sin(th[0]) + p["lc2"] * np.sin(th[0] + th[1])
    return float(p["m1"] * GRAV * y1 + p["m2"] * GRAV * y2)


def total_energy(th, dth, p=DEFAULT_PARAMS):
    return kinetic_energy(th, dth, p) + potential_energy(th, p)


def rk4_step(th, dth, tau, dt, p=DEFAULT_PARAMS):
    """
    固定力矩下推进一步（RK4）。力矩在步内保持不变，
    等价于零阶保持器，也正是数字控制器实际做的事。
    """
    def f(state):
        q, qd = state[:2], state[2:]
        return np.concatenate([qd, forward_dynamics(q, qd, tau, p)])

    s = np.concatenate([np.asarray(th, float), np.asarray(dth, float)])
    k1 = f(s)
    k2 = f(s + 0.5 * dt * k1)
    k3 = f(s + 0.5 * dt * k2)
    k4 = f(s + dt * k3)
    s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return s[:2], s[2:]


def simulate(th, dth, tau_func, t_end, dt=1e-3, p=DEFAULT_PARAMS, record=None):
    """
    闭环仿真：每步用当前状态算力矩，再积分一步。
    tau_func(th, dth) -> tau，这样控制器可以直接塞进来。
    """
    th = np.asarray(th, float).copy()
    dth = np.asarray(dth, float).copy()
    n = int(round(t_end / dt))
    if record is not None:
        record.append((th.copy(), dth.copy()))
    for _ in range(n):
        tau = tau_func(th, dth)
        th, dth = rk4_step(th, dth, tau, dt, p)
        if record is not None:
            record.append((th.copy(), dth.copy()))
    return th, dth


# ---------------- 自检 ----------------
def selfcheck(verbose=True):
    """
    五项物理一致性检查。这一层如果没有自检，写出来的东西只能算"看起来在动"。
    """
    p = DEFAULT_PARAMS
    rows = []
    ok = True

    # 1. 惯性矩阵对称正定
    worst_sym, worst_eig = 0.0, np.inf
    for t1 in (0.0, 0.7, -1.9, 2.6):
        for t2 in (0.0, 0.4, -1.2, np.pi):
            M = mass_matrix(np.array([t1, t2]), p)
            worst_sym = max(worst_sym, abs(M[0, 1] - M[1, 0]))
            worst_eig = min(worst_eig, float(np.min(np.linalg.eigvalsh(M))))
    rows.append(("M 对称且正定", max(worst_sym, max(0.0, -worst_eig)), worst_sym < 1e-12 and worst_eig > 0))
    ok &= worst_sym < 1e-12 and worst_eig > 0

    # 2. 逆动力学 -> 正动力学 往返一致
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(20):
        th = rng.uniform(-np.pi, np.pi, 2)
        dth = rng.uniform(-3, 3, 2)
        ddth = rng.uniform(-10, 10, 2)
        tau = inverse_dynamics(th, dth, ddth, p)
        worst = max(worst, float(np.max(np.abs(forward_dynamics(th, dth, tau, p) - ddth))))
    rows.append(("逆/正动力学往返误差", worst, worst < 1e-9))
    ok &= worst < 1e-9

    # 3. 重力补偿后静态平衡：tau = G(q) 且 qd = 0 时角加速度应为零
    worst = 0.0
    for t1 in (0.2, 1.4, -0.8):
        for t2 in (0.5, -1.1):
            ddth = forward_dynamics(np.array([t1, t2]), np.zeros(2), gravity_vec(np.array([t1, t2]), p), p)
            worst = max(worst, float(np.max(np.abs(ddth))))
    rows.append(("重力补偿静态平衡", worst, worst < 1e-12))
    ok &= worst < 1e-12

    # 4. 无外力自由摆动时总能量守恒（RK4 数值积分误差）
    th0, dth0 = np.array([0.3, 0.6]), np.array([0.0, 0.0])
    E0 = total_energy(th0, dth0, p)
    th, dth = simulate(th0, dth0, lambda q, qd: np.zeros(2), 10.0, dt=1e-3, p=p)
    drift = abs(total_energy(th, dth, p) - E0) / abs(E0)
    rows.append(("自由摆动能量守恒(10s)", drift, drift < 1e-6))
    ok &= drift < 1e-6

    # 5. 由动能反推惯性矩阵（Hessian of T wrt qd 必须等于 M）
    worst = 0.0
    for _ in range(5):
        th = rng.uniform(-np.pi, np.pi, 2)
        M = mass_matrix(th, p)
        # T = 0.5*qd^T*M*qd，取三个基向量即可反解出 M 的全部四个元素
        a = kinetic_energy(th, np.array([1.0, 0.0]), p)
        b = kinetic_energy(th, np.array([0.0, 1.0]), p)
        c = kinetic_energy(th, np.array([1.0, 1.0]), p)
        est = np.array([[2 * a, c - a - b], [c - a - b, 2 * b]])
        worst = max(worst, float(np.max(np.abs(est - M))))
    rows.append(("动能反推惯性矩阵", worst, worst < 1e-9))
    ok &= worst < 1e-9

    if verbose:
        print("动力学自检 —— 五项" + ("全部通过" if ok else "存在失败"))
        for name, val, good in rows:
            print(f"  [{'OK' if good else 'XX'}] {name:22s} = {val:.3e}")
        a1, a2, a3 = coeffs(p)
        print(f"  惯量常数 a1={a1:.4f}  a2={a2:.4f}  a3={a3:.4f}")
        print(f"  重力矩峰值 |G| = {max(np.linalg.norm(gravity_vec(np.array([t, 0.0]), p)) for t in np.linspace(-np.pi, np.pi, 99)):.2f} N*m")
    return ok


if __name__ == "__main__":
    selfcheck()
