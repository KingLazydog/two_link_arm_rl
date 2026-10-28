"""
arm.py —— 二连杆机械臂的运动学核心（纯 numpy，无外部依赖）

这一层是整个项目的地基，四个入口：
    fk(t1, t2)          正运动学：关节角 -> 末端位置
    jacobian(t1, t2)    雅可比：关节速度 -> 末端速度
    ik_analytic(x, y)   解析逆运动学（两组解：肘上 / 肘下）
    ik_numeric(x, y,..) 数值逆运动学（雅可比迭代 + DLS 阻尼）

约定：角度单位一律弧度，逆时针为正。
"""

import numpy as np

L1 = 1.0          # 第一节杆长
L2 = 0.8          # 第二节杆长
MAX_ANGLE = np.pi  # 关节角归一化到 (-pi, pi]


# ---------------- 正运动学 ----------------
def rot2(theta):
    """2D 旋转矩阵。它的两列 = 原 x/y 轴转过 theta 后的新方向。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def homogeneous(theta, tx, ty):
    """
    3x3 齐次变换 T = R(theta) @ Trans(tx, ty)。
    右边先作用：点先在自己的坐标系里被推到 (tx, ty)，再整体旋转 theta。
    所以平移那一列必须是 R @ t，而不是裸的 t —— 写成裸的 t 就变成
    "先旋转后平移"，得到的是完全不同的位姿。
    """
    T = np.eye(3)
    R = rot2(theta)
    T[:2, :2] = R
    T[:2, 2] = R @ np.array([tx, ty])
    return T


def fk(t1, t2, l1=L1, l2=L2):
    """正运动学。返回 (肘位置, 末端位置)，都是长度 2 的数组。"""
    elbow = np.array([l1 * np.cos(t1), l1 * np.sin(t1)])
    tip = elbow + np.array([l2 * np.cos(t1 + t2), l2 * np.sin(t1 + t2)])
    return elbow, tip


def wrap(a):
    """把角度归一化到 (-pi, pi]。机器人的关节不会为了够到脚下而转一整圈。"""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------- 雅可比 ----------------
def jacobian(t1, t2, l1=L1, l2=L2):
    """
    雅可比矩阵：末端速度 v = J @ 关节速度。
    第一列 = 只有 theta1 动时末端的速度方向；第二列 = 只有 theta2 动时。
    """
    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
    return np.array([
        [-l1 * s1 - l2 * s12, -l2 * s12],
        [l1 * c1 + l2 * c12, l2 * c12],
    ])


def det_jacobian(t1, t2, l1=L1, l2=L2):
    """det J = l1*l2*sin(t2)。为 0 即奇异位形（伸直或完全折叠）。"""
    return l1 * l2 * np.sin(t2)


def dls_step(J, err, lam=0.05):
    """
    阻尼最小二乘求解 J @ dq = err。
    纯伪逆在奇异附近会给出疯狂的大步长；lambda 就是那根缰绳。
    """
    if lam <= 0:
        return np.linalg.pinv(J) @ err
    m, n = J.shape
    return J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(m), err)


# ---------------- 逆运动学 ----------------
def ik_analytic(x, y, l1=L1, l2=L2):
    """
    解析逆运动学，返回两组解 [(t1,t2), (t1',t2')]，不可达时返回空列表。
    两组解来自 sin(t2) 的正负两支 —— 同一个目标，肘可以朝上也可以朝下。
    """
    r = np.hypot(x, y)
    if r > l1 + l2 or r < abs(l1 - l2):
        return []                                  # 够不到：太远或太近
    c2 = (r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    c2 = float(np.clip(c2, -1.0, 1.0))
    s2_pos = np.sqrt(max(0.0, 1.0 - c2 * c2))
    sols = []
    for s2 in (s2_pos, -s2_pos):                   # +/- 两支 = 肘上 / 肘下
        t2 = np.arctan2(s2, c2)
        t1 = np.arctan2(y, x) - np.arctan2(l2 * s2, l1 + l2 * c2)
        sols.append((wrap(t1), wrap(t2)))
    return sols


def ik_numeric(target, th0, alpha=0.6, iters=200, tol=1e-10, lam=0.05):
    """
    数值逆运动学（雅可比迭代）。返回 (关节角, 误差历史, 是否收敛)。
    真实 6 轴机器人没有闭式解，工业控制器跑的就是这个。
    """
    target = np.asarray(target, dtype=float)
    th = np.array(th0, dtype=float)
    errs = []
    for _ in range(iters):
        _, p = fk(*th)
        e = target - p
        err = float(np.linalg.norm(e))
        errs.append(err)
        if err < tol:
            return th, np.array(errs), True
        J = jacobian(*th)
        dq = dls_step(J, e, lam=lam)
        th = th + alpha * dq
        th = wrap(th)          # 不归一化会累积圈数，真机上就是绞线缆
    return th, np.array(errs), False


# ---------------- 自检 ----------------
def selfcheck(verbose=True):
    """跑一遍前后一致的检查，全部返回 True 才算这块地基可信。"""
    ok = True
    rows = []

    # 1. 齐次矩阵连乘 vs 手推公式（T1 的两级连乘应当等于 fk 的末端）
    t1d, t2d = 30.0, 45.0
    T = (homogeneous(np.radians(t1d), L1, 0)
         @ homogeneous(np.radians(t2d), L2, 0)
         @ np.array([0, 0, 1]))
    _, tip = fk(np.radians(t1d), np.radians(t2d))
    d = np.linalg.norm(T[:2] - tip)
    rows.append(("齐次矩阵连乘 vs 三角公式", d, d < 1e-10))
    ok &= d < 1e-10

    # 2. 雅可比 vs 数值差分
    worst = 0.0
    for t1, t2 in [(0.3, 0.5), (1.1, -0.7), (-0.4, 2.0)]:
        Ja = jacobian(t1, t2)
        for i in range(2):
            dq = np.zeros(2)
            dq[i] = 1e-6
            a = fk(*(np.array([t1, t2]) + dq))[1]
            b = fk(*(np.array([t1, t2]) - dq))[1]
            Jn = (a - b) / 2e-6
            worst = max(worst, np.linalg.norm(Ja[:, i] - Jn))
    rows.append(("解析雅可比 vs 数值差分", worst, worst < 1e-5))
    ok &= worst < 1e-5

    # 3. 解析 IK 回代到 FK
    worst = 0.0
    for tgt in [(1.2, 0.5), (0.6, 1.0), (-0.9, 0.7)]:
        for sol in ik_analytic(*tgt):
            worst = max(worst, np.linalg.norm(fk(*sol)[1] - np.array(tgt)))
    rows.append(("解析 IK 回代误差", worst, worst < 1e-9))
    ok &= worst < 1e-9

    # 4. 数值 IK 能否走到目标
    th, errs, conv = ik_numeric((1.2, 0.5), np.array([0.0, 0.1]))
    rows.append(("数值 IK 收敛", errs[-1], bool(conv)))
    ok &= bool(conv)

    # 5. det J 公式
    worst = 0.0
    for t2 in [0.3, 1.0, -1.7, 2.5]:
        worst = max(worst, abs(np.linalg.det(jacobian(0.4, t2)) - det_jacobian(0.4, t2)))
    rows.append(("det J 公式一致性", worst, worst < 1e-10))
    ok &= worst < 1e-10

    if verbose:
        print("运动学自检 —— 五项" + ("全部通过" if ok else "存在失败"))
        for name, val, good in rows:
            print(f"  [{'OK' if good else 'XX'}] {name:22s} = {val:.3e}")
    return ok


if __name__ == "__main__":
    selfcheck()
