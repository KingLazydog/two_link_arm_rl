/*
 * dyn_core.js —— 二连杆刚体动力学的浏览器实现
 *
 * 与 Python 侧逐行对应：
 *   dynamics.py : coeffs / mass_matrix / coriolis_vec / gravity_vec /
 *                 forward_dynamics / rk4_step
 *   env.py      : _step_torque（ZOH + RK4 子步 + 角速度限幅）
 *   task_dynamics.py : make_pd / make_computed_torque / _nearest_ik
 *   arm.py      : fk / wrap / ik_analytic
 *
 * 之所以要逐行照抄，是因为这个项目全部结论都建立在"数字可信"上。
 * 浏览器里若换成任何一处近似（比如用欧拉积分、或不限幅角速度），
 * 采样率下限、下垂量这些结论会立刻变味。
 *
 * 同时兼容 Node（module.exports）以便与 Python 对拍。
 */
(function (root) {
  'use strict';

  var GRAV = 9.81;

  // arm.py
  var L1 = 1.0, L2 = 0.8;
  var MAX_DTH = 6.0;   // env.py：关节角速度上限，对应驱动器过流保护
  var TAU_MAX = 60.0;  // env.py：力矩上限，重力矩峰值约 43 N*m

  function params(mScale) {
    var s = (mScale === undefined) ? 1.0 : mScale;
    var m1 = 3.0 * s, m2 = 2.0 * s;
    return {
      m1: m1, m2: m2,
      l1: L1, l2: L2,
      lc1: 0.5 * L1, lc2: 0.5 * L2,
      I1: m1 * L1 * L1 / 12.0,
      I2: m2 * L2 * L2 / 12.0
    };
  }
  var DEFAULT_PARAMS = params(1.0);

  // ---------------- 角度 ----------------
  function wrap(a) {
    return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  }

  // ---------------- 正 / 逆运动学 ----------------
  function fk(t1, t2) {
    var c1 = Math.cos(t1), s1 = Math.sin(t1);
    var c12 = Math.cos(t1 + t2), s12 = Math.sin(t1 + t2);
    var ex = L1 * c1, ey = L1 * s1;
    return {
      elbow: [ex, ey],
      tip: [ex + L2 * c12, ey + L2 * s12]
    };
  }

  function ikAnalytic(x, y) {
    var r = Math.hypot(x, y);
    if (r > L1 + L2 || r < Math.abs(L1 - L2)) return [];
    var c2 = (r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2);
    c2 = Math.max(-1.0, Math.min(1.0, c2));
    var s2pos = Math.sqrt(Math.max(0.0, 1.0 - c2 * c2));
    var out = [];
    var branches = [s2pos, -s2pos];
    for (var i = 0; i < 2; i++) {
      var s2 = branches[i];
      var t2 = Math.atan2(s2, c2);
      var t1 = Math.atan2(y, x) - Math.atan2(L2 * s2, L1 + L2 * c2);
      out.push([wrap(t1), wrap(t2)]);
    }
    return out;
  }

  function nearestIK(q, target) {
    var sols = ikAnalytic(target[0], target[1]);
    if (sols.length === 0) return null;
    var best = null, bestD = Infinity;
    for (var i = 0; i < sols.length; i++) {
      var d1 = wrap(sols[i][0] - q[0]);
      var d2 = wrap(sols[i][1] - q[1]);
      var d = Math.hypot(d1, d2);
      if (d < bestD) { bestD = d; best = sols[i]; }
    }
    return best;
  }

  // ---------------- 刚体动力学 ----------------
  function coeffs(p) {
    p = p || DEFAULT_PARAMS;
    var a1 = p.I1 + p.I2 + p.m1 * p.lc1 * p.lc1
           + p.m2 * (p.l1 * p.l1 + p.lc2 * p.lc2);
    var a2 = p.m2 * p.l1 * p.lc2;
    var a3 = p.I2 + p.m2 * p.lc2 * p.lc2;
    return [a1, a2, a3];
  }

  function massMatrix(th, p) {
    var c = coeffs(p);
    var a1 = c[0], a2 = c[1], a3 = c[2];
    var c2 = Math.cos(th[1]);
    return [
      [a1 + 2.0 * a2 * c2, a3 + a2 * c2],
      [a3 + a2 * c2, a3]
    ];
  }

  function coriolisVec(th, dth, p) {
    var a2 = coeffs(p)[1];
    var s2 = Math.sin(th[1]);
    return [
      -a2 * s2 * (2.0 * dth[0] * dth[1] + dth[1] * dth[1]),
      a2 * s2 * dth[0] * dth[0]
    ];
  }

  function gravityVec(th, p) {
    p = p || DEFAULT_PARAMS;
    var c1 = Math.cos(th[0]);
    var c12 = Math.cos(th[0] + th[1]);
    var g1 = (p.m1 * p.lc1 + p.m2 * p.l1) * GRAV * c1
           + p.m2 * p.lc2 * GRAV * c12;
    var g2 = p.m2 * p.lc2 * GRAV * c12;
    return [g1, g2];
  }

  // 2x2 显式求逆，与 dynamics.forward_dynamics 同一写法
  function forwardDynamics(th, dth, tau, p) {
    p = p || DEFAULT_PARAMS;
    var c = coeffs(p);
    var a1 = c[0], a2 = c[1], a3 = c[2];
    var c2 = Math.cos(th[1]), s2 = Math.sin(th[1]);
    var m11 = a1 + 2.0 * a2 * c2;
    var m12 = a3 + a2 * c2;
    var m22 = a3;
    var rhs0 = tau[0] + a2 * s2 * (2.0 * dth[0] * dth[1] + dth[1] * dth[1])
             - (p.m1 * p.lc1 + p.m2 * p.l1) * GRAV * Math.cos(th[0])
             - p.m2 * p.lc2 * GRAV * Math.cos(th[0] + th[1]);
    var rhs1 = tau[1] - a2 * s2 * dth[0] * dth[0]
             - p.m2 * p.lc2 * GRAV * Math.cos(th[0] + th[1]);
    var det = m11 * m22 - m12 * m12;
    return [(m22 * rhs0 - m12 * rhs1) / det,
            (m11 * rhs1 - m12 * rhs0) / det];
  }

  function rk4Step(th, dth, tau, dt, p) {
    p = p || DEFAULT_PARAMS;
    function f(q, qd) {
      var dd = forwardDynamics(q, qd, tau, p);
      return [qd[0], qd[1], dd[0], dd[1]];
    }
    var y = [th[0], th[1], dth[0], dth[1]];
    var k1 = f([y[0], y[1]], [y[2], y[3]]);
    var y2 = [y[0] + 0.5 * dt * k1[0], y[1] + 0.5 * dt * k1[1],
              y[2] + 0.5 * dt * k1[2], y[3] + 0.5 * dt * k1[3]];
    var k2 = f([y2[0], y2[1]], [y2[2], y2[3]]);
    var y3 = [y[0] + 0.5 * dt * k2[0], y[1] + 0.5 * dt * k2[1],
              y[2] + 0.5 * dt * k2[2], y[3] + 0.5 * dt * k2[3]];
    var k3 = f([y3[0], y3[1]], [y3[2], y3[3]]);
    var y4 = [y[0] + dt * k3[0], y[1] + dt * k3[1],
              y[2] + dt * k3[2], y[3] + dt * k3[3]];
    var k4 = f([y4[0], y4[1]], [y4[2], y4[3]]);
    var out = [0, 0, 0, 0];
    for (var i = 0; i < 4; i++) {
      out[i] = y[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
    return { th: [out[0], out[1]], dth: [out[2], out[3]] };
  }

  // ---------------- 环境（力矩模式）----------------
  // 对应 env.py 的 _step_torque：ZOH + RK4 子步 + 角速度限幅
  var TOL = 0.05;       // 5 cm 判定到达
  var VEL_DT = 0.05;
  var CTRL_DT = 0.01;

  function makeEnv(opt) {
    opt = opt || {};
    var self = {
      dt: opt.dt || CTRL_DT,
      substeps: opt.substeps || Math.max(1, Math.round((opt.dt || CTRL_DT) / 0.002)),
      tauMax: opt.tauMax === undefined ? TAU_MAX : opt.tauMax,
      tol: opt.tol === undefined ? TOL : opt.tol,
      maxDth: MAX_DTH,
      params: opt.params || DEFAULT_PARAMS,
      th: [0, 0],
      dth: [0, 0],
      target: [0, 0],
      steps: 0,
      saturated: 0,

      setState: function (th, dth) {
        this.th = [th[0], th[1]];
        this.dth = [(dth || [0, 0])[0], (dth || [0, 0])[1]];
        this.steps = 0;
        this.saturated = 0;
      },

      dist: function () {
        var p = fk(this.th[0], this.th[1]).tip;
        return Math.hypot(this.target[0] - p[0], this.target[1] - p[1]);
      },

      step: function (action) {
        var a0 = clamp(action[0], -1, 1);
        var a1 = clamp(action[1], -1, 1);
        if (Math.abs(a0) > 0.999 || Math.abs(a1) > 0.999) this.saturated++;

        var tau = [a0 * this.tauMax, a1 * this.tauMax];
        var h = this.dt / this.substeps;
        var th = [this.th[0], this.th[1]];
        var dth = [this.dth[0], this.dth[1]];
        for (var i = 0; i < this.substeps; i++) {
          var s = rk4Step(th, dth, tau, h, this.params);
          th = s.th;
          dth = [clamp(s.dth[0], -this.maxDth, this.maxDth),
                 clamp(s.dth[1], -this.maxDth, this.maxDth)];
        }
        this.th = [wrap(th[0]), wrap(th[1])];
        this.dth = dth;
        this.steps++;
        return this.dist();
      }
    };
    return self;
  }

  // ---------------- 控制器 ----------------
  // env.py / task_dynamics.py：控制器输出归一化到 [-1,1]，环境再乘回 tauMax。
  // 两边必须用同一个 tauMax，否则力矩会被静默放大或缩小。
  function makePD(kp, kd, gravity, params, tauMax) {
    kp = kp === undefined ? 100 : kp;
    kd = kd === undefined ? 20 : kd;
    tauMax = tauMax === undefined ? TAU_MAX : tauMax;
    return function (q, qd, target) {
      var qDes = nearestIK(q, target);
      if (!qDes) return [0, 0];
      var e1 = wrap(qDes[0] - q[0]);
      var e2 = wrap(qDes[1] - q[1]);
      var t1 = kp * e1 - kd * qd[0];
      var t2 = kp * e2 - kd * qd[1];
      if (gravity) {
        var g = gravityVec(q, params || DEFAULT_PARAMS);
        t1 += g[0];
        t2 += g[1];
      }
      return [clamp(t1 / tauMax, -1, 1), clamp(t2 / tauMax, -1, 1)];
    };
  }

  // 计算力矩控制：tau = M*(Kp*e - Kd*qd) + C + G
  function makeComputedTorque(kp, kd, tauMax) {
    kp = kp === undefined ? 100 : kp;
    kd = kd === undefined ? 20 : kd;
    tauMax = tauMax === undefined ? TAU_MAX : tauMax;
    return function (q, qd, target) {
      var qDes = nearestIK(q, target);
      if (!qDes) return [0, 0];
      var v1 = kp * wrap(qDes[0] - q[0]) - kd * qd[0];
      var v2 = kp * wrap(qDes[1] - q[1]) - kd * qd[1];
      var M = massMatrix(q, DEFAULT_PARAMS);
      var C = coriolisVec(q, qd, DEFAULT_PARAMS);
      var G = gravityVec(q, DEFAULT_PARAMS);
      var t1 = M[0][0] * v1 + M[0][1] * v2 + C[0] + G[0];
      var t2 = M[1][0] * v1 + M[1][1] * v2 + C[1] + G[1];
      return [clamp(t1 / tauMax, -1, 1), clamp(t2 / tauMax, -1, 1)];
    };
  }

  function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }

  // ---------------- ZOH 稳定性（谱半径）----------------
  // 对应 task_dynamics.py 的 zoh_radius。
  //
  // 为什么必须单独算这个数：一个已经发散的控制环在乱撞中照样可能穿过目标球，
  // 成功率看上去还不错。成功率回答"快不快"，谱半径回答"稳不稳"。
  function inv2(M) {
    var det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    return [[M[1][1] / det, -M[0][1] / det],
            [-M[1][0] / det, M[0][0] / det]];
  }

  function dGdq(q, p) {
    var h = 1e-6, J = [[0, 0], [0, 0]];
    for (var i = 0; i < 2; i++) {
      var ep = [0, 0], em = [0, 0];
      ep[i] = h; em[i] = -h;
      var gp = gravityVec([q[0] + ep[0], q[1] + ep[1]], p);
      var gm = gravityVec([q[0] + em[0], q[1] + em[1]], p);
      J[0][i] = (gp[0] - gm[0]) / (2 * h);
      J[1][i] = (gp[1] - gm[1]) / (2 * h);
    }
    return J;
  }

  // 4x4 矩阵乘法
  function mul4(A, B) {
    var C = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
    for (var i = 0; i < 4; i++)
      for (var j = 0; j < 4; j++) {
        var s = 0;
        for (var k = 0; k < 4; k++) s += A[i][k] * B[k][j];
        C[i][j] = s;
      }
    return C;
  }

  // 同时积分 Phi = e^{AT} 与 Gamma = ∫e^{A(T-s)}B ds（RK4）
  function propagate(A, B, T, h) {
    h = h || 2e-5;
    var X = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
    var Z = [[0,0],[0,0],[0,0],[0,0]];

    // R = P + s*Q（4x4）
    function axpy4(P, Q, s) {
      var R = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
      for (var i = 0; i < 4; i++)
        for (var j = 0; j < 4; j++) R[i][j] = P[i][j] + s * Q[i][j];
      return R;
    }
    // R = P + s*Q（4x2）
    function axpyZ(P, Q, s) {
      var R = [[0,0],[0,0],[0,0],[0,0]];
      for (var i = 0; i < 4; i++)
        for (var j = 0; j < 2; j++) R[i][j] = P[i][j] + s * Q[i][j];
      return R;
    }
    // A@Z + B（4x2）
    function fZ(Zin) {
      var R = [[0,0],[0,0],[0,0],[0,0]];
      for (var i = 0; i < 4; i++)
        for (var j = 0; j < 2; j++) {
          var s = B[i][j];
          for (var k = 0; k < 4; k++) s += A[i][k] * Zin[k][j];
          R[i][j] = s;
        }
      return R;
    }

    var t = 0;
    while (t < T - 1e-15) {
      var s = Math.min(h, T - t);
      var k1x = mul4(A, X),                       k1z = fZ(Z);
      var k2x = mul4(A, axpy4(X, k1x, s / 2)),    k2z = fZ(axpyZ(Z, k1z, s / 2));
      var k3x = mul4(A, axpy4(X, k2x, s / 2)),    k3z = fZ(axpyZ(Z, k2z, s / 2));
      var k4x = mul4(A, axpy4(X, k3x, s)),        k4z = fZ(axpyZ(Z, k3z, s));

      // X <- X + s/6*(k1 + 2k2 + 2k3 + k4)
      var sx = axpy4(axpy4(axpy4(axpy4(k4x, k3x, 2), k2x, 2), k1x, 1),
                     [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], 1);
      X = axpy4(X, sx, s / 6);
      var sz = axpyZ(axpyZ(axpyZ(axpyZ(k4z, k3z, 2), k2z, 2), k1z, 1),
                     [[0,0],[0,0],[0,0],[0,0]], 1);
      Z = axpyZ(Z, sz, s / 6);
      t += s;
    }
    return { Phi: X, Gam: Z };
  }

  // 4x4 复矩阵特征值模的最大值：Faddeev-LeVerrier 求特征多项式，Durand-Kerner 求根
  function spectralRadius(A) {
    var n = 4;
    var c = new Array(n + 1).fill(0);
    c[n] = 1;
    var Mk = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];   // M_0 = 0
    for (var k = 1; k <= n; k++) {
      // M_k = A @ (M_{k-1} + c_{n-k+1} * I)
      var S = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
      for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) S[i][j] = Mk[i][j];
      var addC = c[n - k + 1];
      for (var i2 = 0; i2 < n; i2++) S[i2][i2] += addC;
      Mk = mul4(A, S);
      var tr = 0;
      for (var i3 = 0; i3 < n; i3++) tr += Mk[i3][i3];
      c[n - k] = -tr / k;
    }
    return maxRootMod(c);
  }

  function maxRootMod(c) {
    // 求 x^n + c[1]x^{n-1} + ... + c[n] = 0 的全部根，返回最大模
    var n = c.length - 1;
    function peval(zr, zi) {
      // 多项式 c[n]x^n + c[n-1]x^{n-1} + ... + c[0]，最高次 c[n]=1
      var pr = c[n], pi = 0;
      for (var i = n - 1; i >= 0; i--) {
        var nr = pr * zr - pi * zi + c[i];
        var ni = pr * zi + pi * zr;
        pr = nr; pi = ni;
      }
      return [pr, pi];
    }
    var zr = new Array(n), zi = new Array(n);
    for (var i = 0; i < n; i++) {
      var ang = 0.4 + 2 * Math.PI * i / n;
      var r = 0.4 + i * 0.1;
      zr[i] = r * Math.cos(ang);
      zi[i] = r * Math.sin(ang);
    }
    for (var it = 0; it < 500; it++) {
      var maxDelta = 0;
      for (var i2 = 0; i2 < n; i2++) {
        var p = peval(zr[i2], zi[i2]);
        var dr = 1, di = 0;
        for (var j = 0; j < n; j++) {
          if (j === i2) continue;
          var ar = zr[i2] - zr[j], ai = zi[i2] - zi[j];
          var nr2 = dr * ar - di * ai;
          var ni2 = dr * ai + di * ar;
          dr = nr2; di = ni2;
        }
        var den = dr * dr + di * di;
        if (den < 1e-300) continue;
        var qr = (p[0] * dr + p[1] * di) / den;
        var qi = (p[1] * dr - p[0] * di) / den;
        zr[i2] -= qr; zi[i2] -= qi;
        maxDelta = Math.max(maxDelta, Math.hypot(qr, qi));
      }
      if (maxDelta < 1e-15) break;
    }
    var best = 0;
    for (var i4 = 0; i4 < n; i4++) best = Math.max(best, Math.hypot(zr[i4], zi[i4]));
    return best;
  }

  function zohRadius(kp, kd, dt, pose, p) {
    p = p || DEFAULT_PARAMS;
    var q0 = pose || [0.0, 0.6];
    var Minv = inv2(massMatrix(q0, p));
    var dG = dGdq(q0, p);
    var MG = [[0, 0], [0, 0]];
    for (var i = 0; i < 2; i++) for (var j = 0; j < 2; j++) {
      var s = 0;
      for (var k = 0; k < 2; k++) s += Minv[i][k] * dG[k][j];
      MG[i][j] = s;
    }
    var A = [[0,0,1,0],[0,0,0,1],[-MG[0][0],-MG[0][1],0,0],[-MG[1][0],-MG[1][1],0,0]];
    var B = [[0,0],[0,0],[Minv[0][0],Minv[0][1]],[Minv[1][0],Minv[1][1]]];
    var pr = propagate(A, B, dt);
    // K = [-kp*I, -kd*I]，形状 2x4；闭环 = Phi + Gam @ K
    // Gam 是 4x2，K 是 2x4，Gam@K 的第 cc 列 = Gam[:,0]*K[0][cc] + Gam[:,1]*K[1][cc]
    var CL = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
    for (var r = 0; r < 4; r++) for (var cc = 0; cc < 4; cc++) {
      // K = [-kp*I, -kd*I]（2x4），Gam@K 的第 cc 列 = Gam[:, cc%2] * gain
      var gain = (cc < 2) ? -kp : -kd;
      var gcol = cc % 2;
      CL[r][cc] = pr.Phi[r][cc] + pr.Gam[r][gcol] * gain;
    }
    return spectralRadius(CL);
  }


  // ---------------- 评测 ----------------
  // 与 env.py 的任务定义一致：姿态在 (-pi,pi) 均匀取，目标在工作空间环内取
  function makeRng(seed) {
    // 与 numpy 不同的发生器，故"浏览器实测"与 Python 的具体样本不同，
    // 但统计口径（采样分布、判定容差、回合长度）完全一致。
    var s = (seed >>> 0) || 1;
    return function () {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  }

  function sampleTarget(rng) {
    var r = 0.35 + rng() * (1.65 - 0.35);   // |L1-L2|+0.15 .. L1+L2-0.15
    var a = -Math.PI + rng() * 2 * Math.PI;
    return [r * Math.cos(a), r * Math.sin(a)];
  }

  function evaluate(ctrl, opt) {
    opt = opt || {};
    var episodes = opt.episodes || 20;
    var horizon = opt.horizon || 10.0;
    var dt = opt.dt || CTRL_DT;
    var maxSteps = Math.round(horizon / dt);
    var env = makeEnv({
      dt: dt,
      substeps: opt.substeps || Math.max(1, Math.round(dt / 0.002)),
      tauMax: opt.tauMax,
      params: opt.params
    });
    var rng = makeRng(opt.seed || 12345);

    var succ = 0, times = [], maxRate = 0, satRatio = 0;

    for (var ep = 0; ep < episodes; ep++) {
      var th0 = [-Math.PI + rng() * 2 * Math.PI, -Math.PI + rng() * 2 * Math.PI];
      var tgt = sampleTarget(rng);
      env.setState(th0, [0, 0]);
      env.target = tgt;

      var reached = false;
      for (var k = 0; k < maxSteps; k++) {
        var a = ctrl([env.th[0], env.th[1]], [env.dth[0], env.dth[1]], tgt);
        var d = env.step(a);
        var rate = Math.max(Math.abs(env.dth[0]), Math.abs(env.dth[1]));
        if (rate > maxRate) maxRate = rate;
        if (d <= env.tol) { reached = true; times.push((k + 1) * dt); break; }
      }
      if (reached) succ++;
      satRatio += env.saturated / Math.max(1, env.steps);
    }

    return {
      success: succ / episodes,
      succCount: succ,
      episodes: episodes,
      meanTime: times.length ? times.reduce(function (a, b) { return a + b; }, 0) / times.length : null,
      maxRate: maxRate,
      satRatio: satRatio / episodes
    };
  }

  // ---------------- 静态悬停下垂量 ----------------
  // 对应 task_dynamics.py 的 static_droop：把手臂放到 pose，只命令"保持不动"，
  // 看它 5 秒后实际停在离 pose 多远。唯一外力是重力，唯一抵抗力是 Kp 的弹簧力。
  function staticDroop(kp, kd, gravity, pose, dt, tauMax) {
    kd = kd === undefined ? 20 : kd;
    pose = pose || [0.0, 0.6];
    dt = dt || CTRL_DT;
    tauMax = tauMax === undefined ? TAU_MAX : tauMax;
    var env = makeEnv({ dt: dt, substeps: Math.max(1, Math.round(dt / 0.002)), tauMax: tauMax });
    env.setState(pose, [0, 0]);
    var p0 = fk(pose[0], pose[1]).tip;
    var n = Math.round(5.0 / dt);
    for (var i = 0; i < n; i++) {
      var q = [env.th[0], env.th[1]];
      var qd = [env.dth[0], env.dth[1]];
      var t1 = kp * wrap(pose[0] - q[0]) - kd * qd[0];
      var t2 = kp * wrap(pose[1] - q[1]) - kd * qd[1];
      if (gravity) { var g = gravityVec(q); t1 += g[0]; t2 += g[1]; }
      env.step([clamp(t1 / tauMax, -1, 1), clamp(t2 / tauMax, -1, 1)]);
    }
    var p1 = fk(env.th[0], env.th[1]).tip;
    return Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
  }

  var DYN = {
    GRAV: GRAV, L1: L1, L2: L2, TOL: TOL, TAU_MAX: TAU_MAX, MAX_DTH: MAX_DTH,
    CTRL_DT: CTRL_DT, VEL_DT: VEL_DT,
    params: params, DEFAULT_PARAMS: DEFAULT_PARAMS,
    wrap: wrap, clamp: clamp, fk: fk, ikAnalytic: ikAnalytic, nearestIK: nearestIK,
    coeffs: coeffs, massMatrix: massMatrix, coriolisVec: coriolisVec,
    gravityVec: gravityVec, forwardDynamics: forwardDynamics, rk4Step: rk4Step,
    makeEnv: makeEnv, makePD: makePD, makeComputedTorque: makeComputedTorque,
    staticDroop: staticDroop,
    makeRng: makeRng, sampleTarget: sampleTarget, evaluate: evaluate,
    zohRadius: zohRadius
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = DYN;
  else root.DYN = DYN;
})(typeof window !== 'undefined' ? window : global);
