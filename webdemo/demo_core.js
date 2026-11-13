/* demo_core.js —— 二连杆机械臂仿真核心
 * 与 two_link_arm_rl/arm.py、env.py 逐行对应，可在浏览器与 Node 中运行。
 * 所有物理常量与原项目一致：L1=1.0 L2=0.8 maxW=3.0 dt=0.05 tol=0.05 steps=200
 */
(function (root) {
  var L1 = 1.0, L2 = 0.8, MAXW = 3.0, DT = 0.05, TOL = 0.05, MAXSTEPS = 200;

  function wrap(a) {
    return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  }

  function fk(t1, t2) {
    var ex = L1 * Math.cos(t1), ey = L1 * Math.sin(t1);
    return {
      ex: ex, ey: ey,
      tx: ex + L2 * Math.cos(t1 + t2),
      ty: ey + L2 * Math.sin(t1 + t2)
    };
  }

  /* 返回 [j00, j01, j10, j11] */
  function jac(t1, t2) {
    var s1 = Math.sin(t1), c1 = Math.cos(t1);
    var s12 = Math.sin(t1 + t2), c12 = Math.cos(t1 + t2);
    return [-L1 * s1 - L2 * s12, -L2 * s12, L1 * c1 + L2 * c12, L2 * c12];
  }

  /* 阻尼最小二乘：(J J^T + lam^2 I) y = err,  dq = J^T y */
  function dls(J, ex, ey, lam) {
    var a00 = J[0], a01 = J[1], a10 = J[2], a11 = J[3];
    if (lam <= 0) {
      var det = a00 * a11 - a01 * a10;
      if (Math.abs(det) < 1e-14) return [0, 0];
      var i00 = a11 / det, i01 = -a01 / det, i10 = -a10 / det, i11 = a00 / det;
      return [i00 * ex + i01 * ey, i10 * ex + i11 * ey];
    }
    var l2 = lam * lam;
    var m00 = a00 * a00 + a01 * a01 + l2;
    var m01 = a00 * a10 + a01 * a11;
    var m11 = a10 * a10 + a11 * a11 + l2;
    var d = m00 * m11 - m01 * m01;
    if (Math.abs(d) < 1e-16) return [0, 0];
    var y0 = (m11 * ex - m01 * ey) / d;
    var y1 = (-m01 * ex + m00 * ey) / d;
    return [a00 * y0 + a10 * y1, a01 * y0 + a11 * y1];
  }

  function clip(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /* ---------------- 环境 ---------------- */
  function createEnv(cfg) {
    cfg = cfg || {};
    var e = {
      th: [0, 0], target: [0, 0], steps: 0,
      deadzone: cfg.deadzone || 0,
      delay: Math.round(cfg.actionDelay || 0),
      noise: cfg.obsNoise || 0,
      rng: cfg.rng || Math.random,
      aq: []
    };

    e.reset = function (th, target) {
      e.th = [th[0], th[1]];
      e.target = [target[0], target[1]];
      e.steps = 0;
      e.aq = [];
      for (var i = 0; i <= e.delay; i++) e.aq.push([0, 0]);
      return e.obs();
    };

    e.dist = function () {
      var p = fk(e.th[0], e.th[1]);
      var dx = e.target[0] - p.tx, dy = e.target[1] - p.ty;
      return Math.sqrt(dx * dx + dy * dy);
    };

    e.obs = function () {
      var p = fk(e.th[0], e.th[1]);
      var dx = e.target[0] - p.tx, dy = e.target[1] - p.ty;
      if (e.noise > 0) {
        dx += gauss(e.rng) * e.noise;
        dy += gauss(e.rng) * e.noise;
      }
      return [
        Math.cos(e.th[0]), Math.sin(e.th[0]),
        Math.cos(e.th[1]), Math.sin(e.th[1]),
        dx, dy,
        e.th[0] / Math.PI, e.th[1] / Math.PI
      ];
    };

    e.step = function (action) {
      var a = [clip(action[0], -1, 1), clip(action[1], -1, 1)];
      if (e.delay > 0) {
        e.aq.push(a);
        a = e.aq.shift();
      }
      a = applyDeadzone(a, e.deadzone);
      var n1 = wrap(e.th[0] + a[0] * MAXW * DT);
      var n2 = wrap(e.th[1] + a[1] * MAXW * DT);
      e.th[0] = n1; e.th[1] = n2;
      e.steps += 1;
      var d = e.dist();
      var term = d <= TOL;
      var trunc = (!term) && e.steps >= MAXSTEPS;
      return { obs: e.obs(), term: term, trunc: trunc, dist: d };
    };

    return e;
  }

  function applyDeadzone(a, d) {
    if (!(d > 0)) return a;
    var out = [0, 0];
    for (var i = 0; i < 2; i++) {
      var m = Math.abs(a[i]);
      out[i] = m >= d ? Math.sign(a[i]) * (m - d) / (1 - d) : 0;
    }
    return out;
  }

  /* Box-Muller，只取一个分量 */
  function gauss(rng) {
    var u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ---------------- 策略 ---------------- */
  function servoAction(obs, gain, lam) {
    var t1 = Math.atan2(obs[1], obs[0]);
    var t2 = Math.atan2(obs[3], obs[2]);
    var J = jac(t1, t2);
    var dq = dls(J, obs[4], obs[5], lam);
    return [clip(dq[0] * gain / MAXW, -1, 1), clip(dq[1] * gain / MAXW, -1, 1)];
  }

  function mv(W, x) {
    var n = W.length, out = new Array(n);
    for (var i = 0; i < n; i++) {
      var row = W[i], s = 0;
      for (var j = 0; j < row.length; j++) s += row[j] * x[j];
      out[i] = s;
    }
    return out;
  }

  function tanhArr(v) {
    var o = new Array(v.length);
    for (var i = 0; i < v.length; i++) o[i] = Math.tanh(v[i]);
    return o;
  }

  function addB(v, b) {
    for (var i = 0; i < v.length; i++) v[i] += b[i];
    return v;
  }

  /* 与 train_ppo.py 一致：obs[4:6] /= 2，三层 MLP + tanh 输出 */
  function netMu(obs, P) {
    var x = obs.slice();
    x[4] /= 2; x[5] /= 2;
    var h = tanhArr(addB(mv(P.W1, x), P.b1));
    h = tanhArr(addB(mv(P.W2, h), P.b2));
    return tanhArr(addB(mv(P.W3, h), P.b3));
  }

  function netStd(P) {
    var s = [0, 0];
    for (var i = 0; i < 2; i++) {
      s[i] = 0.10 + 0.50 / (1 + Math.exp(-P.log_std[i]));
    }
    return s;
  }

  function netAction(obs, P, sample, rng) {
    var mu = netMu(obs, P);
    if (!sample) return [clip(mu[0], -1, 1), clip(mu[1], -1, 1)];
    var sd = netStd(P);
    rng = rng || Math.random;
    return [
      clip(mu[0] + gauss(rng) * sd[0], -1, 1),
      clip(mu[1] + gauss(rng) * sd[1], -1, 1)
    ];
  }

  /* ---------------- 回合 / 批量评测 ---------------- */
  function runEpisode(policyFn, ep, envCfg, opts) {
    opts = opts || {};
    var env = createEnv(envCfg);
    var obs = env.reset(ep.th, ep.target);
    var path = [], steps = 0, success = false;
    for (var t = 0; t < MAXSTEPS; t++) {
      if (opts.trace) path.push(fk(env.th[0], env.th[1]));
      var r = env.step(policyFn(obs));
      obs = r.obs;
      steps++;
      if (r.term) { success = true; break; }
      if (r.trunc) break;
    }
    if (opts.trace) path.push(fk(env.th[0], env.th[1]));
    return { steps: steps, success: success, dist: env.dist(), path: path };
  }

  function evaluate(policyFn, episodes, envCfg) {
    var ok = 0, st = 0;
    for (var i = 0; i < episodes.length; i++) {
      var r = runEpisode(policyFn, episodes[i], envCfg);
      if (r.success) ok++;
      st += r.steps;
    }
    return {
      rate: ok / episodes.length,
      avgSteps: st / episodes.length
    };
  }

  var API = {
    L1: L1, L2: L2, MAXW: MAXW, DT: DT, TOL: TOL, MAXSTEPS: MAXSTEPS,
    wrap: wrap, fk: fk, jac: jac, dls: dls,
    createEnv: createEnv, servoAction: servoAction,
    netMu: netMu, netStd: netStd, netAction: netAction,
    runEpisode: runEpisode, evaluate: evaluate
  };

  root.ArmCore = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
})(typeof window !== 'undefined' ? window : globalThis);
