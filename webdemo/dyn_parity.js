/*
 * dyn_parity.js —— 与 Python 侧 dyn_parity.py 配套的逐步对拍脚本
 * 用同一组用例跑 JS 实现，输出 parity_js.json 供 Python 比对。
 */
'use strict';

const fs = require('fs');
const path = require('path');
const DYN = require('./dyn_core.js');

const CASES = [
  ['pd_nograv', 100.0, 20.0, false, [0.4, -0.7], [0.9, 0.5], 300],
  ['pd_grav',   100.0, 20.0, true,  [0.4, -0.7], [0.9, 0.5], 300],
  ['pd_soft',    25.0,  8.0, true,  [-1.2, 2.1], [-0.4, -0.9], 400],
];

const TAU_MAX = 60.0;
const out = {};

for (const [name, kp, kd, grav, th0, tgt, n] of CASES) {
  const env = DYN.makeEnv({ dt: 0.01, substeps: 5, tauMax: TAU_MAX });
  env.setState(th0, [0, 0]);
  env.target = tgt;

  const rows = [[env.th[0], env.th[1], env.dth[0], env.dth[1]]];
  for (let i = 0; i < n; i++) {
    const q = [env.th[0], env.th[1]];
    const qd = [env.dth[0], env.dth[1]];
    const qDes = DYN.nearestIK(q, tgt);
    let t1 = kp * DYN.wrap(qDes[0] - q[0]) - kd * qd[0];
    let t2 = kp * DYN.wrap(qDes[1] - q[1]) - kd * qd[1];
    if (grav) {
      const g = DYN.gravityVec(q);
      t1 += g[0];
      t2 += g[1];
    }
    const a = [DYN.clamp(t1 / TAU_MAX, -1, 1), DYN.clamp(t2 / TAU_MAX, -1, 1)];
    env.step(a);
    rows.push([env.th[0], env.th[1], env.dth[0], env.dth[1]]);
  }
  out[name] = rows;
}

const dst = path.join(__dirname, 'parity_js.json');
fs.writeFileSync(dst, JSON.stringify(out));
console.log(`[js] 写出 ${dst}`);
