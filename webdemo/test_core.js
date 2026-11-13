/* test_core.js —— 校验网页端的 JS 物理与 Python 端逐位一致
 * 用法： node webdemo/test_core.js
 */
const fs = require('fs');
const path = require('path');
const C = require('./demo_core.js');

const dataPath = path.join(__dirname, 'demo_data.json');
const DATA = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
const EPS = DATA.episodes;

function servo(gain) { return o => C.servoAction(o, gain, 0.05); }
function net(P, sample) { return o => C.netAction(o, P, sample, Math.random); }

function run(label, pol, cfg, expected) {
  const r = C.evaluate(pol, EPS, cfg);
  const ok = Math.abs(r.rate - expected) <= 0.06;
  console.log(
    (ok ? '  OK  ' : '  ??  ') +
    label.padEnd(26) +
    (r.rate * 100).toFixed(1).padStart(6) + '%   ' +
    r.avgSteps.toFixed(1).padStart(6) + ' 步   python ' +
    (expected * 100).toFixed(1) + '%'
  );
  return ok;
}

console.log('回合数', EPS.length, '  权重 ppo/ bc 维度',
  DATA.policyPPO.W1.length + 'x' + DATA.policyPPO.W1[0].length);

let all = true;
all &= run('伺服 gain=1.0 dz=0', servo(1.0), { deadzone: 0 }, 0.98);
all &= run('伺服 gain=32 dz=0.25', servo(32), { deadzone: 0.25 }, 0.80);
all &= run('伺服 gain=1.0 dz=0.25', servo(1.0), { deadzone: 0.25 }, 0.00);
all &= run('BC 确定性 dz=0', net(DATA.policyBC, false), { deadzone: 0 }, 0.78);
all &= run('BC 确定性 dz=0.25', net(DATA.policyBC, false), { deadzone: 0.25 }, 0.00);
all &= run('PPO 确定性 dz=0.25', net(DATA.policyPPO, false), { deadzone: 0.25 }, 0.00);
all &= run('PPO 采样 dz=0.25', net(DATA.policyPPO, true), { deadzone: 0.25 }, 0.96);
all &= run('PPO 采样 dz=0', net(DATA.policyPPO, true), { deadzone: 0 }, 0.98);

console.log(all ? '\n全部一致' : '\n存在偏差，请检查 demo_core.js');
