/*
 * dyn_check.js —— 检查浏览器实现在「评测口径」下能否复现 Python 的定型数字
 * 不是为了逐位相等（随机样本发生器不同），而是看趋势与量级是否一致。
 */
'use strict';
const DYN = require('./dyn_core.js');

const TAU = 60.0;
const HZ = { 20: 0.05, 50: 0.02, 100: 0.01, 500: 0.002 };

function run(label, ctrlFactory, hz, mScale) {
  const dt = HZ[hz];
  const envParams = DYN.params(mScale === undefined ? 1.0 : mScale);
  // 控制器永远以为质量是标称值（1.0 倍），真实机构由 envParams 决定
  const ctrl = ctrlFactory(1.0);
  const r = DYN.evaluate(ctrl, {
    episodes: 20, horizon: 10.0, dt: dt, tauMax: TAU,
    params: envParams, seed: 20261113
  });
  console.log(
    `  ${label.padEnd(26)}${String(hz).padStart(4)}Hz  ` +
    `成功 ${(r.success * 100).toFixed(1).padStart(5)}%  ` +
    `到达 ${r.meanTime === null ? '  -  ' : r.meanTime.toFixed(2) + 's'}  ` +
    `最大关节速度 ${r.maxRate.toFixed(2)} rad/s  饱和 ${(r.satRatio * 100).toFixed(0)}%`
  );
  return r;
}

console.log('实验 A/B · 重力补偿值多少（100 Hz）');
run('裸 PD（不补重力）', () => DYN.makePD(100, 20, false), 100);
run('PD + 重力补偿',     () => DYN.makePD(100, 20, true), 100);
run('计算力矩（全模型）', () => DYN.makeComputedTorque(100, 20), 100);

console.log('\n实验 C · 采样率（PD + 重力补偿）');
[500, 100, 50, 20].forEach(hz => run('PD + 重力补偿', () => DYN.makePD(100, 20, true), hz));

console.log('\n实验七 · 模型失配（真实质量 = 控制器以为的 2 倍，100 Hz）');
run('PD + 重力补偿（失配）', () => DYN.makePD(100, 20, true), 100, 2.0);
run('裸 PD（失配）',          () => DYN.makePD(100, 20, false), 100, 2.0);
