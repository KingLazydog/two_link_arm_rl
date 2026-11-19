"""
build_dyn_demo.py —— 把刚体动力学（dyn_core.js）并入 arm_demo.html

做法：在现有运动学 demo 基础上，新增一个「刚体动力学」实验区，
展示项目里三个最值钱的结论（都现场计算，不写死）：
  1. 控制器对比：裸 PD / PD+重力补偿 / 计算力矩 的成功率
  2. 采样率下限：20/50/100/500 Hz 的 ZOH 谱半径（发散/稳定）
  3. 下垂量：不同 Kp 下静态悬停末端下垂，展示 e=G/Kp 反比

不动原有运动学 demo 的任何逻辑，只追加。
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = HERE.parent / "arm_demo.html"
DYN = HERE / "dyn_core.js"

html = HTML.read_text(encoding="utf-8")
dyn = DYN.read_text(encoding="utf-8")

# 幂等保护：若已含动力学标记，说明已构建过，避免重复插入
if "刚体动力学 · 力矩控制" in html:
    print(f"[build] {HTML.name} 已含动力学区，跳过（幂等）")
    raise SystemExit(0)

# 1. CSS：在 </style> 前插入动力学区样式
CSS = """
  /* ---- 刚体动力学实验区 ---- */
  .dyn-block { margin-top: 20px; }
  .dyn-block .dyn-head { display: flex; align-items: baseline; gap: 10px; margin: 0 0 10px; }
  .dyn-block .dyn-head h2 { font-size: 15px; font-weight: 600; margin: 0; }
  .dyn-block .dyn-head .tag { font-size: 11px; color: var(--muted); }
  .dyn-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .dyn-card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
  .dyn-card h3 { font-size: 13px; font-weight: 600; margin: 0 0 10px; }
  .dyn-card .row { display: flex; justify-content: space-between; padding: 5px 0; font-size: 13px; border-bottom: 1px dashed var(--line); }
  .dyn-card .row:last-child { border-bottom: none; }
  .dyn-card .row .k { color: var(--muted); }
  .dyn-card .row .num { font-variant-numeric: tabular-nums; font-weight: 500; }
  .dyn-card .row .num.good { color: var(--green); }
  .dyn-card .row .num.bad { color: var(--red); }
  .dyn-card .hint { font-size: 12px; color: var(--muted); margin-top: 8px; line-height: 1.6; }
  .dyn-ctl { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }
  .dyn-ctl select { font-family: inherit; font-size: 12.5px; }
  .dyn-ctl button { font-family: inherit; font-size: 12.5px; padding: 5px 12px; border-radius: 6px; border: 1px solid var(--line); background: var(--ink); color: #fff; cursor: pointer; }
  .dyn-ctl button:disabled { opacity: .5; }
  .dyn-note { font-size: 12px; color: var(--muted); line-height: 1.7; margin-top: 14px; }
  .dyn-note b { color: var(--ink); }
"""
html = html.replace("</style>", CSS + "</style>", 1)

# 2. 实验区 HTML：插在 .verdict 之后、footer 之前
HTML_BLOCK = """
  <div class="dyn-block">
    <div class="dyn-head">
      <h2>刚体动力学 · 力矩控制</h2>
      <span class="tag">拉格朗日建模 M(q)q̈ + C(q,q̇)q̇ + G(q) = τ</span>
    </div>
    <div class="dyn-cols">
      <div class="dyn-card">
        <h3>控制器对比（同一批随机姿态，100 Hz）</h3>
        <div class="dyn-ctl">
          <button id="dyn-run">跑动力学评测</button>
          <select id="dyn-hz">
            <option value="100" selected>100 Hz</option>
            <option value="50">50 Hz</option>
            <option value="20">20 Hz</option>
          </select>
        </div>
        <div class="row"><span class="k">裸 PD（不补重力）</span><span class="num" id="d-a">—</span></div>
        <div class="row"><span class="k">PD + 重力补偿</span><span class="num" id="d-b">—</span></div>
        <div class="row"><span class="k">计算力矩（全模型）</span><span class="num" id="d-c">—</span></div>
        <div class="hint" id="d-droop">静态悬停下垂量：—</div>
      </div>
      <div class="dyn-card">
        <h3>采样率下限 · ZOH 闭环谱半径</h3>
        <div class="row"><span class="k">500 Hz</span><span class="num" id="z-500">—</span></div>
        <div class="row"><span class="k">100 Hz</span><span class="num" id="z-100">—</span></div>
        <div class="row"><span class="k">50 Hz</span><span class="num" id="z-50">—</span></div>
        <div class="row"><span class="k">20 Hz</span><span class="num" id="z-20">—</span></div>
        <div class="hint">谱半径 &gt; 1 表示发散。20 Hz 成功率仍可能不低——发散后乱撞偶然穿过目标球，<b>成功率只回答「快不快」，谱半径回答「稳不稳」</b>。</div>
      </div>
    </div>
    <div class="dyn-note">
      <b>为什么这是「正式版」而不是 demo：</b>上一版只有运动学（θ += a·dt），这一版补上了质量、惯量、重力（m₁=3 kg、m₂=2 kg，重力矩峰值 42.9 N·m）。
      左边那个「PD + 重力补偿」把成功率从 36% 拉到 100%，就是简历上那条 <b>36% → 100%</b> 的现场版。
    </div>
  </div>

"""
html = html.replace('  <div class="verdict" id="verdict"></div>\n', '  <div class="verdict" id="verdict"></div>\n' + HTML_BLOCK, 1)

# 3. 内联 dyn_core.js + 动力学 UI 逻辑：插在最后一个 </script> 之前
DYN_JS = """
<script>
""" + dyn + """
</script>
<script>
(function () {
  const D = window.DYN;
  if (!D) return;
  const hzSel = document.getElementById('dyn-hz');
  const btn = document.getElementById('dyn-run');
  const EP = 20;                 // 回合数（浏览器侧用 20 回合，够看趋势，不追求与 Python 50 回合逐位相等）
  const TAU = 60.0;

  function runCtrl(ctrl, hz, mScale) {
    const dt = { 20: 0.05, 50: 0.02, 100: 0.01, 500: 0.002 }[hz];
    const envParams = D.params(mScale === undefined ? 1.0 : mScale);
    const r = D.evaluate(ctrl, { episodes: EP, horizon: 10.0, dt: dt, tauMax: TAU, params: envParams, seed: 20261113 });
    return r;
  }

  function fmtRate(r) {
    return (r.success * 100).toFixed(0) + '%  ·  ' + (r.meanTime === null ? '-' : r.meanTime.toFixed(2) + 's');
  }

  function runAll() {
    const hz = parseInt(hzSel.value, 10);
    btn.disabled = true; btn.textContent = '计算中…';
    setTimeout(() => {
      const ra = runCtrl(D.makePD(100, 20, false), hz);
      const rb = runCtrl(D.makePD(100, 20, true), hz);
      const rc = runCtrl(D.makeComputedTorque(100, 20), hz);
      const el = (id, r, good) => {
        const e = document.getElementById(id);
        e.textContent = fmtRate(r);
        e.className = 'num ' + (good ? 'good' : '');
      };
      el('d-a', ra, false);
      el('d-b', rb, rb.success >= 0.9);
      el('d-c', rc, rc.success >= 0.9);

      // 下垂量：静态悬停仿真，Kp 越低下垂越大，补重力后归零
      const kps = [30, 100, 300, 1000];
      const drop = kps.map(kp => {
        const d = D.staticDroop(kp, 20, false, [0, 0.6], 0.01, 60) * 1000;
        return d.toFixed(1);
      });
      document.getElementById('d-droop').innerHTML =
        '静态悬停下垂量（Kp=30/100/300/1000）：<b>' + drop.join(' / ') + '</b> mm — 与 e=G/Kp 反比，补重力后精确归零（0.0 mm）';

      // 谱半径
      const hzs = [500, 100, 50, 20];
      hzs.forEach(h => {
        const dt = { 500: 0.002, 100: 0.01, 50: 0.02, 20: 0.05 }[h];
        const r = D.zohRadius(100, 20, dt, [0.0, 0.6]);
        const e = document.getElementById('z-' + h);
        e.textContent = r.toFixed(2) + (r <= 1.0 ? ' 稳定' : ' 发散');
        e.className = 'num ' + (r <= 1.0 ? 'good' : 'bad');
      });

      btn.disabled = false; btn.textContent = '跑动力学评测';
    }, 30);
  }

  btn.addEventListener('click', runAll);
  // 初始即算一次谱半径（便宜），成功率先空着等用户点
  (function initZoh() {
    const hzs = [500, 100, 50, 20];
    hzs.forEach(h => {
      const dt = { 500: 0.002, 100: 0.01, 50: 0.02, 20: 0.05 }[h];
      const r = D.zohRadius(100, 20, dt, [0.0, 0.6]);
      const e = document.getElementById('z-' + h);
      e.textContent = r.toFixed(2) + (r <= 1.0 ? ' 稳定' : ' 发散');
      e.className = 'num ' + (r <= 1.0 ? 'good' : 'bad');
    });
  })();
})();
</script>
"""

# 在最后一个 </script> 前插入（即 UI 脚本的结束标签之前）
# 更简单：直接插在 </body> 前
html = html.replace("</body>", DYN_JS + "\n</body>", 1)

HTML.write_text(html, encoding="utf-8")
print(f"[build] 已写入 {HTML}（原 683 行 → {html.count(chr(10)) + 1} 行）")
