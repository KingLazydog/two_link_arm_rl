"""build_demo.py —— 把权重、回合数据、仿真核心内联进单文件 HTML

依赖 export_demo_data.py 先跑过（生成 webdemo/demo_data.json）
产物：two_link_arm_rl/arm_demo.html（单文件，双击即开，无需服务器）
"""

import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "webdemo", "demo_core.js")
DATA = os.path.join(HERE, "webdemo", "demo_data.json")
TPL = os.path.join(HERE, "webdemo", "template.html")
OUT = os.path.join(HERE, "arm_demo.html")


def main():
    with open(CORE, encoding="utf-8") as f:
        core = f.read()
    with open(DATA, encoding="utf-8") as f:
        data = json.load(f)
    with open(TPL, encoding="utf-8") as f:
        tpl = f.read()

    html = tpl.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    html = html.replace("__CORE__", core)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print("已生成 %s  (%.1f KB)" % (OUT, os.path.getsize(OUT) / 1024))


if __name__ == "__main__":
    main()
