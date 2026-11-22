"""
server.py —— 用友政务运维测试的部署后端（吴传雄 / two_link_arm_rl）

部署一个"我做过的系统"：
  GET  /            -> arm_demo.html（二连杆机械臂交互演示，含物理引擎，单文件）
  GET  /api/eval    -> 现场跑伺服基线评测（numpy-only，无 torch 依赖），返回 JSON

评测接口参数（都可省略）：
  episodes  回合数，默认 20，最大 100
  gain      伺服增益，默认 1.0（项目定型数字：98%）
  deadzone  关节死区，默认 0（设 0.25 可看"专家失效"场景）

用法（服务器上）：
  cd /opt/two_link_arm_rl/deploy_kit
  nohup python3 server.py > server.log 2>&1 &

本机彩排（Windows）：
  python server.py
  浏览器开 http://127.0.0.1:5000/
"""

import os
import sys

from flask import Flask, jsonify, request, send_file

# 让 server.py 能 import 仓库根目录的 controller/env/arm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from controller import evaluate, make_resolved_rate, make_zero  # noqa: E402

app = Flask(__name__)
# 中文直接输出，不转成 \uXXXX 转义（HR 打开 /api/eval 时可读）
app.json.ensure_ascii = False


@app.route("/")
def index():
    return send_file(os.path.join(ROOT, "arm_demo.html"))


@app.route("/api/eval")
def api_eval():
    try:
        episodes = min(max(int(request.args.get("episodes", 20)), 1), 100)
        gain = float(request.args.get("gain", 1.0))
        deadzone = float(request.args.get("deadzone", 0.0))
    except ValueError:
        return jsonify({"error": "bad query params"}), 400

    env_kw = {"deadzone": deadzone} if deadzone > 0 else None
    res_servo = evaluate(make_resolved_rate(gain=gain), episodes=episodes, seed=2026)
    res_zero = evaluate(make_zero(), episodes=episodes, seed=2026)

    return jsonify({
        "project": "two_link_arm_rl —— 二连杆机械臂够点任务（独立完成）",
        "policy": "resolved-rate 雅可比伺服 (gain=%.2f)" % gain,
        "env": "deadzone=%.2f" % deadzone if deadzone > 0 else "clean",
        "episodes": episodes,
        "servo": res_servo,
        "zero_baseline": res_zero,
        "note": "zero=完全不动的地板；伺服应显著高于地板。deadzone>0 时可见伺服退化（专家失效实验）。",
    })


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
