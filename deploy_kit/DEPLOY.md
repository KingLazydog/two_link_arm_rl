# 阿里云部署作战手册（吴传雄 → 用友政务运维测试）

目标：在阿里云 ECS 上部署 `arm_demo.html`（前端）+ `server.py`（Flask 后端），
把两个链接发给 HR：
  http://<服务器IP>/            ← 机械臂交互演示（nginx 静态站）
  http://<服务器IP>:5000/api/eval  ← 现场跑评测的 JSON 接口

---

## 第 0 步：本机彩排（强烈建议，服务器到手前先跑通一遍）

Windows PowerShell 里：

```powershell
pip install flask numpy --index-url https://pypi.org/simple
cd C:\Users\huaji\WorkBuddy\2026-09-01-18-12-20\two_link_arm_rl\deploy_kit
python server.py
```

浏览器开 `http://127.0.0.1:5000/` 能看到机械臂演示、
`http://127.0.0.1:5000/api/eval` 能看到 JSON → 彩排通过。
（每步输出截图发 W）

---

## 第 1 步：登录服务器

```bash
ssh root@<服务器IP>
# 首次会问 fingerprint，输 yes；然后输 HR 给的密码
# 如果给的是密钥文件：ssh -i 密钥路径 root@<服务器IP>
```

## 第 2 步：装基础软件（Ubuntu）

```bash
apt update
apt install -y nginx python3-pip git
```

（如果是 CentOS：把 apt 换成 yum，nginx 包名相同）

## 第 3 步：拉取你的项目（GitHub 上已有）

```bash
git clone https://github.com/KingLazydog/two_link_arm_rl.git /opt/two_link_arm_rl
```

## 第 4 步：前端交给 nginx

```bash
cp /opt/two_link_arm_rl/arm_demo.html /var/www/html/index.html
systemctl restart nginx
curl -s localhost | head -5     # 应看到 html 开头
```

## 第 5 步：起后端服务

```bash
pip3 install flask numpy -i https://mirrors.aliyun.com/pypi/simple/
# 若报 externally-managed-environment，改用：
# pip3 install flask numpy --break-system-packages -i https://mirrors.aliyun.com/pypi/simple/

cd /opt/two_link_arm_rl/deploy_kit
nohup python3 server.py > server.log 2>&1 &
sleep 2
curl -s "localhost:5000/api/health"    # 应返回 {"status":"ok"}
curl -s "localhost:5000/api/eval?episodes=10"
```

## 第 6 步：⚠️ 阿里云安全组放行（最大的坑）

**服务器内开防火墙没用！** 必须在阿里云控制台：

1. 登录阿里云控制台 → ECS 实例 → 点实例 ID → 「安全组」标签 → 点安全组 ID → 「手动添加」：
   - 放行 **80/TCP**（授权对象 0.0.0.0/0）
   - 放行 **5000/TCP**（授权对象 0.0.0.0/0）
2. 保存后立即生效。

**如果 HR 没给你阿里云控制台权限** → 直接回他一句：
"服务器部署好了，本机测试全通，麻烦您在安全组放行 80 和 5000 两个端口，
或者告诉我用哪个端口方便，我马上改。"
——这本身就是一个加分的运维沟通话术（知道该找谁开什么权限）。

## 第 7 步：公网验证 + 交付

手机浏览器（**关掉 WiFi 用流量**，验证公网可达）：
- `http://<服务器IP>/` → 机械臂演示页
- `http://<服务器IP>:5000/api/eval?episodes=30` → JSON 评测结果

都通了，把两个链接发范先生，附一句：
> 部署完成。第一个链接是项目演示（机械臂交互仿真），第二个是后端评测接口，
> 现场跑 30 回合伺服基线返回成功率。服务器上用的 nginx + Flask + git，
> 有需要再调整的随时说。

---

## 常见故障速查

| 症状 | 原因 | 处理 |
|---|---|---|
| ssh 连不上 | IP 错/密码错/安全组没放 22 | 确认 IP；安全组放行 22/TCP |
| 页面打不开但 ssh 能连 | 安全组没放 80/5000 | 第 6 步 |
| pip 报 externally-managed | 新版 Ubuntu 限制 | 加 `--break-system-packages` |
| git clone 慢/失败 | 服务器到 github 不稳 | 重试 2-3 次；还不行用 `scp` 从本机上传整个目录 |
| 5000 有响应但 curl / 不对 | nginx 没重启 | `systemctl restart nginx` |
| server.py 起不来 | 端口被占 | `kill %1` 后重跑，或换 `port=8000` |

**每一步的输出都截图发 W，卡住立刻停手别自己试错。**
