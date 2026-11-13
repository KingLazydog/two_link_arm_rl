"""export_policy_c.py —— 把训练好的策略导出成可在 MCU 上跑的 C 代码

产物（webdemo/c/）：
    policy_ppo.h     接口声明
    policy_ppo.c     权重（float 数组）+ 前向推理
    test_policy.c    与 Python 对拍的测试（tcc 可直接跑）

为什么值得做：
    训练在 Python（PyTorch），执行在 MCU（C）。这条"训练→部署"的链路
    才是工业界 Sim2Real 的真实形态。权重是纯数据，与框架无关。

体积：64*8 + 64*64 + 2*64 + 130 = 4866 个 float ≈ 19.5 KB Flash
    （STM32F103C8T6 有 64 KB Flash，装得下）
"""

import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "webdemo", "c")


def load_actor(path):
    ck = torch.load(path, map_location="cpu")
    return {
        "W1": ck["actor.0.weight"].numpy().astype(np.float64),
        "b1": ck["actor.0.bias"].numpy().astype(np.float64),
        "W2": ck["actor.2.weight"].numpy().astype(np.float64),
        "b2": ck["actor.2.bias"].numpy().astype(np.float64),
        "W3": ck["actor.4.weight"].numpy().astype(np.float64),
        "b3": ck["actor.4.bias"].numpy().astype(np.float64),
        "log_std": ck["log_std"].numpy().astype(np.float64),
    }


def forward(obs, P):
    x = np.array(obs, dtype=np.float64)
    x[4] /= 2.0
    x[5] /= 2.0
    h = np.tanh(P["W1"] @ x + P["b1"])
    h = np.tanh(P["W2"] @ h + P["b2"])
    return np.tanh(P["W3"] @ h + P["b3"])


def fmt_mat(name, arr, per_line=6):
    rows, cols = arr.shape
    lines = ["static const float %s[%d][%d] = {" % (name, rows, cols)]
    for r in range(rows):
        vals = ["%.8ff" % v for v in arr[r]]
        chunks = [", ".join(vals[i:i + per_line]) for i in range(0, cols, per_line)]
        lines.append("    { " + ",\n      ".join(chunks) + " }" +
                     ("," if r < rows - 1 else ""))
    lines.append("};")
    return "\n".join(lines)


def fmt_vec(name, arr):
    vals = ["%.8ff" % v for v in arr]
    return "static const float %s[%d] = {\n    %s\n};" % (
        name, len(arr), ", ".join(vals))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    P = load_actor(os.path.join(HERE, "sim2real_policy.pt"))

    header = """/* policy_ppo.h —— 二连杆机械臂策略（由 PyTorch 权重导出，请勿手改）
 *
 * obs[8] = { cos(th1), sin(th1), cos(th2), sin(th2), dx, dy, th1/pi, th2/pi }
 * act[2] = 归一化关节角速度，已裁剪到 [-1, 1]
 *
 * 生成脚本：export_policy_c.py
 */
#ifndef POLICY_PPO_H
#define POLICY_PPO_H

#ifdef __cplusplus
extern "C" {
#endif

/* 确定性输出（等于网络均值）。实机部署建议用 sample 版本，
 * 因为该策略是在带摩擦死区的环境里用采样方式训练/验证的。 */
void policy_ppo_mu(const float obs[8], float act[2]);

/* 采样输出。seed 由调用方维护（每次传入同一个变量地址即可）。 */
void policy_ppo_sample(const float obs[8], float act[2], unsigned int *seed);

/* 策略标准差（两个关节），供调试观察 */
void policy_ppo_std(float std_out[2]);

#ifdef __cplusplus
}
#endif

#endif /* POLICY_PPO_H */
"""

    body = """/* policy_ppo.c —— 由 export_policy_c.py 自动生成，请勿手改
 *
 * 网络：8 -> 64 (tanh) -> 64 (tanh) -> 2 (tanh)
 * 权重：%d 个 float，约 %.1f KB Flash
 */
#include <math.h>
#include "policy_ppo.h"

/* Windows 下的 Tiny C Compiler 没有 float 版的 tanh/exp，用 double 版包一层。
 * Keil / arm-none-eabi-gcc 有 tanhf/expf，不受影响。 */
#if defined(__TINYC__)
#define tanhf(x) ((float)tanh((double)(x)))
#define expf(x)  ((float)exp((double)(x)))
#endif

#define HID 64
#define OBS 8

%s

%s

%s

%s

%s

%s

static const float LOG_STD[2] = { %.8ff, %.8ff };

/* 标准差软映射到 [0.10, 0.60]，与训练脚本一致 */
void policy_ppo_std(float std_out[2])
{
    int i;
    for (i = 0; i < 2; i++) {
        std_out[i] = 0.10f + 0.50f / (1.0f + expf(-LOG_STD[i]));
    }
}

void policy_ppo_mu(const float obs[8], float act[2])
{
    float x[OBS];
    float h1[HID], h2[HID];
    int i, j;
    float s;

    for (i = 0; i < OBS; i++) x[i] = obs[i];
    x[4] /= 2.0f;
    x[5] /= 2.0f;

    for (i = 0; i < HID; i++) {
        s = B1[i];
        for (j = 0; j < OBS; j++) s += W1[i][j] * x[j];
        h1[i] = tanhf(s);
    }
    for (i = 0; i < HID; i++) {
        s = B2[i];
        for (j = 0; j < HID; j++) s += W2[i][j] * h1[j];
        h2[i] = tanhf(s);
    }
    for (i = 0; i < 2; i++) {
        s = B3[i];
        for (j = 0; j < HID; j++) s += W3[i][j] * h2[j];
        s = tanhf(s);
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        act[i] = s;
    }
}

/* 12 个均匀分布求和近似高斯（均值 6、标准差 1），避免依赖 rand() 与 libm */
static float gauss_approx(unsigned int *seed)
{
    unsigned int s = *seed;
    float acc = 0.0f;
    int i;
    for (i = 0; i < 12; i++) {
        s = s * 1103515245u + 12345u;
        acc += (float)((s >> 16) & 0x7FFF) / 32767.0f;
    }
    *seed = s;
    return acc - 6.0f;
}

void policy_ppo_sample(const float obs[8], float act[2], unsigned int *seed)
{
    float std2[2];
    int i;
    policy_ppo_mu(obs, act);
    policy_ppo_std(std2);
    for (i = 0; i < 2; i++) {
        act[i] += gauss_approx(seed) * std2[i];
        if (act[i] >  1.0f) act[i] =  1.0f;
        if (act[i] < -1.0f) act[i] = -1.0f;
    }
}
""" % (
        P["W1"].size + P["b1"].size + P["W2"].size + P["b2"].size +
        P["W3"].size + P["b3"].size,
        (P["W1"].size + P["b1"].size + P["W2"].size + P["b2"].size +
         P["W3"].size + P["b3"].size) * 4 / 1024.0,
        fmt_mat("W1", P["W1"]), fmt_vec("B1", P["b1"]),
        fmt_mat("W2", P["W2"]), fmt_vec("B2", P["b2"]),
        fmt_mat("W3", P["W3"]), fmt_vec("B3", P["b3"]),
        P["log_std"][0], P["log_std"][1],
    )

    # ---- 对拍向量：20 组随机观测 + Python 输出 ----
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(20):
        obs = np.concatenate([
            rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2),
            rng.uniform(-0.6, 0.6, 2), rng.uniform(-1, 1, 2)
        ])
        cases.append((obs.tolist(), forward(obs, P).tolist()))

    with open(os.path.join(OUTDIR, "test_vectors.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f)

    test_c = """/* test_policy.c —— 与 Python 端对拍：C 推理结果应逐位接近 PyTorch 输出
 * 生成：export_policy_c.py
 */
#include <stdio.h>
#include <math.h>
#include "policy_ppo.h"

#define NCASE %d

static const float CASES[NCASE][8] = {
%s
};

static const float EXPECT[NCASE][2] = {
%s
};

int main(void)
{
    int i, k;
    float act[2], maxerr = 0.0f;
    for (i = 0; i < NCASE; i++) {
        float e;
        policy_ppo_mu(CASES[i], act);
        for (k = 0; k < 2; k++) {
            e = (float)fabs((double)(act[k] - EXPECT[i][k]));
            if (e > maxerr) maxerr = e;
        }
        printf("case %%2d  C=(%%+.6f, %%+.6f)  python=(%%+.6f, %%+.6f)\\n",
               i, act[0], act[1], EXPECT[i][0], EXPECT[i][1]);
    }
    printf("\\n最大绝对误差 = %%.3e\\n", (double)maxerr);
    printf("%%s\\n", maxerr < 1e-4f ? "PASS：C 推理与 PyTorch 一致" : "FAIL");
    return maxerr < 1e-4f ? 0 : 1;
}
""" % (
        len(cases),
        ",\n".join("    { " + ", ".join("%.8ff" % v for v in o) + " }" for o, _ in cases),
        ",\n".join("    { " + ", ".join("%.8ff" % v for v in e) + " }" for _, e in cases),
    )

    with open(os.path.join(OUTDIR, "policy_ppo.h"), "w", encoding="utf-8") as f:
        f.write(header)
    with open(os.path.join(OUTDIR, "policy_ppo.c"), "w", encoding="utf-8") as f:
        f.write(body)
    with open(os.path.join(OUTDIR, "test_policy.c"), "w", encoding="utf-8") as f:
        f.write(test_c)

    print("已生成：")
    for fn in ["policy_ppo.h", "policy_ppo.c", "test_policy.c"]:
        p = os.path.join(OUTDIR, fn)
        print("  %-18s %6.1f KB" % (fn, os.path.getsize(p) / 1024))
    print("\n编译并自测： tcc -run test_policy.c policy_ppo.c   (需在 webdemo/c 目录下)")


if __name__ == "__main__":
    main()
