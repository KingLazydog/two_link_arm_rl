/* policy_ppo.h —— 二连杆机械臂策略（由 PyTorch 权重导出，请勿手改）
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
