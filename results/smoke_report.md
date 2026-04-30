# DV Group Proxy Signature Report

本文报告只汇总正式参数轨道 `GPV-S/GPV-M/GPV-L` 的结果。`toy` 轨道仅用于开发阶段正确性打通，不进入最终统计、图表和结论。

审计打开与公开归责的正确性表述按方案要求采用“除可忽略的 LWE 解码失败概率外成立”的口径。真实/模拟签名分布实验用于数值检验不可区分性，不替代理论安全证明。

## Parameter Tracks

| track | n | m | q | sigma1 | sigma2 | sigma_enc | Bz | betaS | stern_rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPV-S | 512 | 23552 | 8380417 | 640 | 640 | 192 | 500000.0 | 3 | 224 |
| GPV-M | 768 | 35328 | 8380417 | 768 | 768 | 224 | 900000.0 | 3 | 256 |
| GPV-L | 1024 | 47104 | 8380417 | 960 | 960 | 256 | 1500000.0 | 3 | 288 |

## Timing Summary

| track | verifier_count | algorithm | mean_ms | std_ms |
| --- | --- | --- | --- | --- |
| GPV-S | 1 | audit_keygen | 47754.2799 | 0.0 |
| GPV-S | 1 | judge | 4781.1468 | 0.0 |
| GPV-S | 1 | keygen | 32295.0205 | 0.0 |
| GPV-S | 1 | open_receipt | 27.8788 | 0.0 |
| GPV-S | 1 | proxy_authorize | 9.8704 | 0.0 |
| GPV-S | 1 | proxy_keygen | 480.8426 | 0.0 |
| GPV-S | 1 | proxy_sign | 250.0576 | 0.0 |
| GPV-S | 1 | receipt_gen | 209.7531 | 0.0 |
| GPV-S | 1 | simulate | 431.8504 | 0.0 |
| GPV-S | 1 | verify | 368.4816 | 0.0 |

## Size Summary

| track | verifier_count | signature_bytes | zk_proof_bytes | receipt_bytes | open_bytes | simulate_signature_bytes |
| --- | --- | --- | --- | --- | --- | --- |
| GPV-S | 1 | 506147.0 | 156500.0 | 321732.0 | 157726.0 | 506092.0 |

## Success And Failure

| track | verifier_count | algorithm | repetitions | success | failure | error |
| --- | --- | --- | --- | --- | --- | --- |
| GPV-S | 1 | audit_keygen | 1 | 1 | 0 | 0 |
| GPV-S | 1 | judge | 1 | 1 | 0 | 0 |
| GPV-S | 1 | keygen | 1 | 1 | 0 | 0 |
| GPV-S | 1 | open_receipt | 1 | 1 | 0 | 0 |
| GPV-S | 1 | proxy_authorize | 1 | 1 | 0 | 0 |
| GPV-S | 1 | proxy_keygen | 1 | 1 | 0 | 0 |
| GPV-S | 1 | proxy_sign | 1 | 1 | 0 | 0 |
| GPV-S | 1 | receipt_gen | 1 | 1 | 0 | 0 |
| GPV-S | 1 | simulate | 1 | 1 | 0 | 0 |
| GPV-S | 1 | verify | 1 | 1 | 0 | 0 |

## Distribution Summary

| track | verifier_count | samples_per_class | max_ks_stat | min_ks_pvalue | max_abs_cohen_d | auc | auc_ci_low | auc_ci_high | mmd_stat | mmd_pvalue | indistinguishable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPV-S | 1 | 2 | 1.0 | 0.333333 | 2.565165 | 1.0 | 1.0 | 1.0 | -0.017323 | 0.691542 | False |

## Figures

下列图表均为最终正式轨道生成结果，图题和坐标轴保持英文。

### All Algorithm Timings

![All Algorithm Timings](smoke_assets/timings_all_algorithms.png)

### Business Path Timings

![Business Path Timings](smoke_assets/timings_business_path.png)

### Signature And ZK Sizes

![Signature And ZK Sizes](smoke_assets/signature_and_zk_sizes.png)

### Time Share at N=1

![Time Share at N=1](smoke_assets/time_share_n1.png)

### Time Share at N=8

![Time Share at N=8](smoke_assets/time_share_n8.png)

### Distribution AUC

![Distribution AUC](smoke_assets/distribution_auc.png)

### Maximum KS Statistic

![Maximum KS Statistic](smoke_assets/distribution_max_ks.png)
