# Learning Acceptance Latest

- generated_at_utc: 2026-03-30T17:59:07+00:00
- current_run_date: 20260330
- overall_pass: PASS

## P_fill

- anchor_trade_date: 20260327
- status: trained
- loaded_trade_dates: 21
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260326
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.714286 | worse (-0.137566) |
| lr_logloss | 0.180540 | improved (-0.236687) |
| lr_brier | 0.053569 | improved (-0.079850) |
| lgbm_auc | 0.675325 | worse (-0.102453) |
| lgbm_logloss | 0.200486 | improved (-0.176628) |
| lgbm_brier | 0.040363 | improved (-0.058483) |

## E_ret

- anchor_trade_date: 20260326
- status: trained
- loaded_trade_dates: 20
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260325
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.071770 | worse (+0.030053) |
| lr_rmse | 0.092111 | worse (+0.038287) |
| lr_corr | 0.501009 | improved (+0.063212) |
| lr_directional_acc | 0.638889 | improved (+0.013889) |
| lgbm_mae | 0.085717 | worse (+0.035297) |
| lgbm_rmse | 0.104302 | worse (+0.040866) |
| lgbm_corr | 0.125175 | worse (-0.030221) |
| lgbm_directional_acc | 0.500000 | improved (+0.012500) |

