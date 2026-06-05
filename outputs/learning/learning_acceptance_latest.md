# Learning Acceptance Latest

- generated_at_utc: 2026-06-05T16:09:41+00:00
- current_run_date: 20260605
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260604
- status: trained
- loaded_trade_dates: 64
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260603
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.923077 | improved (+0.369231) |
| lr_logloss | 0.234085 | improved (-0.193278) |
| lr_brier | 0.070294 | improved (-0.053851) |
| lgbm_auc | 0.961538 | improved (+0.130769) |
| lgbm_logloss | 0.084903 | improved (-0.017220) |
| lgbm_brier | 0.020309 | improved (-0.003918) |

## E_ret

- anchor_trade_date: 20260603
- status: trained
- loaded_trade_dates: 63
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260602
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.060212 | worse (+0.004381) |
| lr_rmse | 0.076319 | worse (+0.007074) |
| lr_corr | -0.272394 | worse (-0.224386) |
| lr_directional_acc | 0.400000 | worse (-0.107937) |
| lgbm_mae | 0.057875 | improved (-0.003996) |
| lgbm_rmse | 0.078001 | worse (+0.002583) |
| lgbm_corr | -0.039279 | worse (-0.027078) |
| lgbm_directional_acc | 0.553846 | improved (+0.077656) |

