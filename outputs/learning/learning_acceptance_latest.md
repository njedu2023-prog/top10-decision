# Learning Acceptance Latest

- generated_at_utc: 2026-04-28T15:37:38+00:00
- current_run_date: 20260428
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260427
- status: trained
- loaded_trade_dates: 39
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260424
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.841270 | worse (-0.062239) |
| lr_logloss | 0.286980 | worse (+0.079406) |
| lr_brier | 0.091281 | worse (+0.029081) |
| lgbm_auc | 0.611111 | worse (-0.134503) |
| lgbm_logloss | 0.209871 | worse (+0.066517) |
| lgbm_brier | 0.038893 | worse (+0.015427) |

## E_ret

- anchor_trade_date: 20260424
- status: trained
- loaded_trade_dates: 38
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260423
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.044707 | improved (-0.003896) |
| lr_rmse | 0.063172 | improved (-0.006741) |
| lr_corr | 0.573287 | worse (-0.039837) |
| lr_directional_acc | 0.714286 | worse (-0.040816) |
| lgbm_mae | 0.035829 | improved (-0.000355) |
| lgbm_rmse | 0.049381 | improved (-0.001542) |
| lgbm_corr | 0.831900 | worse (-0.052876) |
| lgbm_directional_acc | 0.803571 | improved (+0.007653) |

