# Learning Acceptance Latest

- generated_at_utc: 2026-04-27T15:19:09+00:00
- current_run_date: 20260427
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260424
- status: trained
- loaded_trade_dates: 38
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260423
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.903509 | worse (-0.096491) |
| lr_logloss | 0.207574 | improved (-0.030559) |
| lr_brier | 0.062201 | improved (-0.011059) |
| lgbm_auc | 0.745614 | worse (-0.193161) |
| lgbm_logloss | 0.143354 | worse (+0.075926) |
| lgbm_brier | 0.023466 | worse (+0.003039) |

## E_ret

- anchor_trade_date: 20260423
- status: trained
- loaded_trade_dates: 37
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260422
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.048603 | worse (+0.000673) |
| lr_rmse | 0.069913 | worse (+0.008486) |
| lr_corr | 0.613124 | worse (-0.091197) |
| lr_directional_acc | 0.755102 | improved (+0.043564) |
| lgbm_mae | 0.036185 | improved (-0.001778) |
| lgbm_rmse | 0.050923 | worse (+0.004968) |
| lgbm_corr | 0.884776 | improved (+0.020374) |
| lgbm_directional_acc | 0.795918 | improved (+0.045918) |

