# Learning Acceptance Latest

- generated_at_utc: 2026-05-01T13:56:09+00:00
- current_run_date: 20260430
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260429
- status: trained
- loaded_trade_dates: 41
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260428
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.912458 | worse (-0.069998) |
| lr_logloss | 0.231796 | improved (-0.015186) |
| lr_brier | 0.075386 | worse (+0.011727) |
| lgbm_auc | 0.912458 | worse (-0.069998) |
| lgbm_logloss | 0.127614 | worse (+0.042616) |
| lgbm_brier | 0.029305 | worse (+0.003242) |

## E_ret

- anchor_trade_date: 20260428
- status: trained
- loaded_trade_dates: 40
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260427
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.063849 | worse (+0.028431) |
| lr_rmse | 0.087299 | worse (+0.041070) |
| lr_corr | 0.027120 | worse (-0.692835) |
| lr_directional_acc | 0.508772 | worse (-0.265422) |
| lgbm_mae | 0.065489 | worse (+0.035470) |
| lgbm_rmse | 0.088688 | worse (+0.051421) |
| lgbm_corr | -0.103977 | worse (-0.890675) |
| lgbm_directional_acc | 0.561404 | worse (-0.196661) |

