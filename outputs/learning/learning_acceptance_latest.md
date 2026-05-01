# Learning Acceptance Latest

- generated_at_utc: 2026-05-01T13:37:34+00:00
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
| lr_mae | 0.054125 | worse (+0.018707) |
| lr_rmse | 0.075273 | worse (+0.029044) |
| lr_corr | 0.594634 | worse (-0.125321) |
| lr_directional_acc | 0.701754 | worse (-0.072439) |
| lgbm_mae | 0.040722 | worse (+0.010703) |
| lgbm_rmse | 0.054244 | worse (+0.016976) |
| lgbm_corr | 0.808223 | improved (+0.021524) |
| lgbm_directional_acc | 0.807018 | improved (+0.048953) |

