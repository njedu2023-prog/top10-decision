# Learning Acceptance Latest

- generated_at_utc: 2026-05-01T15:13:34+00:00
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
| lr_logloss | 0.208421 | improved (-0.029712) |
| lr_brier | 0.062336 | improved (-0.010924) |
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
| lr_mae | 0.053678 | worse (+0.005748) |
| lr_rmse | 0.077716 | worse (+0.016290) |
| lr_corr | 0.195715 | worse (-0.508606) |
| lr_directional_acc | 0.714286 | improved (+0.002747) |
| lgbm_mae | 0.060358 | worse (+0.022395) |
| lgbm_rmse | 0.087101 | worse (+0.041146) |
| lgbm_corr | -0.198715 | worse (-1.063117) |
| lgbm_directional_acc | 0.428571 | worse (-0.321429) |

