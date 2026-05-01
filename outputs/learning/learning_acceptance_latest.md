# Learning Acceptance Latest

- generated_at_utc: 2026-05-01T14:49:51+00:00
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
| lr_mae | 0.054296 | worse (+0.005693) |
| lr_rmse | 0.073482 | worse (+0.003569) |
| lr_corr | 0.259953 | worse (-0.353171) |
| lr_directional_acc | 0.535714 | worse (-0.219388) |
| lgbm_mae | 0.054836 | worse (+0.018652) |
| lgbm_rmse | 0.073967 | worse (+0.023044) |
| lgbm_corr | 0.245109 | worse (-0.639667) |
| lgbm_directional_acc | 0.571429 | worse (-0.224490) |

