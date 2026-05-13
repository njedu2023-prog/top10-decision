# Learning Acceptance Latest

- generated_at_utc: 2026-05-13T15:50:28+00:00
- current_run_date: 20260513
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260512
- status: trained
- loaded_trade_dates: 47
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260511
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.806604 | worse (-0.120363) |
| lr_logloss | 0.474085 | worse (+0.221725) |
| lr_brier | 0.125251 | worse (+0.048158) |
| lgbm_auc | 0.957547 | worse (-0.005000) |
| lgbm_logloss | 0.139208 | improved (-0.051713) |
| lgbm_brier | 0.043771 | improved (-0.000885) |

## E_ret

- anchor_trade_date: 20260511
- status: trained
- loaded_trade_dates: 46
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260508
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.058609 | worse (+0.013868) |
| lr_rmse | 0.069828 | worse (+0.012389) |
| lr_corr | 0.060404 | worse (-0.083745) |
| lr_directional_acc | 0.516854 | improved (+0.038131) |
| lgbm_mae | 0.063953 | worse (+0.015223) |
| lgbm_rmse | 0.080930 | worse (+0.019640) |
| lgbm_corr | -0.065461 | worse (-0.299013) |
| lgbm_directional_acc | 0.449438 | improved (+0.013268) |

