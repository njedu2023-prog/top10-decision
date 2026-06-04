# Learning Acceptance Latest

- generated_at_utc: 2026-06-04T13:15:30+00:00
- current_run_date: 20260604
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260603
- status: trained
- loaded_trade_dates: 63
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260602
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.553846 | worse (-0.271551) |
| lr_logloss | 0.427362 | worse (+0.087012) |
| lr_brier | 0.124145 | worse (+0.021454) |
| lgbm_auc | 0.830769 | worse (-0.073993) |
| lgbm_logloss | 0.102123 | improved (-0.113592) |
| lgbm_brier | 0.024226 | improved (-0.045327) |

## E_ret

- anchor_trade_date: 20260602
- status: trained
- loaded_trade_dates: 62
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260601
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.055831 | improved (-0.004112) |
| lr_rmse | 0.069245 | improved (-0.010098) |
| lr_corr | -0.048008 | worse (-0.123618) |
| lr_directional_acc | 0.507937 | worse (-0.139122) |
| lgbm_mae | 0.061871 | worse (+0.000818) |
| lgbm_rmse | 0.075418 | improved (-0.004070) |
| lgbm_corr | -0.012201 | worse (-0.165021) |
| lgbm_directional_acc | 0.476190 | worse (-0.095238) |

