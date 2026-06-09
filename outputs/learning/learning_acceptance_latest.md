# Learning Acceptance Latest

- generated_at_utc: 2026-06-09T16:14:22+00:00
- current_run_date: 20260609
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260608
- status: trained
- loaded_trade_dates: 66
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260605
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.895062 | improved (+0.185105) |
| lr_logloss | 0.175921 | improved (-0.107683) |
| lr_brier | 0.051063 | improved (-0.032977) |
| lgbm_auc | 0.697531 | worse (-0.145051) |
| lgbm_logloss | 0.184403 | worse (+0.079099) |
| lgbm_brier | 0.034432 | worse (+0.008905) |

## E_ret

- anchor_trade_date: 20260605
- status: trained
- loaded_trade_dates: 65
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260604
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.064764 | improved (-0.012693) |
| lr_rmse | 0.087438 | improved (-0.006100) |
| lr_corr | 0.037248 | improved (+0.021251) |
| lr_directional_acc | 0.539474 | improved (+0.052294) |
| lgbm_mae | 0.067247 | improved (-0.012776) |
| lgbm_rmse | 0.094900 | improved (-0.003012) |
| lgbm_corr | -0.014034 | worse (-0.024800) |
| lgbm_directional_acc | 0.539474 | improved (+0.077935) |

