# Learning Acceptance Latest

- generated_at_utc: 2026-03-31T14:43:56+00:00
- current_run_date: 20260331
- overall_pass: PASS

## P_fill

- anchor_trade_date: 20260330
- status: trained
- loaded_trade_dates: 22
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260327
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.967742 | improved (+0.253456) |
| lr_logloss | 0.162700 | improved (-0.017840) |
| lr_brier | 0.049821 | improved (-0.003748) |
| lgbm_auc | 0.887097 | improved (+0.211772) |
| lgbm_logloss | 0.072186 | improved (-0.128300) |
| lgbm_brier | 0.019188 | improved (-0.021175) |

## E_ret

- anchor_trade_date: 20260327
- status: trained
- loaded_trade_dates: 21
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260326
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.041728 | improved (-0.030042) |
| lr_rmse | 0.061187 | improved (-0.030924) |
| lr_corr | 0.587734 | improved (+0.086726) |
| lr_directional_acc | 0.753247 | improved (+0.114358) |
| lgbm_mae | 0.057403 | improved (-0.028314) |
| lgbm_rmse | 0.074621 | improved (-0.029681) |
| lgbm_corr | 0.206152 | improved (+0.080977) |
| lgbm_directional_acc | 0.636364 | improved (+0.136364) |

