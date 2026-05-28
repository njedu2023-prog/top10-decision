# Learning Acceptance Latest

- generated_at_utc: 2026-05-28T11:11:22+00:00
- current_run_date: 20260528
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260527
- status: trained
- loaded_trade_dates: 58
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260526
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.808916 | improved (+0.600583) |
| lr_logloss | 0.321673 | improved (-0.016011) |
| lr_brier | 0.087487 | improved (-0.002715) |
| lgbm_auc | 0.850246 | improved (+0.246080) |
| lgbm_logloss | 0.138573 | improved (-0.011261) |
| lgbm_brier | 0.029720 | worse (+0.002949) |

## E_ret

- anchor_trade_date: 20260526
- status: trained
- loaded_trade_dates: 57
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260525
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.078857 | worse (+0.018916) |
| lr_rmse | 0.093533 | worse (+0.019360) |
| lr_corr | 0.238390 | improved (+0.178458) |
| lr_directional_acc | 0.617021 | improved (+0.091769) |
| lgbm_mae | 0.086224 | worse (+0.016656) |
| lgbm_rmse | 0.100553 | worse (+0.016115) |
| lgbm_corr | 0.095701 | improved (+0.127943) |
| lgbm_directional_acc | 0.489362 | improved (+0.034816) |

