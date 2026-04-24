# Learning Acceptance Latest

- generated_at_utc: 2026-04-24T14:50:19+00:00
- current_run_date: 20260424
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260423
- status: trained
- loaded_trade_dates: 37
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260422
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 1.000000 | improved (+0.115385) |
| lr_logloss | 0.238133 | worse (+0.021250) |
| lr_brier | 0.073260 | worse (+0.007981) |
| lgbm_auc | 0.938776 | improved (+0.002878) |
| lgbm_logloss | 0.067428 | improved (-0.113708) |
| lgbm_brier | 0.020427 | improved (-0.018175) |

## E_ret

- anchor_trade_date: 20260422
- status: trained
- loaded_trade_dates: 36
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260421
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.047930 | improved (-0.004083) |
| lr_rmse | 0.061427 | improved (-0.003934) |
| lr_corr | 0.704321 | improved (+0.119294) |
| lr_directional_acc | 0.711538 | improved (+0.071538) |
| lgbm_mae | 0.037963 | improved (-0.007559) |
| lgbm_rmse | 0.045956 | improved (-0.013693) |
| lgbm_corr | 0.864402 | improved (+0.158596) |
| lgbm_directional_acc | 0.750000 | worse (-0.010000) |

