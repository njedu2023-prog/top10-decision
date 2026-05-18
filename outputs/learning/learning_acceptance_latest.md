# Learning Acceptance Latest

- generated_at_utc: 2026-05-18T16:33:36+00:00
- current_run_date: 20260518
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260515
- status: trained
- loaded_trade_dates: 50
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260514
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 1.000000 | flat (+0.000000) |
| lr_logloss | 0.238949 | worse (+0.041816) |
| lr_brier | 0.075151 | worse (+0.017303) |
| lgbm_auc | 0.981481 | worse (-0.018519) |
| lgbm_logloss | 0.126191 | worse (+0.077897) |
| lgbm_brier | 0.021084 | worse (+0.005164) |

## E_ret

- anchor_trade_date: 20260514
- status: trained
- loaded_trade_dates: 49
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260513
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.074811 | improved (-0.003235) |
| lr_rmse | 0.089257 | improved (-0.001771) |
| lr_corr | 0.146124 | improved (+0.059755) |
| lr_directional_acc | 0.574074 | improved (+0.299564) |
| lgbm_mae | 0.075414 | worse (+0.002189) |
| lgbm_rmse | 0.087362 | worse (+0.000146) |
| lgbm_corr | 0.232300 | improved (+0.030714) |
| lgbm_directional_acc | 0.574074 | improved (+0.044662) |

