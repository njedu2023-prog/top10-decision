# Learning Acceptance Latest

- generated_at_utc: 2026-06-08T16:53:40+00:00
- current_run_date: 20260608
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260605
- status: trained
- loaded_trade_dates: 65
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260604
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.709957 | worse (-0.213120) |
| lr_logloss | 0.283604 | worse (+0.049519) |
| lr_brier | 0.084039 | worse (+0.013746) |
| lgbm_auc | 0.842582 | worse (-0.118957) |
| lgbm_logloss | 0.105303 | worse (+0.020400) |
| lgbm_brier | 0.025527 | worse (+0.005218) |

## E_ret

- anchor_trade_date: 20260604
- status: trained
- loaded_trade_dates: 64
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260603
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.077457 | worse (+0.017244) |
| lr_rmse | 0.093538 | worse (+0.017219) |
| lr_corr | 0.015997 | improved (+0.288392) |
| lr_directional_acc | 0.487179 | improved (+0.087179) |
| lgbm_mae | 0.080023 | worse (+0.022148) |
| lgbm_rmse | 0.097912 | worse (+0.019911) |
| lgbm_corr | 0.010767 | improved (+0.050046) |
| lgbm_directional_acc | 0.461538 | worse (-0.092308) |

