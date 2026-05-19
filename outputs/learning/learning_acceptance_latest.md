# Learning Acceptance Latest

- generated_at_utc: 2026-05-19T16:31:43+00:00
- current_run_date: 20260519
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260518
- status: trained
- loaded_trade_dates: 51
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260515
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.706494 | worse (-0.293506) |
| lr_logloss | 0.683640 | worse (+0.444691) |
| lr_brier | 0.166505 | worse (+0.091354) |
| lgbm_auc | 0.971429 | worse (-0.010053) |
| lgbm_logloss | 0.108809 | improved (-0.017382) |
| lgbm_brier | 0.034471 | worse (+0.013387) |

## E_ret

- anchor_trade_date: 20260515
- status: trained
- loaded_trade_dates: 50
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260514
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.059803 | improved (-0.015009) |
| lr_rmse | 0.081299 | improved (-0.007958) |
| lr_corr | 0.232983 | improved (+0.086859) |
| lr_directional_acc | 0.592593 | improved (+0.018519) |
| lgbm_mae | 0.062992 | improved (-0.012422) |
| lgbm_rmse | 0.084040 | improved (-0.003322) |
| lgbm_corr | 0.078192 | worse (-0.154108) |
| lgbm_directional_acc | 0.425926 | worse (-0.148148) |

