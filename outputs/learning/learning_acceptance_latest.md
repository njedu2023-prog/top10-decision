# Learning Acceptance Latest

- generated_at_utc: 2026-06-11T16:13:08+00:00
- current_run_date: 20260611
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260610
- status: trained
- loaded_trade_dates: 68
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260609
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 1.000000 | improved (+0.062400) |
| lr_logloss | 0.217775 | worse (+0.013962) |
| lr_brier | 0.063549 | improved (-0.000240) |
| lgbm_auc | 0.992754 | improved (+0.021554) |
| lgbm_logloss | 0.038025 | improved (-0.041904) |
| lgbm_brier | 0.011404 | improved (-0.010644) |

## E_ret

- anchor_trade_date: 20260609
- status: trained
- loaded_trade_dates: 67
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260608
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.059552 | improved (-0.018358) |
| lr_rmse | 0.076638 | improved (-0.016048) |
| lr_corr | -0.058707 | worse (-0.283054) |
| lr_directional_acc | 0.524194 | worse (-0.086918) |
| lgbm_mae | 0.062174 | improved (-0.017290) |
| lgbm_rmse | 0.080257 | improved (-0.024265) |
| lgbm_corr | -0.039047 | improved (+0.144945) |
| lgbm_directional_acc | 0.548387 | worse (-0.044205) |

