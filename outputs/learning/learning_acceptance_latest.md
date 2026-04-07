# Learning Acceptance Latest

- generated_at_utc: 2026-04-07T14:50:14+00:00
- current_run_date: 20260407
- overall_pass: PASS

## P_fill

- anchor_trade_date: 20260403
- status: trained
- loaded_trade_dates: 26
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260402
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.971429 | improved (+0.317582) |
| lr_logloss | 0.235988 | improved (-0.233830) |
| lr_brier | 0.076620 | improved (-0.045027) |
| lgbm_auc | 1.000000 | improved (+0.307692) |
| lgbm_logloss | 0.031256 | improved (-0.442065) |
| lgbm_brier | 0.007753 | improved (-0.095170) |

## E_ret

- anchor_trade_date: 20260402
- status: trained
- loaded_trade_dates: 25
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260401
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.054553 | worse (+0.013536) |
| lr_rmse | 0.064920 | worse (+0.011173) |
| lr_corr | 0.618269 | improved (+0.059203) |
| lr_directional_acc | 0.692308 | improved (+0.001399) |
| lgbm_mae | 0.062728 | worse (+0.016980) |
| lgbm_rmse | 0.076863 | worse (+0.017609) |
| lgbm_corr | 0.274428 | worse (-0.060550) |
| lgbm_directional_acc | 0.576923 | worse (-0.059441) |

