# Learning Acceptance Latest

- generated_at_utc: 2026-04-14T14:57:32+00:00
- current_run_date: 20260414
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260413
- status: trained
- loaded_trade_dates: 29
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260410
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.928571 | improved (+0.237662) |
| lr_logloss | 0.332780 | improved (-0.005682) |
| lr_brier | 0.109210 | worse (+0.022416) |
| lgbm_auc | 0.994048 | improved (+0.189502) |
| lgbm_logloss | 0.078298 | improved (-0.211002) |
| lgbm_brier | 0.025421 | improved (-0.037542) |

## E_ret

- anchor_trade_date: 20260410
- status: trained
- loaded_trade_dates: 28
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260409
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.039376 | improved (-0.015428) |
| lr_rmse | 0.067170 | improved (-0.003469) |
| lr_corr | 0.475624 | worse (-0.015069) |
| lr_directional_acc | 0.727273 | improved (+0.131119) |
| lgbm_mae | 0.038798 | improved (-0.006465) |
| lgbm_rmse | 0.053836 | improved (-0.007848) |
| lgbm_corr | 0.420544 | worse (-0.043003) |
| lgbm_directional_acc | 0.709091 | worse (-0.002448) |

