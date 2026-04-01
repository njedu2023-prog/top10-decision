# Learning Acceptance Latest

- generated_at_utc: 2026-04-01T14:45:50+00:00
- current_run_date: 20260401
- overall_pass: PASS

## P_fill

- anchor_trade_date: 20260331
- status: trained
- loaded_trade_dates: 23
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260330
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.846667 | worse (-0.121075) |
| lr_logloss | 0.273847 | worse (+0.111147) |
| lr_brier | 0.064741 | worse (+0.014920) |
| lgbm_auc | 0.573333 | worse (-0.313763) |
| lgbm_logloss | 0.333523 | worse (+0.261337) |
| lgbm_brier | 0.057345 | worse (+0.038158) |

## E_ret

- anchor_trade_date: 20260330
- status: trained
- loaded_trade_dates: 22
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260327
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.037814 | improved (-0.003914) |
| lr_rmse | 0.048689 | improved (-0.012498) |
| lr_corr | 0.694594 | improved (+0.106860) |
| lr_directional_acc | 0.790323 | improved (+0.037076) |
| lgbm_mae | 0.045020 | improved (-0.012384) |
| lgbm_rmse | 0.059880 | improved (-0.014741) |
| lgbm_corr | 0.437502 | improved (+0.231350) |
| lgbm_directional_acc | 0.661290 | improved (+0.024927) |

