# Learning Acceptance Latest

- generated_at_utc: 2026-04-08T14:50:00+00:00
- current_run_date: 20260408
- overall_pass: PASS

## P_fill

- anchor_trade_date: 20260407
- status: trained
- loaded_trade_dates: 27
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260403
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.444444 | worse (-0.526984) |
| lr_logloss | 0.287569 | worse (+0.051581) |
| lr_brier | 0.032144 | improved (-0.044476) |
| lgbm_auc | 0.618519 | worse (-0.381481) |
| lgbm_logloss | 0.158898 | worse (+0.127642) |
| lgbm_brier | 0.028651 | worse (+0.020898) |

## E_ret

- anchor_trade_date: 20260403
- status: trained
- loaded_trade_dates: 26
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260402
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.035806 | improved (-0.018747) |
| lr_rmse | 0.054442 | improved (-0.010477) |
| lr_corr | 0.664205 | improved (+0.045936) |
| lr_directional_acc | 0.800000 | improved (+0.107692) |
| lgbm_mae | 0.042039 | improved (-0.020689) |
| lgbm_rmse | 0.059062 | improved (-0.017801) |
| lgbm_corr | 0.664834 | improved (+0.390407) |
| lgbm_directional_acc | 0.800000 | improved (+0.223077) |

