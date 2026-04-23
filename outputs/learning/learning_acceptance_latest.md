# Learning Acceptance Latest

- generated_at_utc: 2026-04-23T15:22:17+00:00
- current_run_date: 20260423
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260422
- status: trained
- loaded_trade_dates: 36
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260421
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.884615 | improved (+0.211282) |
| lr_logloss | 0.216883 | improved (-0.082106) |
| lr_brier | 0.065279 | improved (-0.000380) |
| lgbm_auc | 0.935897 | improved (+0.122564) |
| lgbm_logloss | 0.181136 | worse (+0.020092) |
| lgbm_brier | 0.038602 | worse (+0.009095) |

## E_ret

- anchor_trade_date: 20260421
- status: trained
- loaded_trade_dates: 35
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260420
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.052013 | worse (+0.007374) |
| lr_rmse | 0.065361 | worse (+0.010209) |
| lr_corr | 0.585027 | improved (+0.088073) |
| lr_directional_acc | 0.640000 | improved (+0.072432) |
| lgbm_mae | 0.045522 | worse (+0.011858) |
| lgbm_rmse | 0.059648 | worse (+0.017797) |
| lgbm_corr | 0.705806 | worse (-0.083829) |
| lgbm_directional_acc | 0.760000 | improved (+0.070811) |

