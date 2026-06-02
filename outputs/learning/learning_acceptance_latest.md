# Learning Acceptance Latest

- generated_at_utc: 2026-06-02T10:31:56+00:00
- current_run_date: 20260602
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260601
- status: trained
- loaded_trade_dates: 61
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260529
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 1.000000 | improved (+0.380000) |
| lr_logloss | 0.351915 | worse (+0.040910) |
| lr_brier | 0.110903 | worse (+0.016418) |
| lgbm_auc | 1.000000 | improved (+0.120000) |
| lgbm_logloss | 0.021707 | improved (-0.062455) |
| lgbm_brier | 0.004697 | improved (-0.015673) |

## E_ret

- anchor_trade_date: 20260529
- status: trained
- loaded_trade_dates: 60
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260528
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.058328 | improved (-0.011276) |
| lr_rmse | 0.076399 | improved (-0.009601) |
| lr_corr | -0.079922 | worse (-0.143883) |
| lr_directional_acc | 0.580000 | improved (+0.085051) |
| lgbm_mae | 0.058476 | improved (-0.015883) |
| lgbm_rmse | 0.072883 | improved (-0.018967) |
| lgbm_corr | 0.127340 | improved (+0.236971) |
| lgbm_directional_acc | 0.520000 | improved (+0.014949) |

