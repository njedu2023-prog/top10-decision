# Learning Acceptance Latest

- generated_at_utc: 2026-05-11T16:11:47+00:00
- current_run_date: 20260511
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260508
- status: trained
- loaded_trade_dates: 45
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260507
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.773936 | worse (-0.168241) |
| lr_logloss | 0.377191 | worse (+0.190552) |
| lr_brier | 0.127990 | worse (+0.067915) |
| lgbm_auc | 0.867021 | worse (-0.000326) |
| lgbm_logloss | 0.124475 | worse (+0.007125) |
| lgbm_brier | 0.026981 | worse (+0.003287) |

## E_ret

- anchor_trade_date: 20260507
- status: trained
- loaded_trade_dates: 44
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260506
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.053635 | worse (+0.007509) |
| lr_rmse | 0.069215 | worse (+0.008994) |
| lr_corr | 0.007889 | worse (-0.020972) |
| lr_directional_acc | 0.551020 | flat (+0.000000) |
| lgbm_mae | 0.050079 | improved (-0.015092) |
| lgbm_rmse | 0.067042 | improved (-0.014771) |
| lgbm_corr | 0.025781 | improved (+0.015761) |
| lgbm_directional_acc | 0.612245 | improved (+0.214286) |

