# Learning Acceptance Latest

- generated_at_utc: 2026-05-26T16:52:52+00:00
- current_run_date: 20260526
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260525
- status: trained
- loaded_trade_dates: 56
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260522
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.896465 | improved (+0.015384) |
| lr_logloss | 0.228342 | improved (-0.014968) |
| lr_brier | 0.066339 | improved (-0.011597) |
| lgbm_auc | 0.926768 | improved (+0.132173) |
| lgbm_logloss | 0.148422 | improved (-0.006276) |
| lgbm_brier | 0.035403 | worse (+0.001638) |

## E_ret

- anchor_trade_date: 20260522
- status: trained
- loaded_trade_dates: 55
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260521
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.064593 | worse (+0.000544) |
| lr_rmse | 0.080212 | worse (+0.002221) |
| lr_corr | -0.038152 | worse (-0.051523) |
| lr_directional_acc | 0.540541 | worse (-0.084459) |
| lgbm_mae | 0.066288 | improved (-0.000311) |
| lgbm_rmse | 0.081854 | improved (-0.001373) |
| lgbm_corr | -0.007783 | worse (-0.030753) |
| lgbm_directional_acc | 0.540541 | improved (+0.196791) |

