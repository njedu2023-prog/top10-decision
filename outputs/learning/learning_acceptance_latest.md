# Learning Acceptance Latest

- generated_at_utc: 2026-05-21T16:23:57+00:00
- current_run_date: 20260521
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260520
- status: trained
- loaded_trade_dates: 53
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260519
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.949153 | worse (-0.022438) |
| lr_logloss | 0.251813 | worse (+0.023799) |
| lr_brier | 0.080102 | worse (+0.010400) |
| lgbm_auc | 0.949153 | worse (-0.033802) |
| lgbm_logloss | 0.160106 | worse (+0.115926) |
| lgbm_brier | 0.038331 | worse (+0.024479) |

## E_ret

- anchor_trade_date: 20260519
- status: trained
- loaded_trade_dates: 52
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260518
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.068161 | worse (+0.010986) |
| lr_rmse | 0.081887 | worse (+0.002601) |
| lr_corr | 0.285366 | improved (+0.085832) |
| lr_directional_acc | 0.420455 | worse (-0.137987) |
| lgbm_mae | 0.075519 | worse (+0.018401) |
| lgbm_rmse | 0.090580 | worse (+0.013560) |
| lgbm_corr | 0.035022 | worse (-0.282066) |
| lgbm_directional_acc | 0.431818 | worse (-0.087662) |

