# Learning Acceptance Latest

- generated_at_utc: 2026-06-10T16:46:10+00:00
- current_run_date: 20260610
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260609
- status: trained
- loaded_trade_dates: 67
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260608
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.937600 | improved (+0.042538) |
| lr_logloss | 0.203813 | worse (+0.027891) |
| lr_brier | 0.063789 | worse (+0.012727) |
| lgbm_auc | 0.971200 | improved (+0.273669) |
| lgbm_logloss | 0.079929 | improved (-0.104474) |
| lgbm_brier | 0.022047 | improved (-0.012385) |

## E_ret

- anchor_trade_date: 20260608
- status: trained
- loaded_trade_dates: 66
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260605
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.077910 | worse (+0.013146) |
| lr_rmse | 0.092686 | worse (+0.005248) |
| lr_corr | 0.224347 | improved (+0.187099) |
| lr_directional_acc | 0.611111 | improved (+0.071637) |
| lgbm_mae | 0.079464 | worse (+0.012217) |
| lgbm_rmse | 0.104521 | worse (+0.009621) |
| lgbm_corr | -0.183992 | worse (-0.169958) |
| lgbm_directional_acc | 0.592593 | improved (+0.053119) |

