# Learning Acceptance Latest

- generated_at_utc: 2026-04-15T14:50:58+00:00
- current_run_date: 20260415
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260414
- status: trained
- loaded_trade_dates: 30
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260413
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.897436 | worse (-0.031136) |
| lr_logloss | 0.338248 | worse (+0.005467) |
| lr_brier | 0.107003 | improved (-0.002208) |
| lgbm_auc | 0.961538 | worse (-0.032509) |
| lgbm_logloss | 0.110897 | worse (+0.032599) |
| lgbm_brier | 0.027022 | worse (+0.001602) |

## E_ret

- anchor_trade_date: 20260413
- status: trained
- loaded_trade_dates: 29
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260410
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.046631 | worse (+0.007254) |
| lr_rmse | 0.057495 | improved (-0.009675) |
| lr_corr | 0.670838 | improved (+0.195214) |
| lr_directional_acc | 0.446429 | worse (-0.280844) |
| lgbm_mae | 0.043043 | worse (+0.004246) |
| lgbm_rmse | 0.056762 | worse (+0.002926) |
| lgbm_corr | 0.548754 | improved (+0.128209) |
| lgbm_directional_acc | 0.750000 | improved (+0.040909) |

