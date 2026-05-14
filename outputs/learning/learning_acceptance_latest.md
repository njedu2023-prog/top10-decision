# Learning Acceptance Latest

- generated_at_utc: 2026-05-14T15:31:35+00:00
- current_run_date: 20260514
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260513
- status: trained
- loaded_trade_dates: 48
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260512
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.834225 | improved (+0.027621) |
| lr_logloss | 0.303299 | improved (-0.170787) |
| lr_brier | 0.086083 | improved (-0.039167) |
| lgbm_auc | 0.868093 | worse (-0.089454) |
| lgbm_logloss | 0.234913 | worse (+0.095705) |
| lgbm_brier | 0.044882 | worse (+0.001111) |

## E_ret

- anchor_trade_date: 20260512
- status: trained
- loaded_trade_dates: 47
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260511
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.058202 | improved (-0.000407) |
| lr_rmse | 0.073653 | worse (+0.003825) |
| lr_corr | 0.019270 | worse (-0.041134) |
| lr_directional_acc | 0.433962 | worse (-0.082892) |
| lgbm_mae | 0.059980 | improved (-0.003973) |
| lgbm_rmse | 0.076867 | improved (-0.004063) |
| lgbm_corr | -0.061914 | improved (+0.003548) |
| lgbm_directional_acc | 0.433962 | worse (-0.015476) |

