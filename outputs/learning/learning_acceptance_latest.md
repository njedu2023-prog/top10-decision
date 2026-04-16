# Learning Acceptance Latest

- generated_at_utc: 2026-04-16T15:06:34+00:00
- current_run_date: 20260416
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260415
- status: trained
- loaded_trade_dates: 31
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260414
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.727851 | worse (-0.169585) |
| lr_logloss | 0.266294 | improved (-0.071953) |
| lr_brier | 0.062542 | improved (-0.044461) |
| lgbm_auc | 0.871132 | worse (-0.090407) |
| lgbm_logloss | 0.133021 | worse (+0.022123) |
| lgbm_brier | 0.030429 | worse (+0.003407) |

## E_ret

- anchor_trade_date: 20260414
- status: trained
- loaded_trade_dates: 30
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260413
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.060623 | worse (+0.013992) |
| lr_rmse | 0.111944 | worse (+0.054448) |
| lr_corr | 0.638373 | worse (-0.032465) |
| lr_directional_acc | 0.711538 | improved (+0.265110) |
| lgbm_mae | 0.055863 | worse (+0.012819) |
| lgbm_rmse | 0.076798 | worse (+0.020036) |
| lgbm_corr | 0.538136 | worse (-0.010617) |
| lgbm_directional_acc | 0.634615 | worse (-0.115385) |

