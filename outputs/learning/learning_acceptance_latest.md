# Learning Acceptance Latest

- generated_at_utc: 2026-05-07T15:34:59+00:00
- current_run_date: 20260507
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260506
- status: trained
- loaded_trade_dates: 43
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260430
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.849490 | improved (+0.336503) |
| lr_logloss | 0.340881 | improved (-0.006229) |
| lr_brier | 0.089634 | worse (+0.021825) |
| lgbm_auc | 0.691327 | worse (-0.191790) |
| lgbm_logloss | 0.249203 | worse (+0.139062) |
| lgbm_brier | 0.036846 | worse (+0.013316) |

## E_ret

- anchor_trade_date: 20260430
- status: trained
- loaded_trade_dates: 42
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260429
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.050131 | improved (-0.013992) |
| lr_rmse | 0.063341 | improved (-0.030886) |
| lr_corr | -0.090735 | improved (+0.077511) |
| lr_directional_acc | 0.376623 | improved (+0.053391) |
| lgbm_mae | 0.053074 | improved (-0.006909) |
| lgbm_rmse | 0.065842 | improved (-0.020807) |
| lgbm_corr | -0.089989 | worse (-0.086657) |
| lgbm_directional_acc | 0.415584 | improved (+0.041847) |

