# Learning Acceptance Latest

- generated_at_utc: 2026-04-20T14:58:22+00:00
- current_run_date: 20260420
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260417
- status: trained
- loaded_trade_dates: 33
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260416
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.817164 | improved (+0.251375) |
| lr_logloss | 0.144734 | improved (-0.215680) |
| lr_brier | 0.022783 | improved (-0.066843) |
| lgbm_auc | 0.884328 | improved (+0.182574) |
| lgbm_logloss | 0.203784 | improved (-0.014222) |
| lgbm_brier | 0.041257 | worse (+0.003593) |

## E_ret

- anchor_trade_date: 20260416
- status: trained
- loaded_trade_dates: 32
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260415
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.034799 | improved (-0.018266) |
| lr_rmse | 0.043708 | improved (-0.025938) |
| lr_corr | 0.728356 | improved (+0.194680) |
| lr_directional_acc | 0.697368 | improved (+0.013158) |
| lgbm_mae | 0.037134 | improved (-0.010841) |
| lgbm_rmse | 0.051376 | improved (-0.010727) |
| lgbm_corr | 0.540813 | worse (-0.165686) |
| lgbm_directional_acc | 0.710526 | worse (-0.026316) |

