# Learning Acceptance Latest

- generated_at_utc: 2026-04-17T14:37:28+00:00
- current_run_date: 20260417
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260416
- status: trained
- loaded_trade_dates: 32
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260415
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.565789 | worse (-0.162061) |
| lr_logloss | 0.360414 | worse (+0.094120) |
| lr_brier | 0.089626 | worse (+0.027084) |
| lgbm_auc | 0.701754 | worse (-0.169377) |
| lgbm_logloss | 0.218006 | worse (+0.084985) |
| lgbm_brier | 0.037664 | worse (+0.007235) |

## E_ret

- anchor_trade_date: 20260415
- status: trained
- loaded_trade_dates: 31
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260414
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.053065 | improved (-0.007558) |
| lr_rmse | 0.069646 | improved (-0.042297) |
| lr_corr | 0.533675 | worse (-0.104698) |
| lr_directional_acc | 0.684211 | worse (-0.027328) |
| lgbm_mae | 0.047975 | improved (-0.007888) |
| lgbm_rmse | 0.062102 | improved (-0.014695) |
| lgbm_corr | 0.706500 | improved (+0.168363) |
| lgbm_directional_acc | 0.736842 | improved (+0.102227) |

