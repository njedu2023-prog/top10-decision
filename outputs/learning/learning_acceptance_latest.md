# Learning Acceptance Latest

- generated_at_utc: 2026-05-22T15:53:28+00:00
- current_run_date: 20260522
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260521
- status: trained
- loaded_trade_dates: 54
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260520
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.864583 | worse (-0.084569) |
| lr_logloss | 0.589759 | worse (+0.337946) |
| lr_brier | 0.131964 | worse (+0.051863) |
| lgbm_auc | 0.947917 | worse (-0.001236) |
| lgbm_logloss | 0.199240 | worse (+0.039134) |
| lgbm_brier | 0.057832 | worse (+0.019501) |

## E_ret

- anchor_trade_date: 20260520
- status: trained
- loaded_trade_dates: 53
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260519
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.065216 | improved (-0.002945) |
| lr_rmse | 0.084175 | worse (+0.002288) |
| lr_corr | 0.091502 | worse (-0.193864) |
| lr_directional_acc | 0.474576 | improved (+0.054122) |
| lgbm_mae | 0.063622 | improved (-0.011897) |
| lgbm_rmse | 0.087196 | improved (-0.003384) |
| lgbm_corr | -0.006440 | worse (-0.041462) |
| lgbm_directional_acc | 0.627119 | improved (+0.195300) |

