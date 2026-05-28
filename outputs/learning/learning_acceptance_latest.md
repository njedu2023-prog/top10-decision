# Learning Acceptance Latest

- generated_at_utc: 2026-05-28T05:50:01+00:00
- current_run_date: 20260527
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260526
- status: trained
- loaded_trade_dates: 57
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260525
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.208333 | worse (-0.688131) |
| lr_logloss | 0.337678 | worse (+0.109336) |
| lr_brier | 0.090121 | worse (+0.023782) |
| lgbm_auc | 0.604167 | worse (-0.322601) |
| lgbm_logloss | 0.149834 | worse (+0.001412) |
| lgbm_brier | 0.026772 | improved (-0.008631) |

## E_ret

- anchor_trade_date: 20260525
- status: trained
- loaded_trade_dates: 56
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260522
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.059941 | improved (-0.004652) |
| lr_rmse | 0.074172 | improved (-0.006040) |
| lr_corr | 0.059932 | improved (+0.098084) |
| lr_directional_acc | 0.525253 | worse (-0.015288) |
| lgbm_mae | 0.069568 | worse (+0.003280) |
| lgbm_rmse | 0.084438 | worse (+0.002584) |
| lgbm_corr | -0.032242 | worse (-0.024460) |
| lgbm_directional_acc | 0.454545 | worse (-0.085995) |

