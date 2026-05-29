# Learning Acceptance Latest

- generated_at_utc: 2026-05-29T16:53:58+00:00
- current_run_date: 20260529
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260528
- status: trained
- loaded_trade_dates: 59
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260527
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.727273 | worse (-0.081644) |
| lr_logloss | 0.262314 | improved (-0.059360) |
| lr_brier | 0.080050 | improved (-0.007437) |
| lgbm_auc | 0.902357 | improved (+0.052111) |
| lgbm_logloss | 0.108005 | improved (-0.030568) |
| lgbm_brier | 0.029813 | worse (+0.000093) |

## E_ret

- anchor_trade_date: 20260527
- status: trained
- loaded_trade_dates: 58
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260526
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.061668 | improved (-0.017189) |
| lr_rmse | 0.076742 | improved (-0.016791) |
| lr_corr | 0.005567 | worse (-0.232823) |
| lr_directional_acc | 0.446809 | worse (-0.170213) |
| lgbm_mae | 0.058424 | improved (-0.027800) |
| lgbm_rmse | 0.076554 | improved (-0.023999) |
| lgbm_corr | 0.038807 | worse (-0.056895) |
| lgbm_directional_acc | 0.553191 | improved (+0.063830) |

