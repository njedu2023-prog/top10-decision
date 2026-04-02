# Learning Acceptance Latest

- generated_at_utc: 2026-04-02T14:36:17+00:00
- current_run_date: 20260402
- overall_pass: PASS

## P_fill

- anchor_trade_date: 20260401
- status: trained
- loaded_trade_dates: 24
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260331
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.809091 | worse (-0.037576) |
| lr_logloss | 0.280655 | worse (+0.006808) |
| lr_brier | 0.085245 | worse (+0.020504) |
| lgbm_auc | 0.854545 | improved (+0.281212) |
| lgbm_logloss | 0.148726 | improved (-0.184797) |
| lgbm_brier | 0.034164 | improved (-0.023181) |

## E_ret

- anchor_trade_date: 20260331
- status: trained
- loaded_trade_dates: 23
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260330
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.051640 | worse (+0.013826) |
| lr_rmse | 0.067098 | worse (+0.018409) |
| lr_corr | 0.723879 | improved (+0.029285) |
| lr_directional_acc | 0.860000 | improved (+0.069677) |
| lgbm_mae | 0.060378 | worse (+0.015359) |
| lgbm_rmse | 0.076123 | worse (+0.016243) |
| lgbm_corr | 0.675946 | improved (+0.238444) |
| lgbm_directional_acc | 0.720000 | improved (+0.058710) |

