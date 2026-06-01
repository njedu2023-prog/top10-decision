# Learning Acceptance Latest

- generated_at_utc: 2026-06-01T13:20:00+00:00
- current_run_date: 20260601
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260529
- status: trained
- loaded_trade_dates: 60
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260528
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.620000 | worse (-0.107273) |
| lr_logloss | 0.310788 | worse (+0.048474) |
| lr_brier | 0.094373 | worse (+0.014323) |
| lgbm_auc | 0.880000 | worse (-0.022357) |
| lgbm_logloss | 0.084162 | improved (-0.023843) |
| lgbm_brier | 0.020370 | improved (-0.009443) |

## E_ret

- anchor_trade_date: 20260528
- status: trained
- loaded_trade_dates: 59
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260527
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.069604 | worse (+0.007936) |
| lr_rmse | 0.086000 | worse (+0.009258) |
| lr_corr | 0.063960 | improved (+0.058393) |
| lr_directional_acc | 0.494949 | improved (+0.048141) |
| lgbm_mae | 0.074359 | worse (+0.015935) |
| lgbm_rmse | 0.091850 | worse (+0.015296) |
| lgbm_corr | -0.109631 | worse (-0.148438) |
| lgbm_directional_acc | 0.505051 | worse (-0.048141) |

