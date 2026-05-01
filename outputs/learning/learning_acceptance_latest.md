# Learning Acceptance Latest

- generated_at_utc: 2026-05-01T14:20:48+00:00
- current_run_date: 20260429
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260428
- status: trained
- loaded_trade_dates: 40
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260427
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.982456 | improved (+0.141186) |
| lr_logloss | 0.246982 | improved (-0.039998) |
| lr_brier | 0.063659 | improved (-0.027623) |
| lgbm_auc | 0.982456 | improved (+0.371345) |
| lgbm_logloss | 0.084998 | improved (-0.124873) |
| lgbm_brier | 0.026063 | improved (-0.012830) |

## E_ret

- anchor_trade_date: 20260427
- status: trained
- loaded_trade_dates: 39
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260424
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.048125 | worse (+0.003418) |
| lr_rmse | 0.059925 | improved (-0.003247) |
| lr_corr | 0.051638 | worse (-0.521649) |
| lr_directional_acc | 0.532258 | worse (-0.182028) |
| lgbm_mae | 0.051007 | worse (+0.015178) |
| lgbm_rmse | 0.063369 | worse (+0.013987) |
| lgbm_corr | 0.013033 | worse (-0.818866) |
| lgbm_directional_acc | 0.451613 | worse (-0.351959) |

