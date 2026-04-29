# Learning Acceptance Latest

- generated_at_utc: 2026-04-29T15:26:53+00:00
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
| lr_mae | 0.035418 | improved (-0.009289) |
| lr_rmse | 0.046229 | improved (-0.016943) |
| lr_corr | 0.719956 | improved (+0.146668) |
| lr_directional_acc | 0.774194 | improved (+0.059908) |
| lgbm_mae | 0.030019 | improved (-0.005810) |
| lgbm_rmse | 0.037268 | improved (-0.012113) |
| lgbm_corr | 0.786698 | worse (-0.045201) |
| lgbm_directional_acc | 0.758065 | worse (-0.045507) |

