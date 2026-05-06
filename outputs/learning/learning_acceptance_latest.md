# Learning Acceptance Latest

- generated_at_utc: 2026-05-06T15:32:59+00:00
- current_run_date: 20260506
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260430
- status: trained
- loaded_trade_dates: 42
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260429
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.512987 | worse (-0.399471) |
| lr_logloss | 0.347110 | worse (+0.115314) |
| lr_brier | 0.067810 | improved (-0.007576) |
| lgbm_auc | 0.883117 | worse (-0.029341) |
| lgbm_logloss | 0.110141 | improved (-0.017473) |
| lgbm_brier | 0.023530 | improved (-0.005775) |

## E_ret

- anchor_trade_date: 20260429
- status: trained
- loaded_trade_dates: 41
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260428
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.064123 | worse (+0.000273) |
| lr_rmse | 0.094227 | worse (+0.006929) |
| lr_corr | -0.168246 | worse (-0.195366) |
| lr_directional_acc | 0.323232 | worse (-0.185540) |
| lgbm_mae | 0.059983 | improved (-0.005507) |
| lgbm_rmse | 0.086649 | improved (-0.002039) |
| lgbm_corr | -0.003331 | improved (+0.100645) |
| lgbm_directional_acc | 0.373737 | worse (-0.187666) |

