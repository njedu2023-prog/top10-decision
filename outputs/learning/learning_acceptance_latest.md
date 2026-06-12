# Learning Acceptance Latest

- generated_at_utc: 2026-06-12T16:13:18+00:00
- current_run_date: 20260612
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260611
- status: trained
- loaded_trade_dates: 69
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260610
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.901515 | worse (-0.098485) |
| lr_logloss | 0.388837 | worse (+0.171063) |
| lr_brier | 0.110765 | worse (+0.047216) |
| lgbm_auc | 0.969697 | worse (-0.023057) |
| lgbm_logloss | 0.118300 | worse (+0.080275) |
| lgbm_brier | 0.039757 | worse (+0.028353) |

## E_ret

- anchor_trade_date: 20260610
- status: trained
- loaded_trade_dates: 68
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260609
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.057379 | improved (-0.002174) |
| lr_rmse | 0.073644 | improved (-0.002993) |
| lr_corr | 0.176407 | improved (+0.235114) |
| lr_directional_acc | 0.594203 | improved (+0.070009) |
| lgbm_mae | 0.063987 | worse (+0.001813) |
| lgbm_rmse | 0.078793 | improved (-0.001464) |
| lgbm_corr | 0.227495 | improved (+0.266542) |
| lgbm_directional_acc | 0.420290 | worse (-0.128097) |

