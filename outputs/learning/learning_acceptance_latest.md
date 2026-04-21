# Learning Acceptance Latest

- generated_at_utc: 2026-04-21T14:54:49+00:00
- current_run_date: 20260421
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260420
- status: trained
- loaded_trade_dates: 34
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260417
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.878378 | improved (+0.061214) |
| lr_logloss | 0.254596 | worse (+0.109862) |
| lr_brier | 0.081709 | worse (+0.058927) |
| lgbm_auc | 0.959459 | improved (+0.075131) |
| lgbm_logloss | 0.059704 | improved (-0.144080) |
| lgbm_brier | 0.014175 | improved (-0.027082) |

## E_ret

- anchor_trade_date: 20260417
- status: trained
- loaded_trade_dates: 33
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260416
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.044813 | worse (+0.010014) |
| lr_rmse | 0.061116 | worse (+0.017408) |
| lr_corr | 0.477195 | worse (-0.251161) |
| lr_directional_acc | 0.611940 | worse (-0.085428) |
| lgbm_mae | 0.044058 | worse (+0.006924) |
| lgbm_rmse | 0.060137 | worse (+0.008761) |
| lgbm_corr | 0.542880 | improved (+0.002067) |
| lgbm_directional_acc | 0.701493 | worse (-0.009034) |

