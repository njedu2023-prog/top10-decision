# Learning Acceptance Latest

- generated_at_utc: 2026-04-13T14:56:52+00:00
- current_run_date: 20260413
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260410
- status: trained
- loaded_trade_dates: 28
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260407
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.690909 | improved (+0.246465) |
| lr_logloss | 0.338463 | worse (+0.050894) |
| lr_brier | 0.086794 | worse (+0.054650) |
| lgbm_auc | 0.804545 | improved (+0.186027) |
| lgbm_logloss | 0.289301 | worse (+0.130402) |
| lgbm_brier | 0.062962 | worse (+0.034311) |

## E_ret

- anchor_trade_date: 20260409
- status: trained
- loaded_trade_dates: 27
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260403
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.054804 | worse (+0.018998) |
| lr_rmse | 0.070639 | worse (+0.016197) |
| lr_corr | 0.490693 | worse (-0.173512) |
| lr_directional_acc | 0.596154 | worse (-0.203846) |
| lgbm_mae | 0.045263 | worse (+0.003224) |
| lgbm_rmse | 0.061684 | worse (+0.002623) |
| lgbm_corr | 0.463547 | worse (-0.201287) |
| lgbm_directional_acc | 0.711538 | worse (-0.088462) |

