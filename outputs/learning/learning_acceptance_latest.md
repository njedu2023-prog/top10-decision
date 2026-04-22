# Learning Acceptance Latest

- generated_at_utc: 2026-04-22T14:58:31+00:00
- current_run_date: 20260422
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260421
- status: trained
- loaded_trade_dates: 35
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260420
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.673333 | worse (-0.205045) |
| lr_logloss | 0.298989 | worse (+0.044393) |
| lr_brier | 0.065659 | improved (-0.016050) |
| lgbm_auc | 0.813333 | worse (-0.146126) |
| lgbm_logloss | 0.161045 | worse (+0.101340) |
| lgbm_brier | 0.029507 | worse (+0.015332) |

## E_ret

- anchor_trade_date: 20260420
- status: trained
- loaded_trade_dates: 34
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260417
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.044639 | improved (-0.000174) |
| lr_rmse | 0.055152 | improved (-0.005964) |
| lr_corr | 0.496954 | improved (+0.019759) |
| lr_directional_acc | 0.567568 | worse (-0.044373) |
| lgbm_mae | 0.033664 | improved (-0.010395) |
| lgbm_rmse | 0.041851 | improved (-0.018286) |
| lgbm_corr | 0.789635 | improved (+0.246755) |
| lgbm_directional_acc | 0.689189 | worse (-0.012303) |

