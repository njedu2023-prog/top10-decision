# Learning Acceptance Latest

- generated_at_utc: 2026-05-12T15:46:26+00:00
- current_run_date: 20260512
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260511
- status: trained
- loaded_trade_dates: 46
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260508
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.926966 | improved (+0.153030) |
| lr_logloss | 0.252361 | improved (-0.124831) |
| lr_brier | 0.077093 | improved (-0.050897) |
| lgbm_auc | 0.962547 | improved (+0.095526) |
| lgbm_logloss | 0.190921 | worse (+0.066446) |
| lgbm_brier | 0.044656 | worse (+0.017675) |

## E_ret

- anchor_trade_date: 20260508
- status: trained
- loaded_trade_dates: 45
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260507
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.044741 | improved (-0.008894) |
| lr_rmse | 0.057439 | improved (-0.011776) |
| lr_corr | 0.144149 | improved (+0.136260) |
| lr_directional_acc | 0.478723 | worse (-0.072297) |
| lgbm_mae | 0.048729 | improved (-0.001350) |
| lgbm_rmse | 0.061290 | improved (-0.005752) |
| lgbm_corr | 0.233552 | improved (+0.207771) |
| lgbm_directional_acc | 0.436170 | worse (-0.176075) |

