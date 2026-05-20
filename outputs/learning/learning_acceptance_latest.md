# Learning Acceptance Latest

- generated_at_utc: 2026-05-20T16:36:14+00:00
- current_run_date: 20260520
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260519
- status: trained
- loaded_trade_dates: 52
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260518
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.971591 | improved (+0.265097) |
| lr_logloss | 0.228014 | improved (-0.455625) |
| lr_brier | 0.069701 | improved (-0.096804) |
| lgbm_auc | 0.982955 | improved (+0.011526) |
| lgbm_logloss | 0.044180 | improved (-0.064629) |
| lgbm_brier | 0.013852 | improved (-0.020619) |

## E_ret

- anchor_trade_date: 20260518
- status: trained
- loaded_trade_dates: 51
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260515
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.057176 | improved (-0.002627) |
| lr_rmse | 0.079285 | improved (-0.002014) |
| lr_corr | 0.199533 | worse (-0.033449) |
| lr_directional_acc | 0.558442 | worse (-0.034151) |
| lgbm_mae | 0.057118 | improved (-0.005874) |
| lgbm_rmse | 0.077020 | improved (-0.007020) |
| lgbm_corr | 0.317088 | improved (+0.238895) |
| lgbm_directional_acc | 0.519481 | improved (+0.093555) |

