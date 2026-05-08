# Learning Acceptance Latest

- generated_at_utc: 2026-05-08T15:03:15+00:00
- current_run_date: 20260508
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260507
- status: trained
- loaded_trade_dates: 44
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260506
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.942177 | improved (+0.092687) |
| lr_logloss | 0.186639 | improved (-0.154242) |
| lr_brier | 0.060075 | improved (-0.029560) |
| lgbm_auc | 0.867347 | improved (+0.176020) |
| lgbm_logloss | 0.117350 | improved (-0.131853) |
| lgbm_brier | 0.023693 | improved (-0.013152) |

## E_ret

- anchor_trade_date: 20260506
- status: trained
- loaded_trade_dates: 43
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260430
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.046126 | improved (-0.004005) |
| lr_rmse | 0.060221 | improved (-0.003120) |
| lr_corr | 0.028861 | improved (+0.119596) |
| lr_directional_acc | 0.551020 | improved (+0.174397) |
| lgbm_mae | 0.065171 | worse (+0.012097) |
| lgbm_rmse | 0.081812 | worse (+0.015970) |
| lgbm_corr | 0.010020 | improved (+0.100009) |
| lgbm_directional_acc | 0.397959 | worse (-0.017625) |

