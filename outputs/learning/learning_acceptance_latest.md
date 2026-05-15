# Learning Acceptance Latest

- generated_at_utc: 2026-05-15T15:29:00+00:00
- current_run_date: 20260515
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260514
- status: trained
- loaded_trade_dates: 49
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260513
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 1.000000 | improved (+0.165775) |
| lr_logloss | 0.197134 | improved (-0.106165) |
| lr_brier | 0.057848 | improved (-0.028236) |
| lgbm_auc | 1.000000 | improved (+0.131907) |
| lgbm_logloss | 0.048294 | improved (-0.186620) |
| lgbm_brier | 0.015920 | improved (-0.028962) |

## E_ret

- anchor_trade_date: 20260513
- status: trained
- loaded_trade_dates: 48
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260512
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.078046 | worse (+0.019845) |
| lr_rmse | 0.091028 | worse (+0.017375) |
| lr_corr | 0.086369 | improved (+0.067099) |
| lr_directional_acc | 0.274510 | worse (-0.159452) |
| lgbm_mae | 0.073224 | worse (+0.013245) |
| lgbm_rmse | 0.087216 | worse (+0.010349) |
| lgbm_corr | 0.201586 | improved (+0.263500) |
| lgbm_directional_acc | 0.529412 | improved (+0.095450) |

