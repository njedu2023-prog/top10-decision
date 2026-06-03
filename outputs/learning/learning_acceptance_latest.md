# Learning Acceptance Latest

- generated_at_utc: 2026-06-03T18:05:09+00:00
- current_run_date: 20260603
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260602
- status: trained
- loaded_trade_dates: 62
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260601
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.825397 | worse (-0.174603) |
| lr_logloss | 0.340350 | improved (-0.011565) |
| lr_brier | 0.102691 | improved (-0.008213) |
| lgbm_auc | 0.904762 | worse (-0.095238) |
| lgbm_logloss | 0.215714 | worse (+0.194007) |
| lgbm_brier | 0.069553 | worse (+0.064856) |

## E_ret

- anchor_trade_date: 20260601
- status: trained
- loaded_trade_dates: 61
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260529
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.059943 | worse (+0.001615) |
| lr_rmse | 0.079343 | worse (+0.002944) |
| lr_corr | 0.075610 | improved (+0.155533) |
| lr_directional_acc | 0.647059 | improved (+0.067059) |
| lgbm_mae | 0.061053 | worse (+0.002577) |
| lgbm_rmse | 0.079488 | worse (+0.006605) |
| lgbm_corr | 0.152821 | improved (+0.025480) |
| lgbm_directional_acc | 0.571429 | improved (+0.051429) |

