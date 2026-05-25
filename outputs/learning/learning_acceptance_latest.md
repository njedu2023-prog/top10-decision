# Learning Acceptance Latest

- generated_at_utc: 2026-05-25T16:03:08+00:00
- current_run_date: 20260525
- overall_pass: FAIL

## P_fill

- anchor_trade_date: 20260522
- status: trained
- loaded_trade_dates: 55
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260521
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.881081 | improved (+0.016498) |
| lr_logloss | 0.243310 | improved (-0.346449) |
| lr_brier | 0.077937 | improved (-0.054028) |
| lgbm_auc | 0.794595 | worse (-0.153322) |
| lgbm_logloss | 0.154697 | improved (-0.044542) |
| lgbm_brier | 0.033765 | improved (-0.024067) |

## E_ret

- anchor_trade_date: 20260521
- status: trained
- loaded_trade_dates: 54
- missing_trade_dates: 2
- previous_anchor_trade_date: 20260520
- acceptance_pass: FAIL

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.064050 | improved (-0.001166) |
| lr_rmse | 0.077992 | improved (-0.006183) |
| lr_corr | 0.013371 | worse (-0.078131) |
| lr_directional_acc | 0.625000 | improved (+0.150424) |
| lgbm_mae | 0.066599 | worse (+0.002977) |
| lgbm_rmse | 0.083227 | improved (-0.003969) |
| lgbm_corr | 0.022971 | improved (+0.029410) |
| lgbm_directional_acc | 0.343750 | worse (-0.283369) |

