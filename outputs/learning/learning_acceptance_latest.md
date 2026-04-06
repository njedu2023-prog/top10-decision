# Learning Acceptance Latest

- generated_at_utc: 2026-04-06T14:15:56+00:00
- current_run_date: 20260403
- overall_pass: PASS

## P_fill

- anchor_trade_date: 20260402
- status: trained
- loaded_trade_dates: 25
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260401
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.653846 | worse (-0.155245) |
| lr_logloss | 0.469818 | worse (+0.189163) |
| lr_brier | 0.121647 | worse (+0.036402) |
| lgbm_auc | 0.692308 | worse (-0.162238) |
| lgbm_logloss | 0.473321 | worse (+0.324595) |
| lgbm_brier | 0.102923 | worse (+0.068759) |

## E_ret

- anchor_trade_date: 20260401
- status: trained
- loaded_trade_dates: 24
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260331
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.041017 | improved (-0.010623) |
| lr_rmse | 0.053747 | improved (-0.013351) |
| lr_corr | 0.559066 | worse (-0.164813) |
| lr_directional_acc | 0.690909 | worse (-0.169091) |
| lgbm_mae | 0.045748 | improved (-0.014630) |
| lgbm_rmse | 0.059253 | improved (-0.016869) |
| lgbm_corr | 0.334978 | worse (-0.340968) |
| lgbm_directional_acc | 0.636364 | worse (-0.083636) |

