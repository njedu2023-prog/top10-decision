# Learning Acceptance Latest

- generated_at_utc: 2026-03-30T17:53:07+00:00
- current_run_date: 20260325
- overall_pass: PASS

## P_fill

- anchor_trade_date: 20260324
- status: trained
- loaded_trade_dates: 18
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260323
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_auc | 0.744856 | worse (-0.181070) |
| lr_logloss | 0.351876 | improved (-0.952773) |
| lr_brier | 0.104043 | improved (-0.233438) |
| lgbm_auc | 0.818930 | worse (-0.144033) |
| lgbm_logloss | 0.204998 | worse (+0.057342) |
| lgbm_brier | 0.037420 | improved (-0.010721) |

## E_ret

- anchor_trade_date: 20260323
- status: trained
- loaded_trade_dates: 17
- missing_trade_dates: 0
- previous_anchor_trade_date: 20260320
- acceptance_pass: PASS

| 指标 | 当前 | 对比上次 |
|---|---:|---|
| lr_mae | 0.052749 | worse (+0.009063) |
| lr_rmse | 0.069768 | worse (+0.012977) |
| lr_corr | 0.499329 | improved (+0.581684) |
| lr_directional_acc | 0.666667 | improved (+0.074074) |
| lgbm_mae | 0.065689 | worse (+0.019941) |
| lgbm_rmse | 0.083621 | worse (+0.022651) |
| lgbm_corr | 0.235698 | improved (+0.421534) |
| lgbm_directional_acc | 0.481481 | worse (-0.185185) |

