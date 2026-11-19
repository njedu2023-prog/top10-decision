# Decision Report (20261120)

- signal_date: **20261119**
- exec_date: **20261120**
- exit_date: **20261123**
- requested_trade_date: **20261119**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1982,volatility_mean=0.0474,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.1982**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **23**
- features_base_loaded: **True**
- features_base_rows: **5573**
- features_limit_loaded: **True**
- features_limit_rows: **5573**
- truth_close_loaded: **True**
- truth_close_rows: **5573**
- meta_loaded: **True**

## Engine Status

- p_fill_pred_src: **rule**
- p_fill_model_loaded: **True**
- p_fill_model_kind: **lgbm**
- p_fill_degrade_reason: **model_rejected_by_learning_acceptance:model_meta_not_trained**
- eret_pred_src: **rule**
- eret_model_loaded: **True**
- eret_model_kind: **none**
- eret_degrade_reason: **model_rejected_by_learning_acceptance:selected_model_pass_false**

## Intraday Risk Status

- fields_present: **True**
- available_rows: **0** / **23**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **23**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **23**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.009334**
- max_EV_base: **-0.009334**
- max_E_ret: **0.013592**
- mean_cost: **0.001686**
- mean_risk_penalty: **0.020857**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261119.csv`
- execution_table: `data/decision/decision_execution_20261120.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261120.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261119.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 600520.SH | 三佳科技 | 1→2 | 0 | -0.009334 | 0.622023 | 0.009724 | 0.001132 | 0.014251 |
| 2 | 002165.SZ | 红 宝 丽 | 1→2 | 0 | -0.01085 | 0.614541 | 0.010641 | 0.001142 | 0.016247 |
| 3 | 603078.SH | 江化微 | 1→2 | 0 | -0.011747 | 0.629967 | 0.009023 | 0.001268 | 0.016164 |
| 4 | 603324.SH | 盛剑科技 | 3→4 | 0 | -0.012105 | 0.626917 | 0.006965 | 0.001133 | 0.015338 |
| 5 | 002418.SZ | 康盛股份 | 1→2 | 0 | -0.012577 | 0.606645 | 0.009141 | 0.001295 | 0.016827 |
| 6 | 002579.SZ | 中京电子 | 1→2 | 0 | -0.013276 | 0.623289 | 0.009161 | 0.001286 | 0.0177 |
| 7 | 601616.SH | 广电电气 | 1→2 | 0 | -0.013347 | 0.619367 | 0.007605 | 0.001314 | 0.016743 |
| 8 | 603030.SH | 全筑股份 | 3→4 | 0 | -0.014921 | 0.604322 | 0.012418 | 0.002134 | 0.020292 |
| 9 | 600619.SH | 海立股份 | 1→2 | 0 | -0.015241 | 0.692401 | 0.01032 | 0.00156 | 0.020827 |
| 10 | 002420.SZ | 毅昌科技 | 1→2 | 0 | -0.016137 | 0.457813 | 0.013592 | 0.002024 | 0.020336 |
| 11 | 603928.SH | 兴业股份 | 1→2 | 0 | -0.016384 | 0.721917 | 0.008408 | 0.001722 | 0.020731 |
| 12 | 600156.SH | 华升股份 | 2→3 | 0 | -0.016404 | 0.73648 | 0.009511 | 0.001768 | 0.021641 |
| 13 | 000711.SZ | 京蓝科技 | 2→3 | 0 | -0.016588 | 0.66424 | 0.011244 | 0.001435 | 0.022622 |
| 14 | 605178.SH | 时空科技 | 1→2 | 0 | -0.017919 | 0.601927 | 0.009535 | 0.001825 | 0.021834 |
| 15 | 603931.SH | 格林达 | 1→2 | 0 | -0.01794 | 0.639354 | 0.010074 | 0.001952 | 0.022429 |
| 16 | 002771.SZ | 真视通 | 3→4 | 0 | -0.018492 | 0.574781 | 0.009695 | 0.001946 | 0.022118 |
| 17 | 603813.SH | 原尚股份 | 1→2 | 0 | -0.018498 | 0.600781 | 0.009623 | 0.001924 | 0.022355 |
| 18 | 002077.SZ | 大港股份 | 1→2 | 0 | -0.01863 | 0.6818 | 0.00988 | 0.001366 | 0.024 |
| 19 | 000656.SZ | 金科股份 | 1→2 | 0 | -0.018892 | 0.689283 | 0.00972 | 0.001592 | 0.024 |
| 20 | 002146.SZ | 荣盛发展 | nan | 0 | -0.019061 | 0.662627 | 0.008505 | 0.001473 | 0.023224 |
| 21 | 002242.SZ | 九阳股份 | 1→2 | 0 | -0.019467 | 0.756475 | 0.010717 | 0.002387 | 0.025187 |
| 22 | 001358.SZ | 兴欣新材 | 1→2 | 0 | -0.021703 | 0.680501 | 0.009392 | 0.00225 | 0.025844 |
| 23 | 000678.SZ | 襄阳轴承 | 1→2 | 0 | -0.027195 | 0.769571 | 0.006031 | 0.002837 | 0.029 |

