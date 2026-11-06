# Decision Report (20261109)

- signal_date: **20261106**
- exec_date: **20261109**
- exit_date: **20261110**
- requested_trade_date: **20261106**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.2321,volatility_mean=0.0515**
- guardrail_reason: **CAUTION:tail_risk_mean=0.2321**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **29**
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
- available_rows: **29** / **29**
- hard_risk_rows: **5**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.005032**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **29**
- selected_rows: **1**
- positive_ev_rows: **1**
- positive_ev_base_rows: **1**
- positive_e_ret_rows: **27**
- high_pfill_rows: **0**
- low_risk_rows: **1**
- max_EV: **0.002271**
- max_EV_base: **0.002271**
- max_E_ret: **0.019279**
- mean_cost: **0.001733**
- mean_risk_penalty: **0.019415**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261106.csv`
- execution_table: `data/decision/decision_execution_20261109.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261109.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261106.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 603381.SH | 永臻股份 | 4→5 | 0.07 | 0.002271 | 0.724594 | 0.016286 | 0.001195 | 0.008334 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 2 | 605339.SH | 南侨食品 | 2→3 | 0 | -0.001638 | 0.717796 | 0.016803 | 0.00221 | 0.01149 |
| 3 | 603366.SH | 日出东方 | 1→2 | 0 | -0.002459 | 0.677262 | 0.012827 | 0.001138 | 0.010008 |
| 4 | 000525.SZ | 红太阳 | 1→2 | 0 | -0.005097 | 0.708598 | 0.011646 | 0.001217 | 0.012132 |
| 5 | 600540.SH | 新赛股份 | 1→2 | 0 | -0.005246 | 0.735396 | 0.012944 | 0.001612 | 0.013153 |
| 6 | 002578.SZ | 闽发铝业 | 5→6 | 0 | -0.005298 | 0.745427 | 0.016675 | 0.001378 | 0.01635 |
| 7 | 601599.SH | 浙文影业 | 5→6 | 0 | -0.005695 | 0.785797 | 0.015259 | 0.001819 | 0.015867 |
| 8 | 002889.SZ | 东方嘉盛 | 1→2 | 0 | -0.005984 | 0.766142 | 0.015624 | 0.001438 | 0.016517 |
| 9 | 001308.SZ | 康冠科技 | 1→2 | 0 | -0.006249 | 0.689799 | 0.013317 | 0.001955 | 0.013479 |
| 10 | 002712.SZ | 思美传媒 | 1→2 | 0 | -0.006506 | 0.707195 | 0.011629 | 0.001107 | 0.013623 |
| 11 | 002310.SZ | 东方新能 | 1→2 | 0 | -0.006675 | 0.68173 | 0.013738 | 0.002079 | 0.013962 |
| 12 | 601595.SH | 上海电影 | 1→2 | 0 | -0.007398 | 0.74384 | 0.014285 | 0.001327 | 0.016697 |
| 13 | 002388.SZ | 新亚制程 | 3→4 | 0 | -0.007525 | 0.740971 | 0.011959 | 0.001598 | 0.014789 |
| 14 | 002929.SZ | 润建股份 | 2→3 | 0 | -0.008992 | 0.771369 | 0.011392 | 0.001697 | 0.016082 |
| 15 | 600229.SH | 城市传媒 | 1→2 | 0 | -0.009269 | 0.690876 | 0.015831 | 0.002104 | 0.018103 |
| 16 | 002973.SZ | 侨银股份 | 1→2 | 0 | -0.009679 | 0.774204 | 0.014359 | 0.001423 | 0.019373 |
| 17 | 600207.SH | 安彩高科 | 1→2 | 0 | -0.009753 | 0.737448 | 0.019279 | 0.002027 | 0.021943 |
| 18 | 603657.SH | 春光科技 | 3→4 | 0 | -0.01012 | 0.748098 | 0.011898 | 0.001267 | 0.017754 |
| 19 | 001316.SZ | 润贝航科 | 1→2 | 0 | -0.011591 | 0.748935 | 0.012778 | 0.001428 | 0.019734 |
| 20 | 001258.SZ | 立新能源 | 1→2 | 0 | -0.013891 | 0.703269 | 0.014194 | 0.001876 | 0.021996 |
| 21 | 600278.SH | 东方创业 | 1→2 | 0 | -0.014203 | 0.765092 | 0.010037 | 0.00153 | 0.020351 |
| 22 | 600371.SH | 万向德农 | 1→2 | 0 | -0.015272 | 0.834579 | 0.014251 | 0.002219 | 0.024946 |
| 23 | 002086.SZ | 东方海洋 | 2→3 | 0 | -0.016582 | 0.76926 | 0.012514 | 0.003068 | 0.023141 |
| 24 | 002088.SZ | 鲁阳节能 | 4→5 | 0 | -0.018732 | 0.813801 | 0.008938 | 0.002005 | 0.024 |
| 25 | 600261.SH | 阳光照明 | 2→3 | 0 | -0.028299 | 0.595964 | 0.001603 | 0.00162 | 0.027633 |
| 26 | 600327.SH | 大东方 | 1→2 | 0 | -0.032648 | 0.58842 | -0.000489 | 0.001978 | 0.030382 |
| 27 | 600156.SH | 华升股份 | 1→2 | 0 | -0.032734 | 0.643511 | 0.001946 | 0.001531 | 0.032456 |
| 28 | 000595.SZ | 新能股份 | 1→2 | 0 | -0.033808 | 0.563231 | 0.001839 | 0.002149 | 0.032694 |
| 29 | 601188.SH | 龙江交通 | 1→2 | 0 | -0.039772 | 0.654333 | -0.002221 | 0.002259 | 0.036059 |

