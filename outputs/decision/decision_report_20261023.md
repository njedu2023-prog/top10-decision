# Decision Report (20261023)

- signal_date: **20261022**
- exec_date: **20261023**
- exit_date: **20261026**
- requested_trade_date: **20261022**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.2067,volatility_mean=0.0524**
- guardrail_reason: **CAUTION:open_board_max=7.0,tail_risk_mean=0.2067**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **30**
- features_base_loaded: **True**
- features_base_rows: **5570**
- features_limit_loaded: **True**
- features_limit_rows: **5570**
- truth_close_loaded: **True**
- truth_close_rows: **5570**
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
- available_rows: **30** / **30**
- hard_risk_rows: **7**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.005377**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **30**
- selected_rows: **2**
- positive_ev_rows: **2**
- positive_ev_base_rows: **2**
- positive_e_ret_rows: **25**
- high_pfill_rows: **0**
- low_risk_rows: **3**
- max_EV: **0.001426**
- max_EV_base: **0.001426**
- max_E_ret: **0.023752**
- mean_cost: **0.001849**
- mean_risk_penalty: **0.019544**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261022.csv`
- execution_table: `data/decision/decision_execution_20261023.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261023.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261022.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 600293.SH | 三峡新材 | 3→4 | 0.07 | 0.001426 | 0.634177 | 0.023752 | 0.002129 | 0.011509 |
| 2 | 600698.SH | 湖南天雁 | 1→2 | 0.07 | 0.000632 | 0.737073 | 0.014485 | 0.00115 | 0.008894 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 3 | 603278.SH | 大业股份 | 5→6 | 0 | -0.001145 | 0.706333 | 0.015071 | 0.001304 | 0.010486 |
| 4 | 002249.SZ | 大洋电机 | 1→2 | 0 | -0.00148 | 0.66447 | 0.016959 | 0.001217 | 0.011531 |
| 5 | 600192.SH | 长城电工 | 1→2 | 0 | -0.001981 | 0.712129 | 0.013365 | 0.001167 | 0.010331 |
| 6 | 603528.SH | 多伦科技 | 1→2 | 0 | -0.00266 | 0.710368 | 0.011794 | 0.001154 | 0.009885 |
| 7 | 600838.SH | 上海九百 | 2→3 | 0 | -0.003423 | 0.724389 | 0.010944 | 0.001356 | 0.009994 |
| 8 | 603350.SH | 安乃达 | 1→2 | 0 | -0.004324 | 0.748013 | 0.016035 | 0.001297 | 0.015021 |
| 9 | 603370.SH | 华新精科 | 1→2 | 0 | -0.004565 | 0.683901 | 0.012275 | 0.00132 | 0.011641 |
| 10 | 001360.SZ | 南矿集团 | 1→2 | 0 | -0.005323 | 0.66856 | 0.018238 | 0.002149 | 0.015367 |
| 11 | 600261.SH | 阳光照明 | 1→2 | 0 | -0.005512 | 0.767575 | 0.01393 | 0.001416 | 0.014788 |
| 12 | 002097.SZ | 山河智能 | 1→2 | 0 | -0.00588 | 0.781538 | 0.014872 | 0.00144 | 0.016063 |
| 13 | 603028.SH | 赛福天 | 1→2 | 0 | -0.006163 | 0.701926 | 0.013128 | 0.00174 | 0.013637 |
| 14 | 603280.SH | 南方路机 | 1→2 | 0 | -0.006235 | 0.748344 | 0.014724 | 0.001317 | 0.015937 |
| 15 | 603330.SH | 天洋新材 | 2→3 | 0 | -0.007667 | 0.761597 | 0.013419 | 0.002254 | 0.015633 |
| 16 | 003021.SZ | 兆威机电 | 1→2 | 0 | -0.007801 | 0.786508 | 0.017857 | 0.001996 | 0.019849 |
| 17 | 002535.SZ | 林州重机 | 1→2 | 0 | -0.00781 | 0.741077 | 0.011781 | 0.001461 | 0.01508 |
| 18 | 000678.SZ | 襄阳轴承 | 1→2 | 0 | -0.008573 | 0.764601 | 0.013792 | 0.001498 | 0.01762 |
| 19 | 603933.SH | 睿能科技 | 1→2 | 0 | -0.008673 | 0.696851 | 0.013289 | 0.001891 | 0.016042 |
| 20 | 603089.SH | 正裕工业 | 1→2 | 0 | -0.009399 | 0.768477 | 0.011796 | 0.001375 | 0.017089 |
| 21 | 002983.SZ | 芯瑞达 | 1→2 | 0 | -0.009956 | 0.660064 | 0.008559 | 0.001214 | 0.014392 |
| 22 | 002307.SZ | 北新路桥 | 1→2 | 0 | -0.011686 | 0.856276 | 0.010393 | 0.001963 | 0.018622 |
| 23 | 603728.SH | 鸣志电器 | 1→2 | 0 | -0.016422 | 0.812304 | 0.011093 | 0.00212 | 0.023313 |
| 24 | 601999.SH | 出版传媒 | 2→3 | 0 | -0.035739 | 0.636093 | -0.00016 | 0.002161 | 0.033477 |
| 25 | 603138.SH | 海量数据 | 2→3 | 0 | -0.037169 | 0.713193 | 0.001275 | 0.002237 | 0.035842 |
| 26 | 002724.SZ | 海洋王 | 5→6 | 0 | -0.037657 | 0.601181 | -0.000308 | 0.002089 | 0.035383 |
| 27 | 000530.SZ | 冰山冷热 | 1→2 | 0 | -0.038794 | 0.671293 | -0.00088 | 0.00347 | 0.034733 |
| 28 | 002076.SZ | 星光股份 | 3→4 | 0 | -0.039298 | 0.651107 | 0.000827 | 0.003161 | 0.036676 |
| 29 | 600202.SH | 哈空调 | 2→3 | 0 | -0.043909 | 0.615243 | -0.00406 | 0.002923 | 0.038488 |
| 30 | 601091.SH | 沈鼓集团 | 1→2 | 0 | -0.044422 | 0.617481 | -0.003113 | 0.0035 | 0.039 |

