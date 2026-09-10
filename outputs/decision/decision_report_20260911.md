# Decision Report (20260911)

- signal_date: **20260910**
- exec_date: **20260911**
- exit_date: **20260914**
- requested_trade_date: **20260910**
- regime: **RISK_ON**
- risk_budget: **1**
- regime_reason: **tail_risk_mean=0.1375**
- guardrail_reason: **CAUTION:open_board_max=5.0,tail_risk_mean=0.1375**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **34**
- features_base_loaded: **True**
- features_base_rows: **5549**
- features_limit_loaded: **True**
- features_limit_rows: **5549**
- truth_close_loaded: **True**
- truth_close_rows: **5549**
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
- available_rows: **34** / **34**
- hard_risk_rows: **5**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.004504**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **34**
- selected_rows: **4**
- positive_ev_rows: **4**
- positive_ev_base_rows: **4**
- positive_e_ret_rows: **31**
- high_pfill_rows: **0**
- low_risk_rows: **7**
- max_EV: **0.004504**
- max_EV_base: **0.004504**
- max_E_ret: **0.018645**
- mean_cost: **0.001527**
- mean_risk_penalty: **0.015931**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20260910.csv`
- execution_table: `data/decision/decision_execution_20260911.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260911.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20260910.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 002377.SZ | 国创高新 | 2→3 | 0.1 | 0.004504 | 0.731004 | 0.013811 | 0.001234 | 0.004358 |
| 2 | 600644.SH | 乐山电力 | 1→2 | 0.1 | 0.002172 | 0.730908 | 0.016056 | 0.001295 | 0.008269 |
| 3 | 600744.SH | 华银电力 | 1→2 | 0.1 | 0.001936 | 0.703573 | 0.018645 | 0.001159 | 0.010023 |
| 4 | 600712.SH | 南宁百货 | 1→2 | 0.1 | 0.000195 | 0.733578 | 0.012522 | 0.001283 | 0.007708 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 5 | 603421.SH | 鼎信通讯 | 2→3 | 0 | -0.000235 | 0.714212 | 0.012258 | 0.001134 | 0.007855 |
| 6 | 002204.SZ | 大连重工 | 1→2 | 0 | -0.00096 | 0.689321 | 0.013052 | 0.001222 | 0.008736 |
| 7 | 000978.SZ | 桂林旅游 | 4→5 | 0 | -0.00187 | 0.771799 | 0.013457 | 0.001544 | 0.010712 |
| 8 | 000722.SZ | 湖南发展 | 1→2 | 0 | -0.001977 | 0.731371 | 0.01623 | 0.001259 | 0.012587 |
| 9 | 000993.SZ | 闽东电力 | 2→3 | 0 | -0.002169 | 0.735913 | 0.017806 | 0.001364 | 0.013908 |
| 10 | 002790.SZ | 瑞尔特 | 3→4 | 0 | -0.002295 | 0.662831 | 0.014171 | 0.001853 | 0.009834 |
| 11 | 603105.SH | 芯能科技 | 1→2 | 0 | -0.002523 | 0.771518 | 0.01801 | 0.001403 | 0.015014 |
| 12 | 000565.SZ | 渝三峡Ａ | 2→3 | 0 | -0.00308 | 0.700267 | 0.011173 | 0.001119 | 0.009785 |
| 13 | 002531.SZ | 天顺风能 | 1→2 | 0 | -0.003797 | 0.707906 | 0.011854 | 0.001236 | 0.010952 |
| 14 | 600359.SH | 新农开发 | 2→3 | 0 | -0.004428 | 0.782624 | 0.013555 | 0.001307 | 0.013729 |
| 15 | 002174.SZ | 游族网络 | 1→2 | 0 | -0.005032 | 0.767757 | 0.015753 | 0.001409 | 0.015717 |
| 16 | 600667.SH | 太极实业 | 1→2 | 0 | -0.005185 | 0.582395 | 0.012446 | 0.001139 | 0.011294 |
| 17 | 002543.SZ | 万和电气 | 1→2 | 0 | -0.005881 | 0.708406 | 0.008376 | 0.0012 | 0.010614 |
| 18 | 002201.SZ | 九鼎新材 | 1→2 | 0 | -0.00612 | 0.780047 | 0.01298 | 0.001405 | 0.01484 |
| 19 | 002636.SZ | 金安国纪 | 1→2 | 0 | -0.007151 | 0.728055 | 0.015172 | 0.001374 | 0.016823 |
| 20 | 603316.SH | 诚邦股份 | 1→2 | 0 | -0.008114 | 0.785308 | 0.012245 | 0.001371 | 0.016358 |
| 21 | 600192.SH | 长城电工 | 2→3 | 0 | -0.008275 | 0.801823 | 0.011697 | 0.001567 | 0.016088 |
| 22 | 603601.SH | 再升科技 | 1→2 | 0 | -0.010263 | 0.757756 | 0.011142 | 0.00156 | 0.017146 |
| 23 | 000048.SZ | 京基智农 | 1→2 | 0 | -0.011184 | 0.752044 | 0.010936 | 0.001432 | 0.017976 |
| 24 | 000823.SZ | 超声电子 | 1→2 | 0 | -0.011453 | 0.752111 | 0.013835 | 0.002017 | 0.019842 |
| 25 | 601231.SH | 环旭电子 | 1→2 | 0 | -0.011622 | 0.687117 | 0.010155 | 0.002062 | 0.016538 |
| 26 | 600876.SH | 凯盛新能 | 1→2 | 0 | -0.011767 | 0.744596 | 0.010061 | 0.0021 | 0.017158 |
| 27 | 002661.SZ | 克明食品 | 1→2 | 0 | -0.012071 | 0.828725 | 0.010334 | 0.001769 | 0.018866 |
| 28 | 603318.SH | 水发燃气 | 1→2 | 0 | -0.012696 | 0.788868 | 0.012604 | 0.002038 | 0.020601 |
| 29 | 002912.SZ | 中新赛克 | 1→2 | 0 | -0.013813 | 0.799338 | 0.008219 | 0.001611 | 0.018771 |
| 30 | 600792.SH | 云煤能源 | 3→4 | 0 | -0.027578 | 0.612669 | 0.000138 | 0.001436 | 0.026226 |
| 31 | 000532.SZ | 华金资本 | 1→2 | 0 | -0.028942 | 0.46411 | -0.003761 | 0.001755 | 0.025441 |
| 32 | 001299.SZ | 美能能源 | 1→2 | 0 | -0.034066 | 0.667901 | -0.001162 | 0.001908 | 0.031383 |
| 33 | 600318.SH | 新力金融 | 1→2 | 0 | -0.034739 | 0.699429 | 0.00073 | 0.002323 | 0.032927 |
| 34 | 600488.SH | 津药药业 | 1→2 | 0 | -0.03767 | 0.670489 | -0.003109 | 0.002027 | 0.033558 |

