# Decision Report (20260929)

- signal_date: **20260928**
- exec_date: **20260929**
- exit_date: **20260930**
- requested_trade_date: **20260928**
- regime: **RISK_ON**
- risk_budget: **1**
- regime_reason: **tail_risk_mean=0.1495**
- guardrail_reason: **CAUTION:tail_risk_mean=0.1495**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **31**
- features_base_loaded: **True**
- features_base_rows: **5557**
- features_limit_loaded: **True**
- features_limit_rows: **5557**
- truth_close_loaded: **True**
- truth_close_rows: **5557**
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
- available_rows: **31** / **31**
- hard_risk_rows: **2**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.003922**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **31**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **30**
- high_pfill_rows: **0**
- low_risk_rows: **5**
- max_EV: **-0.000047**
- max_EV_base: **-0.000047**
- max_E_ret: **0.019896**
- mean_cost: **0.00154**
- mean_risk_penalty: **0.015391**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20260928.csv`
- execution_table: `data/decision/decision_execution_20260929.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260929.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20260928.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 605366.SH | 宏柏新材 | 1→2 | 0 | -0.000047 | 0.704479 | 0.012222 | 0.001293 | 0.007363 |
| 2 | 001368.SZ | 通达创智 | 1→2 | 0 | -0.000538 | 0.697319 | 0.010675 | 0.001192 | 0.006789 |
| 3 | 600825.SH | 新华传媒 | 5→6 | 0 | -0.001229 | 0.612178 | 0.019896 | 0.002228 | 0.011181 |
| 4 | 000513.SZ | 丽珠集团 | 1→2 | 0 | -0.00176 | 0.75941 | 0.014383 | 0.001328 | 0.011355 |
| 5 | 000678.SZ | 襄阳轴承 | 2→3 | 0 | -0.002064 | 0.761359 | 0.016607 | 0.001234 | 0.013474 |
| 6 | 001201.SZ | 东瑞股份 | 1→2 | 0 | -0.002191 | 0.733999 | 0.010939 | 0.001152 | 0.009068 |
| 7 | 002232.SZ | 启明信息 | 1→2 | 0 | -0.002613 | 0.697196 | 0.012267 | 0.001166 | 0.009999 |
| 8 | 000020.SZ | 深华发Ａ | 1→2 | 0 | -0.002837 | 0.737699 | 0.012832 | 0.001104 | 0.011199 |
| 9 | 600241.SH | 时代万恒 | 1→2 | 0 | -0.003071 | 0.720872 | 0.009467 | 0.001162 | 0.008733 |
| 10 | 002640.SZ | 跨境通 | 1→2 | 0 | -0.003387 | 0.772021 | 0.014806 | 0.001645 | 0.013173 |
| 11 | 002912.SZ | 中新赛克 | 1→2 | 0 | -0.003584 | 0.722657 | 0.013271 | 0.001208 | 0.011966 |
| 12 | 001330.SZ | 博纳影业 | 1→2 | 0 | -0.003894 | 0.705095 | 0.015764 | 0.001771 | 0.013238 |
| 13 | 603949.SH | 雪龙集团 | 3→4 | 0 | -0.005046 | 0.743108 | 0.015105 | 0.001227 | 0.015044 |
| 14 | 002962.SZ | 五方光电 | 1→2 | 0 | -0.005343 | 0.823955 | 0.013041 | 0.001703 | 0.014385 |
| 15 | 600418.SH | 江淮汽车 | 1→2 | 0 | -0.00609 | 0.71539 | 0.013325 | 0.001221 | 0.014401 |
| 16 | 000503.SZ | 国新健康 | 1→2 | 0 | -0.006489 | 0.758652 | 0.013308 | 0.001324 | 0.015261 |
| 17 | 600802.SH | 福建水泥 | 3→4 | 0 | -0.007372 | 0.777065 | 0.012849 | 0.001655 | 0.015701 |
| 18 | 002347.SZ | 泰尔股份 | 1→2 | 0 | -0.007626 | 0.780672 | 0.011675 | 0.001409 | 0.015331 |
| 19 | 002852.SZ | 道道全 | 1→2 | 0 | -0.00825 | 0.795906 | 0.013049 | 0.001404 | 0.017232 |
| 20 | 000980.SZ | 众泰汽车 | 1→2 | 0 | -0.008676 | 0.74048 | 0.014243 | 0.001476 | 0.017747 |
| 21 | 600488.SH | 津药药业 | 1→2 | 0 | -0.008949 | 0.752588 | 0.011475 | 0.00127 | 0.016316 |
| 22 | 002480.SZ | 新筑股份 | 1→2 | 0 | -0.010105 | 0.803347 | 0.014935 | 0.002085 | 0.020018 |
| 23 | 600032.SH | 浙江新能 | 1→2 | 0 | -0.010164 | 0.744585 | 0.014194 | 0.00204 | 0.018693 |
| 24 | 603968.SH | 醋化股份 | 1→2 | 0 | -0.010309 | 0.65297 | 0.007026 | 0.001205 | 0.013692 |
| 25 | 002342.SZ | 巨力索具 | 1→2 | 0 | -0.011197 | 0.75466 | 0.012326 | 0.00197 | 0.018529 |
| 26 | 601218.SH | 吉鑫科技 | 2→3 | 0 | -0.011981 | 0.780799 | 0.011466 | 0.002086 | 0.018847 |
| 27 | 002242.SZ | 九阳股份 | 1→2 | 0 | -0.012419 | 0.832456 | 0.011742 | 0.002295 | 0.019899 |
| 28 | 000011.SZ | 深物业A | 1→2 | 0 | -0.013625 | 0.725565 | 0.012456 | 0.001883 | 0.020779 |
| 29 | 603396.SH | 金辰股份 | 3→4 | 0 | -0.016896 | 0.736093 | 0.007189 | 0.001654 | 0.020534 |
| 30 | 603278.SH | 大业股份 | 2→3 | 0 | -0.026809 | 0.615025 | 0.001484 | 0.001665 | 0.026056 |
| 31 | 601579.SH | 会稽山 | 1→2 | 0 | -0.033947 | 0.650772 | -0.001752 | 0.001694 | 0.031113 |

