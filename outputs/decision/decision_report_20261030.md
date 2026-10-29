# Decision Report (20261030)

- signal_date: **20261029**
- exec_date: **20261030**
- exit_date: **20261102**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1999,volatility_mean=0.0463,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.1999**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **38**
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
- available_rows: **0** / **38**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **38**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **38**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.009513**
- max_EV_base: **-0.009513**
- max_E_ret: **0.012321**
- mean_cost: **0.001741**
- mean_risk_penalty: **0.021383**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261029.csv`
- execution_table: `data/decision/decision_execution_20261030.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261030.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261029.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 603131.SH | 上海沪工 | 2→3 | 0 | -0.009513 | 0.613121 | 0.010798 | 0.001209 | 0.014924 |
| 2 | 603703.SH | 盛洋科技 | 1→2 | 0 | -0.011117 | 0.584999 | 0.008041 | 0.001109 | 0.014712 |
| 3 | 003032.SZ | 传智教育 | 1→2 | 0 | -0.012546 | 0.620413 | 0.008343 | 0.001118 | 0.016604 |
| 4 | 002679.SZ | 福建金森 | 1→2 | 0 | -0.013479 | 0.616568 | 0.008734 | 0.001164 | 0.0177 |
| 5 | 002275.SZ | 桂林三金 | 1→2 | 0 | -0.013542 | 0.68424 | 0.008674 | 0.001477 | 0.018 |
| 6 | 002086.SZ | 东方海洋 | 1→2 | 0 | -0.013756 | 0.625038 | 0.010901 | 0.001892 | 0.018678 |
| 7 | 002526.SZ | 山东矿机 | 2→3 | 0 | -0.014659 | 0.604339 | 0.009104 | 0.002023 | 0.018138 |
| 8 | 600715.SH | 文投控股 | 3→4 | 0 | -0.015383 | 0.779911 | 0.011385 | 0.002249 | 0.022013 |
| 9 | 600576.SH | 祥源文旅 | 1→2 | 0 | -0.015394 | 0.613808 | 0.008145 | 0.001712 | 0.018681 |
| 10 | 605598.SH | 上海港湾 | 1→2 | 0 | -0.016123 | 0.60067 | 0.010964 | 0.001945 | 0.020763 |
| 11 | 600501.SH | 航天晨光 | 1→2 | 0 | -0.016164 | 0.669748 | 0.009511 | 0.001343 | 0.021192 |
| 12 | 001219.SZ | 青岛食品 | 4→5 | 0 | -0.016385 | 0.58342 | 0.009948 | 0.002064 | 0.020126 |
| 13 | 600261.SH | 阳光照明 | 1→2 | 0 | -0.01644 | 0.631944 | 0.009504 | 0.001521 | 0.020925 |
| 14 | 603006.SH | 联明股份 | 2→3 | 0 | -0.016684 | 0.619468 | 0.008947 | 0.001937 | 0.02029 |
| 15 | 603382.SH | 海阳科技 | 1→2 | 0 | -0.016736 | 0.691059 | 0.008547 | 0.001454 | 0.021188 |
| 16 | 603536.SH | 惠发食品 | 1→2 | 0 | -0.016804 | 0.650695 | 0.008728 | 0.001299 | 0.021184 |
| 17 | 600643.SH | 爱建集团 | 1→2 | 0 | -0.016855 | 0.760163 | 0.007058 | 0.001936 | 0.020284 |
| 18 | 601956.SH | 东贝集团 | 3→4 | 0 | -0.016915 | 0.579958 | 0.009083 | 0.00192 | 0.020263 |
| 19 | 000701.SZ | 厦门信达 | 1→2 | 0 | -0.017109 | 0.652207 | 0.008143 | 0.001244 | 0.021177 |
| 20 | 600997.SH | 开滦股份 | 1→2 | 0 | -0.017232 | 0.711901 | 0.008744 | 0.002368 | 0.021088 |
| 21 | 002865.SZ | 钧达股份 | 1→2 | 0 | -0.017472 | 0.645872 | 0.009292 | 0.001391 | 0.022083 |
| 22 | 601188.SH | 龙江交通 | 1→2 | 0 | -0.017545 | 0.687726 | 0.011127 | 0.001661 | 0.023537 |
| 23 | 600592.SH | 龙溪股份 | 1→2 | 0 | -0.017602 | 0.600736 | 0.009188 | 0.001764 | 0.021358 |
| 24 | 605338.SH | 巴比食品 | 1→2 | 0 | -0.017605 | 0.617115 | 0.007478 | 0.001977 | 0.020243 |
| 25 | 600428.SH | 中远海特 | 1→2 | 0 | -0.017774 | 0.677526 | 0.007882 | 0.002336 | 0.020778 |
| 26 | 002149.SZ | 西部材料 | 1→2 | 0 | -0.017848 | 0.644087 | 0.01058 | 0.001362 | 0.0233 |
| 27 | 002862.SZ | 实丰文化 | 2→3 | 0 | -0.01805 | 0.503453 | 0.012321 | 0.00205 | 0.022203 |
| 28 | 603103.SH | 横店影视 | 1→2 | 0 | -0.018087 | 0.638997 | 0.010883 | 0.002119 | 0.022923 |
| 29 | 002565.SZ | 顺灏股份 | 1→2 | 0 | -0.018128 | 0.645828 | 0.010086 | 0.001342 | 0.0233 |
| 30 | 600118.SH | 中国卫星 | 1→2 | 0 | -0.018361 | 0.734347 | 0.008684 | 0.001811 | 0.022927 |
| 31 | 001225.SZ | 和泰机电 | 1→2 | 0 | -0.018999 | 0.755749 | 0.009065 | 0.00185 | 0.024 |
| 32 | 002277.SZ | 友阿股份 | 1→2 | 0 | -0.019034 | 0.722866 | 0.006877 | 0.001551 | 0.022454 |
| 33 | 002943.SZ | 宇晶股份 | 2→3 | 0 | -0.019803 | 0.697752 | 0.009972 | 0.001743 | 0.025018 |
| 34 | 002531.SZ | 天顺风能 | 1→2 | 0 | -0.019804 | 0.708389 | 0.007991 | 0.001465 | 0.024 |
| 35 | 605179.SH | 一鸣食品 | 1→2 | 0 | -0.01993 | 0.684391 | 0.007992 | 0.0014 | 0.024 |
| 36 | 002172.SZ | 澳洋健康 | 2→3 | 0 | -0.019932 | 0.748698 | 0.009722 | 0.00201 | 0.025201 |
| 37 | 000530.SZ | 冰山冷热 | 1→2 | 0 | -0.024538 | 0.685053 | 0.010507 | 0.002736 | 0.029 |
| 38 | 002708.SZ | 光洋股份 | 4→5 | 0 | -0.025126 | 0.793868 | 0.007266 | 0.002595 | 0.028299 |

