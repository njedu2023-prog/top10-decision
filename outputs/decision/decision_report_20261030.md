# Decision Report (20261030)

- signal_date: **20261029**
- exec_date: **20261030**
- exit_date: **20261102**
- requested_trade_date: **20261029**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1999,volatility_mean=0.0463**
- guardrail_reason: **CAUTION:open_board_max=7.0,tail_risk_mean=0.1999**
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
- available_rows: **38** / **38**
- hard_risk_rows: **8**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.004454**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **38**
- selected_rows: **1**
- positive_ev_rows: **1**
- positive_ev_base_rows: **1**
- positive_e_ret_rows: **34**
- high_pfill_rows: **0**
- low_risk_rows: **3**
- max_EV: **0.00466**
- max_EV_base: **0.00466**
- max_E_ret: **0.01853**
- mean_cost: **0.001741**
- mean_risk_penalty: **0.019232**
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
| 1 | 603131.SH | 上海沪工 | 2→3 | 0.07 | 0.00466 | 0.728232 | 0.01853 | 0.001209 | 0.007625 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 2 | 003032.SZ | 传智教育 | 1→2 | 0 | -0.000513 | 0.737938 | 0.014134 | 0.001118 | 0.009825 |
| 3 | 002526.SZ | 山东矿机 | 2→3 | 0 | -0.000853 | 0.730197 | 0.015715 | 0.002023 | 0.010305 |
| 4 | 603703.SH | 盛洋科技 | 1→2 | 0 | -0.002238 | 0.690909 | 0.011823 | 0.001109 | 0.009297 |
| 5 | 002679.SZ | 福建金森 | 1→2 | 0 | -0.002472 | 0.727861 | 0.013307 | 0.001164 | 0.010994 |
| 6 | 001219.SZ | 青岛食品 | 4→5 | 0 | -0.003429 | 0.711045 | 0.015369 | 0.002064 | 0.012293 |
| 7 | 601956.SH | 东贝集团 | 3→4 | 0 | -0.003837 | 0.706665 | 0.014877 | 0.00192 | 0.01243 |
| 8 | 600501.SH | 航天晨光 | 1→2 | 0 | -0.005132 | 0.777675 | 0.014981 | 0.001343 | 0.01544 |
| 9 | 605598.SH | 上海港湾 | 1→2 | 0 | -0.005502 | 0.714116 | 0.015356 | 0.001945 | 0.014523 |
| 10 | 002862.SZ | 实丰文化 | 2→3 | 0 | -0.005538 | 0.646872 | 0.016823 | 0.00205 | 0.01437 |
| 11 | 603006.SH | 联明股份 | 2→3 | 0 | -0.00558 | 0.725668 | 0.013854 | 0.001937 | 0.013697 |
| 12 | 002943.SZ | 宇晶股份 | 2→3 | 0 | -0.005825 | 0.766719 | 0.017402 | 0.001743 | 0.017425 |
| 13 | 600592.SH | 龙溪股份 | 1→2 | 0 | -0.006247 | 0.701954 | 0.015136 | 0.001764 | 0.015108 |
| 14 | 605338.SH | 巴比食品 | 1→2 | 0 | -0.006294 | 0.70396 | 0.013008 | 0.001977 | 0.013474 |
| 15 | 000701.SZ | 厦门信达 | 1→2 | 0 | -0.006611 | 0.759658 | 0.013071 | 0.001244 | 0.015297 |
| 16 | 002565.SZ | 顺灏股份 | 1→2 | 0 | -0.007463 | 0.7535 | 0.014063 | 0.001342 | 0.016718 |
| 17 | 600261.SH | 阳光照明 | 1→2 | 0 | -0.007518 | 0.731444 | 0.015163 | 0.001521 | 0.017088 |
| 18 | 002865.SZ | 钧达股份 | 1→2 | 0 | -0.007573 | 0.752319 | 0.013617 | 0.001391 | 0.016427 |
| 19 | 002275.SZ | 桂林三金 | 1→2 | 0 | -0.007767 | 0.761424 | 0.012189 | 0.001477 | 0.015571 |
| 20 | 600428.SH | 中远海特 | 1→2 | 0 | -0.009792 | 0.785767 | 0.013778 | 0.002336 | 0.018282 |
| 21 | 002172.SZ | 澳洋健康 | 2→3 | 0 | -0.011598 | 0.853236 | 0.014155 | 0.00201 | 0.021665 |
| 22 | 603382.SH | 海阳科技 | 1→2 | 0 | -0.012847 | 0.747487 | 0.009001 | 0.001454 | 0.018122 |
| 23 | 603103.SH | 横店影视 | 1→2 | 0 | -0.013002 | 0.750285 | 0.014173 | 0.002119 | 0.021517 |
| 24 | 002531.SZ | 天顺风能 | 1→2 | 0 | -0.013176 | 0.793846 | 0.012206 | 0.001465 | 0.021401 |
| 25 | 002149.SZ | 西部材料 | 1→2 | 0 | -0.013468 | 0.690379 | 0.011621 | 0.001362 | 0.020128 |
| 26 | 000530.SZ | 冰山冷热 | 1→2 | 0 | -0.013588 | 0.779698 | 0.015519 | 0.002736 | 0.022952 |
| 27 | 600997.SH | 开滦股份 | 1→2 | 0 | -0.013774 | 0.80045 | 0.012097 | 0.002368 | 0.021088 |
| 28 | 600643.SH | 爱建集团 | 1→2 | 0 | -0.01407 | 0.824145 | 0.009889 | 0.001936 | 0.020284 |
| 29 | 600118.SH | 中国卫星 | 1→2 | 0 | -0.014762 | 0.822521 | 0.012128 | 0.001811 | 0.022927 |
| 30 | 605179.SH | 一鸣食品 | 1→2 | 0 | -0.015487 | 0.729055 | 0.009543 | 0.0014 | 0.021044 |
| 31 | 002086.SZ | 东方海洋 | 1→2 | 0 | -0.02505 | 0.565798 | 0.002449 | 0.001892 | 0.024544 |
| 32 | 600576.SH | 祥源文旅 | 1→2 | 0 | -0.026682 | 0.551781 | -0.00021 | 0.001712 | 0.024854 |
| 33 | 603536.SH | 惠发食品 | 1→2 | 0 | -0.028039 | 0.58088 | 0.001557 | 0.001299 | 0.027644 |
| 34 | 601188.SH | 龙江交通 | 1→2 | 0 | -0.028995 | 0.623794 | 0.003754 | 0.001661 | 0.029676 |
| 35 | 600715.SH | 文投控股 | 3→4 | 0 | -0.034386 | 0.664276 | -0.000185 | 0.002249 | 0.032013 |
| 36 | 001225.SZ | 和泰机电 | 1→2 | 0 | -0.035286 | 0.672617 | 0.00084 | 0.00185 | 0.034 |
| 37 | 002277.SZ | 友阿股份 | 1→2 | 0 | -0.036417 | 0.564602 | -0.004273 | 0.001551 | 0.032454 |
| 38 | 002708.SZ | 光洋股份 | 4→5 | 0 | -0.04451 | 0.615099 | -0.005879 | 0.002595 | 0.038299 |

