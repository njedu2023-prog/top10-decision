# Decision Report (20261015)

- signal_date: **20261014**
- exec_date: **20261015**
- exit_date: **20261016**
- requested_trade_date: **20261014**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1708,volatility_mean=0.0491**
- guardrail_reason: **CAUTION:open_board_max=6.0,tail_risk_mean=0.1708**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **34**
- features_base_loaded: **True**
- features_base_rows: **5563**
- features_limit_loaded: **True**
- features_limit_rows: **5563**
- truth_close_loaded: **True**
- truth_close_rows: **5563**
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
- hard_risk_rows: **6**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.004119**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **34**
- selected_rows: **2**
- positive_ev_rows: **2**
- positive_ev_base_rows: **2**
- positive_e_ret_rows: **32**
- high_pfill_rows: **0**
- low_risk_rows: **2**
- max_EV: **0.002263**
- max_EV_base: **0.002263**
- max_E_ret: **0.020996**
- mean_cost: **0.001637**
- mean_risk_penalty: **0.017116**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261014.csv`
- execution_table: `data/decision/decision_execution_20261015.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261015.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261014.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 002813.SZ | 路畅科技 | 2→3 | 0.07 | 0.002263 | 0.747258 | 0.013925 | 0.00113 | 0.007012 |
| 2 | 002343.SZ | 慈文传媒 | 4→5 | 0.07 | 0.001471 | 0.705105 | 0.015607 | 0.001238 | 0.008295 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 3 | 605189.SH | 富春染织 | 1→2 | 0 | -0.000702 | 0.695805 | 0.016808 | 0.001924 | 0.010472 |
| 4 | 000601.SZ | 韶能股份 | 1→2 | 0 | -0.002614 | 0.777572 | 0.014277 | 0.001471 | 0.012245 |
| 5 | 002552.SZ | 宝鼎科技 | 1→2 | 0 | -0.003123 | 0.683346 | 0.020996 | 0.002104 | 0.015367 |
| 6 | 002029.SZ | 七 匹 狼 | 1→2 | 0 | -0.003238 | 0.710117 | 0.011865 | 0.001131 | 0.010533 |
| 7 | 600391.SH | 航发科技 | 2→3 | 0 | -0.003344 | 0.684102 | 0.014565 | 0.001177 | 0.012131 |
| 8 | 603258.SH | 电魂网络 | 1→2 | 0 | -0.003646 | 0.705948 | 0.012658 | 0.001218 | 0.011364 |
| 9 | 000530.SZ | 冰山冷热 | 2→3 | 0 | -0.003831 | 0.66149 | 0.012515 | 0.001848 | 0.010262 |
| 10 | 000573.SZ | 粤宏远Ａ | 1→2 | 0 | -0.003864 | 0.749868 | 0.013976 | 0.001322 | 0.013022 |
| 11 | 002917.SZ | 金奥博 | 2→3 | 0 | -0.004208 | 0.803825 | 0.013003 | 0.001374 | 0.013286 |
| 12 | 603992.SH | 松霖科技 | 1→2 | 0 | -0.004277 | 0.711155 | 0.011269 | 0.001226 | 0.011065 |
| 13 | 601188.SH | 龙江交通 | 2→3 | 0 | -0.004669 | 0.806717 | 0.012992 | 0.001678 | 0.013472 |
| 14 | 603656.SH | 泰禾智能 | 1→2 | 0 | -0.005057 | 0.755188 | 0.012274 | 0.001311 | 0.013015 |
| 15 | 600202.SH | 哈空调 | 2→3 | 0 | -0.005326 | 0.778659 | 0.01239 | 0.001465 | 0.013509 |
| 16 | 600252.SH | 中恒集团 | 1→2 | 0 | -0.005385 | 0.750812 | 0.013373 | 0.001484 | 0.013942 |
| 17 | 000668.SZ | 荣丰控股 | 2→3 | 0 | -0.005716 | 0.740092 | 0.015862 | 0.002088 | 0.015367 |
| 18 | 002696.SZ | 百洋股份 | 4→5 | 0 | -0.005888 | 0.710592 | 0.016367 | 0.002151 | 0.015367 |
| 19 | 001299.SZ | 美能能源 | 1→2 | 0 | -0.005919 | 0.724798 | 0.010555 | 0.001164 | 0.012405 |
| 20 | 002860.SZ | 星帅尔 | 2→3 | 0 | -0.006145 | 0.637904 | 0.020196 | 0.00219 | 0.016838 |
| 21 | 000823.SZ | 超声电子 | 2→3 | 0 | -0.007139 | 0.719839 | 0.016897 | 0.001322 | 0.01798 |
| 22 | 601858.SH | 中国科传 | 1→2 | 0 | -0.007915 | 0.756827 | 0.014736 | 0.00146 | 0.017607 |
| 23 | 603598.SH | 引力传媒 | 1→2 | 0 | -0.008137 | 0.782465 | 0.012533 | 0.001731 | 0.016213 |
| 24 | 603533.SH | 掌阅科技 | 1→2 | 0 | -0.008183 | 0.745371 | 0.013382 | 0.001475 | 0.016683 |
| 25 | 603096.SH | 新经典 | 4→5 | 0 | -0.008873 | 0.819034 | 0.015327 | 0.002061 | 0.019365 |
| 26 | 001318.SZ | 阳光乳业 | 2→3 | 0 | -0.008907 | 0.838737 | 0.013537 | 0.001798 | 0.018462 |
| 27 | 002469.SZ | 三维化学 | 1→2 | 0 | -0.011585 | 0.737937 | 0.011216 | 0.001825 | 0.018036 |
| 28 | 600121.SH | 郑州煤电 | 1→2 | 0 | -0.014959 | 0.833823 | 0.012808 | 0.002062 | 0.023576 |
| 29 | 600847.SH | 万里股份 | 2→3 | 0 | -0.021677 | 0.564479 | 0.00139 | 0.001123 | 0.021339 |
| 30 | 600097.SH | 开创国际 | 1→2 | 0 | -0.028623 | 0.623931 | 0.001619 | 0.00142 | 0.028213 |
| 31 | 603968.SH | 醋化股份 | 3→4 | 0 | -0.032872 | 0.694976 | 0.001499 | 0.002018 | 0.031896 |
| 32 | 601811.SH | 新华文轩 | 1→2 | 0 | -0.036332 | 0.624473 | -0.002494 | 0.001862 | 0.032913 |
| 33 | 001216.SZ | 华瓷股份 | 1→2 | 0 | -0.037788 | 0.63765 | -0.002726 | 0.002049 | 0.034 |
| 34 | 600408.SH | 安泰集团 | 1→2 | 0 | -0.038417 | 0.664267 | 0.001534 | 0.002754 | 0.036682 |

