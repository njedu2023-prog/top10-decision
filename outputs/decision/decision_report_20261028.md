# Decision Report (20261028)

- signal_date: **20261027**
- exec_date: **20261028**
- exit_date: **20261029**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.2046,volatility_mean=0.0482**
- guardrail_reason: **CAUTION:tail_risk_mean=0.2046**
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
- intraday_execution_penalty_mean: **0.004772**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **38**
- selected_rows: **1**
- positive_ev_rows: **1**
- positive_ev_base_rows: **1**
- positive_e_ret_rows: **32**
- high_pfill_rows: **0**
- low_risk_rows: **3**
- max_EV: **0.001995**
- max_EV_base: **0.001995**
- max_E_ret: **0.021772**
- mean_cost: **0.001614**
- mean_risk_penalty: **0.019228**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261027.csv`
- execution_table: `data/decision/decision_execution_20261028.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261028.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261027.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 600838.SH | 上海九百 | 5→6 | 0.07 | 0.001995 | 0.705167 | 0.016124 | 0.001118 | 0.008257 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 2 | 000058.SZ | 深 赛 格 | 1→2 | 0 | -0.000207 | 0.645686 | 0.021772 | 0.002186 | 0.012079 |
| 3 | 603037.SH | 凯众股份 | 1→2 | 0 | -0.001229 | 0.72904 | 0.013099 | 0.001106 | 0.009673 |
| 4 | 002708.SZ | 光洋股份 | 2→3 | 0 | -0.001599 | 0.704329 | 0.020012 | 0.00172 | 0.013975 |
| 5 | 001400.SZ | 江顺科技 | 1→2 | 0 | -0.002108 | 0.719274 | 0.014037 | 0.001176 | 0.011028 |
| 6 | 002623.SZ | 亚玛顿 | 1→2 | 0 | -0.00239 | 0.692841 | 0.01234 | 0.001145 | 0.009795 |
| 7 | 601086.SH | 国芳集团 | 1→2 | 0 | -0.003779 | 0.69586 | 0.013905 | 0.001341 | 0.012113 |
| 8 | 001219.SZ | 青岛食品 | 2→3 | 0 | -0.003932 | 0.77269 | 0.013575 | 0.001228 | 0.013194 |
| 9 | 002205.SZ | 国统股份 | 1→2 | 0 | -0.004618 | 0.722139 | 0.011796 | 0.001159 | 0.011978 |
| 10 | 603370.SH | 华新精科 | 1→2 | 0 | -0.005542 | 0.777396 | 0.013395 | 0.001385 | 0.01457 |
| 11 | 002093.SZ | 国脉科技 | 2→3 | 0 | -0.005693 | 0.792121 | 0.0153 | 0.001388 | 0.016425 |
| 12 | 002823.SZ | 凯中精密 | 1→2 | 0 | -0.005789 | 0.749864 | 0.013916 | 0.0013 | 0.014924 |
| 13 | 002249.SZ | 大洋电机 | 1→2 | 0 | -0.005859 | 0.739027 | 0.014231 | 0.001296 | 0.01508 |
| 14 | 000888.SZ | 峨眉山Ａ | 1→2 | 0 | -0.007669 | 0.749704 | 0.01044 | 0.001286 | 0.01421 |
| 15 | 000678.SZ | 襄阳轴承 | 4→5 | 0 | -0.007879 | 0.7579 | 0.016714 | 0.001749 | 0.018798 |
| 16 | 600828.SH | 茂业商业 | 1→2 | 0 | -0.008782 | 0.708247 | 0.013907 | 0.002165 | 0.016466 |
| 17 | 002187.SZ | 广百股份 | 1→2 | 0 | -0.009323 | 0.743908 | 0.013421 | 0.001441 | 0.017866 |
| 18 | 601566.SH | 九牧王 | 1→2 | 0 | -0.009391 | 0.748917 | 0.013379 | 0.001322 | 0.018089 |
| 19 | 603376.SH | 大明电子 | 1→2 | 0 | -0.009791 | 0.780016 | 0.01403 | 0.001596 | 0.019138 |
| 20 | 600785.SH | 新华百货 | 1→2 | 0 | -0.009923 | 0.749503 | 0.011408 | 0.001713 | 0.01676 |
| 21 | 600303.SH | 曙光股份 | 1→2 | 0 | -0.010131 | 0.757204 | 0.012437 | 0.001463 | 0.018085 |
| 22 | 001318.SZ | 阳光乳业 | 2→3 | 0 | -0.011493 | 0.848375 | 0.013646 | 0.001844 | 0.021226 |
| 23 | 605333.SH | 沪光股份 | 2→3 | 0 | -0.011684 | 0.768302 | 0.012246 | 0.001624 | 0.019469 |
| 24 | 603813.SH | 原尚股份 | 1→2 | 0 | -0.012173 | 0.718982 | 0.011139 | 0.002017 | 0.018164 |
| 25 | 001331.SZ | 胜通能源 | 2→3 | 0 | -0.012681 | 0.785225 | 0.012753 | 0.001506 | 0.021189 |
| 26 | 603565.SH | 中谷物流 | 1→2 | 0 | -0.013264 | 0.785157 | 0.011263 | 0.002317 | 0.01979 |
| 27 | 603758.SH | 秦安股份 | 1→2 | 0 | -0.013516 | 0.743197 | 0.012494 | 0.001977 | 0.020824 |
| 28 | 002011.SZ | 盾安环境 | 1→2 | 0 | -0.014134 | 0.73524 | 0.013744 | 0.00203 | 0.022208 |
| 29 | 600715.SH | 文投控股 | 1→2 | 0 | -0.015037 | 0.815768 | 0.010659 | 0.001931 | 0.021801 |
| 30 | 600865.SH | 百大集团 | nan | 0 | -0.016954 | 0.834002 | 0.011927 | 0.002158 | 0.024743 |
| 31 | 002820.SZ | 桂发祥 | 2→3 | 0 | -0.020708 | 0.58805 | 0.001001 | 0.001445 | 0.019852 |
| 32 | 603028.SH | 赛福天 | 4→5 | 0 | -0.029979 | 0.538288 | -0.000599 | 0.001301 | 0.028356 |
| 33 | 603528.SH | 多伦科技 | 4→5 | 0 | -0.030454 | 0.518913 | -0.000179 | 0.001348 | 0.029013 |
| 34 | 601956.SH | 东贝集团 | 1→2 | 0 | -0.030526 | 0.582174 | 0.000865 | 0.001355 | 0.029675 |
| 35 | 600826.SH | 兰生股份 | nan | 0 | -0.033515 | 0.659576 | -0.001255 | 0.001725 | 0.030962 |
| 36 | 600626.SH | 申达股份 | 1→2 | 0 | -0.035165 | 0.62429 | -0.003436 | 0.002212 | 0.030808 |
| 37 | 002284.SZ | 亚太股份 | 1→2 | 0 | -0.035657 | 0.591217 | -0.002157 | 0.001622 | 0.03276 |
| 38 | 002724.SZ | 海洋王 | 1→2 | 0 | -0.041086 | 0.681681 | -0.001652 | 0.002642 | 0.037318 |

