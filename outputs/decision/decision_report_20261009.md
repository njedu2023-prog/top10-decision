# Decision Report (20261009)

- signal_date: **20261008**
- exec_date: **20261009**
- exit_date: **20261012**
- requested_trade_date: **20261008**
- regime: **RISK_ON**
- risk_budget: **1**
- regime_reason: **tail_risk_mean=0.1539**
- guardrail_reason: **CAUTION:open_board_max=13.0,tail_risk_mean=0.1539**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **38**
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
- available_rows: **38** / **38**
- hard_risk_rows: **5**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.003997**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **38**
- selected_rows: **2**
- positive_ev_rows: **2**
- positive_ev_base_rows: **2**
- positive_e_ret_rows: **34**
- high_pfill_rows: **0**
- low_risk_rows: **4**
- max_EV: **0.000243**
- max_EV_base: **0.000243**
- max_E_ret: **0.020275**
- mean_cost: **0.001719**
- mean_risk_penalty: **0.016727**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261008.csv`
- execution_table: `data/decision/decision_execution_20261009.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261009.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261008.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 600825.SH | 新华传媒 | 8→9 | 0.1 | 0.000243 | 0.612104 | 0.019325 | 0.002219 | 0.009367 |
| 2 | 605303.SH | 园林股份 | 3→4 | 0.1 | 0.000227 | 0.712484 | 0.020275 | 0.00227 | 0.011949 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 3 | 603928.SH | 兴业股份 | 2→3 | 0 | -0.000003 | 0.717524 | 0.012403 | 0.001115 | 0.007787 |
| 4 | 002242.SZ | 九阳股份 | 4→5 | 0 | -0.000565 | 0.692126 | 0.01891 | 0.002083 | 0.01157 |
| 5 | 002490.SZ | 山东墨龙 | 1→2 | 0 | -0.002236 | 0.734185 | 0.013823 | 0.001257 | 0.011127 |
| 6 | 600310.SH | 广西能源 | 1→2 | 0 | -0.002269 | 0.725434 | 0.012283 | 0.001282 | 0.009898 |
| 7 | 002805.SZ | 丰元股份 | 1→2 | 0 | -0.002335 | 0.76289 | 0.017935 | 0.001248 | 0.014769 |
| 8 | 600241.SH | 时代万恒 | 4→5 | 0 | -0.002344 | 0.694626 | 0.01278 | 0.001201 | 0.01002 |
| 9 | 600408.SH | 安泰集团 | 1→2 | 0 | -0.003192 | 0.734476 | 0.009944 | 0.001255 | 0.00924 |
| 10 | 002702.SZ | 海欣食品 | 1→2 | 0 | -0.003744 | 0.722974 | 0.012233 | 0.001324 | 0.011264 |
| 11 | 600722.SH | 金牛化工 | 1→2 | 0 | -0.00466 | 0.767628 | 0.014812 | 0.001364 | 0.014666 |
| 12 | 002774.SZ | 快意电梯 | 1→2 | 0 | -0.005064 | 0.712269 | 0.014366 | 0.001915 | 0.013381 |
| 13 | 603137.SH | 恒尚节能 | 1→2 | 0 | -0.005291 | 0.696238 | 0.01331 | 0.001701 | 0.012856 |
| 14 | 603863.SH | 松炀资源 | 1→2 | 0 | -0.005934 | 0.701349 | 0.01012 | 0.001233 | 0.011799 |
| 15 | 600026.SH | 中远海能 | 1→2 | 0 | -0.006051 | 0.668294 | 0.014726 | 0.002148 | 0.013744 |
| 16 | 002058.SZ | 紫竹高科 | 3→4 | 0 | -0.006264 | 0.737693 | 0.0174 | 0.001926 | 0.017174 |
| 17 | 600812.SH | 华北制药 | 1→2 | 0 | -0.006871 | 0.806305 | 0.013328 | 0.001436 | 0.016182 |
| 18 | 001300.SZ | 三柏硕 | 1→2 | 0 | -0.007358 | 0.665733 | 0.013065 | 0.001772 | 0.014284 |
| 19 | 002869.SZ | 金溢科技 | 1→2 | 0 | -0.007672 | 0.757121 | 0.011768 | 0.001304 | 0.015278 |
| 20 | 605298.SH | 必得科技 | 1→2 | 0 | -0.008337 | 0.671297 | 0.015105 | 0.00223 | 0.016247 |
| 21 | 002866.SZ | 传艺科技 | 3→4 | 0 | -0.008356 | 0.773879 | 0.011594 | 0.001748 | 0.015579 |
| 22 | 002565.SZ | 顺灏股份 | 1→2 | 0 | -0.008638 | 0.826986 | 0.012314 | 0.001743 | 0.017078 |
| 23 | 002733.SZ | 雄韬股份 | 1→2 | 0 | -0.008729 | 0.795311 | 0.016299 | 0.001585 | 0.020107 |
| 24 | 002687.SZ | 乔治白 | 1→2 | 0 | -0.009059 | 0.770032 | 0.01316 | 0.001321 | 0.017872 |
| 25 | 605378.SH | 野马电池 | 1→2 | 0 | -0.009433 | 0.818111 | 0.014274 | 0.001644 | 0.019466 |
| 26 | 000869.SZ | 张  裕Ａ | 1→2 | 0 | -0.010184 | 0.706224 | 0.008836 | 0.002046 | 0.014377 |
| 27 | 603788.SH | 宁波高发 | 1→2 | 0 | -0.010382 | 0.801327 | 0.011614 | 0.001688 | 0.018 |
| 28 | 600617.SH | 国新能源 | 1→2 | 0 | -0.011565 | 0.749574 | 0.011686 | 0.002078 | 0.018247 |
| 29 | 002951.SZ | 金时科技 | 1→2 | 0 | -0.011888 | 0.741141 | 0.013162 | 0.002191 | 0.019452 |
| 30 | 601975.SH | 招商南油 | 1→2 | 0 | -0.012466 | 0.795355 | 0.011688 | 0.001858 | 0.019905 |
| 31 | 600663.SH | 陆家嘴 | 2→3 | 0 | -0.013235 | 0.803762 | 0.011616 | 0.00251 | 0.020061 |
| 32 | 600400.SH | 红豆股份 | 1→2 | 0 | -0.015774 | 0.813789 | 0.008955 | 0.001912 | 0.02115 |
| 33 | 603906.SH | 龙蟠科技 | 2→3 | 0 | -0.016099 | 0.804281 | 0.009846 | 0.001941 | 0.022077 |
| 34 | 001336.SZ | 楚环科技 | 1→2 | 0 | -0.021583 | 0.564617 | -0.000585 | 0.001153 | 0.0201 |
| 35 | 601956.SH | 东贝集团 | 1→2 | 0 | -0.027229 | 0.547796 | 0.000428 | 0.002008 | 0.025455 |
| 36 | 600513.SH | 联环药业 | 1→2 | 0 | -0.028934 | 0.613797 | -0.001607 | 0.00162 | 0.026328 |
| 37 | 603073.SH | 彩蝶实业 | 1→2 | 0 | -0.037293 | 0.602942 | -0.004528 | 0.001891 | 0.032672 |
| 38 | 603200.SH | 上海洗霸 | 3→4 | 0 | -0.03953 | 0.638225 | -0.003614 | 0.002116 | 0.035108 |

