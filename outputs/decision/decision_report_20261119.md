# Decision Report (20261119)

- signal_date: **20261118**
- exec_date: **20261119**
- exit_date: **20261120**
- requested_trade_date: **20261118**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.2086,volatility_mean=0.0465**
- guardrail_reason: **CAUTION:tail_risk_mean=0.2086**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **34**
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
- available_rows: **34** / **34**
- hard_risk_rows: **1**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.003713**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **34**
- selected_rows: **2**
- positive_ev_rows: **2**
- positive_ev_base_rows: **2**
- positive_e_ret_rows: **34**
- high_pfill_rows: **0**
- low_risk_rows: **2**
- max_EV: **0.001385**
- max_EV_base: **0.001385**
- max_E_ret: **0.018566**
- mean_cost: **0.001692**
- mean_risk_penalty: **0.015974**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261118.csv`
- execution_table: `data/decision/decision_execution_20261119.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261119.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261118.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 002771.SZ | 真视通 | 2→3 | 0.07 | 0.001385 | 0.733721 | 0.017017 | 0.001103 | 0.009998 |
| 2 | 002272.SZ | 川润股份 | 1→2 | 0.07 | 0.000339 | 0.722993 | 0.014139 | 0.001257 | 0.008626 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 3 | 600410.SH | 华胜天成 | 2→3 | 0 | -0.000476 | 0.707754 | 0.01665 | 0.001551 | 0.010709 |
| 4 | 000889.SZ | 中嘉博创 | 1→2 | 0 | -0.001342 | 0.706768 | 0.014052 | 0.001254 | 0.010019 |
| 5 | 000815.SZ | 美利云 | 1→2 | 0 | -0.001768 | 0.76237 | 0.017399 | 0.001308 | 0.013724 |
| 6 | 603030.SH | 全筑股份 | 2→3 | 0 | -0.001781 | 0.731393 | 0.018316 | 0.0023 | 0.012877 |
| 7 | 000707.SZ | 双环科技 | 2→3 | 0 | -0.001985 | 0.798939 | 0.013391 | 0.001463 | 0.01122 |
| 8 | 002194.SZ | 武汉凡谷 | 1→2 | 0 | -0.002769 | 0.722772 | 0.014132 | 0.00178 | 0.011204 |
| 9 | 000158.SZ | 常山北明 | 1→2 | 0 | -0.002849 | 0.687148 | 0.016369 | 0.001212 | 0.012884 |
| 10 | 600156.SH | 华升股份 | 1→2 | 0 | -0.003301 | 0.733296 | 0.011758 | 0.001327 | 0.010597 |
| 11 | 002685.SZ | 华东重机 | 4→5 | 0 | -0.004341 | 0.618355 | 0.017284 | 0.00197 | 0.013059 |
| 12 | 603324.SH | 盛剑科技 | 2→3 | 0 | -0.004449 | 0.746117 | 0.014128 | 0.002098 | 0.012892 |
| 13 | 603887.SH | 城地香江 | 1→2 | 0 | -0.004579 | 0.748799 | 0.015772 | 0.001222 | 0.015167 |
| 14 | 000980.SZ | 众泰汽车 | 1→2 | 0 | -0.004827 | 0.716996 | 0.012548 | 0.001422 | 0.012401 |
| 15 | 600186.SH | 莲花控股 | 1→2 | 0 | -0.004871 | 0.762995 | 0.016309 | 0.001324 | 0.015992 |
| 16 | 603985.SH | 恒润股份 | 1→2 | 0 | -0.005769 | 0.731167 | 0.012367 | 0.001811 | 0.013 |
| 17 | 603220.SH | 中贝通信 | 1→2 | 0 | -0.006105 | 0.74957 | 0.014705 | 0.001282 | 0.015844 |
| 18 | 002955.SZ | 鸿合科技 | 1→2 | 0 | -0.006143 | 0.730321 | 0.014133 | 0.001811 | 0.014654 |
| 19 | 002929.SZ | 润建股份 | 1→2 | 0 | -0.006452 | 0.751566 | 0.016961 | 0.001505 | 0.017694 |
| 20 | 002403.SZ | 爱仕达 | 1→2 | 0 | -0.006466 | 0.700618 | 0.013713 | 0.001873 | 0.0142 |
| 21 | 002197.SZ | 证通电子 | 1→2 | 0 | -0.00728 | 0.788252 | 0.01289 | 0.001368 | 0.016072 |
| 22 | 603629.SH | 利通电子 | 1→2 | 0 | -0.007501 | 0.706618 | 0.015469 | 0.001291 | 0.017141 |
| 23 | 600839.SH | 四川长虹 | 1→2 | 0 | -0.007827 | 0.676496 | 0.015469 | 0.001957 | 0.016335 |
| 24 | 002916.SZ | 深南电路 | 1→2 | 0 | -0.007929 | 0.656535 | 0.015745 | 0.001989 | 0.016277 |
| 25 | 000711.SZ | 京蓝科技 | 1→2 | 0 | -0.00812 | 0.715322 | 0.01227 | 0.001899 | 0.014998 |
| 26 | 600405.SH | 动力源 | 1→2 | 0 | -0.011174 | 0.727746 | 0.013209 | 0.00202 | 0.018767 |
| 27 | 002843.SZ | 泰嘉股份 | 2→3 | 0 | -0.011314 | 0.783338 | 0.012944 | 0.002098 | 0.019356 |
| 28 | 002042.SZ | 华孚时尚 | 1→2 | 0 | -0.012628 | 0.785764 | 0.010482 | 0.001628 | 0.019237 |
| 29 | 000410.SZ | 沈阳机床 | 1→2 | 0 | -0.013557 | 0.80352 | 0.011703 | 0.001549 | 0.021411 |
| 30 | 002148.SZ | 北纬科技 | 1→2 | 0 | -0.014375 | 0.809014 | 0.018566 | 0.003181 | 0.026214 |
| 31 | 603657.SH | 春光科技 | 1→2 | 0 | -0.014957 | 0.747282 | 0.013282 | 0.00224 | 0.022642 |
| 32 | 002463.SZ | 沪电股份 | 1→2 | 0 | -0.016839 | 0.751121 | 0.010139 | 0.00155 | 0.022904 |
| 33 | 600162.SH | 香江控股 | 1→2 | 0 | -0.018334 | 0.762026 | 0.009634 | 0.001675 | 0.024 |
| 34 | 600336.SH | 澳柯玛 | 1→2 | 0 | -0.032691 | 0.576626 | 0.000916 | 0.002218 | 0.031001 |

