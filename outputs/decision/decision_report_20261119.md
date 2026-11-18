# Decision Report (20261119)

- signal_date: **20261118**
- exec_date: **20261119**
- exit_date: **20261120**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.2086,volatility_mean=0.0465,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.2086**
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
- available_rows: **0** / **34**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **34**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **34**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.010686**
- max_EV_base: **-0.010686**
- max_E_ret: **0.014194**
- mean_cost: **0.001692**
- mean_risk_penalty: **0.020947**
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

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 002272.SZ | 川润股份 | 1→2 | 0 | -0.010686 | 0.634177 | 0.009086 | 0.001257 | 0.015191 |
| 2 | 000889.SZ | 中嘉博创 | 1→2 | 0 | -0.010738 | 0.612967 | 0.009459 | 0.001254 | 0.015282 |
| 3 | 600156.SH | 华升股份 | 1→2 | 0 | -0.011321 | 0.63767 | 0.0083 | 0.001327 | 0.015287 |
| 4 | 002771.SZ | 真视通 | 2→3 | 0 | -0.011549 | 0.621703 | 0.010412 | 0.001103 | 0.016919 |
| 5 | 000158.SZ | 常山北明 | 1→2 | 0 | -0.01356 | 0.576735 | 0.011227 | 0.001212 | 0.018823 |
| 6 | 600410.SH | 华胜天成 | 2→3 | 0 | -0.013603 | 0.591604 | 0.010321 | 0.001551 | 0.018157 |
| 7 | 000980.SZ | 众泰汽车 | 1→2 | 0 | -0.014129 | 0.61655 | 0.008452 | 0.001422 | 0.017918 |
| 8 | 000707.SZ | 双环科技 | 2→3 | 0 | -0.014911 | 0.691049 | 0.008508 | 0.001463 | 0.019328 |
| 9 | 603030.SH | 全筑股份 | 2→3 | 0 | -0.014914 | 0.607822 | 0.01332 | 0.0023 | 0.02071 |
| 10 | 002403.SZ | 爱仕达 | 1→2 | 0 | -0.014948 | 0.598857 | 0.011703 | 0.001873 | 0.020083 |
| 11 | 600405.SH | 动力源 | 1→2 | 0 | -0.015057 | 0.643617 | 0.012139 | 0.00202 | 0.02085 |
| 12 | 000815.SZ | 美利云 | 1→2 | 0 | -0.015467 | 0.651796 | 0.01087 | 0.001308 | 0.021243 |
| 13 | 002194.SZ | 武汉凡谷 | 1→2 | 0 | -0.015653 | 0.602849 | 0.008503 | 0.00178 | 0.019 |
| 14 | 603887.SH | 城地香江 | 1→2 | 0 | -0.015767 | 0.65672 | 0.01023 | 0.001222 | 0.021263 |
| 15 | 603985.SH | 恒润股份 | 1→2 | 0 | -0.016167 | 0.621356 | 0.007853 | 0.001811 | 0.019235 |
| 16 | 002042.SZ | 华孚时尚 | 1→2 | 0 | -0.016227 | 0.716741 | 0.008114 | 0.001628 | 0.020415 |
| 17 | 600336.SH | 澳柯玛 | 1→2 | 0 | -0.01692 | 0.650597 | 0.009683 | 0.002218 | 0.021001 |
| 18 | 603324.SH | 盛剑科技 | 2→3 | 0 | -0.017205 | 0.617692 | 0.009095 | 0.002098 | 0.020724 |
| 19 | 603220.SH | 中贝通信 | 1→2 | 0 | -0.017363 | 0.654515 | 0.009378 | 0.001282 | 0.022218 |
| 20 | 002685.SZ | 华东重机 | 4→5 | 0 | -0.017376 | 0.456918 | 0.012005 | 0.00197 | 0.020892 |
| 21 | 600186.SH | 莲花控股 | 1→2 | 0 | -0.017463 | 0.657956 | 0.010857 | 0.001324 | 0.023282 |
| 22 | 002955.SZ | 鸿合科技 | 1→2 | 0 | -0.017538 | 0.619761 | 0.008667 | 0.001811 | 0.021098 |
| 23 | 002197.SZ | 证通电子 | 1→2 | 0 | -0.01768 | 0.689433 | 0.008217 | 0.001368 | 0.021976 |
| 24 | 000410.SZ | 沈阳机床 | 1→2 | 0 | -0.0178 | 0.708629 | 0.008815 | 0.001549 | 0.022497 |
| 25 | 002929.SZ | 润建股份 | 1→2 | 0 | -0.017835 | 0.664339 | 0.010491 | 0.001505 | 0.0233 |
| 26 | 000711.SZ | 京蓝科技 | 1→2 | 0 | -0.018153 | 0.610478 | 0.007427 | 0.001899 | 0.020788 |
| 27 | 603629.SH | 利通电子 | 1→2 | 0 | -0.018812 | 0.611287 | 0.009455 | 0.001291 | 0.0233 |
| 28 | 002916.SZ | 深南电路 | 1→2 | 0 | -0.019046 | 0.568167 | 0.010669 | 0.001989 | 0.023118 |
| 29 | 603657.SH | 春光科技 | 1→2 | 0 | -0.019066 | 0.648494 | 0.011063 | 0.00224 | 0.024 |
| 30 | 002463.SZ | 沪电股份 | 1→2 | 0 | -0.01908 | 0.711282 | 0.009097 | 0.00155 | 0.024 |
| 31 | 600839.SH | 四川长虹 | 1→2 | 0 | -0.019697 | 0.554881 | 0.00923 | 0.001957 | 0.022862 |
| 32 | 002148.SZ | 北纬科技 | 1→2 | 0 | -0.019707 | 0.758946 | 0.014194 | 0.003181 | 0.027299 |
| 33 | 600162.SH | 香江控股 | 1→2 | 0 | -0.020257 | 0.713511 | 0.007593 | 0.001675 | 0.024 |
| 34 | 002843.SZ | 泰嘉股份 | 2→3 | 0 | -0.022304 | 0.693323 | 0.008543 | 0.002098 | 0.026129 |

