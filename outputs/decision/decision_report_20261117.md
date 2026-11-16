# Decision Report (20261117)

- signal_date: **20261116**
- exec_date: **20261117**
- exit_date: **20261118**
- requested_trade_date: **20261116**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1859,volatility_mean=0.0459**
- guardrail_reason: **CAUTION:open_board_max=11.0,tail_risk_mean=0.1859**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **30**
- features_base_loaded: **True**
- features_base_rows: **5572**
- features_limit_loaded: **True**
- features_limit_rows: **5572**
- truth_close_loaded: **True**
- truth_close_rows: **5572**
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
- hard_risk_rows: **6**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.004789**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **30**
- selected_rows: **2**
- positive_ev_rows: **2**
- positive_ev_base_rows: **2**
- positive_e_ret_rows: **26**
- high_pfill_rows: **0**
- low_risk_rows: **2**
- max_EV: **0.001543**
- max_EV_base: **0.001543**
- max_E_ret: **0.017241**
- mean_cost: **0.001732**
- mean_risk_penalty: **0.018589**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261116.csv`
- execution_table: `data/decision/decision_execution_20261117.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261117.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261116.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 002548.SZ | 金新农 | 2→3 | 0.07 | 0.001543 | 0.749614 | 0.013558 | 0.001244 | 0.007377 |
| 2 | 002187.SZ | 广百股份 | 3→4 | 0.07 | 0.000919 | 0.694047 | 0.017241 | 0.001165 | 0.009882 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 3 | 002058.SZ | 紫竹高科 | 1→2 | 0 | -0.001588 | 0.730085 | 0.014116 | 0.001178 | 0.010717 |
| 4 | 601929.SH | 吉视传媒 | 1→2 | 0 | -0.002802 | 0.723307 | 0.013658 | 0.001545 | 0.011137 |
| 5 | 002861.SZ | 瀛通通讯 | 1→2 | 0 | -0.003242 | 0.735248 | 0.01255 | 0.001271 | 0.011199 |
| 6 | 600386.SH | 北巴传媒 | 1→2 | 0 | -0.003313 | 0.721786 | 0.012671 | 0.001327 | 0.011132 |
| 7 | 002905.SZ | 金逸影视 | 2→3 | 0 | -0.00346 | 0.715763 | 0.016858 | 0.002106 | 0.013421 |
| 8 | 000533.SZ | 顺钠股份 | 1→2 | 0 | -0.003494 | 0.687695 | 0.012224 | 0.001149 | 0.010752 |
| 9 | 600822.SH | 上海物贸 | 1→2 | 0 | -0.004062 | 0.709648 | 0.013507 | 0.001236 | 0.012411 |
| 10 | 002959.SZ | 小熊电器 | 1→2 | 0 | -0.004885 | 0.715788 | 0.015071 | 0.001735 | 0.013937 |
| 11 | 002685.SZ | 华东重机 | 2→3 | 0 | -0.00514 | 0.777485 | 0.0151 | 0.001563 | 0.015317 |
| 12 | 000521.SZ | 长虹美菱 | 1→2 | 0 | -0.006389 | 0.715583 | 0.011384 | 0.001864 | 0.012671 |
| 13 | 002419.SZ | 天虹股份 | 1→2 | 0 | -0.006413 | 0.764862 | 0.015489 | 0.00141 | 0.01685 |
| 14 | 000628.SZ | 高新发展 | 1→2 | 0 | -0.007148 | 0.748361 | 0.013786 | 0.001261 | 0.016204 |
| 15 | 600936.SH | 北投科技 | 1→2 | 0 | -0.007884 | 0.697032 | 0.013398 | 0.001984 | 0.015238 |
| 16 | 001318.SZ | 阳光乳业 | 1→2 | 0 | -0.008143 | 0.776484 | 0.011952 | 0.00148 | 0.015944 |
| 17 | 605378.SH | 野马电池 | 1→2 | 0 | -0.008233 | 0.695856 | 0.013316 | 0.001949 | 0.015551 |
| 18 | 002748.SZ | 世龙实业 | 2→3 | 0 | -0.009496 | 0.785444 | 0.012684 | 0.001524 | 0.017934 |
| 19 | 002356.SZ | 鹏飞绿能 | 1→2 | 0 | -0.010708 | 0.7861 | 0.011861 | 0.001718 | 0.018313 |
| 20 | 002425.SZ | 凯撒文化 | 2→3 | 0 | -0.010877 | 0.769805 | 0.012039 | 0.001834 | 0.01831 |
| 21 | 603983.SH | 丸美生物 | 1→2 | 0 | -0.01223 | 0.756843 | 0.014263 | 0.002232 | 0.020793 |
| 22 | 002035.SZ | 华帝股份 | 1→2 | 0 | -0.012976 | 0.78361 | 0.010159 | 0.001593 | 0.019344 |
| 23 | 603118.SH | 共进股份 | 1→2 | 0 | -0.013384 | 0.725958 | 0.011584 | 0.001678 | 0.020116 |
| 24 | 000156.SZ | 华数传媒 | 1→2 | 0 | -0.013589 | 0.804009 | 0.012406 | 0.001536 | 0.022027 |
| 25 | 601188.SH | 龙江交通 | 1→2 | 0 | -0.027799 | 0.643474 | 0.002663 | 0.001744 | 0.027768 |
| 26 | 000819.SZ | 岳阳兴长 | 2→3 | 0 | -0.033975 | 0.655255 | -0.001365 | 0.001978 | 0.031102 |
| 27 | 000595.SZ | 新能股份 | 2→3 | 0 | -0.036108 | 0.610914 | -0.0041 | 0.001905 | 0.031699 |
| 28 | 002205.SZ | 国统股份 | 3→4 | 0 | -0.038216 | 0.664078 | -0.001431 | 0.00222 | 0.035046 |
| 29 | 002148.SZ | 北纬科技 | 4→5 | 0 | -0.04073 | 0.70739 | 0.000203 | 0.003329 | 0.037544 |
| 30 | 600715.SH | 文投控股 | 2→3 | 0 | -0.042647 | 0.658991 | -0.002304 | 0.003191 | 0.037938 |

