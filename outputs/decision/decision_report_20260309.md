# Decision Report (20260309)

- signal_date: **20260306**
- exec_date: **20260309**
- regime: **RISK_ON**
- risk_budget: **1**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **74**
- features_base_loaded: **True**
- features_base_rows: **5477**
- features_limit_loaded: **True**
- features_limit_rows: **5477**
- truth_close_loaded: **True**
- truth_close_rows: **5477**
- meta_loaded: **True**

## Engine Status

- p_fill_pred_src: **rule**
- p_fill_model_loaded: **True**
- p_fill_model_kind: **lgbm**
- p_fill_degrade_reason: **model_predict_failed:ValueError:train and valid dataset categorical_feature do not match.**
- eret_pred_src: **rule**
- eret_model_loaded: **True**
- eret_model_kind: **lgbm**
- eret_degrade_reason: **model_predict_failed:ValueError**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20260306.csv`
- execution_table: `data/decision/decision_execution_20260309.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260309.csv`

## TopN Targets

| rank | ts_code | name | weight | EV | P_fill | E_ret |
|---:|---|---|---:|---:|---:|---:|
| 1 | 002498.SZ | 汉缆股份 | 0.1 | 0.019569 | 0.605761 | 0.033625 |
| 2 | 301205.SZ | 联特科技 | 0.1 | 0.00932 | 0.380295 | 0.026611 |
| 3 | 603285.SH | 键邦股份 | 0.1 | 0.008576 | 0.298503 | 0.031409 |
| 4 | 002165.SZ | 红 宝 丽 | 0.1 | 0.00833 | 0.283876 | 0.032163 |
| 5 | 605399.SH | 晨光新材 | 0.1 | 0.008307 | 0.286421 | 0.031797 |
| 6 | 600397.SH | 江钨装备 | 0.1 | 0.007985 | 0.688231 | 0.012765 |
| 7 | 601868.SH | 中国能建 | 0.1 | 0.007844 | 0.426679 | 0.020258 |
| 8 | 003023.SZ | 彩虹集团 | 0.1 | 0.007236 | 0.519833 | 0.015458 |
| 9 | 600645.SH | 中源协和 | 0.1 | 0.006725 | 0.687519 | 0.010945 |
| 10 | 600482.SH | 中国动力 | 0.1 | 0.006599 | 0.396612 | 0.018656 |
