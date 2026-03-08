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

- p_fill_pred_src: **model:lgbm**
- p_fill_model_loaded: **True**
- p_fill_model_kind: **lgbm**
- p_fill_degrade_reason: **none**
- eret_pred_src: **model:lgbm**
- eret_model_loaded: **True**
- eret_model_kind: **lgbm**
- eret_degrade_reason: **none**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20260306.csv`
- execution_table: `data/decision/decision_execution_20260309.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260309.csv`

## TopN Targets

| rank | ts_code | name | weight | EV | P_fill | E_ret |
|---:|---|---|---:|---:|---:|---:|
| 1 | 000533.SZ | 顺钠股份 | 0.1 | 0.001383 | 0.98 | 0.002228 |
| 2 | 601567.SH | 三星医疗 | 0.1 | 0.001383 | 0.98 | 0.002228 |
| 3 | 002498.SZ | 汉缆股份 | 0.1 | 0.001383 | 0.98 | 0.002228 |
| 4 | 000818.SZ | 航锦科技 | 0.1 | 0.001383 | 0.98 | 0.002228 |
| 5 | 002165.SZ | 红 宝 丽 | 0.1 | 0.001383 | 0.98 | 0.002228 |
| 6 | 605399.SH | 晨光新材 | 0.1 | 0.001383 | 0.98 | 0.002228 |
| 7 | 301205.SZ | 联特科技 | 0.1 | 0.001383 | 0.98 | 0.002228 |
| 8 | 002261.SZ | 拓维信息 | 0.1 | 0.001383 | 0.98 | 0.002228 |
| 9 | 600590.SH | 泰豪科技 | 0.1 | 0.001383 | 0.98 | 0.002228 |
| 10 | 002015.SZ | 协鑫能科 | 0.1 | 0.001383 | 0.98 | 0.002228 |
