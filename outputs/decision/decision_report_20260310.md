# Decision Report (20260310)

- signal_date: **20260309**
- exec_date: **20260310**
- requested_trade_date: **auto**
- regime: **RISK_ON**
- risk_budget: **1**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **42**
- features_base_loaded: **True**
- features_base_rows: **5481**
- features_limit_loaded: **True**
- features_limit_rows: **5481**
- truth_close_loaded: **True**
- truth_close_rows: **5481**
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

- candidates_snapshot: `data/decision/decision_candidates_20260309.csv`
- execution_table: `data/decision/decision_execution_20260310.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260310.csv`

## TopN Targets

| rank | ts_code | name | weight | EV | P_fill | E_ret |
|---:|---|---|---:|---:|---:|---:|
| 1 | 002445.SZ | 中南文化 | 0.1 | 0.06482 | 0.933658 | 0.070282 |
| 2 | 601789.SH | 宁波建工 | 0.1 | 0.056291 | 0.98 | 0.058256 |
| 3 | 600227.SH | 赤天化 | 0.1 | 0.036281 | 0.98 | 0.037837 |
| 4 | 605268.SH | 王力安防 | 0.1 | 0.034893 | 0.98 | 0.036421 |
| 5 | 600135.SH | 乐凯胶片 | 0.1 | 0.029006 | 0.98 | 0.030414 |
| 6 | 301179.SZ | 泽宇智能 | 0.1 | 0.019948 | 0.98 | 0.021171 |
| 7 | 688158.SH | 优刻得-W | 0.1 | 0.018828 | 0.98 | 0.020028 |
| 8 | 600851.SH | 海欣股份 | 0.1 | 0.015231 | 0.98 | 0.016358 |
| 9 | 002730.SZ | 电光科技 | 0.1 | 0.014243 | 0.98 | 0.01535 |
| 10 | 600821.SH | 金开新能 | 0.1 | 0.014039 | 0.89753 | 0.016533 |
