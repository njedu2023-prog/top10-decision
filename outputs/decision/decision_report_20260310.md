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

| rank | ts_code | name | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 601789.SH | 宁波建工 | 0.1 | 0.056412 | 0.98 | 0.06449 | 0.001231 | 0.005557 |
| 2 | 688229.SH | 博睿数据 | 0.1 | 0.049016 | 0.98 | 0.058412 | 0.001528 | 0.0067 |
| 3 | 301179.SZ | 泽宇智能 | 0.1 | 0.045306 | 0.98 | 0.050344 | 0.001331 | 0.0027 |
| 4 | 688158.SH | 优刻得-W | 0.1 | 0.041111 | 0.98 | 0.050654 | 0.00183 | 0.0067 |
| 5 | 002445.SZ | 中南文化 | 0.1 | 0.032235 | 0.862621 | 0.049528 | 0.002289 | 0.0082 |
| 6 | 600135.SH | 乐凯胶片 | 0.1 | 0.014409 | 0.98 | 0.021535 | 0.001228 | 0.005467 |
| 7 | 600227.SH | 赤天化 | 0.1 | 0.01398 | 0.98 | 0.032586 | 0.002272 | 0.015682 |
| 8 | 600851.SH | 海欣股份 | 0.1 | 0.010209 | 0.98 | 0.018232 | 0.001711 | 0.005948 |
| 9 | 002575.SZ | 群兴玩具 | 0.1 | 0.00999 | 0.98 | 0.014383 | 0.001405 | 0.0027 |
| 10 | 605268.SH | 王力安防 | 0.1 | 0.008043 | 0.98 | 0.018748 | 0.00213 | 0.0082 |
