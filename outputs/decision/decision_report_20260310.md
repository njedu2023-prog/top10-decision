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
| 1 | 601789.SH | 宁波建工 | 0.1 | 0.058073 | 0.98 | 0.06449 | 0.001231 | 0.003896 |
| 2 | 688229.SH | 博睿数据 | 0.1 | 0.051716 | 0.98 | 0.058412 | 0.001528 | 0.004 |
| 3 | 301179.SZ | 泽宇智能 | 0.1 | 0.048006 | 0.98 | 0.050344 | 0.001331 | 0 |
| 4 | 688158.SH | 优刻得-W | 0.1 | 0.043811 | 0.98 | 0.050654 | 0.00183 | 0.004 |
| 5 | 002445.SZ | 中南文化 | 0.1 | 0.032935 | 0.862621 | 0.049528 | 0.002289 | 0.0075 |
| 6 | 600227.SH | 赤天化 | 0.1 | 0.017152 | 0.98 | 0.032586 | 0.002272 | 0.01251 |
| 7 | 600135.SH | 乐凯胶片 | 0.1 | 0.016103 | 0.98 | 0.021535 | 0.001228 | 0.003773 |
| 8 | 002575.SZ | 群兴玩具 | 0.1 | 0.01269 | 0.98 | 0.014383 | 0.001405 | 0 |
| 9 | 600851.SH | 海欣股份 | 0.1 | 0.011728 | 0.98 | 0.018232 | 0.001711 | 0.004429 |
| 10 | 605268.SH | 王力安防 | 0.1 | 0.008743 | 0.98 | 0.018748 | 0.00213 | 0.0075 |
