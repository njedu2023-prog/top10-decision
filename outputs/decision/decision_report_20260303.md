# Decision Report (20260303)

- signal_date: **20260302**
- exec_date: **20260303**
- regime: **RISK_ON**
- risk_budget: **1**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **88**
- features_base_loaded: **True**
- features_base_rows: **5470**
- features_limit_loaded: **True**
- features_limit_rows: **5470**
- truth_close_loaded: **True**
- truth_close_rows: **5470**
- meta_loaded: **True**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20260302.csv`
- execution_table: `data/decision/decision_execution_20260303.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260303.csv`

## TopN Targets

| rank | ts_code | name | weight | EV |
|---:|---|---|---:|---:|
| 1 | 000960.SZ | 锡业股份 | 0.1 | 0.011665 |
| 2 | 002842.SZ | 翔鹭钨业 | 0.1 | 0.009985 |
| 3 | 300164.SZ | 通源石油 | 0.1 | 0.009425 |
| 4 | 002155.SZ | 湖南黄金 | 0.1 | 0.008883 |
| 5 | 300139.SZ | 晓程科技 | 0.1 | 0.008668 |
| 6 | 600871.SH | 石化油服 | 0.1 | 0.007726 |
| 7 | 603619.SH | 中曼石油 | 0.1 | 0.007639 |
| 8 | 001337.SZ | 四川黄金 | 0.1 | 0.007487 |
| 9 | 600259.SH | 中稀有色 | 0.1 | 0.007267 |
| 10 | 600367.SH | 红星发展 | 0.1 | 0.006214 |
