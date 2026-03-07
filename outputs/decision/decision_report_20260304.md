# Decision Report (20260304)

- signal_date: **20260303**
- exec_date: **20260304**
- regime: **RISK_ON**
- risk_budget: **1**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **83**
- features_base_loaded: **True**
- features_base_rows: **5472**
- features_limit_loaded: **True**
- features_limit_rows: **5472**
- truth_close_loaded: **True**
- truth_close_rows: **5472**
- meta_loaded: **True**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20260303.csv`
- execution_table: `data/decision/decision_execution_20260304.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260304.csv`

## TopN Targets

| rank | ts_code | name | weight | EV |
|---:|---|---|---:|---:|
| 1 | 600367.SH | 红星发展 | 0.1 | 0.018834 |
| 2 | 600938.SH | 中国海油 | 0.1 | 0.018485 |
| 3 | 300164.SZ | 通源石油 | 0.1 | 0.013746 |
| 4 | 600028.SH | 中国石化 | 0.1 | 0.012885 |
| 5 | 600968.SH | 海油发展 | 0.1 | 0.012672 |
| 6 | 600714.SH | 金瑞矿业 | 0.1 | 0.011332 |
| 7 | 600635.SH | 大众公用 | 0.1 | 0.01065 |
| 8 | 300332.SZ | 天壕能源 | 0.1 | 0.009332 |
| 9 | 600722.SH | 金牛化工 | 0.1 | 0.007402 |
| 10 | 603318.SH | 水发燃气 | 0.1 | 0.006982 |
