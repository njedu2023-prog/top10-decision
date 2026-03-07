# Decision Report (20260306)

- signal_date: **20260305**
- exec_date: **20260306**
- regime: **RISK_ON**
- risk_budget: **1**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **69**
- features_base_loaded: **True**
- features_base_rows: **5476**
- features_limit_loaded: **True**
- features_limit_rows: **5476**
- truth_close_loaded: **True**
- truth_close_rows: **5476**
- meta_loaded: **True**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20260305.csv`
- execution_table: `data/decision/decision_execution_20260306.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260306.csv`

## TopN Targets

| rank | ts_code | name | weight | EV |
|---:|---|---|---:|---:|
| 1 | 002195.SZ | 岩山科技 | 0.1 | 0.019848 |
| 2 | 300895.SZ | 铜牛信息 | 0.1 | 0.014782 |
| 3 | 000509.SZ | 华塑控股 | 0.1 | 0.009949 |
| 4 | 601179.SH | 中国西电 | 0.1 | 0.008205 |
| 5 | 300303.SZ | 聚飞光电 | 0.1 | 0.008143 |
| 6 | 600875.SH | 东方电气 | 0.1 | 0.007785 |
| 7 | 600545.SH | 卓郎智能 | 0.1 | 0.007468 |
| 8 | 600744.SH | 华银电力 | 0.1 | 0.006984 |
| 9 | 300269.SZ | 联建光电 | 0.1 | 0.006929 |
| 10 | 300323.SZ | 华灿光电 | 0.1 | 0.006704 |
