# Decision Report (20260403)

- signal_date: **20260402**
- exec_date: **20260403**
- requested_trade_date: **auto**
- regime: **RISK_ON**
- risk_budget: **1**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **28**
- features_base_loaded: **True**
- features_base_rows: **5485**
- features_limit_loaded: **True**
- features_limit_rows: **5485**
- truth_close_loaded: **True**
- truth_close_rows: **5485**
- meta_loaded: **True**

## Engine Status

- p_fill_pred_src: **model:lgbm**
- p_fill_model_loaded: **True**
- p_fill_model_kind: **lgbm**
- p_fill_degrade_reason: **none**
- eret_pred_src: **model:lr**
- eret_model_loaded: **True**
- eret_model_kind: **lr**
- eret_degrade_reason: **none**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20260402.csv`
- execution_table: `data/decision/decision_execution_20260403.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260403.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20260402.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 002263.SZ | 大东南 | 0.1 | 0.02628 | 0.98 | 0.045455 | 0.002325 | 0.0133 |
| 2 | 000720.SZ | 新能泰山 | 0.1 | 0.025843 | 0.967605 | 0.035062 | 0.001776 | 0.006306 |
| 3 | 603222.SH | 济民健康 | 0.1 | 0.023736 | 0.971534 | 0.028384 | 0.001141 | 0.0027 |
| 4 | 000950.SZ | 重药控股 | 0.1 | 0.021586 | 0.944435 | 0.030304 | 0.001228 | 0.005472 |
| 5 | 600339.SH | 中油工程 | 0.1 | 0.021526 | 0.979534 | 0.03867 | 0.002062 | 0.012384 |
| 6 | 002051.SZ | 中工国际 | 0.1 | 0.013419 | 0.98 | 0.017595 | 0.001124 | 0.0027 |
| 7 | 002038.SZ | 双鹭药业 | 0.1 | 0.012669 | 0.98 | 0.019773 | 0.001697 | 0.005012 |
| 8 | 600644.SH | 乐山电力 | 0.1 | 0.008181 | 0.98 | 0.01836 | 0.001511 | 0.0083 |
| 9 | 000788.SZ | 北大医药 | 0.1 | 0.004404 | 0.98 | 0.029801 | 0.002357 | 0.016913 |
| 10 | 002309.SZ | 中利集团 | 0.1 | 0.002594 | 0.98 | 0.032609 | 0.002802 | 0.0192 |

## Full Candidate Pool

| rank | ts_code | name | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 11 | 002828.SZ | 贝肯能源 | 0 | 0.000559 | 0.98 | 0.004679 | 0.001327 | 0.0027 |
| 12 | 605162.SH | 新中港 | 0 | -0.000888 | 0.98 | 0.008936 | 0.001345 | 0.0083 |
| 13 | 603863.SH | 松炀资源 | 0 | -0.005354 | 0.98 | 0.005171 | 0.001371 | 0.00905 |
| 14 | 300839.SZ | 博汇股份 | 0 | -0.008267 | 0.964583 | -0.001307 | 0.001306 | 0.0057 |
| 15 | 603122.SH | 合富中国 | 0 | -0.011242 | 0.978579 | -0.007248 | 0.00145 | 0.0027 |
| 16 | 603798.SH | 康普顿 | 0 | -0.012586 | 0.846633 | 0.000533 | 0.00122 | 0.005211 |
| 17 | 603477.SH | 巨星农牧 | 0 | -0.012643 | 0.98 | -0.003144 | 0.001262 | 0.0083 |
| 18 | 920230.BJ | 欧康医药 | 0 | -0.017023 | 0.98 | -0.002092 | 0.001975 | 0.011665 |
| 19 | 000155.SZ | 川能动力 | 0 | -0.0176 | 0.98 | -0.011768 | 0.00121 | 0.004857 |
| 20 | 002560.SZ | 通达股份 | 0 | -0.018772 | 0.98 | 0.011519 | 0.0035 | 0.0192 |
| 21 | 000586.SZ | 汇源通信 | 0 | -0.022234 | 0.947926 | -0.013246 | 0.001253 | 0.0083 |
| 22 | 600594.SH | 益佰制药 | 0 | -0.024707 | 0.644923 | 0.037785 | 0.001712 | 0.004449 |
| 23 | 603042.SH | 华脉科技 | 0 | -0.030068 | 0.98 | -0.018969 | 0.001693 | 0.009785 |
| 24 | 000968.SZ | 蓝焰控股 | 0 | -0.038512 | 0.68223 | 0.007852 | 0.001712 | 0.005958 |
| 25 | 000762.SZ | 西藏矿业 | 0 | -0.074024 | 0.98 | -0.055724 | 0.001855 | 0.0142 |
| 26 | 603353.SH | 和顺石油 | 0 | -0.080125 | 0.971147 | -0.066972 | 0.001862 | 0.011791 |
| 27 | 300834.SZ | 星辉环材 | 0 | -0.115212 | 0.98 | -0.096903 | 0.001788 | 0.0147 |
| 28 | 600488.SH | 津药药业 | 0 | -0.117723 | 0.227717 | 0.046476 | 0.002096 | 0.0082 |

