# Decision Report (20260907)

- signal_date: **20260904**
- exec_date: **20260907**
- exit_date: **20260908**
- requested_trade_date: **auto**
- regime: **RISK_ON**
- risk_budget: **1**
- regime_reason: **tail_risk_mean=0.1482**
- guardrail_reason: **CAUTION:open_board_max=16.0,tail_risk_mean=0.1482**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **39**
- features_base_loaded: **True**
- features_base_rows: **5548**
- features_limit_loaded: **True**
- features_limit_rows: **5548**
- truth_close_loaded: **True**
- truth_close_rows: **5548**
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
- available_rows: **39** / **39**
- hard_risk_rows: **7**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.004828**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **39**
- selected_rows: **3**
- positive_ev_rows: **3**
- positive_ev_base_rows: **3**
- positive_e_ret_rows: **34**
- high_pfill_rows: **0**
- low_risk_rows: **6**
- max_EV: **0.000267**
- max_EV_base: **0.000267**
- max_E_ret: **0.018214**
- mean_cost: **0.001686**
- mean_risk_penalty: **0.017424**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20260904.csv`
- execution_table: `data/decision/decision_execution_20260907.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260907.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20260904.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 603123.SH | 翠微股份 | 1→2 | 0.1 | 0.000267 | 0.659814 | 0.018214 | 0.00121 | 0.010541 |
| 2 | 000702.SZ | 正虹科技 | 1→2 | 0.1 | 0.000092 | 0.709461 | 0.01352 | 0.001197 | 0.008303 |
| 3 | 002862.SZ | 实丰文化 | 1→2 | 0.1 | 0.000071 | 0.729358 | 0.012111 | 0.001144 | 0.007618 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 4 | 001222.SZ | 源飞宠物 | 1→2 | 0 | -0.000653 | 0.740342 | 0.011947 | 0.001178 | 0.008319 |
| 5 | 003040.SZ | 楚天龙 | 1→2 | 0 | -0.001381 | 0.638455 | 0.015257 | 0.001198 | 0.009924 |
| 6 | 000735.SZ | 罗 牛 山 | 1→2 | 0 | -0.001992 | 0.721207 | 0.01376 | 0.001138 | 0.010778 |
| 7 | 600506.SH | 统一股份 | 1→2 | 0 | -0.002112 | 0.788837 | 0.016391 | 0.001538 | 0.013504 |
| 8 | 600802.SH | 福建水泥 | 1→2 | 0 | -0.002971 | 0.719734 | 0.008756 | 0.001192 | 0.00808 |
| 9 | 002702.SZ | 海欣食品 | 1→2 | 0 | -0.003308 | 0.698855 | 0.010384 | 0.001162 | 0.009403 |
| 10 | 603151.SH | 邦基科技 | 1→2 | 0 | -0.003347 | 0.708854 | 0.013524 | 0.001796 | 0.011137 |
| 11 | 605398.SH | 新炬网络 | 2→3 | 0 | -0.003482 | 0.728538 | 0.013164 | 0.001922 | 0.011152 |
| 12 | 002403.SZ | 爱仕达 | 2→3 | 0 | -0.003853 | 0.700615 | 0.013268 | 0.001237 | 0.011912 |
| 13 | 001366.SZ | 播恩集团 | 1→2 | 0 | -0.003922 | 0.697864 | 0.013717 | 0.001149 | 0.012345 |
| 14 | 000798.SZ | 中水渔业 | 1→2 | 0 | -0.004393 | 0.70652 | 0.010504 | 0.001173 | 0.01064 |
| 15 | 605580.SH | 恒盛能源 | 2→3 | 0 | -0.004761 | 0.720046 | 0.013578 | 0.002106 | 0.012432 |
| 16 | 603122.SH | 合富中国 | 1→2 | 0 | -0.004913 | 0.747554 | 0.014808 | 0.00136 | 0.014623 |
| 17 | 000428.SZ | 华天酒店 | 1→2 | 0 | -0.005391 | 0.722026 | 0.010287 | 0.001267 | 0.011552 |
| 18 | 601949.SH | 中国出版 | 1→2 | 0 | -0.005419 | 0.710157 | 0.010386 | 0.001879 | 0.010915 |
| 19 | 600108.SH | 亚盛集团 | 2→3 | 0 | -0.005633 | 0.733891 | 0.01516 | 0.001449 | 0.01531 |
| 20 | 605577.SH | 龙版传媒 | 5→6 | 0 | -0.007534 | 0.772613 | 0.013219 | 0.001366 | 0.016382 |
| 21 | 000592.SZ | 平潭发展 | 1→2 | 0 | -0.007886 | 0.735038 | 0.013471 | 0.001563 | 0.016225 |
| 22 | 600865.SH | 百大集团 | 2→3 | 0 | -0.00826 | 0.770423 | 0.011364 | 0.001596 | 0.015419 |
| 23 | 002564.SZ | 天沃科技 | 1→2 | 0 | -0.008445 | 0.696079 | 0.011518 | 0.002116 | 0.014347 |
| 24 | 601579.SH | 会稽山 | 1→2 | 0 | -0.008712 | 0.720962 | 0.013261 | 0.001892 | 0.016381 |
| 25 | 603390.SH | 通达电气 | 1→2 | 0 | -0.009329 | 0.744869 | 0.01196 | 0.001837 | 0.016401 |
| 26 | 002124.SZ | 天邦食品 | 1→2 | 0 | -0.009465 | 0.756802 | 0.011002 | 0.001652 | 0.016139 |
| 27 | 001316.SZ | 润贝航科 | 1→2 | 0 | -0.009601 | 0.799131 | 0.012565 | 0.001556 | 0.018085 |
| 28 | 002868.SZ | 绿康生化 | 1→2 | 0 | -0.010879 | 0.73222 | 0.008742 | 0.001393 | 0.015887 |
| 29 | 600059.SH | 古越龙山 | 1→2 | 0 | -0.012665 | 0.846504 | 0.011048 | 0.001853 | 0.020164 |
| 30 | 002827.SZ | 高争民爆 | 1→2 | 0 | -0.016018 | 0.786487 | 0.011759 | 0.002135 | 0.023131 |
| 31 | 600698.SH | 湖南天雁 | 1→2 | 0 | -0.016916 | 0.786387 | 0.008503 | 0.00163 | 0.021972 |
| 32 | 603162.SH | 海通发展 | 2→3 | 0 | -0.017128 | 0.83597 | 0.01054 | 0.002191 | 0.023748 |
| 33 | 600975.SH | 新五丰 | 1→2 | 0 | -0.031036 | 0.624891 | -0.001948 | 0.001819 | 0.028 |
| 34 | 000876.SZ | 新 希 望 | 1→2 | 0 | -0.032966 | 0.598698 | -0.002284 | 0.001989 | 0.02961 |
| 35 | 002949.SZ | 华阳国际 | 1→2 | 0 | -0.033648 | 0.656933 | 0.000013 | 0.001995 | 0.031661 |
| 36 | 603696.SH | 安记食品 | 1→2 | 0 | -0.035845 | 0.678587 | 0.000107 | 0.001918 | 0.034 |
| 37 | 600354.SH | 敦煌种业 | 1→2 | 0 | -0.042565 | 0.639116 | -0.003325 | 0.002597 | 0.037844 |
| 38 | 000560.SZ | 我爱我家 | nan | 0 | -0.043817 | 0.649006 | -0.004936 | 0.002966 | 0.037647 |
| 39 | 000892.SZ | 欢瑞世纪 | 1→2 | 0 | -0.04489 | 0.635916 | -0.005503 | 0.003393 | 0.037998 |

