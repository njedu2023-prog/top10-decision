# Decision Report (20260916)

- signal_date: **20260915**
- exec_date: **20260916**
- exit_date: **20260917**
- requested_trade_date: **20260915**
- regime: **RISK_ON**
- risk_budget: **1**
- regime_reason: **tail_risk_mean=0.1508**
- guardrail_reason: **CAUTION:open_board_max=13.0,tail_risk_mean=0.1508**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **31**
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
- available_rows: **31** / **31**
- hard_risk_rows: **5**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.004508**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **31**
- selected_rows: **1**
- positive_ev_rows: **1**
- positive_ev_base_rows: **1**
- positive_e_ret_rows: **26**
- high_pfill_rows: **0**
- low_risk_rows: **2**
- max_EV: **0.003801**
- max_EV_base: **0.003801**
- max_E_ret: **0.019998**
- mean_cost: **0.001666**
- mean_risk_penalty: **0.017708**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20260915.csv`
- execution_table: `data/decision/decision_execution_20260916.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260916.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20260915.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 002531.SZ | 天顺风能 | 1→2 | 0.1 | 0.003801 | 0.707353 | 0.01421 | 0.001161 | 0.00509 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 2 | 601218.SH | 吉鑫科技 | 1→2 | 0 | -0.0019 | 0.717423 | 0.014602 | 0.001223 | 0.011153 |
| 3 | 603186.SH | 华正新材 | 1→2 | 0 | -0.002636 | 0.699179 | 0.015689 | 0.001278 | 0.012328 |
| 4 | 002487.SZ | 大金重工 | 1→2 | 0 | -0.002643 | 0.777925 | 0.019998 | 0.001508 | 0.016691 |
| 5 | 002232.SZ | 启明信息 | 2→3 | 0 | -0.003099 | 0.712023 | 0.014303 | 0.00195 | 0.011333 |
| 6 | 603090.SH | 宏盛股份 | 2→3 | 0 | -0.004075 | 0.712212 | 0.01117 | 0.001141 | 0.01089 |
| 7 | 002522.SZ | 浙江众成 | 1→2 | 0 | -0.004646 | 0.696277 | 0.0087 | 0.001267 | 0.009437 |
| 8 | 002584.SZ | 西陇科学 | 1→2 | 0 | -0.005059 | 0.763006 | 0.013133 | 0.00141 | 0.013669 |
| 9 | 603248.SH | 锡华科技 | 1→2 | 0 | -0.005157 | 0.763467 | 0.014541 | 0.001451 | 0.014808 |
| 10 | 002213.SZ | 大为股份 | 1→2 | 0 | -0.005381 | 0.707302 | 0.011171 | 0.001116 | 0.012166 |
| 11 | 002846.SZ | 英联股份 | 1→2 | 0 | -0.006663 | 0.704091 | 0.009319 | 0.001212 | 0.012012 |
| 12 | 002882.SZ | 金龙羽 | 1→2 | 0 | -0.006698 | 0.7056 | 0.012507 | 0.001932 | 0.013591 |
| 13 | 002362.SZ | 汉王科技 | 1→2 | 0 | -0.007039 | 0.730876 | 0.012176 | 0.00128 | 0.014657 |
| 14 | 001216.SZ | 华瓷股份 | 1→2 | 0 | -0.007453 | 0.700427 | 0.009437 | 0.002021 | 0.012041 |
| 15 | 600110.SH | 诺德股份 | 1→2 | 0 | -0.007471 | 0.762503 | 0.011921 | 0.001332 | 0.015229 |
| 16 | 603200.SH | 上海洗霸 | 1→2 | 0 | -0.007743 | 0.71065 | 0.012783 | 0.001858 | 0.014969 |
| 17 | 001223.SZ | 欧克科技 | 1→2 | 0 | -0.007767 | 0.742168 | 0.012439 | 0.001303 | 0.015695 |
| 18 | 603353.SH | 和顺石油 | 1→2 | 0 | -0.007934 | 0.7397 | 0.011436 | 0.001274 | 0.015119 |
| 19 | 003026.SZ | 中晶科技 | 1→2 | 0 | -0.008602 | 0.746182 | 0.011845 | 0.001447 | 0.015994 |
| 20 | 605507.SH | 国邦医药 | 1→2 | 0 | -0.008832 | 0.743765 | 0.01218 | 0.001894 | 0.015997 |
| 21 | 002585.SZ | 双星新材 | 3→4 | 0 | -0.009846 | 0.751735 | 0.012663 | 0.001833 | 0.017533 |
| 22 | 605162.SH | 新中港 | 1→2 | 0 | -0.01128 | 0.865872 | 0.012108 | 0.001907 | 0.019857 |
| 23 | 002631.SZ | 德尔未来 | 1→2 | 0 | -0.012534 | 0.741542 | 0.013108 | 0.001478 | 0.020776 |
| 24 | 603266.SH | 天龙股份 | 1→2 | 0 | -0.012544 | 0.755504 | 0.01078 | 0.001454 | 0.019234 |
| 25 | 601579.SH | 会稽山 | 1→2 | 0 | -0.012792 | 0.820451 | 0.012232 | 0.001832 | 0.020996 |
| 26 | 603082.SH | 北自科技 | 1→2 | 0 | -0.018774 | 0.848692 | 0.012809 | 0.003067 | 0.026578 |
| 27 | 600163.SH | 中闽能源 | 1→2 | 0 | -0.031838 | 0.627939 | -0.00328 | 0.001779 | 0.028 |
| 28 | 002912.SZ | 中新赛克 | 4→5 | 0 | -0.035343 | 0.714735 | -0.000447 | 0.002363 | 0.03266 |
| 29 | 605058.SH | 澳弘电子 | 3→4 | 0 | -0.036329 | 0.619567 | -0.007422 | 0.001878 | 0.029853 |
| 30 | 002491.SZ | 通鼎互联 | 1→2 | 0 | -0.040416 | 0.610861 | -0.003051 | 0.002335 | 0.036218 |
| 31 | 000993.SZ | 闽东电力 | 5→6 | 0 | -0.041088 | 0.617161 | -0.006572 | 0.002662 | 0.03437 |

