# Decision Report (20261013)

- signal_date: **20261012**
- exec_date: **20261013**
- exit_date: **20261014**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1720,volatility_mean=0.0456,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.1720**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **39**
- features_base_loaded: **True**
- features_base_rows: **5561**
- features_limit_loaded: **True**
- features_limit_rows: **5561**
- truth_close_loaded: **True**
- truth_close_rows: **5561**
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
- available_rows: **0** / **39**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **39**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **39**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.009016**
- max_EV_base: **-0.009016**
- max_E_ret: **0.011306**
- mean_cost: **0.001651**
- mean_risk_penalty: **0.020409**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261012.csv`
- execution_table: `data/decision/decision_execution_20261013.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261013.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261012.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 002692.SZ | 远程股份 | 2→3 | 0 | -0.009016 | 0.619543 | 0.009973 | 0.00112 | 0.014075 |
| 2 | 002908.SZ | 德生科技 | 1→2 | 0 | -0.010393 | 0.632725 | 0.007543 | 0.001142 | 0.014024 |
| 3 | 601595.SH | 上海电影 | 2→3 | 0 | -0.010838 | 0.629806 | 0.008482 | 0.001158 | 0.015022 |
| 4 | 002343.SZ | 慈文传媒 | 2→3 | 0 | -0.011756 | 0.606864 | 0.009615 | 0.00117 | 0.016421 |
| 5 | 600825.SH | 新华传媒 | 10→11 | 0 | -0.012681 | 0.464077 | 0.010118 | 0.001793 | 0.015584 |
| 6 | 002531.SZ | 天顺风能 | 1→2 | 0 | -0.013019 | 0.570804 | 0.010291 | 0.001175 | 0.017718 |
| 7 | 603230.SH | 内蒙新华 | 1→2 | 0 | -0.013696 | 0.633073 | 0.008554 | 0.001412 | 0.0177 |
| 8 | 002441.SZ | 众业达 | 1→2 | 0 | -0.014364 | 0.689239 | 0.007272 | 0.001376 | 0.018 |
| 9 | 603528.SH | 多伦科技 | 1→2 | 0 | -0.014432 | 0.605881 | 0.010068 | 0.001972 | 0.018559 |
| 10 | 603336.SH | 宏辉果蔬 | 1→2 | 0 | -0.014441 | 0.620626 | 0.007185 | 0.001193 | 0.017707 |
| 11 | 601127.SH | 赛力斯 | 1→2 | 0 | -0.014984 | 0.530929 | 0.009603 | 0.001772 | 0.01831 |
| 12 | 002242.SZ | 九阳股份 | 6→7 | 0 | -0.015057 | 0.778403 | 0.00925 | 0.001982 | 0.020274 |
| 13 | 603968.SH | 醋化股份 | 1→2 | 0 | -0.015446 | 0.672263 | 0.009764 | 0.00139 | 0.02062 |
| 14 | 605136.SH | 丽人丽妆 | 2→3 | 0 | -0.015631 | 0.579854 | 0.011306 | 0.002146 | 0.020041 |
| 15 | 002591.SZ | 恒大高新 | 1→2 | 0 | -0.015699 | 0.68034 | 0.007863 | 0.001535 | 0.019513 |
| 16 | 600831.SH | 广电网络 | 1→2 | 0 | -0.015871 | 0.660956 | 0.009765 | 0.001404 | 0.020921 |
| 17 | 002576.SZ | 通达动力 | 1→2 | 0 | -0.016022 | 0.653241 | 0.007893 | 0.001308 | 0.01987 |
| 18 | 600560.SH | 金自天正 | 1→2 | 0 | -0.016065 | 0.616677 | 0.0066 | 0.00186 | 0.018276 |
| 19 | 603096.SH | 新经典 | 2→3 | 0 | -0.01615 | 0.598568 | 0.009559 | 0.001939 | 0.019933 |
| 20 | 002990.SZ | 盛视科技 | 1→2 | 0 | -0.016664 | 0.64312 | 0.009147 | 0.001256 | 0.021291 |
| 21 | 001231.SZ | 农心科技 | 1→2 | 0 | -0.016671 | 0.633982 | 0.009858 | 0.001483 | 0.021437 |
| 22 | 600812.SH | 华北制药 | 3→4 | 0 | -0.016678 | 0.678557 | 0.010964 | 0.001369 | 0.022749 |
| 23 | 603500.SH | 祥和实业 | 1→2 | 0 | -0.016758 | 0.722142 | 0.007335 | 0.001634 | 0.020421 |
| 24 | 603032.SH | 德新科技 | 1→2 | 0 | -0.016921 | 0.746348 | 0.008644 | 0.001788 | 0.021585 |
| 25 | 603533.SH | 掌阅科技 | 2→3 | 0 | -0.016983 | 0.678881 | 0.00997 | 0.001564 | 0.022188 |
| 26 | 601929.SH | 吉视传媒 | 2→3 | 0 | -0.017235 | 0.694221 | 0.010944 | 0.001777 | 0.023056 |
| 27 | 603992.SH | 松霖科技 | 1→2 | 0 | -0.017362 | 0.672335 | 0.007547 | 0.00228 | 0.020156 |
| 28 | 605577.SH | 龙版传媒 | 2→3 | 0 | -0.017562 | 0.600106 | 0.010102 | 0.001821 | 0.021804 |
| 29 | 600693.SH | 东百集团 | 1→2 | 0 | -0.017722 | 0.542257 | 0.011072 | 0.001832 | 0.021894 |
| 30 | 600851.SH | 海欣股份 | 1→2 | 0 | -0.017798 | 0.658715 | 0.007968 | 0.001353 | 0.021693 |
| 31 | 601811.SH | 新华文轩 | 2→3 | 0 | -0.017858 | 0.710246 | 0.009051 | 0.001486 | 0.022801 |
| 32 | 603232.SH | 格尔软件 | 2→3 | 0 | -0.017889 | 0.676694 | 0.008447 | 0.001624 | 0.021981 |
| 33 | 001330.SZ | 博纳影业 | 2→3 | 0 | -0.018386 | 0.71063 | 0.00958 | 0.001683 | 0.023511 |
| 34 | 002696.SZ | 百洋股份 | 2→3 | 0 | -0.0184 | 0.619095 | 0.00695 | 0.001716 | 0.020987 |
| 35 | 605287.SH | 德才股份 | 2→3 | 0 | -0.018792 | 0.565165 | 0.009764 | 0.002049 | 0.022262 |
| 36 | 600192.SH | 长城电工 | 1→2 | 0 | -0.019267 | 0.639134 | 0.009512 | 0.001347 | 0.024 |
| 37 | 002264.SZ | 新 华 都 | 1→2 | 0 | -0.0199 | 0.774392 | 0.009143 | 0.002285 | 0.024695 |
| 38 | 600241.SH | 时代万恒 | 6→7 | 0 | -0.021478 | 0.743635 | 0.00936 | 0.00256 | 0.025878 |
| 39 | 000892.SZ | 欢瑞世纪 | 2→3 | 0 | -0.023871 | 0.740934 | 0.010499 | 0.00265 | 0.029 |

