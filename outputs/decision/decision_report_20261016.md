# Decision Report (20261016)

- signal_date: **20261015**
- exec_date: **20261016**
- exit_date: **20261019**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1766,volatility_mean=0.0490**
- guardrail_reason: **CAUTION:open_board_max=6.0,tail_risk_mean=0.1766**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **34**
- features_base_loaded: **True**
- features_base_rows: **5565**
- features_limit_loaded: **True**
- features_limit_rows: **5565**
- truth_close_loaded: **True**
- truth_close_rows: **5565**
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
- available_rows: **34** / **34**
- hard_risk_rows: **8**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.005466**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **34**
- selected_rows: **2**
- positive_ev_rows: **2**
- positive_ev_base_rows: **2**
- positive_e_ret_rows: **29**
- high_pfill_rows: **0**
- low_risk_rows: **4**
- max_EV: **0.001384**
- max_EV_base: **0.001384**
- max_E_ret: **0.016155**
- mean_cost: **0.001611**
- mean_risk_penalty: **0.01876**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261015.csv`
- execution_table: `data/decision/decision_execution_20261016.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261016.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261015.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 002343.SZ | 慈文传媒 | 5→6 | 0.07 | 0.001384 | 0.742849 | 0.014178 | 0.001386 | 0.007762 |
| 2 | 603329.SH | 上海雅仕 | 1→2 | 0.07 | 0.000103 | 0.736361 | 0.012306 | 0.001183 | 0.007776 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 3 | 001318.SZ | 阳光乳业 | 3→4 | 0 | -0.000097 | 0.712124 | 0.012153 | 0.001158 | 0.007593 |
| 4 | 605189.SH | 富春染织 | 2→3 | 0 | -0.000912 | 0.795496 | 0.015108 | 0.001539 | 0.011391 |
| 5 | 603386.SH | 骏亚科技 | 1→2 | 0 | -0.001307 | 0.719486 | 0.016155 | 0.001115 | 0.011816 |
| 6 | 002889.SZ | 东方嘉盛 | 1→2 | 0 | -0.001614 | 0.716027 | 0.011423 | 0.001205 | 0.008589 |
| 7 | 000722.SZ | 湖南发展 | 1→2 | 0 | -0.003506 | 0.702099 | 0.01345 | 0.001189 | 0.01176 |
| 8 | 600668.SH | 尖峰集团 | 1→2 | 0 | -0.0045 | 0.778013 | 0.014361 | 0.001351 | 0.014323 |
| 9 | 002748.SZ | 世龙实业 | 1→2 | 0 | -0.005655 | 0.764669 | 0.012569 | 0.001402 | 0.013864 |
| 10 | 600721.SH | 百花医药 | 1→2 | 0 | -0.00608 | 0.721956 | 0.011881 | 0.001345 | 0.013313 |
| 11 | 000995.SZ | 皇台酒业 | 1→2 | 0 | -0.006135 | 0.758794 | 0.012862 | 0.001464 | 0.01443 |
| 12 | 603172.SH | 万丰股份 | 1→2 | 0 | -0.006325 | 0.714086 | 0.011886 | 0.00173 | 0.013083 |
| 13 | 000626.SZ | 远大控股 | 1→2 | 0 | -0.006391 | 0.73753 | 0.014652 | 0.001231 | 0.015966 |
| 14 | 002836.SZ | 新宏泽 | 1→2 | 0 | -0.006873 | 0.689742 | 0.013387 | 0.001719 | 0.014387 |
| 15 | 603200.SH | 上海洗霸 | 1→2 | 0 | -0.00695 | 0.738632 | 0.014744 | 0.001451 | 0.016389 |
| 16 | 002815.SZ | 崇达技术 | 1→2 | 0 | -0.006974 | 0.668596 | 0.015595 | 0.002033 | 0.015367 |
| 17 | 603999.SH | 读者传媒 | 1→2 | 0 | -0.007443 | 0.67718 | 0.013013 | 0.001811 | 0.014444 |
| 18 | 001225.SZ | 和泰机电 | 1→2 | 0 | -0.0085 | 0.690104 | 0.014701 | 0.001271 | 0.017375 |
| 19 | 003029.SZ | 吉大正元 | 1→2 | 0 | -0.009139 | 0.74052 | 0.010722 | 0.001236 | 0.015843 |
| 20 | 605003.SH | 众望布艺 | 1→2 | 0 | -0.010435 | 0.787566 | 0.012566 | 0.002185 | 0.018146 |
| 21 | 603690.SH | 至纯科技 | 1→2 | 0 | -0.010447 | 0.727448 | 0.012218 | 0.001301 | 0.018034 |
| 22 | 601811.SH | 新华文轩 | 2→3 | 0 | -0.011808 | 0.727186 | 0.010343 | 0.001223 | 0.018106 |
| 23 | 002134.SZ | 天津普林 | 1→2 | 0 | -0.012173 | 0.773344 | 0.011746 | 0.001443 | 0.019814 |
| 24 | 603237.SH | 五芳斋 | 1→2 | 0 | -0.013938 | 0.779372 | 0.009967 | 0.001579 | 0.020128 |
| 25 | 600281.SH | 华阳新材 | 1→2 | 0 | -0.015273 | 0.817315 | 0.009674 | 0.00182 | 0.02136 |
| 26 | 605058.SH | 澳弘电子 | 1→2 | 0 | -0.017375 | 0.809876 | 0.010596 | 0.001957 | 0.024 |
| 27 | 603535.SH | 嘉诚国际 | 1→2 | 0 | -0.028674 | 0.597815 | 0.004798 | 0.001981 | 0.029561 |
| 28 | 002696.SZ | 百洋股份 | 5→6 | 0 | -0.030219 | 0.613889 | 0.001252 | 0.001838 | 0.029149 |
| 29 | 002103.SZ | 广博股份 | 1→2 | 0 | -0.030409 | 0.661417 | -0.000007 | 0.001537 | 0.028867 |
| 30 | 002543.SZ | 万和电气 | 1→2 | 0 | -0.030488 | 0.617415 | 0.000041 | 0.001425 | 0.029089 |
| 31 | 002945.SZ | 华林证券 | 1→2 | 0 | -0.036202 | 0.584015 | -0.004321 | 0.002448 | 0.031231 |
| 32 | 600812.SH | 华北制药 | 1→2 | 0 | -0.037952 | 0.615938 | -0.001968 | 0.001934 | 0.034806 |
| 33 | 000530.SZ | 冰山冷热 | 3→4 | 0 | -0.039233 | 0.676984 | -0.001706 | 0.003252 | 0.034825 |
| 34 | 603230.SH | 内蒙新华 | 1→2 | 0 | -0.040144 | 0.601235 | -0.004791 | 0.002017 | 0.035247 |

