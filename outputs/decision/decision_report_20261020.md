# Decision Report (20261020)

- signal_date: **20261019**
- exec_date: **20261020**
- exit_date: **20261021**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1972,volatility_mean=0.0471,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.1972**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **40**
- features_base_loaded: **True**
- features_base_rows: **5564**
- features_limit_loaded: **True**
- features_limit_rows: **5564**
- truth_close_loaded: **True**
- truth_close_rows: **5564**
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
- available_rows: **0** / **40**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **40**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **40**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.009928**
- max_EV_base: **-0.009928**
- max_E_ret: **0.014366**
- mean_cost: **0.00177**
- mean_risk_penalty: **0.020721**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261019.csv`
- execution_table: `data/decision/decision_execution_20261020.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261020.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261019.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 002164.SZ | 宁波东力 | 1→2 | 0 | -0.009928 | 0.621537 | 0.00985 | 0.001187 | 0.014863 |
| 2 | 003029.SZ | 吉大正元 | 3→4 | 0 | -0.011612 | 0.581968 | 0.012463 | 0.001236 | 0.017629 |
| 3 | 001211.SZ | 双枪科技 | 1→2 | 0 | -0.011642 | 0.637477 | 0.009017 | 0.001258 | 0.016132 |
| 4 | 002085.SZ | 万丰奥威 | 2→3 | 0 | -0.012058 | 0.585334 | 0.011221 | 0.001211 | 0.017414 |
| 5 | 603330.SH | 天洋新材 | 1→2 | 0 | -0.012151 | 0.612558 | 0.009246 | 0.001117 | 0.016697 |
| 6 | 603278.SH | 大业股份 | 2→3 | 0 | -0.012922 | 0.65314 | 0.008783 | 0.001311 | 0.017347 |
| 7 | 600178.SH | 东安动力 | 2→3 | 0 | -0.013309 | 0.601512 | 0.011099 | 0.001797 | 0.018189 |
| 8 | 603626.SH | 科森科技 | 1→2 | 0 | -0.013742 | 0.618534 | 0.008234 | 0.001134 | 0.0177 |
| 9 | 600267.SH | 海正药业 | 1→2 | 0 | -0.013858 | 0.561688 | 0.014366 | 0.002175 | 0.019751 |
| 10 | 603348.SH | 文灿股份 | 2→3 | 0 | -0.014202 | 0.625868 | 0.009082 | 0.001853 | 0.018032 |
| 11 | 600329.SH | 达仁堂 | 1→2 | 0 | -0.014214 | 0.64667 | 0.009062 | 0.002074 | 0.018 |
| 12 | 002889.SZ | 东方嘉盛 | 3→4 | 0 | -0.014463 | 0.701425 | 0.009137 | 0.001744 | 0.019128 |
| 13 | 000576.SZ | 甘化科工 | 2→3 | 0 | -0.015203 | 0.691817 | 0.008867 | 0.00141 | 0.019928 |
| 14 | 001323.SZ | 慕思股份 | 1→2 | 0 | -0.015466 | 0.643776 | 0.00745 | 0.002263 | 0.018 |
| 15 | 603286.SH | 日盈电子 | 1→2 | 0 | -0.015764 | 0.612177 | 0.0093 | 0.001233 | 0.020224 |
| 16 | 600860.SH | 京城股份 | 2→3 | 0 | -0.015855 | 0.618925 | 0.011872 | 0.002067 | 0.021136 |
| 17 | 600148.SH | 长春一东 | 2→3 | 0 | -0.015978 | 0.609008 | 0.01013 | 0.002073 | 0.020074 |
| 18 | 603693.SH | 江苏新能 | 1→2 | 0 | -0.01615 | 0.603624 | 0.009907 | 0.001828 | 0.020302 |
| 19 | 603188.SH | 亚邦股份 | 1→2 | 0 | -0.016275 | 0.700496 | 0.010068 | 0.001636 | 0.021692 |
| 20 | 603197.SH | 保隆科技 | 2→3 | 0 | -0.016455 | 0.532852 | 0.01145 | 0.002019 | 0.020537 |
| 21 | 601778.SH | 晶科科技 | 1→2 | 0 | -0.01648 | 0.657902 | 0.009898 | 0.001412 | 0.02158 |
| 22 | 600032.SH | 浙江新能 | 1→2 | 0 | -0.016774 | 0.607095 | 0.010905 | 0.001927 | 0.021467 |
| 23 | 002343.SZ | 慈文传媒 | 1→2 | 0 | -0.017256 | 0.653071 | 0.007363 | 0.0029 | 0.019165 |
| 24 | 603859.SH | 能科科技 | 1→2 | 0 | -0.017295 | 0.589169 | 0.008963 | 0.001862 | 0.020714 |
| 25 | 001216.SZ | 华瓷股份 | 1→2 | 0 | -0.017342 | 0.748885 | 0.011441 | 0.00191 | 0.024 |
| 26 | 605488.SH | 福莱新材 | 1→2 | 0 | -0.018018 | 0.600362 | 0.010199 | 0.001879 | 0.022261 |
| 27 | 002724.SZ | 海洋王 | 2→3 | 0 | -0.018823 | 0.597165 | 0.010118 | 0.002124 | 0.022741 |
| 28 | 002368.SZ | 太极股份 | 1→2 | 0 | -0.01883 | 0.626914 | 0.009385 | 0.002112 | 0.022601 |
| 29 | 002886.SZ | 沃特股份 | 1→2 | 0 | -0.019137 | 0.69425 | 0.009262 | 0.001567 | 0.024 |
| 30 | 002626.SZ | 金达威 | 1→2 | 0 | -0.019195 | 0.747847 | 0.004762 | 0.001776 | 0.02098 |
| 31 | 603040.SH | 新坐标 | 1→2 | 0 | -0.019387 | 0.617618 | 0.009371 | 0.001997 | 0.023178 |
| 32 | 600686.SH | 金龙汽车 | 1→2 | 0 | -0.019599 | 0.651565 | 0.007809 | 0.001329 | 0.023358 |
| 33 | 605058.SH | 澳弘电子 | 1→2 | 0 | -0.019685 | 0.756951 | 0.008186 | 0.001882 | 0.024 |
| 34 | 605258.SH | 协和电子 | 1→2 | 0 | -0.01973 | 0.694485 | 0.008237 | 0.00145 | 0.024 |
| 35 | 603119.SH | 浙江荣泰 | 1→2 | 0 | -0.019757 | 0.667425 | 0.0096 | 0.002164 | 0.024 |
| 36 | 003021.SZ | 兆威机电 | 1→2 | 0 | -0.01992 | 0.641015 | 0.008513 | 0.002051 | 0.023325 |
| 37 | 600487.SH | 亨通光电 | 1→2 | 0 | -0.020091 | 0.559265 | 0.007933 | 0.001228 | 0.0233 |
| 38 | 002929.SZ | 润建股份 | 1→2 | 0 | -0.020707 | 0.650172 | 0.008252 | 0.002072 | 0.024 |
| 39 | 605255.SH | 天普股份 | 1→2 | 0 | -0.021093 | 0.654573 | 0.007737 | 0.0023 | 0.023857 |
| 40 | 603337.SH | 杰克科技 | nan | 0 | -0.021411 | 0.657753 | 0.006615 | 0.002242 | 0.023521 |

