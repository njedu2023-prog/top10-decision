# Decision Report (20261116)

- signal_date: **20261113**
- exec_date: **20261116**
- exit_date: **20261117**
- requested_trade_date: **20261113**
- regime: **RISK_ON**
- risk_budget: **1**
- regime_reason: **tail_risk_mean=0.1720**
- guardrail_reason: **CAUTION:open_board_max=7.0,tail_risk_mean=0.1720**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **35**
- features_base_loaded: **True**
- features_base_rows: **5571**
- features_limit_loaded: **True**
- features_limit_rows: **5571**
- truth_close_loaded: **True**
- truth_close_rows: **5571**
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
- available_rows: **35** / **35**
- hard_risk_rows: **5**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.004492**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **35**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **32**
- high_pfill_rows: **0**
- low_risk_rows: **2**
- max_EV: **-0.000966**
- max_EV_base: **-0.000966**
- max_E_ret: **0.017546**
- mean_cost: **0.001691**
- mean_risk_penalty: **0.018065**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261113.csv`
- execution_table: `data/decision/decision_execution_20261116.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261116.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261113.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 000819.SZ | 岳阳兴长 | 1→2 | 0 | -0.000966 | 0.720788 | 0.012813 | 0.001121 | 0.00908 |
| 2 | 002148.SZ | 北纬科技 | 3→4 | 0 | -0.001112 | 0.71411 | 0.013058 | 0.001228 | 0.009209 |
| 3 | 002425.SZ | 凯撒文化 | 1→2 | 0 | -0.001602 | 0.723612 | 0.014658 | 0.001399 | 0.01081 |
| 4 | 603166.SH | 福达股份 | 1→2 | 0 | -0.001723 | 0.722643 | 0.016053 | 0.001232 | 0.012092 |
| 5 | 000595.SZ | 新能股份 | 1→2 | 0 | -0.002249 | 0.719467 | 0.012654 | 0.001187 | 0.010167 |
| 6 | 002739.SZ | 儒意电影 | 1→2 | 0 | -0.002423 | 0.753454 | 0.011964 | 0.001235 | 0.010202 |
| 7 | 001358.SZ | 兴欣新材 | 2→3 | 0 | -0.00319 | 0.622832 | 0.017546 | 0.001844 | 0.012274 |
| 8 | 603686.SH | 福龙马 | 1→2 | 0 | -0.003265 | 0.807361 | 0.013268 | 0.001506 | 0.012472 |
| 9 | 600715.SH | 文投控股 | 1→2 | 0 | -0.003869 | 0.672375 | 0.012847 | 0.001301 | 0.011206 |
| 10 | 603396.SH | 金辰股份 | 3→4 | 0 | -0.004209 | 0.785619 | 0.014023 | 0.00144 | 0.013785 |
| 11 | 605318.SH | 法狮龙 | 1→2 | 0 | -0.005102 | 0.715665 | 0.013385 | 0.001802 | 0.012878 |
| 12 | 002685.SZ | 华东重机 | 1→2 | 0 | -0.005978 | 0.793556 | 0.013378 | 0.001448 | 0.015146 |
| 13 | 603103.SH | 横店影视 | 1→2 | 0 | -0.006087 | 0.666668 | 0.017114 | 0.002134 | 0.015363 |
| 14 | 002819.SZ | 东方中科 | 1→2 | 0 | -0.006677 | 0.717616 | 0.01287 | 0.001746 | 0.014166 |
| 15 | 603286.SH | 日盈电子 | 1→2 | 0 | -0.007088 | 0.761737 | 0.012073 | 0.00126 | 0.015025 |
| 16 | 001393.SZ | 维通利 | 1→2 | 0 | -0.007524 | 0.778637 | 0.013106 | 0.001824 | 0.015905 |
| 17 | 002905.SZ | 金逸影视 | 1→2 | 0 | -0.008034 | 0.6946 | 0.013497 | 0.001946 | 0.015463 |
| 18 | 002748.SZ | 世龙实业 | 1→2 | 0 | -0.008693 | 0.749806 | 0.012406 | 0.001353 | 0.016642 |
| 19 | 603767.SH | 中马传动 | 1→2 | 0 | -0.009276 | 0.728285 | 0.014178 | 0.001256 | 0.018345 |
| 20 | 600651.SH | 飞乐音响 | 1→2 | 0 | -0.010248 | 0.739188 | 0.014313 | 0.002133 | 0.018695 |
| 21 | 600977.SH | 中国电影 | 1→2 | 0 | -0.011582 | 0.81608 | 0.012082 | 0.001768 | 0.019675 |
| 22 | 603928.SH | 兴业股份 | 1→2 | 0 | -0.011939 | 0.739847 | 0.012337 | 0.001873 | 0.019194 |
| 23 | 600410.SH | 华胜天成 | 1→2 | 0 | -0.012291 | 0.76387 | 0.013484 | 0.001751 | 0.02084 |
| 24 | 002853.SZ | 皮阿诺 | 1→2 | 0 | -0.012771 | 0.717528 | 0.012178 | 0.001854 | 0.019655 |
| 25 | 603636.SH | 南威软件 | 1→2 | 0 | -0.013609 | 0.787453 | 0.012754 | 0.001736 | 0.021916 |
| 26 | 002187.SZ | 广百股份 | 2→3 | 0 | -0.01369 | 0.85295 | 0.012552 | 0.001983 | 0.022413 |
| 27 | 002124.SZ | 天邦食品 | 1→2 | 0 | -0.014727 | 0.814375 | 0.011175 | 0.002241 | 0.021586 |
| 28 | 605255.SH | 天普股份 | 4→5 | 0 | -0.015547 | 0.748599 | 0.013039 | 0.002265 | 0.023043 |
| 29 | 000766.SZ | 通化金马 | 1→2 | 0 | -0.01577 | 0.789089 | 0.009427 | 0.001798 | 0.021411 |
| 30 | 002520.SZ | 日发精机 | 1→2 | 0 | -0.016313 | 0.830293 | 0.01351 | 0.002644 | 0.024886 |
| 31 | 000665.SZ | 湖北广电 | 1→2 | 0 | -0.025763 | 0.557687 | 0.001116 | 0.001315 | 0.025071 |
| 32 | 002548.SZ | 金新农 | 1→2 | 0 | -0.030117 | 0.585303 | -0.00026 | 0.001441 | 0.028524 |
| 33 | 600467.SH | 好当家 | 2→3 | 0 | -0.033168 | 0.699007 | 0.000836 | 0.002334 | 0.031418 |
| 34 | 002042.SZ | 华孚时尚 | 1→2 | 0 | -0.035053 | 0.616196 | -0.004315 | 0.001816 | 0.030578 |
| 35 | 002205.SZ | 国统股份 | 2→3 | 0 | -0.037211 | 0.605176 | -0.003472 | 0.001972 | 0.033139 |

