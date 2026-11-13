# Decision Report (20261116)

- signal_date: **20261113**
- exec_date: **20261116**
- exit_date: **20261117**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1720,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.1720**
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
- available_rows: **0** / **35**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **35**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **35**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.010735**
- max_EV_base: **-0.010735**
- max_E_ret: **0.012258**
- mean_cost: **0.001691**
- mean_risk_penalty: **0.020688**
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
| 1 | 000819.SZ | 岳阳兴长 | 1→2 | 0 | -0.010735 | 0.619255 | 0.00812 | 0.001121 | 0.014642 |
| 2 | 002425.SZ | 凯撒文化 | 1→2 | 0 | -0.012368 | 0.608494 | 0.010541 | 0.001399 | 0.017383 |
| 3 | 002148.SZ | 北纬科技 | 3→4 | 0 | -0.012588 | 0.601878 | 0.007945 | 0.001228 | 0.016142 |
| 4 | 600715.SH | 文投控股 | 1→2 | 0 | -0.01283 | 0.57562 | 0.008817 | 0.001301 | 0.016604 |
| 5 | 000595.SZ | 新能股份 | 1→2 | 0 | -0.01307 | 0.602167 | 0.007902 | 0.001187 | 0.016641 |
| 6 | 002739.SZ | 儒意电影 | 1→2 | 0 | -0.013509 | 0.656356 | 0.007894 | 0.001235 | 0.017456 |
| 7 | 603166.SH | 福达股份 | 1→2 | 0 | -0.013584 | 0.613449 | 0.0097 | 0.001232 | 0.018302 |
| 8 | 600977.SH | 中国电影 | 1→2 | 0 | -0.014293 | 0.744645 | 0.009601 | 0.001768 | 0.019675 |
| 9 | 603686.SH | 福龙马 | 1→2 | 0 | -0.015703 | 0.69662 | 0.008779 | 0.001506 | 0.020313 |
| 10 | 603286.SH | 日盈电子 | 1→2 | 0 | -0.015723 | 0.659686 | 0.010053 | 0.00126 | 0.021095 |
| 11 | 600651.SH | 飞乐音响 | 1→2 | 0 | -0.015852 | 0.633725 | 0.009455 | 0.002133 | 0.019712 |
| 12 | 000665.SZ | 湖北广电 | 1→2 | 0 | -0.016097 | 0.613882 | 0.007477 | 0.001315 | 0.019372 |
| 13 | 001358.SZ | 兴欣新材 | 2→3 | 0 | -0.016254 | 0.464814 | 0.012258 | 0.001844 | 0.020107 |
| 14 | 605318.SH | 法狮龙 | 1→2 | 0 | -0.016315 | 0.597801 | 0.009058 | 0.001802 | 0.019928 |
| 15 | 002819.SZ | 东方中科 | 1→2 | 0 | -0.016454 | 0.614809 | 0.010022 | 0.001746 | 0.02087 |
| 16 | 002205.SZ | 国统股份 | 2→3 | 0 | -0.016459 | 0.773907 | 0.01118 | 0.001972 | 0.023139 |
| 17 | 600467.SH | 好当家 | 2→3 | 0 | -0.016504 | 0.789999 | 0.009176 | 0.002334 | 0.021418 |
| 18 | 002042.SZ | 华孚时尚 | 1→2 | 0 | -0.016798 | 0.75 | 0.007461 | 0.001816 | 0.020578 |
| 19 | 002124.SZ | 天邦食品 | 1→2 | 0 | -0.01691 | 0.770469 | 0.008978 | 0.002241 | 0.021586 |
| 20 | 603396.SH | 金辰股份 | 3→4 | 0 | -0.017056 | 0.682024 | 0.008519 | 0.00144 | 0.021426 |
| 21 | 001393.SZ | 维通利 | 1→2 | 0 | -0.017109 | 0.706319 | 0.009635 | 0.001824 | 0.02209 |
| 22 | 002853.SZ | 皮阿诺 | 1→2 | 0 | -0.01723 | 0.641884 | 0.009394 | 0.001854 | 0.021406 |
| 23 | 002548.SZ | 金新农 | 1→2 | 0 | -0.017472 | 0.647432 | 0.007277 | 0.001441 | 0.020742 |
| 24 | 002685.SZ | 华东重机 | 1→2 | 0 | -0.017497 | 0.696314 | 0.007939 | 0.001448 | 0.021577 |
| 25 | 002905.SZ | 金逸影视 | 1→2 | 0 | -0.017499 | 0.592014 | 0.009125 | 0.001946 | 0.020956 |
| 26 | 000766.SZ | 通化金马 | 1→2 | 0 | -0.01771 | 0.741201 | 0.007419 | 0.001798 | 0.021411 |
| 27 | 002187.SZ | 广百股份 | 2→3 | 0 | -0.017714 | 0.780726 | 0.00921 | 0.001983 | 0.022922 |
| 28 | 603103.SH | 横店影视 | 1→2 | 0 | -0.01792 | 0.582755 | 0.009963 | 0.002134 | 0.021593 |
| 29 | 603928.SH | 兴业股份 | 1→2 | 0 | -0.018134 | 0.63809 | 0.008699 | 0.001873 | 0.021812 |
| 30 | 603767.SH | 中马传动 | 1→2 | 0 | -0.018251 | 0.644931 | 0.009777 | 0.001256 | 0.0233 |
| 31 | 605255.SH | 天普股份 | 4→5 | 0 | -0.018473 | 0.679796 | 0.011014 | 0.002265 | 0.023695 |
| 32 | 600410.SH | 华胜天成 | 1→2 | 0 | -0.018634 | 0.700778 | 0.010155 | 0.001751 | 0.024 |
| 33 | 002748.SZ | 世龙实业 | 1→2 | 0 | -0.018938 | 0.670101 | 0.008528 | 0.001353 | 0.0233 |
| 34 | 603636.SH | 南威软件 | 1→2 | 0 | -0.019495 | 0.716098 | 0.008715 | 0.001736 | 0.024 |
| 35 | 002520.SZ | 日发精机 | 1→2 | 0 | -0.019918 | 0.768104 | 0.009911 | 0.002644 | 0.024886 |

