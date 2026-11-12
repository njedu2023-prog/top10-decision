# Decision Report (20261113)

- signal_date: **20261112**
- exec_date: **20261113**
- exit_date: **20261116**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1952,volatility_mean=0.0498,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.1952**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **37**
- features_base_loaded: **True**
- features_base_rows: **5573**
- features_limit_loaded: **True**
- features_limit_rows: **5573**
- truth_close_loaded: **True**
- truth_close_rows: **5573**
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
- available_rows: **0** / **37**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **37**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **37**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.00741**
- max_EV_base: **-0.00741**
- max_E_ret: **0.01385**
- mean_cost: **0.001985**
- mean_risk_penalty: **0.022764**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261112.csv`
- execution_table: `data/decision/decision_execution_20261113.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261113.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261112.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 603706.SH | 东方环宇 | 4→5 | 0 | -0.00741 | 0.611859 | 0.008855 | 0.001128 | 0.0117 |
| 2 | 600853.SH | 龙建股份 | 1→2 | 0 | -0.009752 | 0.62024 | 0.008682 | 0.001263 | 0.013874 |
| 3 | 600467.SH | 好当家 | 1→2 | 0 | -0.01018 | 0.616642 | 0.008225 | 0.001346 | 0.013906 |
| 4 | 002021.SZ | 中捷资源 | 1→2 | 0 | -0.010688 | 0.617532 | 0.010649 | 0.001322 | 0.015942 |
| 5 | 002187.SZ | 广百股份 | 1→2 | 0 | -0.01478 | 0.597214 | 0.009348 | 0.001224 | 0.019139 |
| 6 | 001358.SZ | 兴欣新材 | 1→2 | 0 | -0.015358 | 0.601116 | 0.007717 | 0.001183 | 0.018813 |
| 7 | 600569.SH | 安阳钢铁 | 1→2 | 0 | -0.015656 | 0.643344 | 0.009637 | 0.001543 | 0.020312 |
| 8 | 600327.SH | 大东方 | 5→6 | 0 | -0.015752 | 0.65435 | 0.008404 | 0.002531 | 0.01872 |
| 9 | 002623.SZ | 亚玛顿 | 4→5 | 0 | -0.016383 | 0.578958 | 0.01385 | 0.002182 | 0.022219 |
| 10 | 603396.SH | 金辰股份 | 2→3 | 0 | -0.016598 | 0.554651 | 0.012624 | 0.002166 | 0.021434 |
| 11 | 600857.SH | 宁波中百 | 1→2 | 0 | -0.016957 | 0.599156 | 0.01 | 0.001841 | 0.021107 |
| 12 | 603135.SH | 中重科技 | 2→3 | 0 | -0.017048 | 0.681848 | 0.007244 | 0.001333 | 0.020654 |
| 13 | 002628.SZ | 成都路桥 | 1→2 | 0 | -0.017269 | 0.631076 | 0.009864 | 0.002062 | 0.021432 |
| 14 | 605169.SH | 洪通燃气 | 1→2 | 0 | -0.017283 | 0.635352 | 0.008744 | 0.001937 | 0.020901 |
| 15 | 601566.SH | 九牧王 | 1→2 | 0 | -0.017684 | 0.729254 | 0.00852 | 0.001753 | 0.022145 |
| 16 | 601606.SH | 长城军工 | 1→2 | 0 | -0.017843 | 0.679331 | 0.009899 | 0.001393 | 0.023175 |
| 17 | 002205.SZ | 国统股份 | 1→2 | 0 | -0.018324 | 0.647801 | 0.008113 | 0.001268 | 0.022312 |
| 18 | 603366.SH | 日出东方 | 5→6 | 0 | -0.018477 | 0.766553 | 0.009615 | 0.002193 | 0.023655 |
| 19 | 002693.SZ | 双成药业 | 1→2 | 0 | -0.019179 | 0.733545 | 0.008908 | 0.001714 | 0.024 |
| 20 | 603958.SH | 哈森股份 | 1→2 | 0 | -0.019232 | 0.651893 | 0.007055 | 0.001343 | 0.022488 |
| 21 | 603725.SH | 天安新材 | 2→3 | 0 | -0.019943 | 0.655708 | 0.008156 | 0.001291 | 0.024 |
| 22 | 605222.SH | 起帆电缆 | 1→2 | 0 | -0.020099 | 0.63111 | 0.007679 | 0.001997 | 0.022948 |
| 23 | 603175.SH | 超颖电子 | 1→2 | 0 | -0.020128 | 0.649271 | 0.00745 | 0.001324 | 0.023641 |
| 24 | 000573.SZ | 粤宏远Ａ | 1→2 | 0 | -0.02023 | 0.605407 | 0.008628 | 0.002253 | 0.0232 |
| 25 | 601929.SH | 吉视传媒 | 1→2 | 0 | -0.021283 | 0.717855 | 0.008764 | 0.002173 | 0.025402 |
| 26 | 600558.SH | 大西洋 | 1→2 | 0 | -0.021298 | 0.775079 | 0.007994 | 0.002305 | 0.02519 |
| 27 | 603536.SH | 惠发食品 | 3→4 | 0 | -0.021364 | 0.685276 | 0.007118 | 0.001921 | 0.02432 |
| 28 | 601188.SH | 龙江交通 | 1→2 | 0 | -0.021559 | 0.69518 | 0.008653 | 0.00226 | 0.025314 |
| 29 | 605255.SH | 天普股份 | 3→4 | 0 | -0.021652 | 0.673703 | 0.006347 | 0.002325 | 0.023604 |
| 30 | 601091.SH | 沈鼓集团 | 1→2 | 0 | -0.022502 | 0.691592 | 0.012991 | 0.002486 | 0.029 |
| 31 | 002148.SZ | 北纬科技 | 2→3 | 0 | -0.02486 | 0.775904 | 0.006686 | 0.0027 | 0.027347 |
| 32 | 002172.SZ | 澳洋健康 | 1→2 | 0 | -0.024987 | 0.741903 | 0.008508 | 0.002543 | 0.028756 |
| 33 | 002593.SZ | 日上集团 | 2→3 | 0 | -0.025015 | 0.755073 | 0.008497 | 0.002643 | 0.028787 |
| 34 | 600802.SH | 福建水泥 | 1→2 | 0 | -0.02544 | 0.770869 | 0.006042 | 0.003387 | 0.026711 |
| 35 | 002086.SZ | 东方海洋 | 1→2 | 0 | -0.026567 | 0.793343 | 0.005691 | 0.002957 | 0.028125 |
| 36 | 000995.SZ | 皇台酒业 | 1→2 | 0 | -0.027136 | 0.762741 | 0.006522 | 0.00311 | 0.029 |
| 37 | 002696.SZ | 百洋股份 | 1→2 | 0 | -0.027587 | 0.781393 | 0.005681 | 0.003026 | 0.029 |

