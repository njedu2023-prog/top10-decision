# Decision Report (20261113)

- signal_date: **20261112**
- exec_date: **20261113**
- exit_date: **20261116**
- requested_trade_date: **20261112**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1952,volatility_mean=0.0498**
- guardrail_reason: **CAUTION:open_board_max=6.0,tail_risk_mean=0.1952**
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
- available_rows: **37** / **37**
- hard_risk_rows: **9**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.005037**

## Decision Diagnostics

- primary_no_trade_reason: **selected_positive_weight**
- rows_scored: **37**
- selected_rows: **2**
- positive_ev_rows: **2**
- positive_ev_base_rows: **2**
- positive_e_ret_rows: **30**
- high_pfill_rows: **0**
- low_risk_rows: **3**
- max_EV: **0.002397**
- max_EV_base: **0.002397**
- max_E_ret: **0.020153**
- mean_cost: **0.001985**
- mean_risk_penalty: **0.021664**
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
| 1 | 603706.SH | 东方环宇 | 4→5 | 0.07 | 0.002397 | 0.722434 | 0.013027 | 0.001128 | 0.005886 |
| 2 | 600853.SH | 龙建股份 | 1→2 | 0.07 | 0.000334 | 0.730653 | 0.013017 | 0.001263 | 0.007914 |

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 3 | 002021.SZ | 中捷资源 | 1→2 | 0 | -0.00079 | 0.724323 | 0.014637 | 0.001322 | 0.01007 |
| 4 | 600467.SH | 好当家 | 1→2 | 0 | -0.001981 | 0.70548 | 0.012173 | 0.001346 | 0.009222 |
| 5 | 603396.SH | 金辰股份 | 2→3 | 0 | -0.001988 | 0.683733 | 0.020153 | 0.002166 | 0.013601 |
| 6 | 002623.SZ | 亚玛顿 | 4→5 | 0 | -0.002541 | 0.703235 | 0.019946 | 0.002182 | 0.014386 |
| 7 | 002187.SZ | 广百股份 | 1→2 | 0 | -0.002592 | 0.715404 | 0.015756 | 0.001224 | 0.01264 |
| 8 | 603135.SH | 中重科技 | 2→3 | 0 | -0.005165 | 0.776318 | 0.011668 | 0.001333 | 0.01289 |
| 9 | 600857.SH | 宁波中百 | 1→2 | 0 | -0.005901 | 0.674793 | 0.016046 | 0.001841 | 0.014887 |
| 10 | 000573.SZ | 粤宏远Ａ | 1→2 | 0 | -0.007637 | 0.72546 | 0.013761 | 0.002253 | 0.015367 |
| 11 | 600569.SH | 安阳钢铁 | 1→2 | 0 | -0.008223 | 0.752146 | 0.012695 | 0.001543 | 0.016229 |
| 12 | 002205.SZ | 国统股份 | 1→2 | 0 | -0.008331 | 0.757238 | 0.012103 | 0.001268 | 0.016228 |
| 13 | 601091.SH | 沈鼓集团 | 1→2 | 0 | -0.008366 | 0.799007 | 0.020137 | 0.002486 | 0.021969 |
| 14 | 001358.SZ | 兴欣新材 | 1→2 | 0 | -0.008943 | 0.688791 | 0.01105 | 0.001183 | 0.015371 |
| 15 | 605169.SH | 洪通燃气 | 1→2 | 0 | -0.010255 | 0.746233 | 0.013232 | 0.001937 | 0.018192 |
| 16 | 603958.SH | 哈森股份 | 1→2 | 0 | -0.010533 | 0.761884 | 0.012173 | 0.001343 | 0.018465 |
| 17 | 600327.SH | 大东方 | 5→6 | 0 | -0.011002 | 0.697879 | 0.009481 | 0.002531 | 0.015088 |
| 18 | 601606.SH | 长城军工 | 1→2 | 0 | -0.01109 | 0.746318 | 0.012101 | 0.001393 | 0.018729 |
| 19 | 605222.SH | 起帆电缆 | 1→2 | 0 | -0.011381 | 0.742634 | 0.013604 | 0.001997 | 0.019486 |
| 20 | 603175.SH | 超颖电子 | 1→2 | 0 | -0.011499 | 0.727615 | 0.011807 | 0.001324 | 0.018767 |
| 21 | 002628.SZ | 成都路桥 | 1→2 | 0 | -0.01208 | 0.723145 | 0.012806 | 0.002062 | 0.019279 |
| 22 | 601188.SH | 龙江交通 | 1→2 | 0 | -0.012298 | 0.779127 | 0.012487 | 0.00226 | 0.019767 |
| 23 | 603366.SH | 日出东方 | 5→6 | 0 | -0.013822 | 0.838133 | 0.012736 | 0.002193 | 0.022304 |
| 24 | 603725.SH | 天安新材 | 2→3 | 0 | -0.015705 | 0.685012 | 0.010046 | 0.001291 | 0.021295 |
| 25 | 605255.SH | 天普股份 | 3→4 | 0 | -0.015749 | 0.782961 | 0.010996 | 0.002325 | 0.022034 |
| 26 | 002693.SZ | 双成药业 | 1→2 | 0 | -0.016733 | 0.806762 | 0.01112 | 0.001714 | 0.02399 |
| 27 | 002172.SZ | 澳洋健康 | 1→2 | 0 | -0.021278 | 0.818536 | 0.011278 | 0.002543 | 0.027966 |
| 28 | 000995.SZ | 皇台酒业 | 1→2 | 0 | -0.025521 | 0.817011 | 0.008065 | 0.00311 | 0.029 |
| 29 | 603536.SH | 惠发食品 | 3→4 | 0 | -0.032136 | 0.594308 | -0.001294 | 0.001921 | 0.029445 |
| 30 | 601566.SH | 九牧王 | 1→2 | 0 | -0.036279 | 0.555965 | -0.004284 | 0.001753 | 0.032145 |
| 31 | 601929.SH | 吉视传媒 | 1→2 | 0 | -0.036725 | 0.6462 | 0.000141 | 0.002173 | 0.034643 |
| 32 | 002593.SZ | 日上集团 | 2→3 | 0 | -0.0399 | 0.702069 | 0.00099 | 0.002643 | 0.037952 |
| 33 | 600558.SH | 大西洋 | 1→2 | 0 | -0.04008 | 0.665587 | -0.003885 | 0.002305 | 0.03519 |
| 34 | 600802.SH | 福建水泥 | 1→2 | 0 | -0.04162 | 0.652356 | -0.002332 | 0.003387 | 0.036711 |
| 35 | 002148.SZ | 北纬科技 | 2→3 | 0 | -0.042785 | 0.660798 | -0.004142 | 0.0027 | 0.037347 |
| 36 | 002086.SZ | 东方海洋 | 1→2 | 0 | -0.045162 | 0.651228 | -0.006266 | 0.002957 | 0.038125 |
| 37 | 002696.SZ | 百洋股份 | 1→2 | 0 | -0.045496 | 0.663694 | -0.005229 | 0.003026 | 0.039 |

