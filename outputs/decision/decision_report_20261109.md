# Decision Report (20261109)

- signal_date: **20261106**
- exec_date: **20261109**
- exit_date: **20261110**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.2321,volatility_mean=0.0515,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.2321**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **29**
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
- available_rows: **0** / **29**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **29**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **29**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.011768**
- max_EV_base: **-0.011768**
- max_E_ret: **0.013792**
- mean_cost: **0.001733**
- mean_risk_penalty: **0.02187**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261106.csv`
- execution_table: `data/decision/decision_execution_20261109.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261109.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261106.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 603381.SH | 永臻股份 | 4→5 | 0 | -0.011768 | 0.596334 | 0.009381 | 0.001195 | 0.016167 |
| 2 | 603366.SH | 日出东方 | 1→2 | 0 | -0.012197 | 0.565258 | 0.007736 | 0.001138 | 0.015432 |
| 3 | 000525.SZ | 红太阳 | 1→2 | 0 | -0.012763 | 0.612727 | 0.008315 | 0.001217 | 0.016641 |
| 4 | 605339.SH | 南侨食品 | 2→3 | 0 | -0.013327 | 0.61377 | 0.012988 | 0.00221 | 0.019088 |
| 5 | 002712.SZ | 思美传媒 | 1→2 | 0 | -0.014052 | 0.618472 | 0.007688 | 0.001107 | 0.0177 |
| 6 | 600540.SH | 新赛股份 | 1→2 | 0 | -0.015491 | 0.638463 | 0.007676 | 0.001612 | 0.01878 |
| 7 | 002310.SZ | 东方新能 | 1→2 | 0 | -0.01554 | 0.577527 | 0.009599 | 0.002079 | 0.019006 |
| 8 | 002388.SZ | 新亚制程 | 3→4 | 0 | -0.015742 | 0.641383 | 0.008119 | 0.001598 | 0.019351 |
| 9 | 001308.SZ | 康冠科技 | 1→2 | 0 | -0.016299 | 0.59666 | 0.008295 | 0.001955 | 0.019293 |
| 10 | 600278.SH | 东方创业 | 1→2 | 0 | -0.016728 | 0.704174 | 0.007318 | 0.00153 | 0.020351 |
| 11 | 600229.SH | 城市传媒 | 1→2 | 0 | -0.017048 | 0.598638 | 0.013792 | 0.002104 | 0.0232 |
| 12 | 601595.SH | 上海电影 | 1→2 | 0 | -0.017163 | 0.631916 | 0.00935 | 0.001327 | 0.021745 |
| 13 | 002578.SZ | 闽发铝业 | 5→6 | 0 | -0.017201 | 0.6319 | 0.011832 | 0.001378 | 0.0233 |
| 14 | 002889.SZ | 东方嘉盛 | 1→2 | 0 | -0.01725 | 0.657778 | 0.011384 | 0.001438 | 0.0233 |
| 15 | 600261.SH | 阳光照明 | 2→3 | 0 | -0.017959 | 0.667178 | 0.009757 | 0.00162 | 0.022848 |
| 16 | 001316.SZ | 润贝航科 | 1→2 | 0 | -0.017992 | 0.670789 | 0.009763 | 0.001428 | 0.023113 |
| 17 | 600327.SH | 大东方 | 1→2 | 0 | -0.017993 | 0.700115 | 0.009179 | 0.001978 | 0.022442 |
| 18 | 600156.SH | 华升股份 | 1→2 | 0 | -0.018037 | 0.709236 | 0.009027 | 0.001531 | 0.022908 |
| 19 | 601599.SH | 浙文影业 | 5→6 | 0 | -0.018345 | 0.67109 | 0.00979 | 0.001819 | 0.023096 |
| 20 | 603657.SH | 春光科技 | 3→4 | 0 | -0.01868 | 0.668097 | 0.008812 | 0.001267 | 0.0233 |
| 21 | 600207.SH | 安彩高科 | 1→2 | 0 | -0.018807 | 0.629991 | 0.011461 | 0.002027 | 0.024 |
| 22 | 000595.SZ | 新能股份 | 1→2 | 0 | -0.019029 | 0.627817 | 0.009262 | 0.002149 | 0.022694 |
| 23 | 002973.SZ | 侨银股份 | 1→2 | 0 | -0.01908 | 0.671561 | 0.009445 | 0.001423 | 0.024 |
| 24 | 001258.SZ | 立新能源 | 1→2 | 0 | -0.019339 | 0.605372 | 0.009993 | 0.001876 | 0.023512 |
| 25 | 002088.SZ | 鲁阳节能 | 4→5 | 0 | -0.020128 | 0.75941 | 0.007739 | 0.002005 | 0.024 |
| 26 | 601188.SH | 龙江交通 | 1→2 | 0 | -0.021637 | 0.754002 | 0.008862 | 0.002259 | 0.026059 |
| 27 | 002929.SZ | 润建股份 | 2→3 | 0 | -0.021867 | 0.655289 | 0.006158 | 0.001697 | 0.024205 |
| 28 | 600371.SH | 万向德农 | 1→2 | 0 | -0.021887 | 0.742492 | 0.009328 | 0.002219 | 0.026594 |
| 29 | 002086.SZ | 东方海洋 | 2→3 | 0 | -0.024582 | 0.697529 | 0.009447 | 0.003068 | 0.028104 |

