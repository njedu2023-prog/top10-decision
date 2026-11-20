# Decision Report (20261123)

- signal_date: **20261120**
- exec_date: **20261123**
- exit_date: **20261124**
- requested_trade_date: **20261120**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.2518,volatility_mean=0.0542,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.2518**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **25**
- features_base_loaded: **True**
- features_base_rows: **5572**
- features_limit_loaded: **True**
- features_limit_rows: **5572**
- truth_close_loaded: **True**
- truth_close_rows: **5572**
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
- available_rows: **0** / **25**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **25**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **25**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.009646**
- max_EV_base: **-0.009646**
- max_E_ret: **0.011244**
- mean_cost: **0.001665**
- mean_risk_penalty: **0.020821**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261120.csv`
- execution_table: `data/decision/decision_execution_20261123.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261123.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261120.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 603004.SH | 鼎龙科技 | 1→2 | 0 | -0.009646 | 0.64495 | 0.009437 | 0.001284 | 0.014449 |
| 2 | 002548.SZ | 金新农 | 1→2 | 0 | -0.012141 | 0.636648 | 0.010807 | 0.001219 | 0.017803 |
| 3 | 002579.SZ | 中京电子 | 2→3 | 0 | -0.012583 | 0.586874 | 0.010288 | 0.001274 | 0.017347 |
| 4 | 002771.SZ | 真视通 | 4→5 | 0 | -0.012606 | 0.621405 | 0.009621 | 0.001436 | 0.017148 |
| 5 | 605058.SH | 澳弘电子 | 1→2 | 0 | -0.013271 | 0.609382 | 0.009112 | 0.001123 | 0.0177 |
| 6 | 002741.SZ | 光华科技 | 1→2 | 0 | -0.0135 | 0.634449 | 0.008079 | 0.00115 | 0.017476 |
| 7 | 000823.SZ | 超声电子 | 1→2 | 0 | -0.014353 | 0.620584 | 0.009726 | 0.001598 | 0.018791 |
| 8 | 600371.SH | 万向德农 | 1→2 | 0 | -0.014744 | 0.627699 | 0.006577 | 0.001172 | 0.0177 |
| 9 | 002749.SZ | 国光股份 | 1→2 | 0 | -0.014854 | 0.613263 | 0.00898 | 0.001774 | 0.018587 |
| 10 | 000607.SZ | 华媒控股 | 1→2 | 0 | -0.01514 | 0.618836 | 0.008699 | 0.001291 | 0.019232 |
| 11 | 603948.SH | 建业股份 | 1→2 | 0 | -0.015213 | 0.727534 | 0.009692 | 0.001654 | 0.02061 |
| 12 | 603324.SH | 盛剑科技 | 4→5 | 0 | -0.015577 | 0.765283 | 0.010343 | 0.0019 | 0.021592 |
| 13 | 002642.SZ | 荣联科技 | 1→2 | 0 | -0.015796 | 0.589523 | 0.009043 | 0.001812 | 0.019315 |
| 14 | 002086.SZ | 东方海洋 | 1→2 | 0 | -0.016519 | 0.617831 | 0.009738 | 0.001892 | 0.020644 |
| 15 | 603386.SH | 骏亚科技 | 1→2 | 0 | -0.017186 | 0.654131 | 0.011244 | 0.001241 | 0.0233 |
| 16 | 603960.SH | 克来机电 | 1→2 | 0 | -0.0173 | 0.595041 | 0.009862 | 0.001889 | 0.021279 |
| 17 | 000593.SZ | 德龙汇能 | 1→2 | 0 | -0.017512 | 0.595435 | 0.010831 | 0.001859 | 0.022102 |
| 18 | 600360.SH | 华微电子 | 1→2 | 0 | -0.018094 | 0.735476 | 0.009458 | 0.001802 | 0.023249 |
| 19 | 605018.SH | 长华集团 | 1→2 | 0 | -0.01815 | 0.592093 | 0.009413 | 0.00193 | 0.021793 |
| 20 | 002199.SZ | 东晶电子 | 1→2 | 0 | -0.018853 | 0.597497 | 0.009463 | 0.002033 | 0.022474 |
| 21 | 605577.SH | 龙版传媒 | 1→2 | 0 | -0.018927 | 0.655611 | 0.008562 | 0.001241 | 0.0233 |
| 22 | 603466.SH | 风语筑 | 1→2 | 0 | -0.019114 | 0.668504 | 0.008456 | 0.001467 | 0.0233 |
| 23 | 600467.SH | 好当家 | 1→2 | 0 | -0.020592 | 0.705814 | 0.008347 | 0.002167 | 0.024316 |
| 24 | 600327.SH | 大东方 | 1→2 | 0 | -0.024843 | 0.734478 | 0.007982 | 0.002449 | 0.028258 |
| 25 | 000530.SZ | 冰山冷热 | 1→2 | 0 | -0.025185 | 0.683334 | 0.009594 | 0.002972 | 0.028769 |

