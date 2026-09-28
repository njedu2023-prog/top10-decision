# Decision Report (20260929)

- signal_date: **20260928**
- exec_date: **20260929**
- exit_date: **20260930**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1495,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.1495**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **31**
- features_base_loaded: **True**
- features_base_rows: **5557**
- features_limit_loaded: **True**
- features_limit_rows: **5557**
- truth_close_loaded: **True**
- truth_close_rows: **5557**
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
- available_rows: **0** / **31**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **31**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **31**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.008797**
- max_EV_base: **-0.008797**
- max_E_ret: **0.014575**
- mean_cost: **0.00154**
- mean_risk_penalty: **0.019847**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20260928.csv`
- execution_table: `data/decision/decision_execution_20260929.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20260929.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20260928.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 605366.SH | 宏柏新材 | 1→2 | 0 | -0.008797 | 0.631678 | 0.009193 | 0.001293 | 0.013311 |
| 2 | 001368.SZ | 通达创智 | 1→2 | 0 | -0.010048 | 0.637457 | 0.006305 | 0.001192 | 0.012875 |
| 3 | 600241.SH | 时代万恒 | 1→2 | 0 | -0.011956 | 0.620257 | 0.005644 | 0.001162 | 0.014295 |
| 4 | 002232.SZ | 启明信息 | 1→2 | 0 | -0.012521 | 0.596052 | 0.007528 | 0.001166 | 0.015842 |
| 5 | 001201.SZ | 东瑞股份 | 1→2 | 0 | -0.012575 | 0.62221 | 0.006805 | 0.001152 | 0.015656 |
| 6 | 000513.SZ | 丽珠集团 | 1→2 | 0 | -0.012992 | 0.651187 | 0.009729 | 0.001328 | 0.018 |
| 7 | 002962.SZ | 五方光电 | 1→2 | 0 | -0.013339 | 0.730929 | 0.008708 | 0.001703 | 0.018 |
| 8 | 603968.SH | 醋化股份 | 1→2 | 0 | -0.014047 | 0.614312 | 0.006189 | 0.001205 | 0.016644 |
| 9 | 000020.SZ | 深华发Ａ | 1→2 | 0 | -0.014047 | 0.628099 | 0.007573 | 0.001104 | 0.0177 |
| 10 | 600825.SH | 新华传媒 | 5→6 | 0 | -0.014749 | 0.445486 | 0.014575 | 0.002228 | 0.019014 |
| 11 | 002480.SZ | 新筑股份 | 1→2 | 0 | -0.0152 | 0.705225 | 0.010019 | 0.002085 | 0.02018 |
| 12 | 603949.SH | 雪龙集团 | 3→4 | 0 | -0.015457 | 0.668588 | 0.009285 | 0.001227 | 0.020437 |
| 13 | 000678.SZ | 襄阳轴承 | 2→3 | 0 | -0.015644 | 0.658194 | 0.008778 | 0.001234 | 0.020188 |
| 14 | 002640.SZ | 跨境通 | 1→2 | 0 | -0.015692 | 0.671056 | 0.009738 | 0.001645 | 0.020581 |
| 15 | 002912.SZ | 中新赛克 | 1→2 | 0 | -0.016136 | 0.599118 | 0.00813 | 0.001208 | 0.019798 |
| 16 | 002347.SZ | 泰尔股份 | 1→2 | 0 | -0.016261 | 0.693717 | 0.008448 | 0.001409 | 0.020712 |
| 17 | 603278.SH | 大业股份 | 2→3 | 0 | -0.016341 | 0.688762 | 0.008923 | 0.001665 | 0.020822 |
| 18 | 000503.SZ | 国新健康 | 1→2 | 0 | -0.016679 | 0.660666 | 0.008699 | 0.001324 | 0.021102 |
| 19 | 601579.SH | 会稽山 | 1→2 | 0 | -0.016774 | 0.754714 | 0.007993 | 0.001694 | 0.021113 |
| 20 | 600418.SH | 江淮汽车 | 1→2 | 0 | -0.016875 | 0.611786 | 0.008745 | 0.001221 | 0.021003 |
| 21 | 002242.SZ | 九阳股份 | 1→2 | 0 | -0.0169 | 0.731343 | 0.007238 | 0.002295 | 0.019899 |
| 22 | 600488.SH | 津药药业 | 1→2 | 0 | -0.016918 | 0.664387 | 0.008439 | 0.00127 | 0.021255 |
| 23 | 600032.SH | 浙江新能 | 1→2 | 0 | -0.017392 | 0.630336 | 0.008855 | 0.00204 | 0.020934 |
| 24 | 000980.SZ | 众泰汽车 | 1→2 | 0 | -0.018051 | 0.652525 | 0.0093 | 0.001476 | 0.022644 |
| 25 | 600802.SH | 福建水泥 | 3→4 | 0 | -0.01823 | 0.677806 | 0.008299 | 0.001655 | 0.022201 |
| 26 | 001330.SZ | 博纳影业 | 1→2 | 0 | -0.018587 | 0.581127 | 0.007322 | 0.001771 | 0.021071 |
| 27 | 002852.SZ | 道道全 | 1→2 | 0 | -0.018821 | 0.696775 | 0.009207 | 0.001404 | 0.023833 |
| 28 | 603396.SH | 金辰股份 | 3→4 | 0 | -0.018983 | 0.721934 | 0.007797 | 0.001654 | 0.022958 |
| 29 | 002342.SZ | 巨力索具 | 1→2 | 0 | -0.019661 | 0.67438 | 0.008124 | 0.00197 | 0.02317 |
| 30 | 000011.SZ | 深物业A | 1→2 | 0 | -0.020819 | 0.619769 | 0.00817 | 0.001883 | 0.024 |
| 31 | 601218.SH | 吉鑫科技 | 2→3 | 0 | -0.022835 | 0.687762 | 0.007649 | 0.002086 | 0.026009 |

