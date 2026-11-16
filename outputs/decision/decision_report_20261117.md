# Decision Report (20261117)

- signal_date: **20261116**
- exec_date: **20261117**
- exit_date: **20261118**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1859,volatility_mean=0.0459,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.1859**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **30**
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
- available_rows: **0** / **30**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **30**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **30**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.011362**
- max_EV_base: **-0.011362**
- max_E_ret: **0.01102**
- mean_cost: **0.001732**
- mean_risk_penalty: **0.020968**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261116.csv`
- execution_table: `data/decision/decision_execution_20261117.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261117.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261116.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 000533.SZ | 顺钠股份 | 1→2 | 0 | -0.011362 | 0.597847 | 0.009108 | 0.001149 | 0.015658 |
| 2 | 002548.SZ | 金新农 | 2→3 | 0 | -0.011369 | 0.63794 | 0.007084 | 0.001244 | 0.014644 |
| 3 | 002187.SZ | 广百股份 | 3→4 | 0 | -0.011471 | 0.583741 | 0.01102 | 0.001165 | 0.016738 |
| 4 | 002861.SZ | 瀛通通讯 | 1→2 | 0 | -0.012883 | 0.632902 | 0.008555 | 0.001271 | 0.017027 |
| 5 | 600386.SH | 北巴传媒 | 1→2 | 0 | -0.013295 | 0.618194 | 0.007888 | 0.001327 | 0.016844 |
| 6 | 600822.SH | 上海物贸 | 1→2 | 0 | -0.013354 | 0.604116 | 0.009814 | 0.001236 | 0.018046 |
| 7 | 601929.SH | 吉视传媒 | 1→2 | 0 | -0.013356 | 0.612005 | 0.009128 | 0.001545 | 0.017397 |
| 8 | 002058.SZ | 紫竹高科 | 1→2 | 0 | -0.013368 | 0.632083 | 0.008716 | 0.001178 | 0.0177 |
| 9 | 000521.SZ | 长虹美菱 | 1→2 | 0 | -0.015432 | 0.616092 | 0.007555 | 0.001864 | 0.018223 |
| 10 | 000819.SZ | 岳阳兴长 | 2→3 | 0 | -0.015564 | 0.780602 | 0.009629 | 0.001978 | 0.021102 |
| 11 | 002356.SZ | 鹏飞绿能 | 1→2 | 0 | -0.015614 | 0.706816 | 0.009348 | 0.001718 | 0.020503 |
| 12 | 002959.SZ | 小熊电器 | 1→2 | 0 | -0.01562 | 0.608408 | 0.01005 | 0.001735 | 0.019999 |
| 13 | 002685.SZ | 华东重机 | 2→3 | 0 | -0.016044 | 0.670575 | 0.010821 | 0.001563 | 0.021737 |
| 14 | 002419.SZ | 天虹股份 | 1→2 | 0 | -0.016084 | 0.671583 | 0.010437 | 0.00141 | 0.021683 |
| 15 | 002035.SZ | 华帝股份 | 1→2 | 0 | -0.016342 | 0.7119 | 0.008316 | 0.001593 | 0.020669 |
| 16 | 605378.SH | 野马电池 | 1→2 | 0 | -0.016707 | 0.601246 | 0.009557 | 0.001949 | 0.020504 |
| 17 | 600936.SH | 北投科技 | 1→2 | 0 | -0.016985 | 0.597473 | 0.009652 | 0.001984 | 0.020768 |
| 18 | 002905.SZ | 金逸影视 | 2→3 | 0 | -0.017086 | 0.594044 | 0.010561 | 0.002106 | 0.021254 |
| 19 | 000628.SZ | 高新发展 | 1→2 | 0 | -0.017097 | 0.640624 | 0.010008 | 0.001261 | 0.022248 |
| 20 | 000595.SZ | 新能股份 | 2→3 | 0 | -0.017294 | 0.767282 | 0.008223 | 0.001905 | 0.021699 |
| 21 | 601188.SH | 龙江交通 | 1→2 | 0 | -0.017695 | 0.69302 | 0.009273 | 0.001744 | 0.022377 |
| 22 | 603983.SH | 丸美生物 | 1→2 | 0 | -0.018001 | 0.657815 | 0.008697 | 0.002232 | 0.02149 |
| 23 | 002205.SZ | 国统股份 | 3→4 | 0 | -0.018865 | 0.778517 | 0.010791 | 0.00222 | 0.025046 |
| 24 | 002748.SZ | 世龙实业 | 2→3 | 0 | -0.018983 | 0.704647 | 0.009283 | 0.001524 | 0.024 |
| 25 | 001318.SZ | 阳光乳业 | 1→2 | 0 | -0.019052 | 0.67841 | 0.008443 | 0.00148 | 0.0233 |
| 26 | 603118.SH | 共进股份 | 1→2 | 0 | -0.019446 | 0.663557 | 0.009502 | 0.001678 | 0.024073 |
| 27 | 000156.SZ | 华数传媒 | 1→2 | 0 | -0.019771 | 0.705682 | 0.008169 | 0.001536 | 0.024 |
| 28 | 002425.SZ | 凯撒文化 | 2→3 | 0 | -0.021539 | 0.679075 | 0.007559 | 0.001834 | 0.024839 |
| 29 | 600715.SH | 文投控股 | 2→3 | 0 | -0.024706 | 0.783925 | 0.008194 | 0.003191 | 0.027938 |
| 30 | 002148.SZ | 北纬科技 | 4→5 | 0 | -0.024944 | 0.789333 | 0.007512 | 0.003329 | 0.027544 |

