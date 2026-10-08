# Decision Report (20261009)

- signal_date: **20261008**
- exec_date: **20261009**
- exit_date: **20261012**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1539,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.1539**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **38**
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
- available_rows: **0** / **38**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **38**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **38**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.010211**
- max_EV_base: **-0.010211**
- max_E_ret: **0.014475**
- mean_cost: **0.001719**
- mean_risk_penalty: **0.019834**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261008.csv`
- execution_table: `data/decision/decision_execution_20261009.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261009.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261008.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 603928.SH | 兴业股份 | 2→3 | 0 | -0.010211 | 0.616812 | 0.007313 | 0.001115 | 0.013606 |
| 2 | 600241.SH | 时代万恒 | 4→5 | 0 | -0.010398 | 0.604309 | 0.00871 | 0.001201 | 0.014461 |
| 3 | 001336.SZ | 楚环科技 | 1→2 | 0 | -0.010855 | 0.636945 | 0.006943 | 0.001153 | 0.014125 |
| 4 | 600408.SH | 安泰集团 | 1→2 | 0 | -0.012628 | 0.628467 | 0.006066 | 0.001255 | 0.015185 |
| 5 | 600310.SH | 广西能源 | 1→2 | 0 | -0.012871 | 0.614982 | 0.00747 | 0.001282 | 0.016183 |
| 6 | 600825.SH | 新华传媒 | 8→9 | 0 | -0.013094 | 0.445608 | 0.014194 | 0.002219 | 0.0172 |
| 7 | 002490.SZ | 山东墨龙 | 1→2 | 0 | -0.013428 | 0.625537 | 0.0083 | 0.001257 | 0.017362 |
| 8 | 603788.SH | 宁波高发 | 1→2 | 0 | -0.013556 | 0.725864 | 0.008448 | 0.001688 | 0.018 |
| 9 | 605303.SH | 园林股份 | 3→4 | 0 | -0.013566 | 0.586203 | 0.014475 | 0.00227 | 0.019782 |
| 10 | 002702.SZ | 海欣食品 | 1→2 | 0 | -0.014205 | 0.636336 | 0.007573 | 0.001324 | 0.0177 |
| 11 | 002242.SZ | 九阳股份 | 4→5 | 0 | -0.014951 | 0.561711 | 0.011635 | 0.002083 | 0.019403 |
| 12 | 603863.SH | 松炀资源 | 1→2 | 0 | -0.015044 | 0.614821 | 0.006043 | 0.001233 | 0.017527 |
| 13 | 605378.SH | 野马电池 | 1→2 | 0 | -0.015136 | 0.723836 | 0.010019 | 0.001644 | 0.020743 |
| 14 | 603137.SH | 恒尚节能 | 1→2 | 0 | -0.015664 | 0.604599 | 0.009101 | 0.001701 | 0.019465 |
| 15 | 001300.SZ | 三柏硕 | 1→2 | 0 | -0.015719 | 0.5666 | 0.00939 | 0.001772 | 0.019267 |
| 16 | 002733.SZ | 雄韬股份 | 1→2 | 0 | -0.015986 | 0.688464 | 0.009563 | 0.001585 | 0.020985 |
| 17 | 002058.SZ | 紫竹高科 | 3→4 | 0 | -0.016258 | 0.633713 | 0.009725 | 0.001926 | 0.020495 |
| 18 | 600513.SH | 联环药业 | 1→2 | 0 | -0.016402 | 0.715334 | 0.007592 | 0.00162 | 0.020213 |
| 19 | 002805.SZ | 丰元股份 | 1→2 | 0 | -0.016407 | 0.658446 | 0.01014 | 0.001248 | 0.021835 |
| 20 | 002774.SZ | 快意电梯 | 1→2 | 0 | -0.016556 | 0.597349 | 0.008885 | 0.001915 | 0.019948 |
| 21 | 601975.SH | 招商南油 | 1→2 | 0 | -0.016709 | 0.755922 | 0.008431 | 0.001858 | 0.021224 |
| 22 | 601956.SH | 东贝集团 | 1→2 | 0 | -0.016901 | 0.606358 | 0.007602 | 0.002008 | 0.019502 |
| 23 | 002869.SZ | 金溢科技 | 1→2 | 0 | -0.017128 | 0.652531 | 0.007471 | 0.001304 | 0.020699 |
| 24 | 002565.SZ | 顺灏股份 | 1→2 | 0 | -0.017475 | 0.720702 | 0.007779 | 0.001743 | 0.021338 |
| 25 | 002866.SZ | 传艺科技 | 3→4 | 0 | -0.017477 | 0.673114 | 0.007735 | 0.001748 | 0.020935 |
| 26 | 600026.SH | 中远海能 | 1→2 | 0 | -0.017479 | 0.573597 | 0.01039 | 0.002148 | 0.021291 |
| 27 | 002951.SZ | 金时科技 | 1→2 | 0 | -0.017993 | 0.631207 | 0.008087 | 0.002191 | 0.020906 |
| 28 | 000869.SZ | 张  裕Ａ | 1→2 | 0 | -0.018046 | 0.60815 | 0.00574 | 0.002046 | 0.01949 |
| 29 | 600663.SH | 陆家嘴 | 2→3 | 0 | -0.018082 | 0.70088 | 0.006406 | 0.00251 | 0.020061 |
| 30 | 600400.SH | 红豆股份 | 1→2 | 0 | -0.018133 | 0.752317 | 0.006551 | 0.001912 | 0.02115 |
| 31 | 603906.SH | 龙蟠科技 | 2→3 | 0 | -0.018234 | 0.76935 | 0.008193 | 0.001941 | 0.022596 |
| 32 | 605298.SH | 必得科技 | 1→2 | 0 | -0.018614 | 0.586105 | 0.010306 | 0.00223 | 0.022424 |
| 33 | 600617.SH | 国新能源 | 1→2 | 0 | -0.018899 | 0.649539 | 0.006702 | 0.002078 | 0.021175 |
| 34 | 600812.SH | 华北制药 | 1→2 | 0 | -0.019081 | 0.698087 | 0.006977 | 0.001436 | 0.022516 |
| 35 | 600722.SH | 金牛化工 | 1→2 | 0 | -0.019133 | 0.6735 | 0.008213 | 0.001364 | 0.0233 |
| 36 | 603073.SH | 彩蝶实业 | 1→2 | 0 | -0.019365 | 0.728569 | 0.007134 | 0.001891 | 0.022672 |
| 37 | 603200.SH | 上海洗霸 | 3→4 | 0 | -0.019469 | 0.772178 | 0.010043 | 0.002116 | 0.025108 |
| 38 | 002687.SZ | 乔治白 | 1→2 | 0 | -0.020019 | 0.661617 | 0.007747 | 0.001321 | 0.023823 |

