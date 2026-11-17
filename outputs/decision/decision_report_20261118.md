# Decision Report (20261118)

- signal_date: **20261117**
- exec_date: **20261118**
- exit_date: **20261119**
- requested_trade_date: **auto**
- regime: **CAUTION**
- risk_budget: **0.7**
- regime_reason: **tail_risk_mean=0.1989,intraday_risk_mean=0.750**
- guardrail_reason: **CAUTION:intraday_risk_mean=0.750,tail_risk_mean=0.1989**
- input_mode: **pred_plus_fs**
- fs_degrade_reason: **none**

## Input Status

- pred_loaded: **True**
- pred_rows: **33**
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
- available_rows: **0** / **33**
- hard_risk_rows: **0**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.012**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **33**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **33**
- high_pfill_rows: **0**
- low_risk_rows: **0**
- max_EV: **-0.010437**
- max_EV_base: **-0.010437**
- max_E_ret: **0.012437**
- mean_cost: **0.001674**
- mean_risk_penalty: **0.020835**
- mean_extra_penalty_total: **0**

## Artifacts

- candidates_snapshot: `data/decision/decision_candidates_20261117.csv`
- execution_table: `data/decision/decision_execution_20261118.csv`
- learning_table: `data/decision/decision_learning.csv`
- weights_latest: `docs/weights/weights_latest.csv`
- weights_dated: `docs/weights/weights_20261118.csv`
- top_evr_latest: `docs/signals/TopEVR_latest.csv`
- top_evr_dated: `docs/signals/TopEVR_20261117.csv`

## EV > 3% & RiskPenalty < 1%

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## TopN Targets

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|

## Full Candidate Pool

| rank | ts_code | name | 晋阶 | weight | EV | P_fill | E_ret | Cost | RiskPenalty |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 600619.SH | 海立股份 | 1→2 | 0 | -0.010437 | 0.604713 | 0.008336 | 0.001159 | 0.014319 |
| 2 | 000521.SZ | 长虹美菱 | 2→3 | 0 | -0.011917 | 0.598561 | 0.008336 | 0.001282 | 0.015625 |
| 3 | 002959.SZ | 小熊电器 | 2→3 | 0 | -0.012284 | 0.630331 | 0.008799 | 0.001112 | 0.016718 |
| 4 | 002771.SZ | 真视通 | 1→2 | 0 | -0.012429 | 0.617287 | 0.01011 | 0.001145 | 0.017525 |
| 5 | 002889.SZ | 东方嘉盛 | 1→2 | 0 | -0.014152 | 0.613612 | 0.007769 | 0.00122 | 0.0177 |
| 6 | 002912.SZ | 中新赛克 | 1→2 | 0 | -0.014222 | 0.612589 | 0.008632 | 0.00118 | 0.01833 |
| 7 | 000707.SZ | 双环科技 | 1→2 | 0 | -0.014324 | 0.66885 | 0.009771 | 0.001365 | 0.019494 |
| 8 | 600202.SH | 哈空调 | 1→2 | 0 | -0.014388 | 0.626866 | 0.007074 | 0.001123 | 0.0177 |
| 9 | 002584.SZ | 西陇科学 | 1→2 | 0 | -0.015718 | 0.646812 | 0.00947 | 0.001224 | 0.020619 |
| 10 | 002242.SZ | 九阳股份 | 1→2 | 0 | -0.015829 | 0.660802 | 0.009304 | 0.001632 | 0.020345 |
| 11 | 603078.SH | 江化微 | 1→2 | 0 | -0.01583 | 0.592688 | 0.008779 | 0.001232 | 0.019802 |
| 12 | 603297.SH | 永新光学 | 1→2 | 0 | -0.015885 | 0.614812 | 0.008534 | 0.001747 | 0.019384 |
| 13 | 002226.SZ | 江南化工 | 1→2 | 0 | -0.016047 | 0.644131 | 0.009313 | 0.001912 | 0.020133 |
| 14 | 603324.SH | 盛剑科技 | 1→2 | 0 | -0.016123 | 0.594892 | 0.011898 | 0.002052 | 0.021148 |
| 15 | 600881.SH | 亚泰集团 | 1→2 | 0 | -0.016661 | 0.618852 | 0.010084 | 0.002245 | 0.020657 |
| 16 | 002685.SZ | 华东重机 | 3→4 | 0 | -0.016767 | 0.558769 | 0.008023 | 0.001794 | 0.019456 |
| 17 | 002451.SZ | 摩恩电气 | 1→2 | 0 | -0.017056 | 0.482758 | 0.012437 | 0.002121 | 0.020939 |
| 18 | 603206.SH | 嘉环科技 | 1→2 | 0 | -0.017312 | 0.649055 | 0.00949 | 0.00189 | 0.021581 |
| 19 | 600467.SH | 好当家 | 1→2 | 0 | -0.017584 | 0.718498 | 0.008229 | 0.001963 | 0.021533 |
| 20 | 002652.SZ | 扬子新材 | 1→2 | 0 | -0.017702 | 0.671835 | 0.008617 | 0.001478 | 0.022014 |
| 21 | 600936.SH | 北投科技 | 2→3 | 0 | -0.017783 | 0.769603 | 0.008965 | 0.002024 | 0.022659 |
| 22 | 603030.SH | 全筑股份 | 1→2 | 0 | -0.018148 | 0.676357 | 0.007922 | 0.001621 | 0.021885 |
| 23 | 002843.SZ | 泰嘉股份 | 1→2 | 0 | -0.018207 | 0.715057 | 0.008367 | 0.001684 | 0.022506 |
| 24 | 600410.SH | 华胜天成 | 1→2 | 0 | -0.018304 | 0.579136 | 0.010762 | 0.001237 | 0.0233 |
| 25 | 002535.SZ | 林州重机 | 1→2 | 0 | -0.018563 | 0.737886 | 0.010091 | 0.002009 | 0.024 |
| 26 | 603931.SH | 格林达 | 1→2 | 0 | -0.018751 | 0.641756 | 0.009396 | 0.001935 | 0.022846 |
| 27 | 002855.SZ | 捷荣技术 | 1→2 | 0 | -0.018901 | 0.608984 | 0.008929 | 0.001937 | 0.022402 |
| 28 | 002983.SZ | 芯瑞达 | 1→2 | 0 | -0.019042 | 0.681553 | 0.009022 | 0.001448 | 0.023743 |
| 29 | 600261.SH | 阳光照明 | 1→2 | 0 | -0.019337 | 0.638956 | 0.006556 | 0.001934 | 0.021592 |
| 30 | 600206.SH | 有研新材 | 1→2 | 0 | -0.019592 | 0.721542 | 0.008781 | 0.001928 | 0.024 |
| 31 | 600770.SH | 综艺股份 | nan | 0 | -0.019914 | 0.72747 | 0.007868 | 0.002374 | 0.023264 |
| 32 | 002861.SZ | 瀛通通讯 | 2→3 | 0 | -0.020683 | 0.686358 | 0.008494 | 0.001773 | 0.02474 |
| 33 | 000533.SZ | 顺钠股份 | 2→3 | 0 | -0.021266 | 0.786208 | 0.008667 | 0.002475 | 0.025604 |

