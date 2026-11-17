# Decision Report (20261118)

- signal_date: **20261117**
- exec_date: **20261118**
- exit_date: **20261119**
- requested_trade_date: **20261117**
- regime: **RISK_ON**
- risk_budget: **1**
- regime_reason: **tail_risk_mean=0.1989**
- guardrail_reason: **CAUTION:open_board_max=9.0,tail_risk_mean=0.1989**
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
- available_rows: **33** / **33**
- hard_risk_rows: **5**
- intraday_ev_bonus_mean: **0**
- intraday_penalty_extra_mean: **0**
- intraday_execution_penalty_mean: **0.004927**

## Decision Diagnostics

- primary_no_trade_reason: **positive_e_ret_cannot_cover_cost_and_risk**
- rows_scored: **33**
- selected_rows: **0**
- positive_ev_rows: **0**
- positive_ev_base_rows: **0**
- positive_e_ret_rows: **30**
- high_pfill_rows: **0**
- low_risk_rows: **1**
- max_EV: **-0.000699**
- max_EV_base: **-0.000699**
- max_E_ret: **0.017784**
- mean_cost: **0.001674**
- mean_risk_penalty: **0.018012**
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
| 1 | 600619.SH | 海立股份 | 1→2 | 0 | -0.000699 | 0.714111 | 0.012414 | 0.001159 | 0.008405 |
| 2 | 600881.SH | 亚泰集团 | 1→2 | 0 | -0.001834 | 0.744192 | 0.017784 | 0.002245 | 0.012824 |
| 3 | 002771.SZ | 真视通 | 1→2 | 0 | -0.001946 | 0.718938 | 0.015524 | 0.001145 | 0.011962 |
| 4 | 000707.SZ | 双环科技 | 1→2 | 0 | -0.001981 | 0.772647 | 0.014595 | 0.001365 | 0.011892 |
| 5 | 000521.SZ | 长虹美菱 | 2→3 | 0 | -0.002592 | 0.697656 | 0.012497 | 0.001282 | 0.010029 |
| 6 | 002685.SZ | 华东重机 | 3→4 | 0 | -0.002716 | 0.687295 | 0.01557 | 0.001794 | 0.011623 |
| 7 | 002912.SZ | 中新赛克 | 1→2 | 0 | -0.002853 | 0.725525 | 0.013088 | 0.00118 | 0.011169 |
| 8 | 002451.SZ | 摩恩电气 | 1→2 | 0 | -0.003964 | 0.633623 | 0.017776 | 0.002121 | 0.013106 |
| 9 | 002959.SZ | 小熊电器 | 2→3 | 0 | -0.003982 | 0.722017 | 0.012039 | 0.001112 | 0.011562 |
| 10 | 002242.SZ | 九阳股份 | 1→2 | 0 | -0.004487 | 0.76628 | 0.014324 | 0.001632 | 0.01383 |
| 11 | 002584.SZ | 西陇科学 | 1→2 | 0 | -0.004939 | 0.751558 | 0.014923 | 0.001224 | 0.01493 |
| 12 | 603078.SH | 江化微 | 1→2 | 0 | -0.004993 | 0.703946 | 0.014224 | 0.001232 | 0.013774 |
| 13 | 002889.SZ | 东方嘉盛 | 1→2 | 0 | -0.005006 | 0.713347 | 0.01172 | 0.00122 | 0.012147 |
| 14 | 603297.SH | 永新光学 | 1→2 | 0 | -0.00561 | 0.707266 | 0.013634 | 0.001747 | 0.013505 |
| 15 | 600202.SH | 哈空调 | 1→2 | 0 | -0.005674 | 0.72306 | 0.011406 | 0.001123 | 0.012798 |
| 16 | 603324.SH | 盛剑科技 | 1→2 | 0 | -0.0081 | 0.681204 | 0.015908 | 0.002052 | 0.016884 |
| 17 | 002652.SZ | 扬子新材 | 1→2 | 0 | -0.008693 | 0.760303 | 0.011381 | 0.001478 | 0.015868 |
| 18 | 002226.SZ | 江南化工 | 1→2 | 0 | -0.008777 | 0.711743 | 0.014525 | 0.001912 | 0.017202 |
| 19 | 600261.SH | 阳光照明 | 1→2 | 0 | -0.010176 | 0.744739 | 0.010709 | 0.001934 | 0.016217 |
| 20 | 002855.SZ | 捷荣技术 | 1→2 | 0 | -0.010743 | 0.697651 | 0.012726 | 0.001937 | 0.017684 |
| 21 | 002843.SZ | 泰嘉股份 | 1→2 | 0 | -0.010952 | 0.777374 | 0.010457 | 0.001684 | 0.017397 |
| 22 | 600410.SH | 华胜天成 | 1→2 | 0 | -0.010959 | 0.669489 | 0.013508 | 0.001237 | 0.018766 |
| 23 | 600467.SH | 好当家 | 1→2 | 0 | -0.011 | 0.825373 | 0.011855 | 0.001963 | 0.018821 |
| 24 | 603931.SH | 格林达 | 1→2 | 0 | -0.011359 | 0.725918 | 0.01488 | 0.001935 | 0.020226 |
| 25 | 002861.SZ | 瀛通通讯 | 2→3 | 0 | -0.014198 | 0.759522 | 0.01101 | 0.001773 | 0.020788 |
| 26 | 603206.SH | 嘉环科技 | 1→2 | 0 | -0.015014 | 0.718755 | 0.010339 | 0.00189 | 0.020555 |
| 27 | 002983.SZ | 芯瑞达 | 1→2 | 0 | -0.016353 | 0.735268 | 0.010759 | 0.001448 | 0.022816 |
| 28 | 600770.SH | 综艺股份 | nan | 0 | -0.01782 | 0.81068 | 0.009644 | 0.002374 | 0.023264 |
| 29 | 603030.SH | 全筑股份 | 1→2 | 0 | -0.030641 | 0.595206 | 0.000776 | 0.001621 | 0.029482 |
| 30 | 002535.SZ | 林州重机 | 1→2 | 0 | -0.034497 | 0.669455 | 0.000154 | 0.002009 | 0.032591 |
| 31 | 600936.SH | 北投科技 | 2→3 | 0 | -0.036624 | 0.640742 | -0.00303 | 0.002024 | 0.032659 |
| 32 | 600206.SH | 有研新材 | 1→2 | 0 | -0.037636 | 0.585057 | -0.00292 | 0.001928 | 0.034 |
| 33 | 000533.SZ | 顺钠股份 | 2→3 | 0 | -0.039957 | 0.644898 | -0.002911 | 0.002475 | 0.035604 |

