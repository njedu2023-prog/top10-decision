from __future__ import annotations


ERET_TARGET_COLUMN = "realized_ret_open_to_tplus1_timed_exit"
ERET_COMPAT_TARGET_COLUMN = "realized_ret_t1_to_t2"
ERET_TRUTH_VERSION = "eret_truth_v3_tplus1_first_touch_exit"
ERET_HOLDING_MODE = "manual_buy_t_0925_auction_to_tplus1_first_touch_exit"
DECISION_EXECUTION_CONTRACT = "D_signal_manual_T_0925_auction_guidance_Tplus1_timed_exit"
PFILL_TRUTH_VERSION = "pfill_truth_v6_manual_0925_proxy"
PFILL_EXECUTION_CONTRACT = "manual_guidance_0925_public_market_fillability_proxy"
EXIT_POLICY_VERSION = "tplus1_first_touch_tp3_sl2p5_time1450_v1"
EXIT_TAKE_PROFIT_PCT = 0.03
EXIT_STOP_LOSS_PCT = -0.025
EXIT_LATEST_TIME = "14:50"


__all__ = [
    "DECISION_EXECUTION_CONTRACT",
    "ERET_COMPAT_TARGET_COLUMN",
    "ERET_HOLDING_MODE",
    "ERET_TARGET_COLUMN",
    "ERET_TRUTH_VERSION",
    "EXIT_LATEST_TIME",
    "EXIT_POLICY_VERSION",
    "EXIT_STOP_LOSS_PCT",
    "EXIT_TAKE_PROFIT_PCT",
    "PFILL_EXECUTION_CONTRACT",
    "PFILL_TRUTH_VERSION",
]
