口径：PremiumRet(2→3)=Close[3]/Close[2]-1
当前已完成：Premium 子系统骨架 + P0 可跑闭环（train/predict/CSV+MD）
下一步主线：接入 close 真值数据源 → features 升级（两阶段结构特征）→ run_premium.yml 日跑与提交产物

--------------------------------------------------------------------------------
衔接口令：

口径：PremiumRet(2→3)=Close[3]/Close[2]-1
当前已完成：Premium 子系统骨架 + P0 可跑闭环（train/predict/CSV+MD）
下一步主线：接入 close 真值数据源 → features 升级（两阶段结构特征）→ run_premium.yml 日跑与提交产物

--------------------------------------------------------------------------------
线路文件：

. src/top10decision/premium/*
. scripts/run_premium.py
. github/workflows/run_premium.yml
. outputs/premium/rank/
. outputs/premium/learning/premium_eval_history.csv
. outputs/premium/_last_run.txt

--------------------------------------------------------------------------------
期望产物：

. outputs/premium/rank/premium_rank_YYYYMMDD.csv
. outputs/premium/rank/premium_rank_YYYYMMDD.md

--------------------------------------------------------------------------------

 ↑ END: 2026-03-01:08:30
