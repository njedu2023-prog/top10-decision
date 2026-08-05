# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-08-05T16:32:01
- 原始预测样本数：1100
- 已验证样本数：1066
- 未验证样本数：34
- 上涨数量：616
- 上涨率：57.79%
- 涨停数量：246
- 涨停率：23.08%
- 下跌数量：432
- 下跌率：40.53%
- 平均涨跌幅：1.93%
- 中位涨跌幅：1.31%
- 平均上涨幅度：6.40%
- 平均下跌幅度：-4.36%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.041053
- corr_RiskPenalty_return：0.020923
- corr_EV_up：0.026517
- corr_RiskPenalty_down：-0.022556
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
