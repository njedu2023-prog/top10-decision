# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-07-06T17:40:15
- 原始预测样本数：880
- 已验证样本数：846
- 未验证样本数：34
- 上涨数量：492
- 上涨率：58.16%
- 涨停数量：198
- 涨停率：23.40%
- 下跌数量：338
- 下跌率：39.95%
- 平均涨跌幅：2.04%
- 中位涨跌幅：1.31%
- 平均上涨幅度：6.46%
- 平均下跌幅度：-4.29%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.034521
- corr_RiskPenalty_return：0.034585
- corr_EV_up：0.020723
- corr_RiskPenalty_down：-0.030466
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
