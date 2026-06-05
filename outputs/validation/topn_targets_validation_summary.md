# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-06-05T17:12:18
- 原始预测样本数：680
- 已验证样本数：648
- 未验证样本数：32
- 上涨数量：378
- 上涨率：58.33%
- 涨停数量：168
- 涨停率：25.93%
- 下跌数量：254
- 下跌率：39.20%
- 平均涨跌幅：2.16%
- 中位涨跌幅：1.24%
- 平均上涨幅度：6.53%
- 平均下跌幅度：-4.20%
- 最大涨幅：20.21%
- 最大跌幅：-15.31%

## EV / RiskPenalty 相关性

- corr_EV_return：-0.002465
- corr_RiskPenalty_return：0.018791
- corr_EV_up：-0.001024
- corr_RiskPenalty_down：-0.02733
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
