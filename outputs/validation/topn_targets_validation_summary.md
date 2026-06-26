# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-06-26T16:59:57
- 原始预测样本数：820
- 已验证样本数：786
- 未验证样本数：34
- 上涨数量：463
- 上涨率：58.91%
- 涨停数量：190
- 涨停率：24.17%
- 下跌数量：307
- 下跌率：39.06%
- 平均涨跌幅：2.14%
- 中位涨跌幅：1.41%
- 平均上涨幅度：6.44%
- 平均下跌幅度：-4.22%
- 最大涨幅：20.21%
- 最大跌幅：-15.31%

## EV / RiskPenalty 相关性

- corr_EV_return：0.034497
- corr_RiskPenalty_return：0.044911
- corr_EV_up：0.026951
- corr_RiskPenalty_down：-0.053204
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
