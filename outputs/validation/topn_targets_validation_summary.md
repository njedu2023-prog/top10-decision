# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-22T17:06:03
- 原始预测样本数：580
- 已验证样本数：548
- 未验证样本数：32
- 上涨数量：321
- 上涨率：58.58%
- 涨停数量：139
- 涨停率：25.36%
- 下跌数量：211
- 下跌率：38.50%
- 平均涨跌幅：2.16%
- 中位涨跌幅：1.17%
- 平均上涨幅度：6.37%
- 平均下跌幅度：-4.08%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：0.04237
- corr_RiskPenalty_return：0.004948
- corr_EV_up：0.030597
- corr_RiskPenalty_down：-0.024927
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
