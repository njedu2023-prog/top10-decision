# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-06-12T17:36:34
- 原始预测样本数：730
- 已验证样本数：698
- 未验证样本数：32
- 上涨数量：405
- 上涨率：58.02%
- 涨停数量：179
- 涨停率：25.64%
- 下跌数量：277
- 下跌率：39.68%
- 平均涨跌幅：2.10%
- 中位涨跌幅：1.17%
- 平均上涨幅度：6.51%
- 平均下跌幅度：-4.24%
- 最大涨幅：20.21%
- 最大跌幅：-15.31%

## EV / RiskPenalty 相关性

- corr_EV_return：0.004901
- corr_RiskPenalty_return：0.018626
- corr_EV_up：0.002259
- corr_RiskPenalty_down：-0.023312
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
