# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-07-01T17:12:24
- 原始预测样本数：850
- 已验证样本数：816
- 未验证样本数：34
- 上涨数量：482
- 上涨率：59.07%
- 涨停数量：196
- 涨停率：24.02%
- 下跌数量：318
- 下跌率：38.97%
- 平均涨跌幅：2.19%
- 中位涨跌幅：1.43%
- 平均上涨幅度：6.48%
- 平均下跌幅度：-4.21%
- 最大涨幅：20.21%
- 最大跌幅：-15.31%

## EV / RiskPenalty 相关性

- corr_EV_return：0.049053
- corr_RiskPenalty_return：0.058338
- corr_EV_up：0.04087
- corr_RiskPenalty_down：-0.055502
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
