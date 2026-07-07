# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-07-07T17:11:14
- 原始预测样本数：890
- 已验证样本数：856
- 未验证样本数：34
- 上涨数量：496
- 上涨率：57.94%
- 涨停数量：198
- 涨停率：23.13%
- 下跌数量：344
- 下跌率：40.19%
- 平均涨跌幅：2.00%
- 中位涨跌幅：1.25%
- 平均上涨幅度：6.43%
- 平均下跌幅度：-4.31%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.043178
- corr_RiskPenalty_return：0.021886
- corr_EV_up：0.025614
- corr_RiskPenalty_down：-0.022276
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
