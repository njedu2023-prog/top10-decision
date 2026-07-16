# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-07-16T16:15:30
- 原始预测样本数：960
- 已验证样本数：926
- 未验证样本数：34
- 上涨数量：535
- 上涨率：57.78%
- 涨停数量：212
- 涨停率：22.89%
- 下跌数量：375
- 下跌率：40.50%
- 平均涨跌幅：1.95%
- 中位涨跌幅：1.25%
- 平均上涨幅度：6.45%
- 平均下跌幅度：-4.39%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.044561
- corr_RiskPenalty_return：0.014757
- corr_EV_up：0.030451
- corr_RiskPenalty_down：-0.014003
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
