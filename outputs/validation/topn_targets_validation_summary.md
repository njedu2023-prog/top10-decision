# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-10-30T19:56:25
- 原始预测样本数：1660
- 已验证样本数：1626
- 未验证样本数：34
- 上涨数量：920
- 上涨率：56.58%
- 涨停数量：381
- 涨停率：23.43%
- 下跌数量：683
- 下跌率：42.00%
- 平均涨跌幅：1.70%
- 中位涨跌幅：1.05%
- 平均上涨幅度：6.33%
- 平均下跌幅度：-4.49%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.04563
- corr_RiskPenalty_return：-0.004855
- corr_EV_up：0.032325
- corr_RiskPenalty_down：0.002129
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
