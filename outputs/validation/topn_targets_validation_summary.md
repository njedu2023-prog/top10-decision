# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-11-26T23:32:33
- 原始预测样本数：1850
- 已验证样本数：1816
- 未验证样本数：34
- 上涨数量：1031
- 上涨率：56.77%
- 涨停数量：442
- 涨停率：24.34%
- 下跌数量：761
- 下跌率：41.91%
- 平均涨跌幅：1.77%
- 中位涨跌幅：1.20%
- 平均上涨幅度：6.45%
- 平均下跌幅度：-4.51%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.038317
- corr_RiskPenalty_return：0.007042
- corr_EV_up：0.028926
- corr_RiskPenalty_down：-0.000414
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
