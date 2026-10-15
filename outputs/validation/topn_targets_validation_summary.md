# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-10-15T18:56:45
- 原始预测样本数：1550
- 已验证样本数：1516
- 未验证样本数：34
- 上涨数量：851
- 上涨率：56.13%
- 涨停数量：344
- 涨停率：22.69%
- 下跌数量：644
- 下跌率：42.48%
- 平均涨跌幅：1.65%
- 中位涨跌幅：0.91%
- 平均上涨幅度：6.29%
- 平均下跌幅度：-4.43%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.047734
- corr_RiskPenalty_return：-0.005143
- corr_EV_up：0.032785
- corr_RiskPenalty_down：-0.000591
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
