# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-11-12T21:49:04
- 原始预测样本数：1750
- 已验证样本数：1716
- 未验证样本数：34
- 上涨数量：973
- 上涨率：56.70%
- 涨停数量：407
- 涨停率：23.72%
- 下跌数量：720
- 下跌率：41.96%
- 平均涨跌幅：1.75%
- 中位涨跌幅：1.10%
- 平均上涨幅度：6.36%
- 平均下跌幅度：-4.43%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.041062
- corr_RiskPenalty_return：0.005413
- corr_EV_up：0.030212
- corr_RiskPenalty_down：-0.000135
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
