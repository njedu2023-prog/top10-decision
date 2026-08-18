# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-08-18T15:29:51
- 原始预测样本数：1190
- 已验证样本数：1156
- 未验证样本数：34
- 上涨数量：660
- 上涨率：57.09%
- 涨停数量：265
- 涨停率：22.92%
- 下跌数量：477
- 下跌率：41.26%
- 平均涨跌幅：1.85%
- 中位涨跌幅：1.13%
- 平均上涨幅度：6.40%
- 平均下跌幅度：-4.36%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.041376
- corr_RiskPenalty_return：0.021923
- corr_EV_up：0.027145
- corr_RiskPenalty_down：-0.02507
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
