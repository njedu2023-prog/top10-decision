# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-08-14T15:49:27
- 原始预测样本数：1170
- 已验证样本数：1136
- 未验证样本数：34
- 上涨数量：651
- 上涨率：57.31%
- 涨停数量：261
- 涨停率：22.98%
- 下跌数量：466
- 下跌率：41.02%
- 平均涨跌幅：1.87%
- 中位涨跌幅：1.18%
- 平均上涨幅度：6.39%
- 平均下跌幅度：-4.37%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.041336
- corr_RiskPenalty_return：0.021521
- corr_EV_up：0.026767
- corr_RiskPenalty_down：-0.024856
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
