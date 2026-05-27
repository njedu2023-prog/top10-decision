# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-27T17:56:33
- 原始预测样本数：600
- 已验证样本数：568
- 未验证样本数：32
- 上涨数量：331
- 上涨率：58.27%
- 涨停数量：143
- 涨停率：25.18%
- 下跌数量：221
- 下跌率：38.91%
- 平均涨跌幅：2.12%
- 中位涨跌幅：1.04%
- 平均上涨幅度：6.35%
- 平均下跌幅度：-4.07%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：0.044628
- corr_RiskPenalty_return：0.003112
- corr_EV_up：0.033363
- corr_RiskPenalty_down：-0.023634
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
