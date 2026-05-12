# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-12T17:06:10
- 原始预测样本数：500
- 已验证样本数：469
- 未验证样本数：31
- 上涨数量：285
- 上涨率：60.77%
- 涨停数量：124
- 涨停率：26.44%
- 下跌数量：169
- 下跌率：36.03%
- 平均涨跌幅：2.45%
- 中位涨跌幅：1.45%
- 平均上涨幅度：6.28%
- 平均下跌幅度：-3.80%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：-0.023279
- corr_RiskPenalty_return：0.020017
- corr_EV_up：-0.019454
- corr_RiskPenalty_down：-0.020217
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
