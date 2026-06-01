# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-06-01T19:55:41
- 原始预测样本数：640
- 已验证样本数：608
- 未验证样本数：32
- 上涨数量：352
- 上涨率：57.89%
- 涨停数量：154
- 涨停率：25.33%
- 下跌数量：240
- 下跌率：39.47%
- 平均涨跌幅：2.10%
- 中位涨跌幅：1.08%
- 平均上涨幅度：6.46%
- 平均下跌幅度：-4.16%
- 最大涨幅：20.02%
- 最大跌幅：-15.31%

## EV / RiskPenalty 相关性

- corr_EV_return：0.024312
- corr_RiskPenalty_return：0.011144
- corr_EV_up：0.01929
- corr_RiskPenalty_down：-0.030053
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
