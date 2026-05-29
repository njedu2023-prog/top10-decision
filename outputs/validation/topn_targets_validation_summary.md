# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-29T18:10:33
- 原始预测样本数：630
- 已验证样本数：598
- 未验证样本数：32
- 上涨数量：348
- 上涨率：58.19%
- 涨停数量：152
- 涨停率：25.42%
- 下跌数量：234
- 下跌率：39.13%
- 平均涨跌幅：2.11%
- 中位涨跌幅：1.12%
- 平均上涨幅度：6.42%
- 平均下跌幅度：-4.15%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：0.021013
- corr_RiskPenalty_return：0.013083
- corr_EV_up：0.013618
- corr_RiskPenalty_down：-0.034112
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
