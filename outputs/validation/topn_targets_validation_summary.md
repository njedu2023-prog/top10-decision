# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-06-29T17:33:55
- 原始预测样本数：830
- 已验证样本数：796
- 未验证样本数：34
- 上涨数量：469
- 上涨率：58.92%
- 涨停数量：192
- 涨停率：24.12%
- 下跌数量：311
- 下跌率：39.07%
- 平均涨跌幅：2.15%
- 中位涨跌幅：1.41%
- 平均上涨幅度：6.46%
- 平均下跌幅度：-4.23%
- 最大涨幅：20.21%
- 最大跌幅：-15.31%

## EV / RiskPenalty 相关性

- corr_EV_return：0.03194
- corr_RiskPenalty_return：0.049529
- corr_EV_up：0.025747
- corr_RiskPenalty_down：-0.053027
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
