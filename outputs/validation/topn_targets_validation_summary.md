# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-19T17:37:23
- 原始预测样本数：550
- 已验证样本数：518
- 未验证样本数：32
- 上涨数量：309
- 上涨率：59.65%
- 涨停数量：134
- 涨停率：25.87%
- 下跌数量：193
- 下跌率：37.26%
- 平均涨跌幅：2.33%
- 中位涨跌幅：1.41%
- 平均上涨幅度：6.36%
- 平均下跌幅度：-3.94%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：-0.016408
- corr_RiskPenalty_return：0.026047
- corr_EV_up：-0.008196
- corr_RiskPenalty_down：-0.024916
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
