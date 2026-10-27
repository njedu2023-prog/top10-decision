# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-10-27T19:51:40
- 原始预测样本数：1630
- 已验证样本数：1596
- 未验证样本数：34
- 上涨数量：899
- 上涨率：56.33%
- 涨停数量：368
- 涨停率：23.06%
- 下跌数量：675
- 下跌率：42.29%
- 平均涨跌幅：1.66%
- 中位涨跌幅：0.97%
- 平均上涨幅度：6.30%
- 平均下跌幅度：-4.47%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.046829
- corr_RiskPenalty_return：-0.005913
- corr_EV_up：0.033059
- corr_RiskPenalty_down：0.004016
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
