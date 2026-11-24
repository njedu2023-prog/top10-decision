# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-11-24T23:52:12
- 原始预测样本数：1830
- 已验证样本数：1796
- 未验证样本数：34
- 上涨数量：1016
- 上涨率：56.57%
- 涨停数量：431
- 涨停率：24.00%
- 下跌数量：757
- 下跌率：42.15%
- 平均涨跌幅：1.73%
- 中位涨跌幅：1.13%
- 平均上涨幅度：6.42%
- 平均下跌幅度：-4.50%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.041097
- corr_RiskPenalty_return：-0.000885
- corr_EV_up：0.030596
- corr_RiskPenalty_down：0.005709
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
