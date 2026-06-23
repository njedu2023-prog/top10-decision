# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-06-23T17:15:15
- 原始预测样本数：790
- 已验证样本数：756
- 未验证样本数：34
- 上涨数量：448
- 上涨率：59.26%
- 涨停数量：188
- 涨停率：24.87%
- 下跌数量：292
- 下跌率：38.62%
- 平均涨跌幅：2.22%
- 中位涨跌幅：1.42%
- 平均上涨幅度：6.48%
- 平均下跌幅度：-4.19%
- 最大涨幅：20.21%
- 最大跌幅：-15.31%

## EV / RiskPenalty 相关性

- corr_EV_return：0.029328
- corr_RiskPenalty_return：0.072119
- corr_EV_up：0.027075
- corr_RiskPenalty_down：-0.070754
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
