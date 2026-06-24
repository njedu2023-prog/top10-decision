# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-06-24T17:04:35
- 原始预测样本数：800
- 已验证样本数：766
- 未验证样本数：34
- 上涨数量：454
- 上涨率：59.27%
- 涨停数量：189
- 涨停率：24.67%
- 下跌数量：296
- 下跌率：38.64%
- 平均涨跌幅：2.21%
- 中位涨跌幅：1.43%
- 平均上涨幅度：6.47%
- 平均下跌幅度：-4.20%
- 最大涨幅：20.21%
- 最大跌幅：-15.31%

## EV / RiskPenalty 相关性

- corr_EV_return：0.031813
- corr_RiskPenalty_return：0.067601
- corr_EV_up：0.029304
- corr_RiskPenalty_down：-0.072541
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
