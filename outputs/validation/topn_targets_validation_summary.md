# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-07-31T16:45:40
- 原始预测样本数：1070
- 已验证样本数：1036
- 未验证样本数：34
- 上涨数量：590
- 上涨率：56.95%
- 涨停数量：237
- 涨停率：22.88%
- 下跌数量：429
- 下跌率：41.41%
- 平均涨跌幅：1.85%
- 中位涨跌幅：1.09%
- 平均上涨幅度：6.44%
- 平均下跌幅度：-4.39%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.044146
- corr_RiskPenalty_return：0.015929
- corr_EV_up：0.03083
- corr_RiskPenalty_down：-0.015364
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
