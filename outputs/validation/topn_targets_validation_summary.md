# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-04T16:38:03
- 原始预测样本数：410
- 已验证样本数：400
- 未验证样本数：10
- 上涨数量：241
- 上涨率：60.25%
- 涨停数量：109
- 涨停率：27.25%
- 下跌数量：156
- 下跌率：39.00%
- 平均涨跌幅：2.26%
- 中位涨跌幅：1.31%
- 平均上涨幅度：6.29%
- 平均下跌幅度：-3.92%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：0.046962
- corr_RiskPenalty_return：-0.002972
- corr_EV_up：0.036797
- corr_RiskPenalty_down：-0.036657
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
