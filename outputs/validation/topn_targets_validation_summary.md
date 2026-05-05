# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-05T16:21:56
- 原始预测样本数：450
- 已验证样本数：420
- 未验证样本数：30
- 上涨数量：246
- 上涨率：58.57%
- 涨停数量：112
- 涨停率：26.67%
- 下跌数量：161
- 下跌率：38.33%
- 平均涨跌幅：2.23%
- 中位涨跌幅：1.04%
- 平均上涨幅度：6.34%
- 平均下跌幅度：-3.87%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：0.039529
- corr_RiskPenalty_return：0.011082
- corr_EV_up：0.067308
- corr_RiskPenalty_down：-0.009685
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
