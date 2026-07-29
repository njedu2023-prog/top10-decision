# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-07-29T16:23:06
- 原始预测样本数：1050
- 已验证样本数：1016
- 未验证样本数：34
- 上涨数量：584
- 上涨率：57.48%
- 涨停数量：234
- 涨停率：23.03%
- 下跌数量：415
- 下跌率：40.85%
- 平均涨跌幅：1.89%
- 中位涨跌幅：1.23%
- 平均上涨幅度：6.44%
- 平均下跌幅度：-4.44%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.04351
- corr_RiskPenalty_return：0.018207
- corr_EV_up：0.02955
- corr_RiskPenalty_down：-0.018743
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
