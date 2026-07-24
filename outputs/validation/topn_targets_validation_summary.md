# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-07-24T16:39:58
- 原始预测样本数：1020
- 已验证样本数：986
- 未验证样本数：34
- 上涨数量：568
- 上涨率：57.61%
- 涨停数量：228
- 涨停率：23.12%
- 下跌数量：402
- 下跌率：40.77%
- 平均涨跌幅：1.92%
- 中位涨跌幅：1.25%
- 平均上涨幅度：6.45%
- 平均下跌幅度：-4.41%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.042767
- corr_RiskPenalty_return：0.020329
- corr_EV_up：0.029062
- corr_RiskPenalty_down：-0.018693
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
