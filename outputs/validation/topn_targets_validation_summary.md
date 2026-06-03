# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-06-03T18:43:32
- 原始预测样本数：660
- 已验证样本数：628
- 未验证样本数：32
- 上涨数量：368
- 上涨率：58.60%
- 涨停数量：164
- 涨停率：26.11%
- 下跌数量：244
- 下跌率：38.85%
- 平均涨跌幅：2.22%
- 中位涨跌幅：1.37%
- 平均上涨幅度：6.55%
- 平均下跌幅度：-4.16%
- 最大涨幅：20.21%
- 最大跌幅：-15.31%

## EV / RiskPenalty 相关性

- corr_EV_return：-0.010159
- corr_RiskPenalty_return：0.029385
- corr_EV_up：-0.006058
- corr_RiskPenalty_down：-0.040558
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
