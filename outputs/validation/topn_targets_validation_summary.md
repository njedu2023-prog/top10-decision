# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-15T16:50:03
- 原始预测样本数：530
- 已验证样本数：499
- 未验证样本数：31
- 上涨数量：300
- 上涨率：60.12%
- 涨停数量：131
- 涨停率：26.25%
- 下跌数量：184
- 下跌率：36.87%
- 平均涨跌幅：2.36%
- 中位涨跌幅：1.43%
- 平均上涨幅度：6.35%
- 平均下跌幅度：-3.95%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：-0.018849
- corr_RiskPenalty_return：0.016921
- corr_EV_up：-0.015502
- corr_RiskPenalty_down：-0.014604
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
