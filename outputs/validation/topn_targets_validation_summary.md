# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-07T16:47:03
- 原始预测样本数：470
- 已验证样本数：440
- 未验证样本数：30
- 上涨数量：262
- 上涨率：59.55%
- 涨停数量：116
- 涨停率：26.36%
- 下跌数量：164
- 下跌率：37.27%
- 平均涨跌幅：2.34%
- 中位涨跌幅：1.31%
- 平均上涨幅度：6.34%
- 平均下跌幅度：-3.85%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：-0.005372
- corr_RiskPenalty_return：0.012577
- corr_EV_up：0.007991
- corr_RiskPenalty_down：-0.007896
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
