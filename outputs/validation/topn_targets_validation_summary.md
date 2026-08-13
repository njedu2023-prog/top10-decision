# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-08-13T15:57:58
- 原始预测样本数：1160
- 已验证样本数：1126
- 未验证样本数：34
- 上涨数量：646
- 上涨率：57.37%
- 涨停数量：257
- 涨停率：22.82%
- 下跌数量：461
- 下跌率：40.94%
- 平均涨跌幅：1.88%
- 中位涨跌幅：1.18%
- 平均上涨幅度：6.38%
- 平均下跌幅度：-4.36%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.041484
- corr_RiskPenalty_return：0.021835
- corr_EV_up：0.026684
- corr_RiskPenalty_down：-0.02508
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
