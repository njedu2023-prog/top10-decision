# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-08-10T15:58:37
- 原始预测样本数：1130
- 已验证样本数：1096
- 未验证样本数：34
- 上涨数量：635
- 上涨率：57.94%
- 涨停数量：250
- 涨停率：22.81%
- 下跌数量：442
- 下跌率：40.33%
- 平均涨跌幅：1.92%
- 中位涨跌幅：1.35%
- 平均上涨幅度：6.37%
- 平均下跌幅度：-4.38%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.040598
- corr_RiskPenalty_return：0.022558
- corr_EV_up：0.02514
- corr_RiskPenalty_down：-0.027006
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
