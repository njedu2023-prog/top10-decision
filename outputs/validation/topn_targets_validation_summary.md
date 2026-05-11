# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-11T17:23:02
- 原始预测样本数：490
- 已验证样本数：459
- 未验证样本数：31
- 上涨数量：277
- 上涨率：60.35%
- 涨停数量：120
- 涨停率：26.14%
- 下跌数量：167
- 下跌率：36.38%
- 平均涨跌幅：2.39%
- 中位涨跌幅：1.45%
- 平均上涨幅度：6.27%
- 平均下跌幅度：-3.83%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：-0.015297
- corr_RiskPenalty_return：0.021327
- corr_EV_up：-0.012716
- corr_RiskPenalty_down：-0.019268
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
