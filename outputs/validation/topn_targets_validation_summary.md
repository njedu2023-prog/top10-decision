# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-08-20T15:32:19
- 原始预测样本数：1210
- 已验证样本数：1176
- 未验证样本数：34
- 上涨数量：666
- 上涨率：56.63%
- 涨停数量：266
- 涨停率：22.62%
- 下跌数量：491
- 下跌率：41.75%
- 平均涨跌幅：1.77%
- 中位涨跌幅：1.02%
- 平均上涨幅度：6.38%
- 平均下跌幅度：-4.41%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.041958
- corr_RiskPenalty_return：0.024733
- corr_EV_up：0.027551
- corr_RiskPenalty_down：-0.028028
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
