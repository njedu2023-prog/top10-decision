# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-08-26T16:17:46
- 原始预测样本数：1250
- 已验证样本数：1216
- 未验证样本数：34
- 上涨数量：686
- 上涨率：56.41%
- 涨停数量：276
- 涨停率：22.70%
- 下跌数量：511
- 下跌率：42.02%
- 平均涨跌幅：1.76%
- 中位涨跌幅：0.96%
- 平均上涨幅度：6.38%
- 平均下跌幅度：-4.38%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.04235
- corr_RiskPenalty_return：0.022277
- corr_EV_up：0.028371
- corr_RiskPenalty_down：-0.024266
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
