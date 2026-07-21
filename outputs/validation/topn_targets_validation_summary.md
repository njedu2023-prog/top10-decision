# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-07-21T16:22:23
- 原始预测样本数：990
- 已验证样本数：956
- 未验证样本数：34
- 上涨数量：550
- 上涨率：57.53%
- 涨停数量：220
- 涨停率：23.01%
- 下跌数量：390
- 下跌率：40.80%
- 平均涨跌幅：1.90%
- 中位涨跌幅：1.25%
- 平均上涨幅度：6.47%
- 平均下跌幅度：-4.47%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.045934
- corr_RiskPenalty_return：0.012053
- corr_EV_up：0.031695
- corr_RiskPenalty_down：-0.010341
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
