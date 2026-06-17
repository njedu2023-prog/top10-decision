# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-06-17T17:50:11
- 原始预测样本数：760
- 已验证样本数：726
- 未验证样本数：34
- 上涨数量：427
- 上涨率：58.82%
- 涨停数量：180
- 涨停率：24.79%
- 下跌数量：283
- 下跌率：38.98%
- 平均涨跌幅：2.13%
- 中位涨跌幅：1.33%
- 平均上涨幅度：6.40%
- 平均下跌幅度：-4.20%
- 最大涨幅：20.21%
- 最大跌幅：-15.31%

## EV / RiskPenalty 相关性

- corr_EV_return：0.021154
- corr_RiskPenalty_return：0.017879
- corr_EV_up：0.032386
- corr_RiskPenalty_down：-0.043935
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
