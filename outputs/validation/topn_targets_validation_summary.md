# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-11-13T21:15:09
- 原始预测样本数：1760
- 已验证样本数：1726
- 未验证样本数：34
- 上涨数量：979
- 上涨率：56.72%
- 涨停数量：411
- 涨停率：23.81%
- 下跌数量：724
- 下跌率：41.95%
- 平均涨跌幅：1.75%
- 中位涨跌幅：1.12%
- 平均上涨幅度：6.37%
- 平均下跌幅度：-4.44%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.040736
- corr_RiskPenalty_return：0.005811
- corr_EV_up：0.029908
- corr_RiskPenalty_down：-0.000743
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
