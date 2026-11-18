# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-11-18T22:31:51
- 原始预测样本数：1790
- 已验证样本数：1756
- 未验证样本数：34
- 上涨数量：996
- 上涨率：56.72%
- 涨停数量：421
- 涨停率：23.97%
- 下跌数量：737
- 下跌率：41.97%
- 平均涨跌幅：1.75%
- 中位涨跌幅：1.14%
- 平均上涨幅度：6.39%
- 平均下跌幅度：-4.47%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.040756
- corr_RiskPenalty_return：0.002824
- corr_EV_up：0.029929
- corr_RiskPenalty_down：0.001837
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
