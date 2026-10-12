# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-10-12T22:28:06
- 原始预测样本数：1520
- 已验证样本数：1486
- 未验证样本数：34
- 上涨数量：838
- 上涨率：56.39%
- 涨停数量：338
- 涨停率：22.75%
- 下跌数量：628
- 下跌率：42.26%
- 平均涨跌幅：1.69%
- 中位涨跌幅：0.92%
- 平均上涨幅度：6.29%
- 平均下跌幅度：-4.40%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.046474
- corr_RiskPenalty_return：-0.000283
- corr_EV_up：0.031236
- corr_RiskPenalty_down：-0.004819
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
