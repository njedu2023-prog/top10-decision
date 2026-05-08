# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-08T16:12:19
- 原始预测样本数：480
- 已验证样本数：450
- 未验证样本数：30
- 上涨数量：269
- 上涨率：59.78%
- 涨停数量：117
- 涨停率：26.00%
- 下跌数量：166
- 下跌率：36.89%
- 平均涨跌幅：2.33%
- 中位涨跌幅：1.42%
- 平均上涨幅度：6.27%
- 平均下跌幅度：-3.84%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：-0.005605
- corr_RiskPenalty_return：0.016462
- corr_EV_up：-0.001672
- corr_RiskPenalty_down：-0.015939
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
