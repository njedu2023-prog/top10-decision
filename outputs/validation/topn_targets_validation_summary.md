# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-09-09T18:19:47
- 原始预测样本数：1350
- 已验证样本数：1316
- 未验证样本数：34
- 上涨数量：743
- 上涨率：56.46%
- 涨停数量：303
- 涨停率：23.02%
- 下跌数量：553
- 下跌率：42.02%
- 平均涨跌幅：1.77%
- 中位涨跌幅：0.96%
- 平均上涨幅度：6.38%
- 平均下跌幅度：-4.37%
- 最大涨幅：20.21%
- 最大跌幅：-29.99%

## EV / RiskPenalty 相关性

- corr_EV_return：0.043688
- corr_RiskPenalty_return：0.010746
- corr_EV_up：0.029973
- corr_RiskPenalty_down：-0.013074
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
