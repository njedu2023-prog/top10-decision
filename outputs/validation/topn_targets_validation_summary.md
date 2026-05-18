# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-05-18T17:29:48
- 原始预测样本数：540
- 已验证样本数：509
- 未验证样本数：31
- 上涨数量：304
- 上涨率：59.73%
- 涨停数量：131
- 涨停率：25.74%
- 下跌数量：190
- 下跌率：37.33%
- 平均涨跌幅：2.29%
- 中位涨跌幅：1.42%
- 平均上涨幅度：6.31%
- 平均下跌幅度：-3.97%
- 最大涨幅：20.02%
- 最大跌幅：-13.65%

## EV / RiskPenalty 相关性

- corr_EV_return：-0.009779
- corr_RiskPenalty_return：0.017749
- corr_EV_up：-0.010139
- corr_RiskPenalty_down：-0.018848
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
