# top10-decision 最终 weights Top10 后验验证摘要

- 数据源：docs/weights/weights_YYYYMMDD.csv::target_rank_then_backup_rank
- 生成时间：2026-07-02T16:49:21
- 原始预测样本数：860
- 已验证样本数：826
- 未验证样本数：34
- 上涨数量：486
- 上涨率：58.84%
- 涨停数量：198
- 涨停率：23.97%
- 下跌数量：324
- 下跌率：39.23%
- 平均涨跌幅：2.17%
- 中位涨跌幅：1.42%
- 平均上涨幅度：6.49%
- 平均下跌幅度：-4.22%
- 最大涨幅：20.21%
- 最大跌幅：-15.31%

## EV / RiskPenalty 相关性

- corr_EV_return：0.021815
- corr_RiskPenalty_return：0.052805
- corr_EV_up：0.008647
- corr_RiskPenalty_down：-0.046779
- 样本说明：样本>=300，可作为较稳定评估依据

已生成基于 docs/weights 最终排序的后验验证统计；核心看真实上涨率、涨停率、平均/中位涨跌幅、最大涨跌幅。
