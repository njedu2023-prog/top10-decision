# TopN Targets 验证系统需求规格 V1

> 所属仓库：`njedu2023-prog/top10-decision`  
> 模块定位：TopN Targets 后验验证与统计分析模块  
> 版本：V1  
> 编写日期：2026-05-04  
> 核心原则：**独立新增、非侵入式、不扰乱原预测主链路、不改变原 TopN Targets 输出逻辑**

---

## 0. 背景与目标

当前 `top10-decision` 系统每天会生成 `TopN Targets` 预测名单，用于展示系统认为具备较高执行价值和收益期望的股票标的。

本模块的目标不是修改原预测模型，也不是改变 TopN Targets 的生成逻辑，而是在原系统旁边新增一套**独立后验验证系统**：

- D 日：读取当日已生成的 TopN Targets 预测名单；
- T 日：严格按照 A 股交易日历，找到 D 日之后的下一个真实交易日；
- T 日收盘后：验证 D 日 TopN Targets 中每只股票在 T 日的真实涨跌表现；
- 长期累计：统计 TopN Targets 的历史上涨率、涨停率、下跌率、平盘率、平均收益、中位收益；
- 进一步分析：验证 `EV`、`RiskPenalty`、TopN 排名等字段与真实上涨/收益之间的关系。

最终目标是把 TopN Targets 从“每日预测输出”升级为“可验证、可统计、可回测、可校准”的预测系统组件，为后续实盘选 Top1、Top2、Top4、Top10 提供数据证据。

---

## 1. 总体设计原则

### 1.1 非侵入式原则

本模块必须独立新增，不破坏现有预测系统。

禁止：

- 修改 TopN Targets 原有生成逻辑；
- 修改 `decision_candidates_YYYYMMDD.csv` 的原始字段结构；
- 修改原有 `decision_report_YYYYMMDD.md` 的生成逻辑；
- 修改原有 `decision.html` 的主展示逻辑；
- 改动 P_fill、E_ret、Cost、RiskPenalty、EV 主计算逻辑。

允许：

- 新增独立 Python 文件；
- 新增独立 HTML 页面；
- 新增独立输出目录；
- 在 GitHub Actions 中新增独立执行步骤或新增独立 workflow；
- 从原有输出文件读取数据进行验证，但不得反向污染原文件。

---

## 2. 推荐新增文件结构

建议新增以下文件：

```text
scripts/validate_topn_targets.py
scripts/backfill_topn_targets_validation.py

outputs/validation/topn_targets_validation_latest.csv
outputs/validation/topn_targets_validation_history.csv
outputs/validation/topn_targets_validation_summary.json
outputs/validation/topn_targets_validation_summary.md
outputs/validation/topn_targets_validation_by_date.csv
outputs/validation/topn_targets_validation_by_rank.csv
outputs/validation/topn_targets_validation_by_ev_risk.csv

docs/topn_targets_validation.html
```

说明：

- `validate_topn_targets.py`：每日增量验证入口；
- `backfill_topn_targets_validation.py`：历史一次性回放验证入口；
- `topn_targets_validation_latest.csv`：最近一次验证结果；
- `topn_targets_validation_history.csv`：历史累计逐票验证明细；
- `topn_targets_validation_summary.json/md`：累计统计摘要；
- `topn_targets_validation_by_date.csv`：按 D 日分组统计；
- `topn_targets_validation_by_rank.csv`：按 TopN 排名分组统计；
- `topn_targets_validation_by_ev_risk.csv`：按 EV × RiskPenalty 分层统计；
- `docs/topn_targets_validation.html`：独立 HTML 展示页面。

---

## 3. 核心时间轴定义

### 3.1 D 日

D 日是 TopN Targets 预测名单生成日。

例如：

```text
D = 2026-05-06
```

对应文件可能包括：

```text
data/decision/decision_candidates_20260506.csv
outputs/decision/decision_report_20260506.md
```

### 3.2 T 日

T 日是 D 日之后的下一个真实 A 股交易日。

例如：

```text
D = 2026-05-06
T = 2026-05-07
```

如果 D 日之后遇到周末、法定节假日、调休休市，则 T 日必须自动跳转到下一个真实 A 股交易日。

### 3.3 严格交易日历约束

本模块必须严格使用 A 股交易日历，不允许按自然日 `D + 1` 简单推断。

硬性要求：

- D 日必须是 A 股交易日；
- T 日必须是 D 日后的下一个 A 股交易日；
- 若 T 日行情数据尚不存在，则该 D 日状态为“待验证”；
- 若个股停牌、缺行情、数据异常，必须标注，不得静默填 0；
- 所有日期展示必须使用标准格式，例如：

```text
D：2026-05-06 → T：2026-05-07
```

---

## 4. 数据输入

### 4.1 TopN Targets 来源

优先从以下文件读取 D 日候选数据：

```text
data/decision/decision_candidates_YYYYMMDD.csv
```

筛选逻辑：

- 读取 D 日 `decision_candidates`；
- 按原系统 TopN Targets 的规则识别 TopN；
- 保持原 TopN Targets 排序不变；
- 不重新排序；
- 不重新打分。

如果现有文件中已有明确字段可识别 TopN，例如：

```text
is_topn
rank
decision_rank
EV_rank
```

则优先使用已有字段。

如果没有明确字段，则按照原报告 TopN Targets 的排序逻辑筛选前 N 名，默认 N=10。

### 4.2 行情数据来源

行情数据应优先复用当前仓库或上游数据仓库已经存在的标准行情文件。

可选来源包括：

```text
data/market/
data/pred/archive/
data/raw/
上游 a-share-top3-data 快照
```

必须读取到至少以下字段：

| 字段 | 含义 |
|---|---|
| ts_code | 股票代码 |
| trade_date | 交易日 |
| close | 收盘价 |
| pre_close 或 D_close | 前收盘价 / D 日收盘价 |
| pct_chg | 涨跌幅 |
| up_limit | 涨停价，若有 |
| down_limit | 跌停价，若有 |
| name | 股票名称，若有 |

若无法读取 `pct_chg`，则按以下方式计算：

```text
T_return_pct = (T_close / D_close - 1) × 100
```

---

## 5. 验证明细表字段设计

验证表必须保留 D 日 TopN Targets 的全部原字段，并在后面追加验证字段。

### 5.1 原字段保留

原 TopN Targets 字段全部保留，包括但不限于：

```text
ts_code
name
P_fill
E_ret
Cost
RiskPenalty
EV
rank
weight
reason
```

具体字段以 D 日 `decision_candidates_YYYYMMDD.csv` 实际字段为准。

### 5.2 新增验证字段

建议追加以下字段：

| 字段名 | 含义 |
|---|---|
| D_trade_date | 预测日 |
| T_trade_date | 验证日 |
| D_close | D 日收盘价 |
| T_close | T 日收盘价 |
| T_return_pct | T 日相对 D 日涨跌幅 |
| T_result | 上涨 / 下跌 / 平盘 / 涨停 |
| T_limit_hit | 是否涨停 |
| T_down_limit_hit | 是否跌停 |
| validation_status | 已验证 / 待验证 / 缺行情 / 停牌 / 异常 |
| validation_note | 验证说明 |

### 5.3 涨跌分类口径

建议口径：

```text
上涨：T_return_pct > 0.05%
下跌：T_return_pct < -0.05%
平盘：-0.05% <= T_return_pct <= 0.05%
涨停：T_close >= up_limit × 0.999，或 T_return_pct 达到当日涨停规则附近
跌停：T_close <= down_limit × 1.001，或 T_return_pct 达到当日跌停规则附近
```

如缺少 `up_limit/down_limit` 字段，可按股票交易规则估算，但必须标注：

```text
涨停状态：估算
```

---

## 6. 页面展示要求

### 6.1 独立 HTML 页面

新增独立页面：

```text
docs/topn_targets_validation.html
```

该页面不替代原 `decision.html`，只作为 TopN Targets 验证页面。

### 6.2 页面主要区块

页面建议包括以下区块：

1. 页面标题；
2. 最新验证日期；
3. D→T 交易日对标说明；
4. 最新一期 TopN Targets 验证明细表；
5. 历史累计统计表；
6. 按日期统计表；
7. 按 TopN 排名统计表；
8. EV 分层统计表；
9. RiskPenalty 分层统计表；
10. EV × RiskPenalty 二维组合统计表；
11. 数据缺失与异常说明。

### 6.3 涨跌颜色规则

A 股展示习惯：

- 上涨：红色；
- 下跌：绿色；
- 平盘：灰色；
- 涨停：红色加粗，可标注“涨停”；
- 跌停：绿色加粗，可标注“跌停”。

示例：

```text
+3.25%    红色
-2.10%    绿色
0.00%     灰色
+10.01%   红色加粗，标注涨停
```

---

## 7. 累计统计表

### 7.1 总累计统计

必须输出：

| 指标 | 说明 |
|---|---|
| 历史累计样本总数 | 所有已验证股票行数 |
| 上涨数量 | T_return_pct > 0 |
| 上涨率 | 上涨数量 / 总样本数 |
| 涨停数量 | T_limit_hit = True |
| 涨停率 | 涨停数量 / 总样本数 |
| 下跌数量 | T_return_pct < 0 |
| 下跌率 | 下跌数量 / 总样本数 |
| 平盘数量 | T_return_pct ≈ 0 |
| 平盘率 | 平盘数量 / 总样本数 |
| 平均涨跌幅 | T_return_pct 均值 |
| 中位涨跌幅 | T_return_pct 中位数 |
| 最大涨幅 | 样本最大 T_return_pct |
| 最大跌幅 | 样本最小 T_return_pct |

### 7.2 按 D 日统计

按 D 日统计每天 TopN 的整体表现：

| D日 | T日 | TopN数量 | 上涨数 | 上涨率 | 涨停数 | 涨停率 | 平均涨跌幅 | 中位涨跌幅 |
|---|---|---:|---:|---:|---:|---:|---:|---:|

该表用于判断每一天的 TopN Targets 是否整体有效。

---

## 8. TopN 排名有效性统计

必须新增排名层级统计。

建议统计：

| 排名层级 | 样本数 | 上涨率 | 涨停率 | 平均涨跌幅 | 中位涨跌幅 |
|---|---:|---:|---:|---:|---:|
| Top1 |  |  |  |  |  |
| Top1-2 |  |  |  |  |  |
| Top1-3 |  |  |  |  |  |
| Top1-4 |  |  |  |  |  |
| Top1-5 |  |  |  |  |  |
| Top6-10 |  |  |  |  |  |

该表用于回答实盘问题：

- 只买 Top1 是否更优？
- 买前 2 支是否比买前 4 支更稳？
- Top10 后排是否拉低整体胜率？
- 前排排序是否具备真实解释力？

---

## 9. EV 与 RiskPenalty 分层统计

### 9.1 专业口径修正

本模块不应简单假设 EV 和 RiskPenalty 都与上涨正相关。

正确假设应为：

```text
EV 越高，理论上越应对应更高上涨率或更高平均收益；
RiskPenalty 越高，理论上越应对应更高风险、更低稳定性或更大回撤概率；
最优组合通常应是：高 EV + 低 RiskPenalty。
```

### 9.2 EV 三阶梯统计

按历史样本的 EV 分布三分位分组：

| EV分组 | EV范围 | 样本数 | 上涨率 | 涨停率 | 平均涨跌幅 | 中位涨跌幅 |
|---|---|---:|---:|---:|---:|---:|
| 低EV组 | P0-P33 |  |  |  |  |  |
| 中EV组 | P33-P66 |  |  |  |  |  |
| 高EV组 | P66-P100 |  |  |  |  |  |

### 9.3 RiskPenalty 三阶梯统计

按历史样本的 RiskPenalty 分布三分位分组：

| RiskPenalty分组 | RiskPenalty范围 | 样本数 | 上涨率 | 涨停率 | 平均涨跌幅 | 中位涨跌幅 |
|---|---|---:|---:|---:|---:|---:|
| 低风险组 | P0-P33 |  |  |  |  |  |
| 中风险组 | P33-P66 |  |  |  |  |  |
| 高风险组 | P66-P100 |  |  |  |  |  |

### 9.4 EV × RiskPenalty 二维组合统计

建议输出二维组合表：

| 组合 | 样本数 | 上涨率 | 涨停率 | 平均涨跌幅 | 结论 |
|---|---:|---:|---:|---:|---|
| 高EV + 低Risk |  |  |  |  | 最优候选池 |
| 高EV + 中Risk |  |  |  |  | 可观察 |
| 高EV + 高Risk |  |  |  |  | 高收益高波动 |
| 中EV + 低Risk |  |  |  |  | 稳健但弹性不足 |
| 中EV + 中Risk |  |  |  |  | 中性 |
| 中EV + 高Risk |  |  |  |  | 风险偏高 |
| 低EV + 低Risk |  |  |  |  | 防守型但弹性不足 |
| 低EV + 中Risk |  |  |  |  | 弱解释力 |
| 低EV + 高Risk |  |  |  |  | 应弱化或剔除 |

---

## 10. 相关性与解释力分析

### 10.1 相关系数

建议计算：

| 指标 | 说明 |
|---|---|
| corr_EV_return | EV 与 T_return_pct 的 Pearson 相关系数 |
| corr_EV_up | EV 与是否上涨的 Spearman 或点二列相关 |
| corr_Risk_return | RiskPenalty 与 T_return_pct 的 Pearson 相关系数 |
| corr_Risk_down | RiskPenalty 与是否下跌的相关系数 |

### 10.2 注意事项

当历史样本不足时，相关系数容易失真。

建议设置最低样本阈值：

```text
总样本数 < 30：只展示，不下结论；
30 <= 总样本数 < 100：展示初步趋势；
总样本数 >= 100：可输出阶段性统计判断；
总样本数 >= 300：可作为较稳定评估依据。
```

---

## 11. 历史一次性回放验证

### 11.1 是否可实现

可以实现。

建议单独新增：

```text
scripts/backfill_topn_targets_validation.py
```

该文件用于一次性扫描历史数据并生成累计验证结果。

### 11.2 回放逻辑

执行逻辑：

1. 扫描历史 `data/decision/decision_candidates_*.csv`；
2. 逐个解析 D 日；
3. 按 A 股交易日历找到 D 日之后的 T 日；
4. 检查 T 日行情是否存在；
5. 若行情存在，则计算 D→T 涨跌幅；
6. 若行情不存在，则标注为待验证或缺行情；
7. 汇总全部历史样本；
8. 输出历史验证明细和统计文件；
9. 生成 HTML 页面。

### 11.3 推荐命令

```bash
python scripts/backfill_topn_targets_validation.py
```

可选参数：

```bash
python scripts/backfill_topn_targets_validation.py --start-date 20260301 --end-date 20260504
python scripts/backfill_topn_targets_validation.py --force
python scripts/backfill_topn_targets_validation.py --topn 10
```

### 11.4 历史回放注意事项

历史回放能否完整取决于以下数据是否齐全：

- 历史 `decision_candidates_YYYYMMDD.csv` 是否存在；
- 历史 D 日 TopN Targets 是否能从 candidates 文件识别；
- D 日和 T 日行情数据是否存在；
- A 股交易日历是否能正确识别；
- 历史股票是否存在停牌、退市、名称变更、代码异常等情况。

若部分历史数据缺失，脚本不得报错中断，应输出：

```text
validation_status = 缺行情
validation_note = 未找到 T 日行情
```

---

## 12. 每日增量验证

建议新增：

```text
scripts/validate_topn_targets.py
```

每日执行逻辑：

1. 获取当前最新交易日；
2. 找出已经成熟但尚未验证的 D 日；
3. 读取 D 日 TopN Targets；
4. 找出对应 T 日；
5. 如果 T 日行情已经存在，则完成验证；
6. 追加写入历史验证文件；
7. 重新生成统计摘要；
8. 重新生成 HTML 页面。

每日增量验证应具备幂等性：

- 同一 D 日重复运行，不应重复追加；
- 已验证记录可覆盖更新；
- 历史汇总文件可重算生成；
- 不依赖手动删除旧文件。

---

## 13. GitHub Actions 集成建议

建议在现有 workflow 后面新增独立步骤，或新建独立 workflow。

### 13.1 推荐新建独立 workflow

```text
.github/workflows/validate_topn_targets.yml
```

优点：

- 不影响主预测链；
- 验证失败不阻断主预测输出；
- 便于单独手动触发；
- 便于历史回放。

### 13.2 执行时机

建议：

```text
北京时间 T 日 17:30 后运行
```

因为 T 日收盘数据需要完成更新后才能验证。

如果行情数据是仓库内数据快照，则应在上游数据更新完成后运行。

---

## 14. 输出文件定义

### 14.1 `topn_targets_validation_history.csv`

逐票历史明细。

每行 = 某 D 日 TopN 中的一只股票。

### 14.2 `topn_targets_validation_latest.csv`

最近完成验证的 D 日明细。

### 14.3 `topn_targets_validation_summary.json`

机器可读统计摘要。

### 14.4 `topn_targets_validation_summary.md`

人类可读统计摘要。

### 14.5 `topn_targets_validation_by_date.csv`

按 D 日聚合。

### 14.6 `topn_targets_validation_by_rank.csv`

按 TopN 排名聚合。

### 14.7 `topn_targets_validation_by_ev_risk.csv`

按 EV × RiskPenalty 组合聚合。

### 14.8 `docs/topn_targets_validation.html`

GitHub Pages 独立展示页面。

---

## 15. 异常处理规则

| 异常 | 处理方式 |
|---|---|
| D 日 candidates 文件不存在 | 跳过并记录 |
| 无法识别 TopN | 标注异常，不中断 |
| T 日交易日历不存在 | 标注交易日历错误 |
| T 日行情不存在 | 标注待验证或缺行情 |
| 个股停牌 | 标注停牌 |
| D_close 缺失 | 尝试从 D 日行情补齐 |
| T_close 缺失 | 标注缺行情 |
| up_limit 缺失 | 涨停判断标注为估算 |
| 字段名变动 | 使用兼容字段映射，不改原文件 |

---

## 16. 验收标准

本模块完成后，应满足以下验收条件：

1. 原 TopN Targets 预测系统不受影响；
2. 原有报告和信号文件正常生成；
3. 新增验证脚本可单独运行；
4. 新增历史回放脚本可一次性扫描历史数据；
5. 验证表保留 TopN Targets 全部原字段；
6. 验证表追加 T 日真实涨跌幅；
7. D→T 严格使用 A 股交易日历；
8. 上涨红色、下跌绿色、平盘灰色；
9. 输出累计上涨率、涨停率、下跌率、平盘率；
10. 输出按日期、排名、EV、RiskPenalty、EV×RiskPenalty 的分层统计；
11. 输出独立 HTML 页面；
12. 缺数据不报错中断，必须标注状态；
13. 历史回放和每日增量都具备幂等性；
14. 所有新文件路径清晰，不污染原系统主链路。

---

## 17. 阶段性结论

本模块不是简单页面功能，而是 `top10-decision` 系统从“预测输出”走向“预测验证闭环”的关键组件。

其最高价值在于：

- 判断 TopN Targets 是否真的有预测价值；
- 判断 EV 是否具有真实收益解释力；
- 判断 RiskPenalty 是否能有效识别风险；
- 判断 Top1、Top2、Top4、Top10 哪种实盘组合更优；
- 为后续仓位分配、隔夜挂单、实盘执行提供统计依据；
- 为未来模型权重调整和自学习闭环提供真实后验证据。

后续开发应优先完成：

1. `scripts/backfill_topn_targets_validation.py` 历史回放脚本；
2. `scripts/validate_topn_targets.py` 每日增量验证脚本；
3. `docs/topn_targets_validation.html` 独立展示页面；
4. GitHub Actions 独立验证 workflow。

