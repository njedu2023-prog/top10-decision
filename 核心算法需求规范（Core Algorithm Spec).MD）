下面给你一份**《核心算法需求规范（MD锚定文档）》**的“完整草案”。它的定位是：**系统级共识锚定**，用于后续任何开发/中断/续开发时“看它就能无缝对齐”，避免跑偏。
（注意：这是需求与契约，不是程序代码对接说明。）

---

# 核心算法需求规范（Core Algorithm Spec）

**项目：top10-decision / Premium + Decision**
**版本：V1.0（锚定版）**
**目标：让 Premium 与 Decision 同时使用【Feature Store（FS）+ pred_source_latest（上游浓缩特征）】，在贝叶斯视角下实现可回测、可在线更新的 EV 最大化决策体系。**

---

## 0. 总目标与边界

### 0.1 总目标（全系统价值）

在给定候选池（pred_source_latest）的前提下，对每只候选票估计并优化：

[
EV = P_{fill} \times E_{ret} - Cost - RiskPenalty
]

其中：

* **P_fill**：T+1 买入是否可成交（或在限定窗口内可成交的概率）
* **E_ret**：从买入到卖出（默认 T+1→T+2）的预期收益（可扩展到分布/分位数）
* **Cost**：交易成本（手续费、滑点、冲击成本的近似）
* **RiskPenalty**：风险惩罚（波动、流动性、尾部风险、题材拥挤度等）

最终输出：

* Decision：TopN 交易信号（权重/风险预算/执行策略）
* Premium：T→T+2 收盘价分布预测（分位数）+ 溢价预期 + 风险解释

### 0.2 约束（必须遵守）

* **FS 必须达到“≥75% 丰富度”**：覆盖基础行情、流动性、波动、资金流、估值、市值、涨停微结构、题材热度、龙虎榜/热度等主要类别（详见第2章）。
* **融合采用方案C（贝叶斯视角）**：pred_source_latest 作为 prior，FS 提供 likelihood 修正；模型采用 **LR + GBDT**（可包含 LightGBM）。
* **MD 是需求锚定，不作为代码接口**：代码实现可迭代，但必须满足本 MD 的输出契约与评估闭环。

---

## 1. 两类输入：pred_source_latest 与 Feature Store（FS）

### 1.1 pred_source_latest（上游浓缩特征）定位

* 角色：**强 prior（先验）** + **候选池入口**
* 本质：上游模型/因子系统对“上涨/强势/题材”的信息压缩结果
* 必须字段（最小集）：

  * `trade_date, ts_code, name, rank`
  * `prob`（上游上涨概率/强度概率）
  * `StrengthScore`
  * `ThemeBoost`
  * `board`
  * 追溯：`run_id, commit_sha, generated_at_utc`

> 价值评估结论（锚定）：在 FS 存在后，pred_source_latest 仍属**高价值**输入，因为它提供“上游压缩后的后验/embedding式信息”，与 FS 的“物理世界特征”互补而非替代。

### 1.2 Feature Store（FS）定位

* 角色：**likelihood（证据）** + **真值层（Truth Layer）** + **全系统共享特征层**
* 要求：

  * 主键统一：`trade_date + ts_code`
  * 可追溯：每行带 `source_snapshot_id/commit_sha/created_at_utc`
  * 可回测：必须能复现历史某日的特征与真值
  * 与 Premium/Decision 解耦：两者只读 FS，不重复各自拉原始数据

---

## 2. FS 的“≥75% 丰富度”字段要求（按类别）

> 说明：这里是“字段类别 + 代表字段”，实现时允许列名不同，但必须能一一映射到等价信息。

### 2.1 基础行情与价格行为（必需）

* `open, high, low, close, pre_close`
* `pct_chg / returns_1d`
* `high_low_range, candle_body, gap_open`（可派生）
* 多周期收益：`ret_2d, ret_5d, ret_10d`（可派生）

### 2.2 成交与流动性（必需）

* `vol, amount`
* `turnover_rate`（换手率）
* `volume_ratio`（量比）
* `amihud_illiquidity`（可派生）
* `bid_ask_proxy / spread_proxy`（若无盘口，用代理指标）

### 2.3 波动率与风险结构（必需）

* `volatility_5d/10d/20d`（滚动标准差）
* `ATR`（可派生）
* `downside_vol`（可派生）
* `max_drawdown_20d`（可派生）
* 尾部代理：`tail_risk_score`（可由历史分位/极值派生）

### 2.4 估值与市值结构（建议必需，若无则降级但需审计）

* `total_mv, circ_mv`
* `pe_ttm, pb`
* `float_share`（可选）

### 2.5 资金流与情绪（建议强烈引入）

* `moneyflow`（主力净流入/分级资金流，若有）
* `northbound_net` / `hsgt`（若有）
* 市场情绪指标：`market_regime`（系统内部生成：强势/震荡/弱势）

### 2.6 涨停/开板微结构（Decision 的 P_fill 核心，必需）

从 `limit_list_d / limit_break_d / stk_limit / limit_up_tags` 等提取：

* `is_limit_up, limit_type`
* `open_times`（开板次数）
* `seal_amount`（封单金额/封单量）
* `first_seal_time / last_seal_time`（若有）
* `break_count`（炸板次数）
* `limit_up_strength`（综合强度派生）

> 锚定：若缺失该类字段，Decision 的 P_fill 只能退化为弱模型，系统 EV 价值显著下降。

### 2.7 题材/板块热度（建议强烈引入）

* 板块热度：`hot_boards_score`（来自 hot_boards）
* 题材标签：`limit_up_tags`（多标签可展开或聚合）
* `board_crowding`（拥挤度/集中度，派生）

### 2.8 龙虎榜/异动（可选但对策略有价值）

* `top_list`（龙虎榜特征：上榜/机构参与/净买入）
* `abnormal_volume`（异动代理）

---

## 3. 方案C：贝叶斯视角融合（LR + GBDT）

### 3.1 核心思想（锚定）

* **pred_source_latest 提供 prior**：如 `prob`（可转 logit）
* **FS 提供 likelihood 修正**：用市场特征解释“在该票该日，该先验是否可信、风险/成交是否支持、收益分布如何偏移”
* 模型形式（锚定）：

  * **LR**：用于校准与可解释融合（稳定、抗过拟合）
  * **GBDT/LGBM**：用于非线性修正与交互（提升表达力）

### 3.2 数学表述（锚定）

以二分类为例（上涨/可成交）：
[
\text{logit}(P(y=1|FS,pred)) = \text{logit}(P(y=1|pred)) + g(FS)
]
其中：

* `logit(P(y=1|pred))` 来自 pred_source_latest 的 `prob`
* `g(FS)` 由 LR/GBDT 学得（修正项）

### 3.3 两条核心子模型（必须）

* **P_fill 模型**：目标 `y_fill`（是否可成交/是否在窗口内成交）

  * 特征：FS_limit（主）+ FS_base（辅）+ pred_summary（辅）
  * 模型：LR（校准）+ GBDT（主）
* **E_ret / Premium 分布模型**：目标可取两层：

  * `E_ret`（期望收益/回归）
  * `Quantiles`（分位数回归/分布拟合）
  * 特征：FS_base（主）+ pred_summary（辅）+（可选）FS_limit（尾部修正）
  * 模型：GBDT（主）+ LR（校准/融合）

---

## 4. Premium 与 Decision 的职责分工（必须同时用 FS + pred）

### 4.1 Premium（溢价/分布层）必须输出

* Top30 表：`close_T2_p05/p25/p50/p75/p95`（分位预测）
* `up_prob`（上涨概率，融合后）
* `expected_premium`（预期溢价）
* `premium_score`（综合评分）
* 审计：packs_used/missing/degrade_mode + 数据可用等级说明

### 4.2 Decision（执行层）必须输出

* TopN 信号：`jq_code/ts_code, target_weight, risk_budget, buy_window, sell_window`
* 模型中间量可追溯：`p_fill, e_ret, EV, cost, risk_penalty`
* 交易约束：最大持仓、单票上限、流动性过滤、风险 regime

---

## 5. pred_source_latest 与 FS 的合并规则（契约）

### 5.1 Join 主键

* `trade_date + ts_code`
* 允许 `name` 做辅助校验，不作主键

### 5.2 冲突优先级（锚定）

* 真值/行情类：以 FS 为准（Truth Layer）
* 上游信号类：以 pred_source_latest 为准（prob/StrengthScore/ThemeBoost）
* 同名字段冲突：必须重命名（例如 `prob_prior` vs `prob_post`），禁止静默覆盖

---

## 6. 评估与自学习闭环（必须）

### 6.1 必须评估指标

* `hit_rate_up`（方向命中）
* `fill_rate`（成交率）
* `avg_ret` / `median_ret`（收益）
* `EV_after_cost`（扣成本期望）
* `tail_loss`（尾部损失/回撤）

### 6.2 在线更新策略（锚定）

* 滚动窗口训练：建议 60/120/180 交易日可配置
* 训练数据：必须来自 FS + pred 的历史快照，保证可复现
* 模型产物：版本化落盘（含训练窗口/数据hash）

---

## 7. 工程输出契约（路径/命名/落盘）

> 这里只定义“必须存在与可追溯”，具体格式可迭代，但路径命名必须稳定。

### 7.1 Premium 输出

* `outputs/premium/premium_top30_{trade_date}.csv`
* `outputs/premium/premium_full_{trade_date}.csv`
* `outputs/premium/premium_verify_{trade_date}.csv`（pending 也必须产出）
* `docs/reports/premium_latest.md`
* `outputs/premium/_last_run.txt`（必须包含：trade_date、target_date、packs_used、数据等级、commit_sha）

### 7.2 Decision 输出

* `docs/signals/top10_latest.csv`（或系统锁死的信号路径）
* `outputs/decision/decision_report_{exec_date}.md`
* `data/decision/decision_candidates_{trade_date}.csv`（候选快照）

### 7.3 Feature Store 输出（新增契约，必须）

* `data/market/features_base_{trade_date}.csv|parquet`
* `data/market/features_limit_{trade_date}.csv|parquet`
* `data/market/truth_close_{trade_date}.csv|parquet`
* 元信息：`data/market/_meta_{trade_date}.json`（记录来源快照、字段覆盖率、缺失率）

---

## 8. 关键原则（防跑偏）

1. **FS 是全系统唯一真值与特征源**：Premium/Decision 不应各自到处抓数据导致不一致。
2. **pred_source_latest 是强 prior，不是可替代的噪声**：必须保留并进入融合模型。
3. **权重不拍脑袋**：通过 LR+GBDT 学习融合（贝叶斯修正项）。
4. **可追溯 > 漂亮结果**：每次输出必须能回答“用了哪些数据/哪个版本/为何降级/可信度等级”。

---

## 9. 当前状态与下一阶段里程碑（用于无缝续开发）

### 9.1 当前状态（以你目前系统为基准）

* Premium 已跑通并产出报告；Pack2 已固定 soft mode
* pred_source_latest 已包含 prob/StrengthScore/ThemeBoost/board 等浓缩特征
* 当前缺口：FS 尚未达到 ≥75% 丰富度；P_fill 关键微结构特征未统一接入 FS；Tushare token/真值链路需稳定注入

### 9.2 下一阶段里程碑（先后顺序锁死）

M1：FS 75% 丰富度落地（features_base + features_limit + truth_close）
M2：Decision 的 P_fill 模型接入 FS_limit（实现可成交概率）
M3：Premium 的分布模型接入 FS_base（实现非零 expected_premium / premium_score）
M4：LR 校准 + GBDT 非线性修正上线（贝叶斯融合完成）
M5：闭环评估与在线更新（滚动训练 + 版本化）

---

如果你同意这份锚定文档的结构，我下一步会让它变成仓库内的一个文件（建议路径）：

* `docs/specs/CORE_ALGO_SPEC.md`（或你指定位置）

然后我们按这份 MD 的里程碑推进开发：先把 **FS 75%**做出来（这是你要求的硬前提），再把 LR+GBDT 的贝叶斯融合落到 Premium/Decision。
