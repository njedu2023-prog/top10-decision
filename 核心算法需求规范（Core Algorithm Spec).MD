核心算法需求规范（Core Algorithm Spec）

项目：top10-decision / Premium + Decision

版本：V1.2（锚定补充版）

定位：系统级共识锚定（需求/契约），用于中断/续开发时无缝对齐，避免跑偏。

注意：本 MD 不是程序代码接口说明，代码可迭代，但必须满足本 MD 的目标、契约与闭环。

---

## 0. 总目标与边界

### 0.1 总目标（全系统价值）

在给定候选池（pred_source_latest）的前提下，对每只候选票估计并优化：

[
EV = P_{fill} \times E_{ret} - Cost - RiskPenalty
]

其中：

- P_fill：T+1 买入是否可成交（或限定窗口内可成交概率）
- E_ret：买入→卖出（默认 T+1→T+2）的预期收益（可扩展到分布/分位数）
- Cost：交易成本（手续费、滑点、冲击成本近似）
- RiskPenalty：风险惩罚（波动、流动性、尾部风险、题材拥挤度等）

最终输出：

- Decision：TopN 交易信号（权重/风险预算/执行策略）
- Premium：T→T+2 收盘价分布预测（分位数）+ 溢价预期 + 风险解释

### 0.2 约束（必须遵守）

- FS 必须达到 ≥75% 丰富度：覆盖基础行情、流动性、波动、资金流、估值、市值、涨停微结构、题材热度、龙虎榜/异动等主要类别（详见第2章）。
- 融合采用方案C（贝叶斯视角）：pred_source_latest 作为 prior，FS 提供 likelihood 修正；模型采用 LR + GBDT（可包含 LightGBM）。
- Premium 与 Decision 必须使用学习模型并形成闭环：训练—落盘—加载—滚动更新—评估门控（详见第10章）。
- MD 是需求锚定，不作为代码接口：代码实现可迭代，但不得破坏本 MD 的输出契约与评估闭环。

---

## 1. 两类输入：pred_source_latest 与 Feature Store（FS）

### 1.1 pred_source_latest（上游浓缩特征）定位

角色：强 prior（先验） + 候选池入口

本质：上游模型/因子系统对“上涨/强势/题材”的信息压缩结果

必须字段（最小集）：

- trade_date, ts_code, name, rank
- prob（上游上涨概率/强度概率）
- StrengthScore
- ThemeBoost
- board
- 追溯：run_id, commit_sha, generated_at_utc

价值结论（锚定）：在 FS 存在后，pred_source_latest 仍属高价值输入（非替代关系，而是与 FS 互补）。

### 1.2 Feature Store（FS）定位

角色：likelihood（证据） + 真值层（Truth Layer） + 全系统共享特征层

要求：

- 主键统一：trade_date + ts_code
- 可追溯：每行带 source_snapshot_id/commit_sha/created_at_utc
- 可回测：必须能复现历史某日的特征与真值
- 与 Premium/Decision 解耦：两者只读 FS，不重复各自拉原始数据

---

## 2. FS 的“≥75% 丰富度”字段要求（按类别）

说明：这里是“字段类别 + 代表字段”。实现时允许列名不同，但必须能映射到等价信息；并在 _meta_{trade_date}.json 记录覆盖率/缺失率。

### 2.1 基础行情与价格行为（必需）

- open, high, low, close, pre_close
- pct_chg / returns_1d
- 派生：high_low_range, candle_body, gap_open
- 多周期收益：ret_2d, ret_5d, ret_10d（可派生）

### 2.2 成交与流动性（必需）

- vol, amount
- turnover_rate
- volume_ratio
- 派生：amihud_illiquidity
- 若无盘口：bid_ask_proxy / spread_proxy（代理指标）

### 2.3 波动率与风险结构（必需）

- volatility_5d/10d/20d
- 派生：ATR, downside_vol, max_drawdown_20d
- tail_risk_score（由历史分位/极值派生）

### 2.4 估值与市值结构（建议强制，缺失需降级审计）

- total_mv, circ_mv
- pe_ttm, pb
- float_share（可选）

### 2.5 资金流与情绪（建议强烈引入）

- moneyflow（若有）
- northbound_net / hsgt（若有）
- market_regime（系统内部生成：强势/震荡/弱势）

### 2.6 涨停/开板微结构（Decision 的 P_fill 核心，必需）

从 limit_list_d / limit_break_d / stk_limit / limit_up_tags 等提取：

- is_limit_up, limit_type
- open_times（开板次数）
- seal_amount（封单金额/封单量）
- first_seal_time / last_seal_time（若有）
- break_count（炸板次数）
- 派生：limit_up_strength

锚定：若缺失该类字段，Decision 的 P_fill 只能退化为弱模型，系统 EV 价值显著下降。

### 2.7 题材/板块热度（建议强烈引入）

- hot_boards_score（来自 hot_boards）
- limit_up_tags（多标签可展开/聚合）
- 派生：board_crowding（拥挤度/集中度）

### 2.8 龙虎榜/异动（可选但高价值）

- top_list（上榜/机构参与/净买入）
- abnormal_volume（异动代理）

---

## 3. 方案C：贝叶斯视角融合（LR + GBDT）

### 3.1 核心思想（锚定）

- pred_source_latest 提供 prior：如 prob（建议转 logit(prob)）
- FS 提供 likelihood 修正：用市场特征解释“先验是否可信、成交/风险是否支持、收益分布如何偏移”

模型形态（锚定）：

- LR：校准与可解释融合（稳定、抗过拟合）
- GBDT/LGBM：非线性修正与交互（表达力）

### 3.2 数学表述（锚定）

以二分类为例（上涨/可成交）：

[
\text{logit}(P(y=1|FS,pred)) = \text{logit}(P(y=1|pred)) + g(FS)
]

其中：

- logit(P(y=1|pred)) 来自 pred 的 prob
- g(FS) 由 LR/GBDT 学得（修正项）

### 3.3 两条核心子模型（必须）

- P_fill 模型：目标 y_fill（是否可成交/窗口内成交）
  - 特征：FS_limit（主）+ FS_base（辅）+ pred_summary（辅）
  - 模型：GBDT（主）+ LR（校准）

- E_ret / Premium 分布模型
  - 目标：E_ret（回归）+ Quantiles（分位回归/分布拟合）
  - 特征：FS_base（主）+ pred_summary（辅）+（可选）FS_limit（尾部修正）
  - 模型：GBDT/LGBM（主）+ LR（校准/融合）

---

## 4. Premium 与 Decision 的职责分工（必须同时用 FS + pred）

### 4.1 Premium（溢价/分布层）必须输出

- Top30 表：close_T2_p05/p25/p50/p75/p95（分位预测）
- up_prob（融合后上涨概率）
- expected_premium（预期溢价）
- premium_score（综合评分）
- 审计：packs_used/missing/degrade_mode + 数据可用等级说明 + model_mode 说明（详见第10章）

### 4.2 Decision（执行层）必须输出

- TopN 信号：jq_code/ts_code, target_weight, risk_budget, buy_window, sell_window
- 中间量可追溯：p_fill, e_ret, EV, cost, risk_penalty
- 交易约束：最大持仓、单票上限、流动性过滤、风险 regime

---

## 5. pred_source_latest 与 FS 的合并规则（契约）

### 5.1 Join 主键

- trade_date + ts_code
- name 仅做辅助校验，不作主键

### 5.2 冲突优先级（锚定）

- 真值/行情类：以 FS 为准（Truth Layer）
- 上游信号类：以 pred_source_latest 为准（prob/StrengthScore/ThemeBoost）
- 同名冲突：必须重命名（例如 prob_prior vs prob_post），禁止静默覆盖

---

## 6. 评估与自学习闭环（必须）

### 6.1 必须评估指标

- hit_rate_up（方向命中）
- fill_rate（成交率）
- avg_ret / median_ret（收益）
- EV_after_cost（扣成本期望）
- tail_loss（尾部损失/回撤）

### 6.2 在线更新策略（锚定）

- 滚动窗口训练：60/120/180 交易日可配置（默认建议 120）
- 训练数据：必须来自 FS + pred 的历史快照，保证可复现
- 模型产物：版本化落盘（含训练窗口/数据hash）

---

## 7. 工程输出契约（路径/命名/落盘）

### 7.1 Premium 输出

- outputs/premium/premium_top30_{trade_date}.csv
- outputs/premium/premium_full_{trade_date}.csv
- outputs/premium/premium_verify_{trade_date}.csv（pending 也必须产出）
- docs/reports/premium_latest.md
- outputs/premium/_last_run.txt（必须包含：trade_date、target_date、packs_used、数据等级、commit_sha、model_mode）

### 7.2 Decision 输出

- docs/signals/top10_latest.csv（或系统锁死的信号路径）
- outputs/decision/decision_report_{exec_date}.md
- data/decision/decision_candidates_{trade_date}.csv（候选快照）

### 7.3 Feature Store 输出（新增契约，必须）

- data/market/features_base_{trade_date}.csv|parquet
- data/market/features_limit_{trade_date}.csv|parquet
- data/market/truth_close_{trade_date}.csv|parquet
- data/market/_meta_{trade_date}.json（来源快照、字段覆盖率、缺失率、版本）

### 7.4 raw 原料层输出（新增锚定）

raw 层是 FS 的唯一上游原料层，目录结构锁定为：

- data/market/raw/{YYYY}/{YYYYMMDD}/{filename}
- data/market/raw/latest/{filename}

其中：

- latest/：最新镜像目录，每日覆盖更新
- {YYYY}/{YYYYMMDD}/：历史冻结快照目录，保留当日原始记录用于审计/复现/回测

旧扁平结构：

- data/market/raw/daily_{trade_date}.csv
- data/market/raw/daily_latest.csv
- data/market/raw/_sync_meta_{trade_date}.json
- data/market/raw/_sync_meta_latest.json

属于历史过渡形态，不得作为后续主结构长期保留。

---

## 8. 关键原则（防跑偏）

- FS 是全系统唯一真值与特征源：Premium/Decision 不应各自到处抓数据导致不一致。
- pred_source_latest 是强 prior：必须保留并进入融合模型。
- 权重不拍脑袋：通过 LR+GBDT 学习融合（贝叶斯修正项），不允许固定常数权重长期运行。
- 可追溯 > 漂亮结果：每次输出必须能回答“用了哪些数据/哪个版本/为何降级/可信度等级”。

---

## 9. 当前状态与下一阶段里程碑（用于无缝续开发）

### 9.1 当前状态（以你目前系统为基准）

- Premium 已跑通并产出报告；Pack2 已固定 soft mode
- pred_source_latest 已包含 prob/StrengthScore/ThemeBoost/board 等浓缩特征
- 当前缺口：FS 尚未达到 ≥75% 丰富度；P_fill 关键微结构特征未统一接入 FS；Tushare token/真值链路需稳定注入

### 9.2 下一阶段里程碑（先后顺序锁死）

- M1：FS ≥75% 丰富度落地（features_base + features_limit + truth_close + meta）
- M2：Decision 的 P_fill 学习模型接入 FS_limit（输出可成交概率 + 校准）
- M3：Premium 的分布学习模型接入 FS_base（输出分位预测 + 覆盖率校准）
- M4：LR 校准 + GBDT 非线性修正上线（贝叶斯融合完成：logit(prob_prior)+g(FS)）
- M5：闭环评估与在线更新（滚动训练 + 门控覆盖 + 版本化回滚）

---

## 10. 学习模型（必须）与训练闭环（锁死）

### 10.1 总原则（必须）

Premium 与 Decision 都必须使用学习模型（LR + GBDT/LGBM），不得长期停留在纯规则/纯展示形态。

规则允许存在，但只能作为：

- 冷启动兜底（样本不足）
- 风控硬闸门（流动性、涨停不可买等）
- 模型输出后处理（阈值、截断、权重归一）

### 10.2 必须存在的模型（两条主线）

系统必须至少包含并持续更新以下学习模型：

A) Decision 模型组（交易可落地 EV）

- P_fill：预测“能否成交/成交概率”
- E_ret：预测“买入→卖出期望收益/或收益分布参数”
- EV 组合层：使用贝叶斯视角融合 prior 与 likelihood，输出 EV_after_cost

B) Premium 模型组（价格分布预测）

- 分位预测：输出 close_T2_p05/p25/p50/p75/p95（分位回归/分布拟合）
- 校准模型（LR/Isotonic）：保证分位覆盖率与概率校准

锚定：Premium 不是“画分位表”，而是“可训练、可回测的分布预测系统”。

### 10.3 模型输入（锁死）

训练与推理必须同时包含：

- Prior（pred）：logit(prob) + StrengthScore + ThemeBoost + 其它浓缩特征
- Likelihood（FS）：FS_base + FS_limit（按任务选择）

禁止长期只用单一来源运行。

### 10.4 训练数据落库与可复现（必须）

必须落盘训练样本与元信息，保证可复现：

训练样本表：

- data/learning/decision_train_{asof_date}.parquet|csv
- data/learning/premium_train_{asof_date}.parquet|csv

特征元信息：

- data/learning/feature_meta_{asof_date}.json（字段覆盖率/缺失率/版本）

真值元信息：

- data/learning/truth_meta_{asof_date}.json（交易日历、收盘价、成交判定）

### 10.5 模型落盘（必须）

训练后模型必须落盘并版本化：

- models/decision/p_fill_lr.joblib
- models/decision/p_fill_gbdt.joblib
- models/decision/e_ret_lr.joblib
- models/decision/e_ret_gbdt.joblib
- models/premium/q05_gbdt.joblib ... q95_gbdt.joblib（或多输出模型）
- models/_meta.json（训练窗口、特征集合hash、评估指标、commit_sha、创建时间）

### 10.6 滚动训练与更新策略（锁死）

- 滚动窗口默认 120 交易日（可配置 60/120/180）
- 更新频率：每日/每周（默认每日）
- 冷启动：样本不足允许回退，但必须在审计中标记 model_mode=cold_start

### 10.7 评估与门控（必须）

每次训练必须产出评估报告，并设置门控规则：

Decision：

- fill_rate、EV_after_cost、hit_rate_up、tail_loss

Premium：

- 分位覆盖率（实际落在 [p05,p95] 的比例）
- 中位误差（p50误差）
- 尾部误差（p05/p95误差）

门控规则（锚定示例）：

若新模型在滚动回测窗口内 EV_after_cost 低于旧模型超过阈值，则拒绝覆盖，保留旧模型，并记录原因。

### 10.8 在线推理必须加载模型（锁死）

每日预测必须：

- 优先加载 latest 模型
- 模型缺失允许降级，但必须在报告审计中写明：
  - model_loaded=false
  - degrade_reason=model_missing
  - fallback_mode=rule_only

---

## 11. 新增锚定：时间轴、交易语义与两条主线口径

### 11.1 时间轴定义（锁死）

本系统统一采用以下时间轴定义：

- T：信号生成日 / 预测日 / 系统运行日
- T+1：实际买入执行日
- T+2：持有后的下一交易日，用于卖出/验证/收益结算

锚定：P_fill 对应 T+1；Premium 主收益定义对应 T+1→T+2，不得混淆。

### 11.2 P_fill 的严格定义（锁死）

P_fill 预测的是：

> T 日生成的候选票，在 T+1 实际执行时，是否能买到 / 成交概率多大。

因此，P_fill 解决的是“可交易性 / 可执行性”问题，不是 T+2 的价格问题。

### 11.3 系统测的“溢价”定义（锁死）

本系统当前主溢价口径定义为：

[
PremiumRet_{T+1 \to T+2} = \frac{Price_{T+2,\text{exit}}}{Price_{T+1,\text{entry}}} - 1
]

其中默认建议：

- entry 口径：T+1 买入价代理（见 11.6）
- exit 口径：T+2 收盘价（Premium 主展示层）

锚定：

- 不是 T+1 相对 T
- 不是 T+2 相对 T
- 而是 T+2 相对于 T+1 的溢价

### 11.4 Premium 与 Decision 的主口径分线（锁死）

系统必须严格区分两条输出主线：

A) Premium（面向人）

- 主要服务人工判断
- 主展示口径：PredClose_T+2（T+2 预测收盘价）
- 输出重点：价格区间、分位数、未来空间、溢价预期、风险解释
- 作用：指导人类在 T+1 如何做交易，而不是直接给机器下单

B) Decision（面向机器）

- 主要服务自动执行（聚宽）
- 主输出口径：可交易性 / 可执行性 / 风险约束 / 权重分配
- 输出重点：P_fill、EntryPriceProxy_T+1、E_ret、EV、target_weight、risk_budget、regime、最终可执行信号
- 作用：给机器“能直接交易”的结果，而不是报告展示

锚定：

- Premium 主口径可以显示 T+2 预测价格
- Decision 主口径必须偏向可交易与可执行
- P_fill 属于 Decision，不属于 Premium 主展示口径

### 11.5 为什么 T 日可以预测 T+2（锁死）

系统在 T 日运行时，只知道 T 日收盘信息，不知道 T+1 的真实价格与 T+2 的真实价格。

因此系统不是“直接知道 T+2”，而是做条件预测。核心链条为：

第一层：估计 T+1 执行相关变量

- P_fill_T+1
- PredOpen_T+1（可作为价格桥变量）
- EntryPriceProxy_T+1（真实买入价代理）

第二层：估计 T+2 价格/收益变量

- PredClose_T+2
- PremiumRet_{T+1→T+2}

第三层：Decision/EV 融合

[
EV = P_{fill} \times E_{ret} - Cost - RiskPenalty
]

锚定：系统并非“直接从 T 硬推绝对 T+2”，而是通过“先近后远”的链式预测完成。

### 11.6 四个必须锁死的中间变量（新增）

后续实现中，必须显式区分并可审计以下四个变量：

- PredOpen_T+1：预测的 T+1 开盘价（价格路径桥变量）
- PredClose_T+2：预测的 T+2 收盘价（Premium 主展示变量）
- P_fill_T+1：T+1 可成交概率（Decision 主执行变量）
- EntryPriceProxy_T+1：T+1 真实买入价代理（Decision 收益/EV 核心变量）

锚定：

- PredOpen_T+1 可以作为价格预测锚点
- 但 PredOpen_T+1 不得直接偷换成真实成交买入价
- 真正用于收益与 EV 的买入价格，应使用 EntryPriceProxy_T+1

### 11.7 “T→预测 T+1 开盘→预测 T+2 收盘”的定位（新增）

以下思路被正式确认可作为系统中的价格预测链：

- T 日，先预测 T+1 开盘价
- 再用 T 日信息 + PredOpen_T+1 去预测 T+2 收盘价

其定位为：

- 合理
- 值得做
- 更适合作为“价格路径预测链”
- 不应直接等同于“真实可成交买入价链”

因此：

- Premium 可优先使用该链来增强价格展示与分位预测
- Decision 仍需独立保留执行链：P_fill + EntryPriceProxy + E_ret + EV

### 11.8 Premium 与 Decision 的最终职责边界（新增）

Premium 的本质：

- 面向人类的价格/空间判断层
- 可以用 T+2 预测收盘价作为最直观主口径
- 强调未来空间、分位区间、上沿/中位/下沿、风险说明

Decision 的本质：

- 面向机器的执行决策层
- 必须围绕“明天能不能买、买到多少钱、值不值得下单”组织输出
- 不能只吃 PredClose_T+2，而必须纳入可交易性约束

---

## 12. 续开发“无缝衔接口令”（快速定位）

本文档为最高优先级锚定：任何代码改动不得破坏第0/2/3/7/10/11章的契约。

后续开发顺序锁死：

先 FS≥75% → 再 P_fill 学习模型 → 再 Premium 分布模型 → 再融合校准 → 再滚动训练门控
