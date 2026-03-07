# 核心算法需求规范（Core Algorithm Spec）

项目：top10-decision / Premium + Decision

版本：V1.3（正式硬规则整合版）

定位：系统级共识锚定（需求/契约），用于中断/续开发时无缝对齐，避免跑偏。  
注意：本 MD 不是程序代码接口说明；代码可以迭代，但必须满足本 MD 的目标、契约、时间机制与闭环规则。

---

## 0. 总目标与边界

### 0.1 总目标（全系统价值）

在给定候选池（`pred_source_latest`）的前提下，对每只候选票估计并优化：

\[
EV = P_{fill} \times E_{ret} - Cost - RiskPenalty
\]

其中：

- **P_fill**：T+1 买入是否可成交（或限定窗口内可成交概率）
- **E_ret**：买入→卖出（默认 T+1→T+2）的预期收益
- **Cost**：交易成本（手续费、滑点、冲击成本近似）
- **RiskPenalty**：风险惩罚（波动、流动性、尾部风险、题材拥挤度等）

最终输出：

- **Decision**：TopN 交易信号（权重 / 风险预算 / 执行策略）
- **Premium**：T→T+2 收盘价分布预测（分位数）+ 溢价预期 + 风险解释

### 0.2 总边界（必须遵守）

- 候选池入口只能来自 `pred_source_latest` 或其按日归档版本
- 后续所有计算对象，必须以候选池名单为主，不得被 FS / raw 反向扩池
- FS 是全系统唯一真值与共享特征层
- Premium 与 Decision 都必须使用学习模型并形成训练—落盘—加载—更新闭环
- 本 MD 高于具体脚本实现；若实现与本 MD 冲突，以本 MD 为准

---

## 1. 两条主线与职责边界

### 1.1 Premium（面向人）

定位：面向人类的价格 / 空间判断层

主口径：

- `PredClose_T+2`
- 分位数区间
- 溢价预期
- 风险解释

作用：

- 帮助人类理解未来空间
- 指导 T+1 如何做交易判断
- 不直接等同于机器可执行信号

### 1.2 Decision（面向机器）

定位：面向机器执行的交易决策层

主口径：

- `P_fill`
- `EntryPriceProxy_T+1`
- `E_ret`
- `Cost`
- `RiskPenalty`
- `EV`
- `target_weight / risk_budget / regime`

作用：

- 输出能直接进入自动交易系统的可执行结果
- 必须优先保证可交易性与可执行性

### 1.3 当前开发主线

开发顺序锁死：

1. FS ≥ 75%
2. `P_fill` 学习模块
3. `E_ret` 学习模块
4. `Cost / RiskPenalty` 模块
5. `EV` 融合与校准
6. 滚动训练与门控更新

---

## 2. 候选池与输入层硬规则

### 2.1 候选池（pred_source_latest）定位

角色：强 prior（先验） + 候选池入口

必须字段最小集：

- `trade_date`
- `ts_code`
- `name`
- `rank`
- `prob`
- `StrengthScore`
- `ThemeBoost`
- `board`

追溯字段：

- `run_id`
- `commit_sha`
- `generated_at_utc`

### 2.2 候选池主导规则（正式锁死）

后续所有计算对象必须以候选池名单为主。

允许的流程：

1. 先读取 `pred_source_{trade_date}` / `pred_source_latest`
2. 再用 FS / raw 为这批股票补字段
3. 再生成 `fill_truth / trainset / decision inputs`

禁止的流程：

1. 先读取 `features_limit / features_base / raw`
2. 再把全市场或大池股票反向扩回训练或决策对象集

一句话锚定：

**pred_source 决定“算谁”，FS/raw 只负责“给这些股票补什么”。**

---

## 3. Feature Store（FS）定位与要求

### 3.1 FS 定位

FS 是：

- likelihood（证据层）
- truth layer（真值层）
- Premium / Decision 的共享特征层

### 3.2 FS 输出契约

必须存在四件套：

- `data/market/features_base_{trade_date}.csv|parquet`
- `data/market/features_limit_{trade_date}.csv|parquet`
- `data/market/truth_close_{trade_date}.csv|parquet`
- `data/market/meta_{trade_date}.json`

### 3.3 raw 层定位

raw 层是 FS 的唯一上游原料层，目录锁定为：

- `data/market/raw/{YYYY}/{YYYYMMDD}/{filename}`
- `data/market/raw/latest/{filename}`

raw 的职责：

- 提供原始快照
- 供 FS 构建使用
- 供候选池股票补充真值与微结构字段

raw 不得独立决定训练或决策对象全集。

---

## 4. 五模块决策引擎（正式命名）

系统统一采用“五模块决策引擎”结构：

1. **P_fill / Execution Fill Probability Module**  
   可成交概率模块

2. **E_ret / Expected Return Module**  
   预期收益模块

3. **Cost / Transaction Cost Module**  
   交易成本模块

4. **RiskPenalty / Risk Penalty Module**  
   风险惩罚模块

5. **EV / Expected Value Module**  
   综合价值模块

其中：

- 前四个更适合独立建模 / 独立训练
- EV 是融合层，不是最底层原始标签模块

---

## 5. 五模块之间的关系

### 5.1 前四个是基础子模块

#### A. P_fill
回答：T+1 是否可成交 / 成交概率多大

#### B. E_ret
回答：若 T+1 买到，T+1→T+2 预期收益多少

#### C. Cost
回答：执行这笔交易的成本有多高

#### D. RiskPenalty
回答：这笔交易应该扣除多少风险惩罚

### 5.2 EV 是融合层

\[
EV = P_{fill} \times E_{ret} - Cost - RiskPenalty
\]

EV 是上层融合结果，不应在系统第一阶段就替代前四个独立学习模块。

---

## 6. 时间轴定义（正式锁死）

统一时间定义：

- `T`：信号生成日 / 预测日 / 系统运行日
- `T+1`：实际买入执行日
- `T+2`：持有后的下一交易日，用于卖出 / 验证 / 收益结算

### 6.1 P_fill 的严格定义

`P_fill` 预测的是：

- T 日生成候选票
- 在 T+1 执行时是否能买到

所以 `P_fill` 解决的是：

- 可交易性
- 可执行性

不是 T+2 价格问题。

### 6.2 Premium / E_ret 的严格定义

`E_ret / PremiumRet` 预测的是：

- T+1 买入后
- 持有到 T+2 的收益 / 溢价

---

## 7. 四个必须显式区分的中间变量

后续实现必须显式区分以下四个变量：

- `PredOpen_T+1`：预测的 T+1 开盘价
- `PredClose_T+2`：预测的 T+2 收盘价
- `P_fill_T+1`：T+1 可成交概率
- `EntryPriceProxy_T+1`：T+1 真实买入价代理

锚定：

- `PredOpen_T+1` 可以作为价格路径桥变量
- 但 `PredOpen_T+1` 不得偷换成真实成交买入价
- 真正用于收益与 EV 的买入价格，应使用 `EntryPriceProxy_T+1`

---

## 8. 学习模型总原则（必须）

### 8.1 学习模型必须存在

Premium 与 Decision 都不得长期停留在纯规则系统。  
系统必须逐步形成以下学习模型：

- `P_fill`
- `E_ret`
- `Cost`（可先规则后学习）
- `RiskPenalty`（可先规则后学习）
- Premium 分位预测
- EV 融合 / 校准层

### 8.2 学习模块优先级

当前阶段优先级锁死：

1. `P_fill`
2. `E_ret`
3. `Cost`
4. `RiskPenalty`
5. `EV` 组合校准层

---

## 9. 候选池、标签、训练集的统一对象约束

### 9.1 fill_truth

`fill_truth_{trade_date}.csv` 必须满足：

- 主表对象来自 `pred_source`
- FS / raw 只补字段
- 不得扩池

### 9.2 pfill_trainset

`pfill_trainset_{trade_date}.csv` 必须满足：

- 主表对象来自 `fill_truth`
- 再 merge `features_base / features_limit / prior`
- 不得重新扩池

### 9.3 train_pfill

`train_pfill.py` 只允许吃：

- `pfill_trainset_{trade_date}.csv`

不得自行改写对象池。

---

## 10. 训练闭环与模型落盘

### 10.1 模型落盘

训练后模型必须落盘并版本化，例如：

- `models/pfill_lr.joblib`
- `models/pfill_lgbm.joblib`
- `models/pfill_meta.json`
- `models/pfill_lr_{trade_date}.joblib`
- `models/pfill_lgbm_{trade_date}.joblib`
- `models/pfill_meta_{trade_date}.json`

### 10.2 跳过训练也是合法状态

若某日出现：

- 单一标签
- 样本不足
- 无成熟样本
- 真值缺失严重

则允许：

- 不训练
- 写 meta
- 记录 `skip_reason`
- 保留旧模型

这是正常状态，不等于系统失效。

---

## 11. 统一时间机制：预测、训练、更新、生效（正式硬规则）

### 11.1 总原则

系统必须严格区分四个动作：

1. 预测
2. 真值落地
3. 学习训练
4. 模型生效

锚定原则：

- `T` 日预测只能使用 `T-1` 及以前已冻结模型
- `T` 日新落地真值，只能训练过去成熟样本
- `T` 日训练出的新模型，不得反向污染 `T` 日预测
- `T` 日训练出的新模型，最早只能从 `T+1` 开始生效

### 11.2 每日统一顺序

#### 第一步：先预测未来
在 `T` 日先用旧模型输出：

- `PredOpen_T+1`
- `PredClose_T+2`
- `P_fill_T+1`
- `E_ret_{T+1→T+2}`
- `Cost`
- `RiskPenalty`
- `EV`

#### 第二步：再用今天刚落地的真值训练过去样本

- 用 `T` 日真实开盘 / 成交，训练 `T-1` 日对 `T` 的 `PredOpen_T+1 / P_fill`
- 用 `T` 日真实收盘，训练 `T-2` 日对 `T` 的 `PredClose_T+2 / E_ret / PremiumRet`

#### 第三步：训练后经过门控再决定是否更新
若满足：

- 样本量达标
- 标签类别达标
- 评估达标
- 新模型不劣于旧模型

则更新 latest；否则跳过更新。

#### 第四步：新模型从下一交易日生效
`T` 日训练出的模型，只允许从 `T+1` 开始服务预测。

### 11.3 分模块成熟节奏

#### A. P_fill / PredOpen_T+1
最小成熟条件：`T+1` 真值落地即可训练

#### B. E_ret / PredClose_T+2 / PremiumRet
最小成熟条件：必须等到 `T+2` 真值落地

#### C. Cost / RiskPenalty
若学习化，也必须遵守同样时间机制

### 11.4 系统不是“当天预测、当天训练同一批样本”

系统必须同时存在两条并行链：

#### 在线预测链
- `T` 日预测未来
- 服务 Decision / Premium

#### 离线学习链
- `T` 日回收刚成熟的过去样本
- 训练并更新模型
- 服务未来 `T+1` 及以后预测

### 11.5 示例

假设连续交易日为：`01, 02, 03, 04, 05, 06, 07, 08`，当前运行日为 `06`

#### 在 `06` 日先预测：
- 预测 `07(T+1)` 开盘价
- 预测 `08(T+2)` 收盘价
- 预测 `07` 的 `P_fill`
- 预测 `07→08` 的 `E_ret`

#### 在 `06` 日再训练：
- 用 `06` 真实开盘 / 成交，训练 `05` 对 `06` 的 `PredOpen_T+1 / P_fill`
- 用 `06` 真实收盘，训练 `04` 对 `06` 的 `PredClose_T+2 / E_ret / PremiumRet`

#### 在 `06` 日训练后：
- 若通过门控，则更新 latest
- 若不通过门控，则保留旧模型

#### 在 `07` 日起：
- `06` 日训练出的新模型才允许参与 `07` 日预测

### 11.6 最终锚定结论

正式锁死：

- `T` 日先预测，再训练过去，不得反向污染 `T` 日预测
- `T` 日训练出的模型，只能从 `T+1` 起生效
- 学习系统允许跳过，不允许乱更新
- 所有学习模块必须遵守同一时间机制

---

## 12. 当前阶段正式结论

### 12.1 已经完成

- 决策主链基本跑通
- `P_fill` 学习链首轮打通
- 候选池驱动规则已明确
- 样本成熟度规则已明确
- 统一时间机制已锁死

### 12.2 当前正在做

- 把 `P_fill` 从“单日能跑”升级为“多日稳定”
- 收紧 `y_fill` 口径
- 提升标签区分度
- 处理单一标签日、样本不足日的正常跳过逻辑

### 12.3 下一阶段

1. 把 `P_fill` 扩展到成熟窗口
2. 正式进入 `E_ret`
3. 再进入 `Cost / RiskPenalty`
4. 最后做 `EV` 融合与校准

---

## 13. 工程协作规则（保留）

- 一个文件一个文件来
- 每次先给：
  1. 文件全路径
  2. 简要修改报告
  3. 完整可覆盖代码
- 超过 300 行代码文件，优先基于 Raw 链接修改
- 所有交流与输出继续使用简体中文

---

## 14. 最高锚定总结

本系统不是：

- 纯预测展示系统
- 纯规则交易系统
- 当天预测、当天训练同一批样本的混合系统

本系统必须是：

- 候选池驱动
- FS 共享证据层
- 双主线分工明确
- 五模块分层建模
- 统一时间机制
- 允许跳过但不允许乱更新
- 可持续滚动学习的交易决策系统
