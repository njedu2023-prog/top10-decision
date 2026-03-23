# 核心算法需求规范（Core Algorithm Spec）

项目：top10-decision / Premium + Decision

版本：V1.4（训练日期基础设施锚定版）

定位：系统级共识锚定（需求/契约），用于中断/续开发时无缝对齐，避免跑偏。

注意：本 MD 不是程序代码接口说明；代码可以迭代，但必须满足本 MD 的目标、契约、时间机制、样本成熟度机制与闭环规则。

---

# 0. 总目标与边界

## 0.1 总目标 （全系统价值）  Top10 Decision 的唯一最高目标，是在 T日收盘后，从候选池中输出一组组合，使其在 T+1 具有可成交性、在 T+1→T+2 具有正期望收益、并且整体风险可控,并且具备自学习能力，自迭代提升总目标系统价值。

在给定候选池（pred_source_latest）的前提下，对每只候选票估计并优化：

\[
EV = P_{fill} \times E_{ret} - Cost - RiskPenalty
\]

其中：

- P_fill：T+1 买入是否可成交（或限定窗口内可成交概率）
- E_ret：买入→卖出（默认 T+1→T+2）的预期收益
- Cost：交易成本（手续费、滑点、冲击成本近似）
- RiskPenalty：风险惩罚（波动、流动性、尾部风险、题材拥挤度等）

最终输出：

- Decision：TopN 交易信号（权重 / 风险预算 / 执行策略）
- Premium：T→T+2 收盘价分布预测（分位数）+ 溢价预期 + 风险解释

## 0.2 总边界（必须遵守）

- 候选池入口只能来自 pred_source_latest 或其按日归档版本
- 后续所有计算对象，必须以候选池名单为主，不得被 FS / raw 反向扩池
- FS 是全系统唯一真值与共享特征层
- Premium 与 Decision 都必须使用学习模型并形成训练—落盘—加载—更新闭环
- 本 MD 高于具体脚本实现；若实现与本 MD 冲突，以本 MD 为准

---

# 1. 两条主线与职责边界

## 1.1 Premium（面向人）

定位：面向人类的价格 / 空间判断层

主口径：

- PredClose_T+2
- 分位数区间
- 溢价预期
- 风险解释

作用：

- 帮助人类理解未来空间
- 指导 T+1 如何做交易判断
- 不直接等同于机器可执行信号

## 1.2 Decision（面向机器）

定位：面向机器执行的交易决策层

主口径：

- P_fill
- EntryPriceProxy_T+1
- E_ret
- Cost
- RiskPenalty
- EV
- target_weight / risk_budget / regime

作用：

- 输出能直接进入自动交易系统的可执行结果
- 必须优先保证可交易性与可执行性

## 1.3 当前开发主线

开发顺序锁死：

1. FS ≥ 75%
2. P_fill 学习模块
3. E_ret 学习模块
4. Cost / RiskPenalty 模块
5. EV 融合与校准
6. 滚动训练与门控更新

---

# 2. 候选池与输入层硬规则

## 2.1 候选池（pred_source_latest）定位

角色：强 prior（先验） + 候选池入口

必须字段最小集：

- trade_date
- ts_code
- name
- rank
- prob
- StrengthScore
- ThemeBoost
- board

追溯字段：

- run_id
- commit_sha
- generated_at_utc

## 2.2 候选池主导规则（正式锁死）

后续所有计算对象必须以候选池名单为主。

允许的流程：

- 先读取 pred_source_{trade_date} / pred_source_latest
- 再用 FS / raw 为这批股票补字段
- 再生成 fill_truth / trainset / decision inputs

禁止的流程：

- 先读取 features_limit / features_base / raw
- 再把全市场或大池股票反向扩回训练或决策对象集

一句话锚定：

**pred_source 决定“算谁”，FS/raw 只负责“给这些股票补什么”。**

---

# 3. Feature Store（FS）定位与要求

## 3.1 FS 定位

FS 是：

- likelihood（证据层）
- truth layer（真值层）
- Premium / Decision 的共享特征层

## 3.2 FS 输出契约

必须存在四件套：

- data/market/features_base_{trade_date}.csv|parquet
- data/market/features_limit_{trade_date}.csv|parquet
- data/market/truth_close_{trade_date}.csv|parquet
- data/market/meta_{trade_date}.json

## 3.3 raw 层定位

raw 层是 FS 的唯一上游原料层，目录锁定为：

- data/market/raw/{YYYY}/{YYYYMMDD}/{filename}
- data/market/raw/latest/{filename}

raw 的职责：

- 提供原始快照
- 供 FS 构建使用
- 供候选池股票补充真值与微结构字段

raw 不得独立决定训练或决策对象全集。

---

# 4. 五模块决策引擎（正式命名）

系统统一采用“五模块决策引擎”结构：

- P_fill / Execution Fill Probability Module  
  可成交概率模块

- E_ret / Expected Return Module  
  预期收益模块

- Cost / Transaction Cost Module  
  交易成本模块

- RiskPenalty / Risk Penalty Module  
  风险惩罚模块

- EV / Expected Value Module  
  综合价值模块

其中：

- 前四个更适合独立建模 / 独立训练
- EV 是融合层，不是最底层原始标签模块

---

# 5. 五模块之间的关系

## 5.1 前四个是基础子模块

### A. P_fill

回答：T+1 是否可成交 / 成交概率多大

### B. E_ret

回答：若 T+1 买到，T+1→T+2 预期收益多少

### C. Cost

回答：执行这笔交易的成本有多高

### D. RiskPenalty

回答：这笔交易应该扣除多少风险惩罚

## 5.2 EV 是融合层

\[
EV = P_{fill} \times E_{ret} - Cost - RiskPenalty
\]

EV 是上层融合结果，不应在系统第一阶段就替代前四个独立学习模块。

---

# 6. 时间轴定义（正式锁死）

统一时间定义：

- T：信号生成日 / 预测日 / 系统运行日
- T+1：实际买入执行日
- T+2：持有后的下一交易日，用于卖出 / 验证 / 收益结算

## 6.1 P_fill 的严格定义

P_fill 预测的是：

- T 日生成候选票
- 在 T+1 执行时是否能买到

所以 P_fill 解决的是：

- 可交易性
- 可执行性

不是 T+2 价格问题。

## 6.2 Premium / E_ret 的严格定义

E_ret / PremiumRet 预测的是：

- T+1 买入后
- 持有到 T+2 的收益 / 溢价

---

# 7. 四个必须显式区分的中间变量

后续实现必须显式区分以下四个变量：

- PredOpen_T+1：预测的 T+1 开盘价
- PredClose_T+2：预测的 T+2 收盘价
- P_fill_T+1：T+1 可成交概率
- EntryPriceProxy_T+1：T+1 真实买入价代理

锚定：

- PredOpen_T+1 可以作为价格路径桥变量
- 但 PredOpen_T+1 不得偷换成真实成交买入价
- 真正用于收益与 EV 的买入价格，应使用 EntryPriceProxy_T+1

---

# 8. 学习模型总原则（必须）

## 8.1 学习模型必须存在

Premium 与 Decision 都不得长期停留在纯规则系统。

系统必须逐步形成以下学习模型：

- P_fill
- E_ret
- Cost（可先规则后学习）
- RiskPenalty（可先规则后学习）
- Premium 分位预测
- EV 融合 / 校准层

## 8.2 学习模块优先级

当前阶段优先级锁死：

1. P_fill
2. E_ret
3. Cost
4. RiskPenalty
5. EV 组合校准层

---

# 9. 候选池、标签、训练集的统一对象约束

## 9.1 fill_truth

fill_truth_{trade_date}.csv 必须满足：

- 主表对象来自 pred_source
- FS / raw 只补字段
- 不得扩池

## 9.2 pfill_trainset

pfill_trainset_{trade_date}.csv 必须满足：

- 主表对象来自 fill_truth
- 再 merge features_base / features_limit / prior
- 不得重新扩池

## 9.3 train_pfill

train_pfill.py 只允许吃：

- pfill_trainset_{trade_date}.csv
- 或其基于成熟窗口的多日拼接集合

不得自行改写对象池。

---

# 10. 训练闭环与模型落盘

## 10.1 模型落盘

训练后模型必须落盘并版本化，例如：

- models/pfill_lr.joblib
- models/pfill_lgbm.joblib
- models/pfill_meta.json
- models/pfill_lr_{trade_date}.joblib
- models/pfill_lgbm_{trade_date}.joblib
- models/pfill_meta_{trade_date}.json

## 10.2 跳过训练也是合法状态

若某日出现：

- 单一标签
- 样本不足
- 无成熟样本
- 真值缺失严重

则允许：

- 不训练
- 写 meta
- 记录 skip_reason
- 保留旧模型

这是正常状态，不等于系统失效。

---

# 11. 统一时间机制：预测、训练、更新、生效（正式硬规则）

## 11.1 总原则

系统必须严格区分四个动作：

- 预测
- 真值落地
- 学习训练
- 模型生效

锚定原则：

- T 日预测只能使用 T-1 及以前已冻结模型
- T 日新落地真值，只能训练过去成熟样本
- T 日训练出的新模型，不得反向污染 T 日预测
- T 日训练出的新模型，最早只能从 T+1 开始生效

## 11.2 每日统一顺序

### 第一步：先预测未来

在 T 日先用旧模型输出：

- PredOpen_T+1
- PredClose_T+2
- P_fill_T+1
- E_ret_{T+1→T+2}
- Cost
- RiskPenalty
- EV

### 第二步：再用今天刚落地的真值训练过去样本

- 用 T 日真实开盘 / 成交，训练 T-1 日对 T 的 PredOpen_T+1 / P_fill
- 用 T 日真实收盘，训练 T-2 日对 T 的 PredClose_T+2 / E_ret / PremiumRet

### 第三步：训练后经过门控再决定是否更新

若满足：

- 样本量达标
- 标签类别达标
- 评估达标
- 新模型不劣于旧模型

则更新 latest；否则跳过更新。

### 第四步：新模型从下一交易日生效

T 日训练出的模型，只允许从 T+1 开始服务预测。

## 11.3 分模块成熟节奏

### A. P_fill / PredOpen_T+1

最小成熟条件：T+1 真值落地即可训练

### B. E_ret / PredClose_T+2 / PremiumRet

最小成熟条件：必须等到 T+2 真值落地

### C. Cost / RiskPenalty

若学习化，也必须遵守同样时间机制

## 11.4 系统不是“当天预测、当天训练同一批样本”

系统必须同时存在两条并行链：

### 在线预测链

- T 日预测未来
- 服务 Decision / Premium

### 离线学习链

- T 日回收刚成熟的过去样本
- 训练并更新模型
- 服务未来 T+1 及以后预测

## 11.5 示例

假设连续交易日为：01, 02, 03, 04, 05, 06, 07, 08，当前运行日为 06

在 06 日先预测：

- 预测 07(T+1) 开盘价
- 预测 08(T+2) 收盘价
- 预测 07 的 P_fill
- 预测 07→08 的 E_ret

在 06 日再训练：

- 用 06 真实开盘 / 成交，训练 05 对 06 的 PredOpen_T+1 / P_fill
- 用 06 真实收盘，训练 04 对 06 的 PredClose_T+2 / E_ret / PremiumRet

在 06 日训练后：

- 若通过门控，则更新 latest
- 若不通过门控，则保留旧模型

在 07 日起：

- 06 日训练出的新模型才允许参与 07 日预测

## 11.6 最终锚定结论

正式锁死：

- T 日先预测，再训练过去，不得反向污染 T 日预测
- T 日训练出的模型，只能从 T+1 起生效
- 学习系统允许跳过，不允许乱更新
- 所有学习模块必须遵守同一时间机制

---

# 12. 样本成熟度解析层 / 训练日期基础设施（正式新增并锁死）

## 12.1 设立原因

过去系统中，训练日期机制曾经分散在多个脚本中隐式存在，例如：

- build_fill_truth.py 自己临时猜 exec_date
- train_pfill.py 自己临时猜成熟 trade_date
- workflow 中存在一次性 Python 补丁推断样本是否成熟

这种做法会带来：

- 职责耦合
- 多处重复推断
- 不同脚本口径可能分叉
- 后续 E_ret / Cost / RiskPenalty 容易各自再长出一套日期逻辑

因此，系统必须正式建立：

**样本成熟度解析层 / 训练日期基础设施层**

这是系统级中间层，不是一次性过程副产物。

## 12.2 正式模块

当前正式模块锁定为：

- `scripts/resolve_sample_maturity.py`

它的职责是：

- 给定当前运行日
- 给定 raw 快照实际存在的交易日集合
- 给定候选 trade_date 集合
- 统一输出每个 trade_date 的成熟度状态

它不是训练脚本，也不负责标签构建。

## 12.3 正式输出产物

该层必须产出以下正式中间文件：

- `data/market/sample_maturity_candidates.csv`
- `data/market/sample_maturity_latest.csv`
- `data/market/sample_maturity_latest.json`

锚定：

- 这些文件不是调试文件
- 这些文件不是临时产物
- 这些文件是训练日期机制的正式系统产物
- 后续学习模块应优先消费这些结果，而不是各自再推断

## 12.4 样本成熟度字段定义

成熟度解析层必须至少输出：

- `trade_date`
- `exec_date`
- `target_date`
- `sample_maturity`
- `PFILL_READY`
- `ERET_READY`
- `FULLY_READY`

字段含义锁死如下：

### trade_date

样本所属的信号生成日 / 预测日 T

### exec_date

trade_date 之后的第 1 个真实交易日，即 T+1

### target_date

trade_date 之后的第 2 个真实交易日，即 T+2

### sample_maturity

样本成熟状态的文字表达，当前至少允许：

- `UNREADY`
- `PFILL_READY`
- `FULLY_READY`

### PFILL_READY

是否已具备 P_fill 标签构建 / P_fill 训练条件

规则：

- `exec_date` 已存在
- 且 `exec_date <= current_run_date`

则：

- `PFILL_READY = 1`

否则为 0

### ERET_READY

是否已具备 E_ret / PremiumRet 标签构建与训练条件

规则：

- `target_date` 已存在
- 且 `target_date <= current_run_date`

则：

- `ERET_READY = 1`

否则为 0

### FULLY_READY

是否同时具备：

- `PFILL_READY = 1`
- `ERET_READY = 1`

## 12.5 正式口径

以 raw 快照实际存在的交易日为唯一事实来源。

因此：

- `exec_date` 不得凭自然日历推断
- `target_date` 不得凭自然日历推断
- 必须基于 raw 中真实存在的交易日序列解析

锚定：

**训练日期机制以 raw 事实为准，不以脚本猜测为准。**

## 12.6 统一职责边界（正式锁死）

### A. resolve_sample_maturity.py 的职责

负责：

- 解析交易日序列
- 计算 exec_date / target_date
- 计算 sample_maturity / PFILL_READY / ERET_READY / FULLY_READY
- 生成正式中间产物

不负责：

- 训练
- 真值打标
- 特征拼装
- 模型更新

### B. build_fill_truth.py 的职责

负责：

- 读取候选池
- 读取 T+1 truth
- 生成 fill truth 标签
- 消费成熟度解析结果写入输出

不负责：

- 自己推断 exec_date
- 自己再扫描 raw 判断样本成熟度

正式锁死：

**build_fill_truth.py 不得再承担训练日期推断职责。**

### C. build_pfill_trainset.py 的职责

负责：

- 基于 fill_truth 主表拼装训练集
- merge features_base / features_limit / prior
- 保持对象池不扩张

不负责：

- 重新定义样本成熟逻辑
- 自己推断训练日期

### D. train_pfill.py 的职责

负责：

- 消费 sample_maturity 中 `PFILL_READY=1` 的成熟窗口
- 读取多个 `pfill_trainset_{trade_date}.csv`
- 形成成熟窗口训练集
- 训练 / skip / 写 meta / 落盘模型

不负责：

- 自己再定义成熟样本标准
- 自己再扫描 raw 重建训练日期逻辑

### E. run_decision_daily.yml 的职责

负责：

- 按顺序编排
- 先调用成熟度解析器
- 再调用 fill_truth / trainset / train_pfill

不负责：

- 再嵌入临时的成熟度推断逻辑补丁

正式锁死：

**workflow 不得再长期依赖内联 Python 补丁来决定训练日期。**

## 12.7 该层的系统价值

这一层的存在，目的不是“为了当前 P_fill 跑通”，而是为了：

- 统一训练日期口径
- 降低脚本耦合
- 防止不同模块各写各的时间机制
- 让 P_fill / E_ret / Cost / RiskPenalty 共用同一底层时间语义
- 形成可复用、可追溯、可验收的中间状态层

一句话锚定：

**样本成熟度解析层是学习系统的时间语义底座。**

---

# 13. P_fill 当前正式状态与下一步锚定

## 13.1 当前已完成

P_fill 学习链首轮闭环已经打通，并完成以下演进：

- fill_truth 已改为 pred_source 驱动
- pfill_trainset 已改为以 fill_truth 为主表
- 训练日期机制已从隐式补丁升级为正式基础设施
- train_pfill 已从单日训练升级为成熟窗口训练
- 单一标签 / 样本不足 / 无成熟样本时允许 skip，但不断主链

## 13.2 当前 P_fill 正式链路

正式链路为：

1. 候选池 `pred_source_{trade_date}` / `pred_source_latest`
2. raw / FS
3. `resolve_sample_maturity.py`
4. `fill_truth_{trade_date}.csv`
5. `pfill_trainset_{trade_date}.csv`
6. `train_pfill.py`
7. `models/pfill_*`

## 13.3 后续 P_fill 不得回退

后续不允许再回到以下做法：

- 在 build_fill_truth.py 中临时猜 exec_date
- 在 train_pfill.py 中临时扫 raw 自己找成熟样本
- 在 workflow 中再塞临时 Python 推断逻辑替代正式成熟度层

---

# 14. E_ret / Cost / RiskPenalty 后续复用原则（正式锁死）

## 14.1 E_ret

E_ret / PremiumRet 后续进入学习化时，必须直接复用样本成熟度解析层。

规则：

- 以 `ERET_READY=1` 作为最小训练条件
- 不得再自己重建 T+2 成熟度逻辑
- 若需要更细的收益成熟状态，也应在成熟度层扩展字段，而不是分散到各脚本

## 14.2 Cost

若 Cost 后续学习化：

- 也必须遵守统一时间机制
- 也优先复用样本成熟度解析层
- 若额外需要更长窗口真值，应扩展成熟度层，而不是自己旁路计算

## 14.3 RiskPenalty

若 RiskPenalty 后续学习化：

- 也必须复用同一成熟度基础设施
- 也必须服从统一训练日期机制
- 不允许另建一套并行时间语义

## 14.4 EV

EV 是融合层。

因此：

- EV 不负责底层样本成熟度推断
- EV 只消费前序模块在既定时间机制下产出的结果

---

# 15. 当前阶段正式结论

## 15.1 已经完成

- 决策主链基本跑通
- P_fill 学习链首轮打通
- 候选池驱动规则已明确
- 样本成熟度规则已明确
- 统一时间机制已锁死
- 训练日期基础设施已正式建立
- P_fill 已升级为成熟窗口训练

## 15.2 当前正在做

- 把 P_fill 从“能跑”继续提升到“稳定、可验收、可持续滚动”
- 收紧 y_fill 口径
- 提升标签区分度
- 继续处理单一标签日、样本不足日的正常跳过逻辑
- 为 E_ret 接入做准备

## 15.3 下一阶段

1. 正式进入 E_ret
2. 再进入 Cost / RiskPenalty
3. 最后做 EV 融合与校准

---

# 16. 工程协作规则（保留）

- 一个文件一个文件来
- 每次先给：
  1. 文件全路径
  2. 简要修改报告
  3. 完整可覆盖代码
- 超过 300 行代码文件，优先基于 Raw 链接修改
- 所有交流与输出继续使用简体中文

---

# 17. 最高锚定总结

本系统不是：

- 纯预测展示系统
- 纯规则交易系统
- 当天预测、当天训练同一批样本的混合系统
- 多个脚本各自推断训练日期的松散系统

本系统必须是：

- 候选池驱动
- FS 共享证据层
- 双主线分工明确
- 五模块分层建模
- 统一时间机制
- 样本成熟度中间层明确存在
- 学习模块按正式成熟度结果消费样本
- 允许跳过但不允许乱更新
- 可持续滚动学习的交易决策系统

## 0.3 业务真值边界（正式锁死）
Top10 Decision 的信号文件供第三方使用，第三方实际成交、持仓、滑点与账户收益数据不回流本系统。
因此，系统的验证与学习只允许基于真值层可获得的公开到期数据完成，不得以第三方真实业务收益作为训练标签或验收前提。

系统学习口径正式锁死为：
- P_fill：基于公开行情与固定执行规则定义的可成交代理标签
- EntryPriceProxy_T+1：基于公开行情与固定规则定义的代理买入价
- E_ret：基于 EntryPriceProxy_T+1 与 T+2 真值价格定义的代理收益标签
- RiskPenalty：基于公开行情定义的风险代理标签

一句话锚定：系统追求业务价值，但只能在公开真值可验证边界内学习与迭代。
