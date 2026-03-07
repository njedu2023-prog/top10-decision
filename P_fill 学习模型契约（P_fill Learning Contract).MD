# P_fill 学习模型契约（P_fill Learning Contract）

项目：top10-decision  
适用主线：Decision（机器执行主线）  
状态：主线锚定文件  
优先级：高于训练脚本实现细节，低于《核心算法需求规范（Core Algorithm Spec).MD》

---

## 0. 目标

本文件只锁定三件事：

1. `y_fill` 标签定义  
2. `EntryPriceProxy_T+1` 口径定义  
3. `P_fill` 训练样本结构定义  

目的不是直接产出交易脚本，而是先把 **“可成交性学习问题”** 的监督口径钉死，避免后续训练、回测、执行三层出现偷换。

---

## 1. 时间轴锚定（锁死）

- `T`：信号生成日 / 系统运行日
- `T+1`：实际买入执行日
- `T+2`：卖出/验证日

Decision 主线中：

- `P_fill` 只回答：**该标的在 T+1 是否可买**
- `PremiumRet` 只回答：**若在 T+1 买到，持有到 T+2 的溢价**
- `PredOpen_T+1` 可以作为价格桥变量
- `PredOpen_T+1` 不能直接偷换成真实成交买入价
- 真正用于 EV / 收益计算的是 `EntryPriceProxy_T+1`

---

## 2. P_fill 问题定义（锁死）

### 2.1 问题本质

`P_fill` 是一个 **二分类概率学习问题**：

> 给定 T 日收盘后可见的全部信息，预测该票在 T+1 的设定买入窗口内是否“可成交”。

其输出必须是：

- `p_fill in [0, 1]`

而不是：

- 主观打分
- 固定规则开关
- 未来价格涨跌方向代理

---

## 3. y_fill 标签定义（锁死）

## 3.1 标签名

统一使用：

- `y_fill`

## 3.2 标签含义

`y_fill = 1` 表示：

> 该标的在 T+1 的定义买入窗口内，按系统允许的执行规则，存在“可成交”的现实路径。

`y_fill = 0` 表示：

> 该标的在 T+1 的定义买入窗口内，不具备可成交路径，或应视为不可成交。

---

## 3.3 默认买入窗口定义

默认窗口先锁死为：

- `buy_window_start = 09:30:00`
- `buy_window_end   = 10:30:00`

说明：

1. 这是 **学习默认口径**
2. 执行层未来可参数化，但训练主口径先统一
3. 若未来执行层改为 09:40~14:50 多次重试，那属于执行扩展；`P_fill` 的首版主标签仍先按“上午主买入窗口”建模，避免标签漂移

---

## 3.4 y_fill = 1 的判定条件（首版锁死）

同时满足以下条件时，记为 `y_fill = 1`：

1. T+1 该票不是全天停牌
2. T+1 在买入窗口内存在有效成交机会
3. 该成交机会不要求“买在最低价”，只要求“系统合理可执行”
4. 若开盘即一字涨停且在买入窗口内始终无可进入流动性，则判为不可成交
5. 若盘中开板，并在窗口内出现可成交区间，则判为可成交

---

## 3.5 y_fill = 0 的判定条件（首版锁死）

以下任一成立，记为 `y_fill = 0`：

1. T+1 全天停牌
2. T+1 开盘后一字板封死，至窗口结束前无开板可成交机会
3. T+1 虽有价格打印，但从执行角度仍不可进入（例如全窗口无有效可成交流动性）
4. 关键真值缺失，且无法判断是否可成交时，首版保守记 `0`，同时写入 `fill_label_quality = "weak_missing_truth"`

---

## 3.6 暂不做的复杂标签

首版 **不引入** 以下复杂定义，避免训练集污染：

- 不区分部分成交 / 全部成交
- 不区分买入几成仓位
- 不区分 09:35 成交与 10:25 成交的执行质量差异
- 不把“高滑点成交”单独拆成第三类

这些以后可以升级成：

- `fill_level`
- `fill_ratio`
- `fill_difficulty`

但首版先坚持二分类。

---

## 4. EntryPriceProxy_T+1 定义（锁死）

## 4.1 字段名

统一使用：

- `EntryPriceProxy_T+1`

代码落地时建议列名：

- `entry_price_proxy_t1`

---

## 4.2 定义目标

该值不是“真实成交价”，而是：

> 在 T+1 确认可成交前提下，对系统可实现买入成本的统一代理价格。

用途：

1. 给 `PremiumRet` 提供分母口径
2. 给 `E_ret / EV` 提供可计算的买入价格基准
3. 避免把 `PredOpen_T+1` 偷换成真实买价
4. 避免未来回测时把“理想买点”错当成交价

---

## 4.3 首版口径（锁死）

首版统一采用以下规则：

### A. 若 T+1 开盘非涨停封死，且开盘可进入

则：

- `EntryPriceProxy_T+1 = Open_T+1`

### B. 若 T+1 开盘封板，但买入窗口内首次开板并出现可进入机会

则：

- `EntryPriceProxy_T+1 = FirstTradablePriceInWindow_T+1`

### C. 若全窗口不可成交

则：

- `EntryPriceProxy_T+1 = NaN`
- 同时 `y_fill = 0`

---

## 4.4 FirstTradablePriceInWindow_T+1 的实现口径

若当前数据层只具备日级数据、尚不具备分钟级真实成交轨迹，则首版允许使用近似代理，优先级如下：

### 优先级 1：分钟级首个可成交价
若后续接入分钟数据，直接取：

- 买入窗口内首个可成交分钟 bar 的成交代表价

### 优先级 2：开板后首个可成交参考价
若有开板时刻 + 价格轨迹，则取：

- 首次开板后的首个有效成交参考价

### 优先级 3：日级弱代理（仅冷启动）
若暂无分钟级真值，只允许临时使用：

- `max(Open_T+1, Low_T+1)` 不成立，禁止这种伪逻辑
- 允许定义专门的弱代理列：
  - `entry_price_proxy_mode = "weak_daily_proxy"`
  - `entry_price_proxy_t1 = Open_T+1`（仅限“开盘可买”样本）
- 对于“开盘不可买但盘中可买”的样本，若无分钟真值，禁止伪造精细价格，宁可置空或降级

结论：

> 没有分钟级真值时，首版训练重点先放在 `y_fill`，不要为了凑 EV 强行伪造高精度买入价。

---

## 4.5 明确禁止

以下做法一律禁止：

1. 用 `PredOpen_T+1` 直接替代 `EntryPriceProxy_T+1`
2. 用 `Close_T+1` 直接当买入价
3. 用 `High_T+1 / Low_T+1` 任意挑一个美化收益
4. 在 `y_fill = 0` 时仍给出伪买入价

---

## 5. P_fill 训练样本结构（锁死）

## 5.1 样本粒度

一行样本 = 一只股票在一个 `T` 日的一个候选决策样本

主键：

- `trade_date`
- `ts_code`

可附加：

- `name`
- `rank`
- `run_id`

---

## 5.2 特征时间边界

训练 `P_fill` 时，特征只能来自 **T 日收盘后已知信息**，严禁泄露 T+1 / T+2 未来信息。

允许：

- T 日及以前的 FS 特征
- `pred_source_latest` 在 T 日产出的 prior 特征

禁止：

- T+1 开盘后真实成交结果作为输入特征
- T+1 是否开板后的未来结果字段直接回灌
- T+2 收益字段进入 P_fill 训练特征

---

## 5.3 样本表字段分层

建议训练样本落成一张宽表，逻辑上分 6 层：

### A. 主键与追溯层
- `trade_date`
- `ts_code`
- `name`
- `run_id`
- `commit_sha`
- `source_snapshot_id`

### B. prior 层（来自 pred_source_latest）
- `rank`
- `prob_prior`
- `strength_score`
- `theme_boost`
- `board`
- 其他上游浓缩信号

### C. FS_base 层（基础行情/流动性/风险）
- `open`
- `high`
- `low`
- `close`
- `pct_chg`
- `vol`
- `amount`
- `turnover_rate`
- `volume_ratio`
- `volatility_5d`
- `volatility_10d`
- `ret_2d`
- `ret_5d`
- `ret_10d`
- `total_mv`
- `circ_mv`
- `pe_ttm`
- `pb`
- 其他已入库基础特征

### D. FS_limit 层（P_fill 核心层）
- `is_limit_up`
- `limit_type`
- `open_times`
- `seal_amount`
- `first_seal_time`
- `last_seal_time`
- `break_count`
- `limit_up_strength`
- `limit_up_tags`
- 其他涨停/开板微结构特征

### E. 标签与真值层
- `y_fill`
- `fill_label_quality`
- `entry_price_proxy_t1`
- `entry_price_proxy_mode`

### F. 审计与切分层
- `dataset_split`（train / valid / test）
- `sample_weight`
- `is_cold_start`
- `feature_coverage_score`

---

## 5.4 标签表与特征表的关系

建议逻辑上拆两层，再在训练时合并：

### 特征层
- 来源：`features_base_{trade_date}.csv`
- 来源：`features_limit_{trade_date}.csv`
- 来源：`pred_source_latest` 或其归档快照

### 标签层
新增建议文件：
- `data/market/fill_truth_{trade_date}.csv`

至少包含：
- `trade_date`
- `ts_code`
- `y_fill`
- `fill_label_quality`
- `entry_price_proxy_t1`
- `entry_price_proxy_mode`

训练时按：
- `trade_date + ts_code`
做 join

---

## 6. 冷启动与缺失值策略（锁死）

## 6.1 首版冷启动原则

在分钟级 truth 尚不完整时，先做：

1. `y_fill` 可学优先
2. `entry_price_proxy_t1` 可弱化
3. EV 相关收益学习暂不提前绑死

即：

> 先把“能不能买到”学明白，再学“买到后能赚多少”。

---

## 6.2 缺失处理原则

### 强约束字段缺失
若以下字段缺失严重，应降级样本权重或剔除：

- `open_times`
- `seal_amount`
- `limit_type`
- `is_limit_up`

### 弱约束字段缺失
以下字段可做缺失填充或缺失指示器：

- `theme_boost`
- `moneyflow`
- `pe_ttm`
- `pb`
- 部分题材热度字段

---

## 6.3 样本权重原则

建议后续训练时对以下样本提高权重：

1. Top rank 候选样本
2. 涨停微结构完整样本
3. 实盘最常出现的交易形态样本

不建议一开始对全市场随机平权。

---

## 7. 模型输出契约（先锁死，脚本后接）

P_fill 模型推理后，至少输出：

- `trade_date`
- `ts_code`
- `p_fill`
- `fill_model_version`
- `fill_model_mode`
- `fill_feature_coverage`
- `fill_degrade_reason`

说明：

- `fill_model_mode` 用于标记：
  - `full`
  - `soft`
  - `weak`
- 当 FS_limit 不完整时，允许输出概率，但必须带降级说明

---

## 8. 与后续脚本的接入顺序（锁死）

开发顺序固定为：

### 第一步
先新增标签构建脚本  
目标：生成 `fill_truth_{trade_date}.csv`

### 第二步
新增训练样本拼装脚本  
目标：合并 FS_base + FS_limit + pred + fill_truth

### 第三步
新增 P_fill 训练脚本  
目标：训练并落盘：
- `models/pfill_lgbm.joblib`
- `models/pfill_lr.joblib`
- `models/pfill_meta.json`

### 第四步
再接入 `run_v2.py` 或新的 Decision runner  
只负责：
- 加载模型
- 生成 `p_fill`
- 参与 EV 排序
- 输出执行信号

结论：

> `run_v2.py` 不是 P_fill 口径定义层，它只是消费模型结果的执行层。

---

## 9. 当前主线结论（锁死）

当前阶段，Decision 主线最重要的不是先写复杂交易逻辑，而是先锁死：

1. `y_fill` 到底是什么  
2. `EntryPriceProxy_T+1` 到底是什么  
3. 训练样本怎么构成  

只要这三件事不漂移，后续：
- P_fill 学习模型
- PremiumRet 学习模型
- EV 融合排序
- 滚动训练门控

都能在统一口径下继续推进。

---

## 10. 下一文件建议

下一文件建议直接做：

`scripts/build_fill_truth.py`

职责：

1. 读取 `features_limit_{trade_date}.csv`
2. 读取 T+1 真值/执行辅助数据
3. 生成：
   - `y_fill`
   - `fill_label_quality`
   - `entry_price_proxy_t1`
   - `entry_price_proxy_mode`
4. 落盘到：
   - `data/market/fill_truth_{trade_date}.csv`

这是下一步最专业，也最不跑偏的落点。
