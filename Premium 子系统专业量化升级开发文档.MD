# Premium 子系统专业量化升级开发文档  
## 目标锁定版：D 日分析 → T 日上涨攻击 → T+1 承接兑现  
**文件日期：2026-06-16**  
**适用项目：`njedu2023-prog/top10-decision`**  
**适用范围：仅 Premium 子系统，不推倒 decision 主线，不破坏现有输出链路。**  

---

# 0. 给 Codex 的总指令

本次开发不是修页面、不是改展示字段、不是简单调权重，而是要把 `top10-decision` 的 Premium 子系统升级为一个专业量化级别的双阶段短线预测系统。

Premium 的唯一目标锁定为：

> 在 D 日收盘后，严格按照 A 股交易日历，预测并筛选出 T 日最可能上涨、冲高、触板、封板，且 T+1 仍具备资金承接、冲高卖点和正收益兑现能力的最强标的。

也就是：

```text
D 日收盘后分析
→ T 日竞价/开盘买入或观察
→ T 日上涨/冲板/封板验证
→ T+1 承接/冲高/收盘收益验证
→ 用市场真值回写标签
→ 用标签训练模型
→ 下一轮预测越来越准确
```

Premium 不再追求“单一涨停概率最高”，而是追求：

```text
T 日上涨攻击力
+
T+1 承接兑现能力
+
实盘成交可执行性
+
风险过滤
+
市场真值闭环训练
```

---

# 1. 当前问题诊断

## 1.1 当前 Premium 的主要问题

当前 Premium 已经接入了涨停概率模型、T+1 延续上涨率、涨停接力评分、自适应排序评分等字段，但实盘表现出现明显问题：

1. 部分 Premium Top10 票 T 日或 T+1 表现较差。
2. 一些票只是“涨停概率/攻击评分高”，但 T+1 承接差。
3. Premium 排序绕开了 decision 子系统的稳健过滤能力。
4. `premium_adaptive_score` 权重过大，导致“热度/涨停接力”压过“实际收益/承接/风控”。
5. 市场真值验证链不够强，模型没有及时从失败样本中学习。
6. D / T / T+1 / T+2 的历史字段和报告口径存在残留混杂。
7. `limitup_probability_engine.py` 已有模型，但缺少严格上线门禁。
8. 训练样本、验证样本、市场真值、线上排序之间缺少统一契约。

## 1.2 当前 Premium 应保留的价值

不能推倒重来。当前 Premium 已经有一些有价值模块：

| 模块 | 是否保留 | 说明 |
|---|---:|---|
| `predict.py` 主流程 | 保留并重构排序逻辑 | 当前输出链路完整 |
| `limitup_labels.py` | 保留并强化 | 已有 A 股交易日历意识 |
| `limitup_probability_engine.py` | 保留并加门禁 | 不能无条件参与排序 |
| `premium_verify_*.csv` | 保留并扩展 | 应成为真值回写核心 |
| `premium_top10/top20/top30/full` | 保留 | 但字段要增加 |
| `premium_latest.html/md` | 保留 | 报告结构要专业化 |
| decision merge | 必须强化 | 从“参考字段”升级为“风控门槛” |

---

# 2. Premium 新目标冻结

## 2.1 目标定义

Premium 子系统新目标冻结如下：

```text
Premium 子系统目标：
在 D 日收盘后，从全量候选池中筛选出 T 日最可能上涨/冲板/封板，
且 T+1 仍具备资金承接、冲高卖点和正收益兑现能力的最强标的。

Premium 不追求单日涨停概率最高，
而追求 T 日攻击力 + T+1 承接能力 + 实盘可兑现收益 的综合最优。

最终 Top10 不允许仅凭涨停概率入选；
必须同时满足：
1）T 日上涨攻击力强；
2）T+1 承接概率强；
3）预期收益为正；
4）风险惩罚不过高；
5）decision 层没有明确否决；
6）严格使用 A 股交易日历；
7）所有预测必须能被市场真值验证；
8）所有成熟真值必须回写训练集。
```

## 2.2 时间轴定义

必须严格锁定为：

```text
D   = 分析基准日，D 日收盘后运行预测
T   = D 后第 1 个 A 股交易日，买入/上涨攻击验证日
T+1 = D 后第 2 个 A 股交易日，承接/兑现/卖出验证日
```

禁止把自然日当交易日。  
禁止用单票 shift 直接推断 T / T+1。  
禁止把停牌票向后顺延当成 T / T+1。  
禁止继续把 T+2 字段作为主口径输出。历史兼容字段可以保留，但主报告必须统一为 D / T / T+1。

---

# 3. 严格 A 股交易日历契约

## 3.1 必须使用全市场 A 股交易日历

Premium 所有 D → T → T+1 映射必须来自全市场 A 股交易日历。

优先级：

1. 如果项目已有交易日历文件，例如 `data/calendar/trade_cal.csv`，优先使用；
2. 如果无独立交易日历，则从全市场日线行情 `trade_date` 唯一值构建；
3. 不允许用个股自身行情行号 shift；
4. 不允许用自然日 `+1/+2`；
5. 若交易日历缺失，Premium 主流程必须降级为 pending，不允许输出伪确定结果。

## 3.2 交易日历函数建议

新增或统一函数：

```python
def get_a_share_trade_calendar(cfg) -> list[str]:
    """
    返回升序 YYYYMMDD A股交易日列表。
    来源优先级：
    1. data/calendar/trade_cal.csv
    2. data/market/daily_*.csv 文件名
    3. data/market/raw/**/daily.csv 内 trade_date
    """

def advance_trade_days(trade_date: str, n: int, calendar: list[str]) -> tuple[str | None, str]:
    """
    严格交易日推进。
    n=1 返回 D 后第一个交易日 T。
    n=2 返回 D 后第二个交易日 T+1。
    如果找不到，返回 None 和原因。
    """
```

## 3.3 输出必须带交易日字段

所有 Premium 输出文件必须包含：

| 字段 | 含义 |
|---|---|
| `d_trade_date` / `base_date` | D 日分析日 |
| `t_trade_date` / `buy_date` | T 日买入/攻击验证日 |
| `t1_trade_date` / `target_date` | T+1 承接/兑现验证日 |
| `calendar_source` | 交易日历来源 |
| `calendar_status` | `ok / missing / stale / pending` |
| `calendar_reason` | 交易日历说明 |

---

# 4. 市场真值层 Market Truth Layer

## 4.1 市场真值定义

市场真值不是模型预测，不是前端页面展示，而是 A 股真实行情数据。

必须从真实日线数据生成以下字段：

### D 日真值

| 字段 | 含义 |
|---|---|
| `d_open` | D 日开盘 |
| `d_high` | D 日最高 |
| `d_low` | D 日最低 |
| `d_close` | D 日收盘 |
| `d_pre_close` | D 日昨收 |
| `d_pct_chg` | D 日涨跌幅 |
| `d_amount` | D 日成交额 |
| `d_volume` | D 日成交量 |

### T 日真值

| 字段 | 含义 |
|---|---|
| `t_open` | T 日开盘 |
| `t_high` | T 日最高 |
| `t_low` | T 日最低 |
| `t_close` | T 日收盘 |
| `t_pre_close` | T 日昨收 |
| `t_limit_price` | T 日涨停价 |
| `t_up_hit` | T 日收盘是否上涨 |
| `t_open_up_hit` | T 日开盘是否上涨 |
| `t_high_profit_hit` | T 日盘中是否给过利润 |
| `t_touch_limitup` | T 日盘中是否触板 |
| `t_limitup_hit` | T 日是否收盘涨停 |
| `t_intraday_ret` | T 日最高相对 D 收盘收益 |
| `t_close_ret` | T 日收盘相对 D 收盘收益 |

### T+1 真值

| 字段 | 含义 |
|---|---|
| `t1_open` | T+1 开盘 |
| `t1_high` | T+1 最高 |
| `t1_low` | T+1 最低 |
| `t1_close` | T+1 收盘 |
| `t1_pre_close` | T+1 昨收 |
| `t1_open_ret` | T+1 开盘相对 T 收盘收益 |
| `t1_high_ret` | T+1 最高相对 T 收盘收益 |
| `t1_close_ret` | T+1 收盘相对 T 收盘收益 |
| `t1_up_hit` | T+1 收盘是否上涨 |
| `t1_high_profit_hit` | T+1 是否给过可兑现收益 |
| `t1_accept_hit` | T+1 是否承接成功 |
| `t1_fail_hit` | T+1 是否承接失败 |
| `t1_big_drawdown_hit` | T+1 是否大回撤 |
| `t1_limitdown_risk_hit` | T+1 是否接近/触及跌停风险 |

## 4.2 T+1 承接成功定义

建议第一版定义：

```python
t1_accept_hit = (
    (t1_high_ret >= 0.015)
    & (t1_close_ret >= -0.015)
)
```

含义：

1. T+1 至少给过 1.5% 以上可兑现冲高；
2. T+1 收盘不能明显砸穿；
3. 重点不是必须收盘大涨，而是要有实盘可兑现卖点。

## 4.3 T+1 承接失败定义

建议第一版定义：

```python
t1_fail_hit = (
    (t1_high_ret < 0.008)
    | (t1_close_ret <= -0.025)
)
```

含义：

1. T+1 没有给出足够卖点；
2. 或者收盘明显下跌；
3. 这类票即使 T 日强，也不能进入 Premium 最强票。

## 4.4 大跌风险标签

新增：

```python
t1_big_drawdown_hit = (
    (t1_low / t_close - 1.0) <= -0.04
)
```

如果 T+1 盘中较 T 收盘跌超 4%，判定为明显风险样本。

---

# 5. 新核心字段设计

Premium 必须新增 3 个一级核心字段：

```text
t_up_attack_score      # T 日上涨攻击力
t1_accept_score        # T+1 承接能力
premium_final_score    # 最终综合评分
```

---

## 5.1 `t_up_attack_score`：T 日上涨攻击力

目标：判断 T 日是否能上涨、冲高、触板、封板。

建议组成：

```python
t_up_attack_score =
    0.20 * t_up_prob
  + 0.25 * t_touch_limitup_prob
  + 0.25 * t_limitup_prob
  + 0.15 * t_limitup_strength_score
  + 0.10 * eret_plus_score
  + 0.05 * market_emotion_score
```

字段说明：

| 字段 | 来源 |
|---|---|
| `t_up_prob` | 新增模型目标：T 日收盘上涨概率 |
| `t_touch_limitup_prob` | 当前 `t_touch_limitup_prob_model` |
| `t_limitup_prob` | 当前 `t_limitup_prob` |
| `t_limitup_strength_score` | 当前 `t_limitup_strength / 100` |
| `eret_plus_score` | `eret_plus_value` 横截面分位 |
| `market_emotion_score` | 市场情绪分 |

注意：  
当前模型有 `t_limitup_prob_model` 和 `t_touch_limitup_prob_model`，但缺少单独的 `t_up_prob_model`。必须补上。

---

## 5.2 `t1_accept_score`：T+1 承接能力

目标：判断 T+1 是否还能承接、冲高、给卖点。

建议组成：

```python
t1_accept_score =
    0.25 * t1_up_prob
  + 0.25 * t1_accept_prob
  + 0.20 * t1_high_profit_prob
  + 0.15 * t1_close_ret_pred_score
  + 0.10 * t1_high_ret_pred_score
  + 0.05 * execution_safety_score
```

字段说明：

| 字段 | 来源 |
|---|---|
| `t1_up_prob` | 当前 `t1_up_prob_model` |
| `t1_accept_prob` | 新增模型目标 |
| `t1_high_profit_prob` | 当前 `t1_high_profit_prob_model` |
| `t1_close_ret_pred_score` | `t1_close_ret_pred` 横截面分位 |
| `t1_high_ret_pred_score` | `t1_high_ret_pred` 横截面分位 |
| `execution_safety_score` | 成交/风险/置信度综合 |

注意：  
`t1_accept_score` 是本次升级最重要字段。  
Premium 以后不能只看 T 日强，必须重点看 T+1 是否有承接。

---

## 5.3 `premium_final_score`：最终综合评分

建议公式：

```python
premium_final_score =
    0.30 * t_up_attack_score
  + 0.35 * t1_accept_score
  + 0.20 * eret_plus_score
  + 0.10 * execution_score
  + 0.05 * market_score
  - risk_penalty_score
```

解释：

| 组成 | 权重 | 原因 |
|---|---:|---|
| T 日上涨攻击力 | 30% | 必须能涨、能冲 |
| T+1 承接能力 | 35% | 核心痛点，权重最高 |
| 预期收益能力 | 20% | 不能没有收益空间 |
| 实盘执行能力 | 10% | 买得到、卖得出 |
| 市场情绪 | 5% | 顺势加分 |
| 风险惩罚 | 扣分 | 防止大跌票进入 Top10 |

---

# 6. Top10 入选硬门槛

Premium Top10 不能只靠评分排序，必须先过硬门槛。

## 6.1 建议硬门槛

```python
eligible_for_premium_top10 = (
    (dec_can_buy != 0)
    & (score_ev >= 0)
    & (eret_plus_value >= -0.003)
    & (t_up_attack_score >= 0.55)
    & (t1_accept_score >= 0.52)
    & (t1_high_profit_prob_model >= 0.50)
    & (risk_penalty_score <= 0.60)
)
```

## 6.2 放弃池

满足任一条件，不能进 Top10：

```python
force_exclude_top10 = (
    (dec_can_buy == 0)
    | (score_ev < -0.002)
    | (eret_plus_value < -0.008)
    | (t1_accept_score < 0.45)
    | (t1_big_drawdown_prob >= 0.35)
    | (risk_penalty_score >= 0.75)
)
```

## 6.3 观察池

没有被强排除，但不够强的票进入观察池：

```text
只观察不追
```

观察池可以展示，但不能进入 Premium Top10 主买入表。

---

# 7. 排序逻辑重构

## 7.1 当前排序问题

当前 `predict.py` 的排序逻辑大概率是：

```python
sort by [
    "premium_adaptive_score",
    "t_limitup_prob",
    "t1_alpha_score",
    "t_limitup_strength"
]
```

这会导致涨停概率/自适应分过度主导。

## 7.2 新排序逻辑

必须改成：

```python
df_eligible = df[eligible_for_premium_top10].copy()
df_watch = df[~eligible_for_premium_top10 & ~force_exclude_top10].copy()
df_excluded = df[force_exclude_top10].copy()

df_eligible = df_eligible.sort_values(
    by=[
        "premium_final_score",
        "t1_accept_score",
        "t_up_attack_score",
        "eret_plus_value",
        "t1_high_profit_prob_model",
        "t_limitup_prob",
    ],
    ascending=[False, False, False, False, False, False],
    na_position="last",
)
```

最终：

```python
top10 = df_eligible.head(10)
top20 = concat(top10 + df_eligible.next10)
top30 = concat(eligible remaining + watch strongest)
full = eligible + watch + excluded
```

注意：  
如果合格池不足 10 只，允许用观察池补齐，但必须在报告中标注：

```text
Top10 中存在观察池补位标的，不建议无脑买入。
```

---

# 8. 模型目标升级

## 8.1 当前模型目标

当前 `limitup_probability_engine.py` 目标包括：

```text
t_limitup_hit
t_touch_limitup
t1_up_hit
t1_high_profit_hit
t1_close_ret
t1_high_ret
```

## 8.2 新增模型目标

必须新增：

```text
t_up_hit                  # T 日收盘上涨
t_high_profit_hit          # T 日盘中给过收益
t1_accept_hit              # T+1 承接成功
t1_fail_hit                # T+1 承接失败
t1_big_drawdown_hit        # T+1 大回撤风险
```

## 8.3 新预测字段

模型预测输出必须新增：

| 字段 | 含义 |
|---|---|
| `t_up_prob_model` | T 日上涨概率 |
| `t_high_profit_prob_model` | T 日盘中给利润概率 |
| `t1_accept_prob_model` | T+1 承接成功概率 |
| `t1_fail_prob_model` | T+1 承接失败概率 |
| `t1_big_drawdown_prob_model` | T+1 大回撤风险概率 |

## 8.4 模型类型建议

分类目标：

```text
HistGradientBoostingClassifier / LightGBM Classifier
```

回归目标：

```text
HistGradientBoostingRegressor / LightGBM Regressor
```

短线系统建议保留 LightGBM 优先，但必须有 sklearn fallback。

---

# 9. 训练样本工程

## 9.1 训练样本来源

训练样本必须来自：

```text
历史 D 日候选池
+
D 日可见特征
+
T 日市场真值
+
T+1 市场真值
```

不能使用 T/T+1 之后才知道的字段作为 D 日特征。

## 9.2 严禁未来函数

禁止把以下字段作为特征：

```text
t_open
t_high
t_low
t_close
t_limitup_hit
t_touch_limitup
t_up_hit
t1_open
t1_high
t1_low
t1_close
t1_up_hit
t1_accept_hit
t1_close_ret
t1_high_ret
actual_ret
hit_up
label_valid
label_matured
```

## 9.3 特征白名单机制

当前自动识别数值特征太宽，容易误吸标签或旧模型字段。

必须新增特征白名单/黑名单双机制：

```python
FEATURE_ALLOW_PREFIXES = [
    "d_",
    "factor_",
    "mkt_",
    "dec_",
    "eret_",
    "pfill_",
    "risk_",
    "theme_",
    "amount_",
    "ret_",
    "vol_",
    "close_pos_",
]

FEATURE_DENY_KEYWORDS = [
    "actual",
    "hit",
    "label",
    "t_open",
    "t_high",
    "t_low",
    "t_close",
    "t1_open",
    "t1_high",
    "t1_low",
    "t1_close",
    "verify",
    "future",
]
```

只有通过白名单且不触发黑名单的字段才能进模型。

---

# 10. 模型上线门禁

模型不是训练成功就能参与排序。必须有上线门禁。

## 10.1 分类模型门禁

每个分类目标必须记录：

| 指标 | 说明 |
|---|---|
| `samples` | 验证样本数 |
| `positive_rate` | 正样本比例 |
| `auc` | 排序能力 |
| `brier` | 概率校准误差 |
| `precision_top10` | Top10 命中率 |
| `recall` | 召回 |
| `calibration_ece` | 校准误差 |

## 10.2 回归模型门禁

每个回归目标必须记录：

| 指标 | 说明 |
|---|---|
| `samples` | 样本数 |
| `mae` | 平均绝对误差 |
| `rmse` | 均方根误差 |
| `spearman_ic` | 横截面排序相关 |
| `positive_ic_rate` | IC 为正比例 |

## 10.3 最低上线条件

建议第一版：

```python
model_can_rank = (
    validation_days >= 20
    and validation_samples >= 500
    and t1_accept_auc >= 0.53
    and t1_accept_brier <= 0.25
    and t1_ret_spearman_ic_mean > 0
)
```

如果不满足：

```text
模型可以输出字段，但不能参与 premium_final_score 主排序。
```

报告必须显示：

```text
model_rank_mode = disabled_validation_not_pass
```

---

# 11. 市场真值验证报告

## 11.1 每日验证报告必须输出

文件建议：

```text
outputs/premium/validation/premium_validation_{D}.csv
docs/reports/premium_validation_{D}.md
```

## 11.2 验证报告核心指标

### TopK 表现

| 指标 | 说明 |
|---|---|
| Top1 T 日上涨命中率 | `t_up_hit` |
| Top3 T 日上涨命中率 | `t_up_hit` |
| Top5 T 日上涨命中率 | `t_up_hit` |
| Top10 T 日上涨命中率 | `t_up_hit` |
| Top10 T 日触板率 | `t_touch_limitup` |
| Top10 T 日封板率 | `t_limitup_hit` |
| Top10 T+1 承接成功率 | `t1_accept_hit` |
| Top10 T+1 可兑现收益率 | `t1_high_profit_hit` |
| Top10 T+1 收盘上涨率 | `t1_up_hit` |
| Top10 T+1 大回撤率 | `t1_big_drawdown_hit` |

### 收益指标

| 指标 | 说明 |
|---|---|
| Top1 T+1 平均最高收益 | `mean(t1_high_ret)` |
| Top3 T+1 平均最高收益 | `mean(t1_high_ret)` |
| Top5 T+1 平均最高收益 | `mean(t1_high_ret)` |
| Top10 T+1 平均最高收益 | `mean(t1_high_ret)` |
| Top10 T+1 平均收盘收益 | `mean(t1_close_ret)` |
| Top10 中位数收益 | `median(t1_close_ret)` |
| Top10 最差收益 | `min(t1_close_ret)` |
| Top10 胜率 | `mean(t1_close_ret > 0)` |

### 排序质量

| 指标 | 说明 |
|---|---|
| `premium_final_score_spearman_ic` | 最终评分与 T+1 收益相关 |
| `t_up_attack_score_ic` | T 攻击力与 T 日收益相关 |
| `t1_accept_score_ic` | T+1 承接分与 T+1 收益相关 |
| `limitup_prob_ic` | 涨停概率与封板结果相关 |
| `eret_plus_ic` | E_ret_plus 与真实收益相关 |

---

# 12. 自学习闭环

## 12.1 每日闭环顺序

GitHub Actions / 本地流程必须按顺序：

```text
1. 拉取/确认最新 A 股日线行情
2. 构建或更新 A 股交易日历
3. 对历史已成熟 D 样本生成 T/T+1 真值标签
4. 更新 training_samples
5. 训练/更新 Premium 模型
6. 生成模型验证指标
7. 判断模型是否通过上线门禁
8. 对最新 D 日候选池预测
9. 生成 Premium Top10/20/30/full
10. 生成报告
11. 回写 last_run 审计信息
```

## 12.2 模型训练产物

建议输出：

```text
outputs/premium/models/premium_attack_accept_model.joblib
outputs/premium/models/premium_attack_accept_metrics.csv
outputs/premium/models/premium_attack_accept_feature_importance.csv
outputs/premium/learning/premium_training_samples.csv
outputs/premium/learning/premium_training_manifest.json
```

## 12.3 manifest 必须包含

```json
{
  "created_at": "...",
  "train_start_date": "YYYYMMDD",
  "train_end_date": "YYYYMMDD",
  "valid_start_date": "YYYYMMDD",
  "valid_end_date": "YYYYMMDD",
  "sample_rows": 0,
  "valid_rows": 0,
  "feature_count": 0,
  "label_targets": [],
  "model_can_rank": false,
  "reason": ""
}
```

---

# 13. 报告输出格式升级

## 13.1 Premium 主报告必须分区

报告建议结构：

```text
一、Premium 今日结论
二、Top10 最强标的表
三、T 日上涨攻击力排序
四、T+1 承接能力排序
五、风险剔除/观察池
六、市场真值验证
七、模型训练与上线门禁状态
八、字段解释与审计信息
```

## 13.2 Top10 主表字段

必须包含：

| 字段 | 说明 |
|---|---|
| `rank` | 最终排名 |
| `ts_code` | 股票代码 |
| `name` | 名称 |
| `sector` | 板块 |
| `base_date` | D 日 |
| `buy_date` | T 日 |
| `target_date` | T+1 日 |
| `close_T` | D 日或基准收盘价，字段命名需校正 |
| `premium_final_score` | 最终评分 |
| `t_up_attack_score` | T 日上涨攻击力 |
| `t1_accept_score` | T+1 承接能力 |
| `t_up_prob_model` | T 日上涨概率 |
| `t_touch_limitup_prob_model` | T 日触板概率 |
| `t_limitup_prob` | T 日封板概率 |
| `t1_accept_prob_model` | T+1 承接概率 |
| `t1_high_profit_prob_model` | T+1 可兑现收益概率 |
| `eret_plus_value` | 预期收益 |
| `score_ev` | EV |
| `risk_penalty_score` | 风险惩罚 |
| `T日建议买入方式` | 实盘建议 |
| `T日可接受买入价` | 买入价 |
| `T+1卖出计划` | 卖出计划 |
| `入选理由` | 为什么入选 |
| `风险提示` | 为什么可能失败 |

---

# 14. 代码修改清单

## 14.1 第一优先级：`src/top10decision/premium/predict.py`

必须修改：

1. 新增 `_build_t_up_attack_score(df)`。
2. 新增 `_build_t1_accept_score(df)`。
3. 新增 `_build_premium_final_score(df)`。
4. 新增 `_build_premium_eligibility(df)`。
5. 修改最终排序逻辑。
6. 修改 `out_cols`，加入新字段。
7. 修改 `_write_last_run`，输出新评分审计信息。
8. 修改执行建议逻辑，让 `t1_accept_score` 参与“市价竞价/限价竞价/只观察/放弃”。

建议新增字段：

```text
t_up_prob_model
t_high_profit_prob_model
t1_accept_prob_model
t1_fail_prob_model
t1_big_drawdown_prob_model

t_up_attack_score
t1_accept_score
eret_plus_score
execution_score
market_score
risk_penalty_score
premium_final_score

premium_eligible
premium_exclude_reason
premium_bucket
model_rank_mode
```

---

## 14.2 第二优先级：`src/top10decision/premium/limitup_labels.py`

必须修改：

1. 增加 T 日上涨标签。
2. 增加 T 日可兑现收益标签。
3. 增加 T+1 承接成功标签。
4. 增加 T+1 承接失败标签。
5. 增加 T+1 大回撤风险标签。
6. 严格输出 `d_trade_date / t_trade_date / t1_trade_date`。
7. 输出 `label_valid / label_matured / calendar_source / label_as_of`。
8. 遇到停牌/缺行情，不允许顺延，必须 `label_valid=0`。

新增标签：

```text
t_up_hit
t_high_profit_hit
t1_accept_hit
t1_fail_hit
t1_big_drawdown_hit
```

---

## 14.3 第三优先级：`src/top10decision/premium/limitup_probability_engine.py`

必须修改：

1. 扩展分类目标。
2. 增加模型上线门禁。
3. 增加特征白名单/黑名单。
4. 增加 feature importance 输出。
5. 增加 `model_can_rank` 字段。
6. 预测时如果缺失特征过多，输出 `model_quality_flag=degraded`。
7. 不达标模型只能展示，不能参与最终排序。

分类目标改为：

```python
CLASS_TARGETS = [
    "t_up_hit",
    "t_high_profit_hit",
    "t_limitup_hit",
    "t_touch_limitup",
    "t1_up_hit",
    "t1_high_profit_hit",
    "t1_accept_hit",
    "t1_fail_hit",
    "t1_big_drawdown_hit",
]
```

回归目标保留：

```python
REG_TARGETS = [
    "t_close_ret",
    "t_intraday_ret",
    "t1_close_ret",
    "t1_high_ret",
]
```

---

## 14.4 第四优先级：报告层

涉及文件可能包括：

```text
src/top10decision/premium/report_md.py
src/top10decision/premium/premium_views.py
premium_views.py
docs/reports/premium_latest.md
docs/reports/premium_latest.html
decision.html
```

要求：

1. 主表展示 `premium_final_score`。
2. 明确分开 T 日攻击力和 T+1 承接能力。
3. 显示模型是否通过上线门禁。
4. 显示市场真值验证结果。
5. 显示观察池和剔除池。
6. 不再把 `T+2卖出计划` 作为主字段。
7. 页面不允许把字段错位、换行导致信息不可读。

---

## 14.5 第五优先级：Workflow

可能涉及：

```text
.github/workflows/run_premium.yml
scripts/rebuild_premium_reports.py
scripts/build_*.py
```

要求：

1. 先更新市场真值。
2. 再更新训练样本。
3. 再训练模型。
4. 再预测。
5. 再生成报告。
6. 再提交 outputs/docs。

---

# 15. 验收标准

## 15.1 工程验收

必须满足：

```text
python -m compileall src/top10decision/premium
```

通过。

必须能跑：

```text
python -m top10decision.premium.predict
```

或当前项目已有入口能正常生成：

```text
outputs/premium/premium_top10_{D}.csv
outputs/premium/premium_top20_{D}.csv
outputs/premium/premium_top30_{D}.csv
outputs/premium/premium_full_{D}.csv
outputs/premium/premium_verify_{D}.csv
docs/reports/premium_{D}.md
docs/reports/premium_{D}.html
```

## 15.2 字段验收

`premium_top10_{D}.csv` 必须包含：

```text
premium_final_score
t_up_attack_score
t1_accept_score
t_up_prob_model
t1_accept_prob_model
t1_high_profit_prob_model
t1_big_drawdown_prob_model
premium_eligible
premium_bucket
premium_exclude_reason
model_rank_mode
```

## 15.3 交易日历验收

必须在 `_last_run.txt` 中输出：

```text
calendar_contract: strict_a_share_trade_calendar_only
analysis_date_calendar_reason: ...
buy_date_calendar_reason: ...
target_date_calendar_reason: ...
```

如果交易日历不可用，不能伪造结果。

## 15.4 真值验收

成熟样本必须能生成：

```text
t_up_hit
t_limitup_hit
t_touch_limitup
t1_up_hit
t1_high_profit_hit
t1_accept_hit
t1_fail_hit
t1_big_drawdown_hit
t1_close_ret
t1_high_ret
```

## 15.5 模型验收

必须输出：

```text
model_can_rank
model_rank_mode
validation_days
validation_samples
t1_accept_auc
t1_accept_brier
t1_ret_spearman_ic_mean
```

如果模型不达标：

```text
model_rank_mode = disabled_validation_not_pass
```

且模型字段不得进入最终排序主权重。

## 15.6 排名验收

Top10 不能出现明显以下情况：

1. `score_ev` 明显为负仍排名靠前；
2. `eret_plus_value` 明显为负仍排名靠前；
3. `t1_accept_score` 过低仍排名靠前；
4. `t1_big_drawdown_prob_model` 高仍排名靠前；
5. decision 明确否决仍排名靠前；
6. 只有涨停概率高但 T+1 承接弱仍排名靠前。

---

# 16. 禁止事项

本次开发禁止：

1. 禁止推倒 decision 主线。
2. 禁止改动 Tushare Token / 数据源密钥。
3. 禁止把自然日当交易日。
4. 禁止用个股 shift 替代全市场交易日历。
5. 禁止把 T+1 之后才知道的数据作为 D 日特征。
6. 禁止无门禁让模型接管排序。
7. 禁止只改页面不改内核。
8. 禁止只调权重不加真值验证。
9. 禁止删除旧输出文件路径，必须兼容当前前端。
10. 禁止让 `premium_adaptive_score` 继续作为唯一主排序字段。
11. 禁止把观察池票伪装成强买入票。
12. 禁止报告只说“高概率”，必须说明概率来源、模型状态和验证结果。

---

# 17. 推荐开发顺序

## 阶段一：先锁目标和字段

修改：

```text
predict.py
```

完成：

1. 新增 `t_up_attack_score`。
2. 新增 `t1_accept_score`。
3. 新增 `premium_final_score`。
4. 新增 eligibility 硬门槛。
5. 改最终排序。

这是第一优先级。

---

## 阶段二：补真值标签

修改：

```text
limitup_labels.py
```

完成：

1. 增加 T / T+1 标签。
2. 严格交易日历。
3. 输出承接成功/失败/大回撤标签。

---

## 阶段三：升级模型

修改：

```text
limitup_probability_engine.py
```

完成：

1. 增加模型目标。
2. 增加白名单特征。
3. 增加上线门禁。
4. 增加 metrics/manifest。

---

## 阶段四：报告专业化

修改：

```text
report_md.py
premium_views.py
```

完成：

1. 展示三大分数。
2. 展示真值验证。
3. 展示模型门禁。
4. 展示观察池/剔除池。

---

## 阶段五：工作流闭环

修改：

```text
run_premium.yml
scripts/rebuild_premium_reports.py
```

完成：

1. 市场真值 → 标签 → 训练 → 验证 → 预测 → 报告。
2. 每天自动更新模型学习样本。
3. 每天输出验证报告。

---

# 18. 最终验收口令

Codex 开发完成后，请按以下顺序验收：

```bash
python -m compileall src/top10decision/premium
```

然后运行 Premium 主流程，检查：

```text
outputs/premium/_last_run.txt
outputs/premium/premium_top10_{D}.csv
outputs/premium/premium_top20_{D}.csv
outputs/premium/premium_top30_{D}.csv
outputs/premium/premium_full_{D}.csv
outputs/premium/premium_verify_{D}.csv
docs/reports/premium_latest.md
docs/reports/premium_latest.html
```

重点检查：

```text
1. 是否严格 A 股交易日历；
2. 是否输出 D/T/T+1；
3. 是否新增 premium_final_score；
4. 是否新增 t_up_attack_score；
5. 是否新增 t1_accept_score；
6. 是否新增 t1_accept_prob_model；
7. 是否新增 t1_big_drawdown_prob_model；
8. 是否有 model_can_rank；
9. 模型不达标时是否禁止参与主排序；
10. Top10 是否不再被单一涨停概率主导；
11. Top10 是否同时满足 T 日攻击和 T+1 承接；
12. 报告是否输出市场真值验证。
```

---

# 19. 本次升级一句话总结

Premium 必须从：

```text
涨停接力热度排序系统
```

升级为：

```text
D 日分析 → T 日上涨攻击 → T+1 承接兑现 → 市场真值验证 → 自学习训练 的专业量化预测系统。
```

最终目标不是“看起来分数高”，而是：

```text
能提前找出 T 日最强，且 T+1 还能承接、能卖出、能兑现收益的票。
```
