# 0612 Premium 主线开发无缝衔接文档

## 一、当前主线定位

继续 `top10-decision` 的 **Premium 主线**。

当前阶段已经从：

```text
Premium V3 / E_ret_plus / EHX model_v1 接线完成后的实盘执行层增强
```

进一步切换为：

```text
Premium V3.5：涨停接力实盘评分排序阶段
```

当前最高目标不是继续修 workflow，也不是回头改 Decision，而是：

```text
把 Premium 从“买后溢价预测器”
升级为
“D 日分析 → T 日竞价买入 → T+1 延续上涨/卖出”的涨停接力实盘辅助系统
```

---

## 二、当前系统两层结构不要混淆

`top10-decision` 下仍然是两套系统：

### 1. Decision 主线

Decision 负责判断：

```text
能不能买
买不买得到
成本是否可接受
风险是否可接受
EV 是否值得做
```

当前阶段不要回头重做 Decision。

### 2. Premium 主线

Premium 原始核心负责判断：

```text
买到之后，到预测到期日是否还有可兑现溢价
```

当前已新增目标：

```text
找到 T 日有涨停攻击能力，并且 T+1 还能继续上涨的票
```

所以当前 Premium 的新定位是：

```text
买后溢价判断器 + 涨停接力评分器 + 隔夜竞价执行参考器
```

---

## 三、当前交易时间轴必须锁死

真实操作流程已经重新校准为：

```text
D：分析基准日，使用 D 日收盘价 Close[D]
T：下一交易日早上集合竞价买入
T+1：买入后下一个交易日，盘中择时卖出
```

报告顶部已经改为：

```text
预测日（D）：20260609
竞价买入日（T）：20260610
预测到期日（T+1）：20260611
周期：2 个交易日（D→T+1）
```

注意：

```text
T 是买入日，不是分析日。
T+1 是卖出/验证日，不是买入日。
```

---

## 四、0612 已完成的重要修改

### 1. predict.py 已升级为 V3.5 涨停接力评分排序版

文件：

```text
src/top10decision/premium/predict.py
```

当前版本核心说明：

```text
Premium 子系统 — Predict（V3.5：涨停接力评分排序版）
```

已经新增：

```text
_build_limitup_continuation_fields(df)
```

新增字段：

```text
T日涨停概率
T日涨停强度
T+1延续上涨率
涨停接力评分
```

同步英文程序字段：

```text
t_limitup_prob
t_limitup_strength
t1_continue_up_rate
limitup_continuation_score
```

### 2. 操作排名逻辑已从 E_ret_plus 改为涨停接力评分

旧排序逻辑：

```text
主要按 eret_plus_value / rank_eret_plus 排序
```

新排序逻辑：

```text
操作排名 = 按 涨停接力评分 降序排列后的 rank
```

同分排序顺序：

```text
涨停接力评分
T日涨停概率
T+1延续上涨率
T日涨停强度
```

保留审计字段：

```text
rank_limitup_continuation
rank_eret_plus
rank_r_p50
```

### 3. report_md.py 已升级为 V3.5 / V3.6 展示口径

文件：

```text
src/top10decision/premium/report_md.py
```

当前主表已收窄为实盘 9 列：

```text
操作排名｜代码｜名称｜D日收盘价｜T日涨停概率｜T日涨停强度｜T+1延续上涨率｜涨停接力评分｜T日建议买入方式
```

已修正：

```text
T+1建议买入方式 → T日建议买入方式
```

原因：

```text
T 是竞价买入日
T+1 是卖出日
```

### 4. 表格显示已改为不换行、自由撑开

`report_md.py` 中 `_df_to_html_table()` 已改为手写 HTML 表格：

目标：

```text
表格自由撑开
内容不换行
避免“操作排名”“T日涨停概率”等表头被拆成两行
```

当前报告显示已经基本达到：

```text
横向撑开
字段不换行
表格可读性增强
```

### 5. 新报告已生成

当前报告路径：

```text
docs/reports/premium_20260609.md
docs/reports/premium_latest.md
```

报告已显示：

```text
Premium（涨停接力实盘预测）V3.5（D→T→T+1）
```

报告主表已显示：

```text
操作排名
代码
名称
D日收盘价
T日涨停概率
T日涨停强度
T+1延续上涨率
涨停接力评分
T日建议买入方式
```

---

## 五、当前报告验收状态

### 已通过

```text
workflow 能跑通
报告能生成
CSV 能生成
EHX model_v1 仍接通
D/T/T+1 时间轴已显示正确
涨停接力字段已进入报告
操作排名已按涨停接力评分排序
T日建议买入方式 已修正
表格显示不再明显换行
```

### 需要继续注意

当前报告中出现：

```text
T日涨停概率 90%+
T+1延续上涨率 80%+
T日建议买入方式：市价竞价
```

这些数值当前不是严格真实概率，而是规则评分层映射出来的结果。

所以现在更准确的理解是：

```text
T日涨停概率 ≈ T日涨停倾向评分
T+1延续上涨率 ≈ T+1延续倾向评分
```

不要把 96% 理解成“明天真实有 96% 概率涨停”。

---

## 六、当前可信度判断

当前 Premium 原始内核：

```text
E_ret_plus / EHX model_v1 / 分布预测
```

可信度暂估：

```text
5.5 / 10
```

新增涨停接力模块：

```text
T日涨停概率
T日涨停强度
T+1延续上涨率
涨停接力评分
```

可信度暂估：

```text
3.5 / 10
```

整体实盘可参考度：

```text
4.5 / 10
```

原因：

```text
工程链路已接通
字段方向正确
排序目标正确
但涨停接力模块目前是规则评分层，不是经过历史标签训练的真实概率模型
```

---

## 七、当前最重要的问题

### 1. “概率”命名可能误导

当前字段：

```text
T日涨停概率
T+1延续上涨率
```

从程序底层看更像：

```text
T日涨停评分
T+1延续评分
```

后续建议考虑改名，避免误导实盘。

### 2. 市价竞价建议偏激进

当前前排大量出现：

```text
市价竞价
```

这说明执行安全约束还不够强。

后续需要增加：

```text
高价股惩罚
北交所惩罚
流动性惩罚
高开接盘风险惩罚
连续加速末端惩罚
一字买不到惩罚
```

### 3. E_ret_plus 当前验证不佳

当前报告显示：

```text
Raw MAE：5.9271%
Plus MAE：9.0179%
MAE 改善：-3.0909%
Plus 优于 Raw 比例：30.00%
```

说明当前样本中：

```text
E_ret_plus 比 raw 更差
```

这不是 workflow 问题，而是模型有效性/样本适配问题，后续需要继续验证。

---

## 八、下一阶段建议：从“规则评分”升级为“可验证计算闭环”

当前不要再优先改前端。

下一阶段应优先做离线验证和标签构建。

### 第一优先级：新增历史标签构建文件

建议新建：

```text
src/top10decision/premium/limitup_labels.py
```

作用：

根据历史行情，生成这些真实标签：

```text
t_limitup_hit        # T日是否收盘涨停
t_touch_limitup      # T日是否盘中触及涨停
t1_up_hit            # T+1是否上涨
t1_high_profit_hit   # T+1盘中是否给过可兑现收益
t1_close_ret         # T+1收盘收益
t1_high_ret          # T+1最高价收益
```

### 第二优先级：新增涨停接力回测文件

建议新建：

```text
src/top10decision/premium/limitup_backtest.py
```

作用：

回测当前 `涨停接力评分` 是否真的有效。

固定输出：

```text
Top5 T日涨停率
Top10 T日涨停率
Top5 T日触板率
Top10 T日触板率
Top5 T+1上涨率
Top10 T+1上涨率
Top5 平均T+1最高收益
Top10 平均T+1最高收益
最大单票亏损
市价竞价建议命中率
限价竞价建议命中率
```

### 第三优先级：再回到 predict.py 优化评分公式

等回测有结果后，再优化：

```text
_build_limitup_continuation_fields(df)
```

重点：

```text
权重参数化
加入硬过滤
加入风险惩罚
降低伪概率误导
提高实盘可买性
```

---

## 九、后续不要做的事

当前不要：

```text
不要回头改 Decision 主线
不要重做 train.py
不要重做 run_premium.yml
不要继续堆表头字段
不要把操作结论/风险提示加回来
不要把 Premium 报告重新变成技术审计表
不要把 T+1 买入写回来
```

尤其注意：

```text
T日建议买入方式
```

不能再写成：

```text
T+1建议买入方式
```

---

## 十、下一次直接使用的开发口令

下次继续时，直接输入：

```text
继续 top10-decision Premium 主线。

当前阶段是 Premium V3.5 / 涨停接力实盘评分排序阶段。

上一轮已经完成：
1）predict.py 升级为 V3.5：涨停接力评分排序版；
2）新增 _build_limitup_continuation_fields(df)；
3）新增字段：
T日涨停概率、T日涨停强度、T+1延续上涨率、涨停接力评分；
4）操作排名已改为按 涨停接力评分 降序排列；
5）report_md.py 已改为涨停接力实盘 9 列主表：
操作排名｜代码｜名称｜D日收盘价｜T日涨停概率｜T日涨停强度｜T+1延续上涨率｜涨停接力评分｜T日建议买入方式；
6）已修正 T+1建议买入方式 → T日建议买入方式；
7）表格已改为自由撑开、内容不换行；
8）报告顶部已锁定 D/T/T+1：
D=分析基准日，T=竞价买入日，T+1=卖出/验证日。

当前不要回头改 Decision，不要动 train.py，不要动 run_premium.yml，不要加回 操作结论/风险提示。

当前最大问题：
涨停接力字段目前仍是规则评分层，不是真实概率模型；
T日涨停概率/T+1延续上涨率不应直接当作真实概率使用；
市价竞价建议偏激进；
E_ret_plus 当前验证样本中 Plus MAE 差于 Raw MAE。

下一步优先新建：
src/top10decision/premium/limitup_labels.py

目标：
构建历史真实标签：
t_limitup_hit
t_touch_limitup
t1_up_hit
t1_high_profit_hit
t1_close_ret
t1_high_ret

然后再新建：
src/top10decision/premium/limitup_backtest.py

目标：
回测当前涨停接力评分在 Top5/Top10 的真实表现，包括 T日涨停率、T日触板率、T+1上涨率、T+1最高收益、市价竞价建议命中率等。

先做离线验证闭环，再回头优化 predict.py 的评分公式。
```

---

## 十一、当前阶段判断

当前 Premium 主线没有改乱。

当前状态是：

```text
工程链路：已接通
报告展示：基本完成
涨停接力字段：已落地
排序方向：已切换
实盘可信度：仍需历史回测校准
```

当前系统方向正确，但还不能直接重仓实盘。

下一步重点不是继续做漂亮表格，而是：

```text
用历史标签和离线回测证明“涨停接力评分”到底有没有预测力。
```
