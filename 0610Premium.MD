# 0610 Premium 主线开发无缝衔接文档

## 一、当前主线

继续 `top10-decision` 的 Premium 主线。

当前阶段：

```text
Premium V3 / E_ret_plus / EHX model_v1 接线完成后的实盘表头与执行字段优化阶段
```

当前最高目标不是继续修报错，也不是回头改 Decision 主线，而是：

```text
把 Premium 报告从“模型工程报告”升级为“人类实盘操作参考报告”
```

---

## 二、当前系统定位

`top10-decision` 下面仍然是两层系统：

### 1. Decision 主线

Decision 负责判断：

```text
能不能买
买不买得到
成本是否可接受
风险是否可接受
EV 是否值得做
```

当前不要在 Premium 主线里回头深挖 Decision。

### 2. Premium 主线

Premium 负责判断：

```text
买到之后，从 T+1 到 T+2 是否仍有可兑现溢价
```

当前 Premium 核心变量：

```text
E_ret_plus
```

其本质是：

```text
E_ret_plus = 原始 E_ret + EHX 残差增强修正
```

---

## 三、当前交易时间轴必须锁死

用户真实操作流程是：

```text
T：分析基准日
T+1：早上集合竞价买入
T+2：盘中择时卖出
```

所以 Premium 报告不能只看 `Close[T] → Close[T+2]`，还要逐步补足：

```text
T+1 买入执行层
T+2 卖出执行层
```

---

## 四、0610 已完成的重要进展

### 1. EHX 已从 coldstart_v1 升级到 model_v1

此前 `_last_run.txt` 中长期显示：

```text
eret_plus_src: ehx:coldstart_v1
ehx_reason: ehx_model_missing
ehx_load_reason: ehx_model_missing
```

已经完成训练闭环后，模型文件已生成：

```text
outputs/premium/models/ehx_delta.joblib
outputs/premium/models/ehx_meta.json
outputs/premium/_last_train.txt
```

训练状态显示：

```text
trained: True
ehx_trained: True
ehx_reason: ok
ehx_n_samples: 1432
n_days: 48
```

`premium_full_20260605.csv` 与后续报告已经出现：

```text
eret_plus_src: ehx:model_v1
```

结论：

```text
EHX model_v1 接线已完成。
Premium 已不再停留在 coldstart_v1。
```

---

### 2. run_premium.yml 已修复提交链路

此前 workflow 报错原因不是模型训练失败，而是提交阶段被 Python 缓存文件卡住：

```text
error: cannot pull with rebase: You have unstaged changes.
src/top10decision/__pycache__/...
```

已修复方向：

```text
禁止 pycache
清理 __pycache__ / *.pyc
不再在脏工作区 rebase
只提交 outputs/premium、docs/reports、data/market
```

当前 workflow 已能成功提交预测产物与模型产物。

---

### 3. report_md.py 已改成人类操作表头版

当前线上 `report_md.py` 已是：

```text
Premium 子系统 — Markdown 报告渲染（V3.2：人类操作表头版）
```

预测表与验证表已经从工程长表头改为人类操作表头。

当前最终保留字段为：

```text
操作排名｜代码｜名称｜收盘价｜T+2预期收益｜预期价格区间｜T+2上涨概率｜模型置信度
```

已经删除：

```text
操作结论
风险提示
```

最新 `docs/reports/premium_latest.md` 已确认表头中不再出现这两个字段。

---

## 五、当前最新报告状态

当前报告路径：

```text
docs/reports/premium_latest.md
docs/reports/premium_20260608.md
```

当前报告显示：

```text
预测日（T）：20260608
预测到期日（T+2）：20260610
模型版本：premium_v2
```

预测表当前字段：

```text
操作排名
代码
名称
收盘价
T+2预期收益
预期价格区间
T+2上涨概率
模型置信度
```

验证表当前字段：

```text
操作排名
代码
名称
收盘价
T+2预期收益
预期价格区间
T+2上涨概率
模型置信度
```

---

## 六、当前不要做的事

后续继续时，先不要做这些：

```text
不要回头重做 Decision 主线
不要再修 EHX coldstart 问题
不要再动 train.py，除非模型训练重新报错
不要再动 run_premium.yml，除非提交链路重新失败
不要把操作结论 / 风险提示重新加回报告
不要在报告里堆太多工程字段
不要把 Premium 报告重新变成技术审计表
```

当前 `操作结论`、`风险提示` 已明确不要。

---

## 七、当前核心问题：人类实盘操作还缺什么

现在报告已经适合看方向，但还不完全适合实盘下单。

用户真实流程是：

```text
T 日分析
T+1 早上竞价买入
T+2 盘中择时卖出
```

所以目前还缺两个执行层。

---

## 八、下一阶段建议：补 T+1 买入执行层

当前表只能回答：

```text
如果买到了，到 T+2 有没有溢价？
```

但还不能回答：

```text
T+1 竞价怎么买？
高开多少还能买？
市价还是限价？
超过什么价格就放弃？
```

建议新增预测字段：

```text
T+1建议买入方式
T+1可接受买入价
```

字段含义：

### T+1建议买入方式

可选值建议：

```text
市价竞价
限价竞价
只观察不追
放弃
```

### T+1可接受买入价

示例：

```text
≤2.42
≤2.48
高开>5%放弃
```

生成依据：

```text
T日收盘价
T+2预期收益
T+2预期价格中位数
T+2上涨概率
安全边际
```

---

## 九、下一阶段建议：补 T+2 卖出执行层

用户不是 T+2 收盘卖，而是 T+2 盘中择时卖。

所以还需要字段：

```text
T+2卖出计划
```

示例：

```text
冲高至2.55附近分批卖
弱于2.40止损
接近中位价优先兑现
开盘不强则减仓
```

生成依据：

```text
T+2价格区间 p25 / p50 / p75
T+2上涨概率
E_ret_plus
T+1买入价
安全边际
```

---

## 十、建议最终预测表，不是现在立刻改，但作为下一阶段目标

当前已完成版：

```text
操作排名｜代码｜名称｜收盘价｜T+2预期收益｜预期价格区间｜T+2上涨概率｜模型置信度
```

下一阶段实盘增强版建议：

```text
操作排名｜代码｜名称｜T日收盘价｜T+1建议买入方式｜T+1可接受买入价｜T+2预期收益｜T+2预期价格区间｜T+2上涨概率｜模型置信度｜T+2卖出计划
```

暂时不加：

```text
操作结论
风险提示
```

---

## 十一、验证表下一阶段建议

当前验证表只是跟预测表保持同一表头。

但实盘验证最终应升级为：

```text
操作排名｜代码｜名称｜T日收盘价｜T+1模拟买入价｜T+2实际最高价｜T+2实际收盘价｜T+2预期收益｜实际可兑现收益｜预测是否有效｜误差原因
```

原因：

```text
用户实际是在 T+2 盘中卖出，不是一定等收盘。
只用 T+2 收盘验证，会低估盘中可兑现机会。
```

下一阶段验证应补：

```text
T+2最高价收益
T+2盘中是否给过卖点
按计划是否可盈利
```

---

## 十二、下一步优先改哪个文件

如果只是继续调整报告表头：

```text
src/top10decision/premium/report_md.py
```

但如果要新增：

```text
T+1建议买入方式
T+1可接受买入价
T+2卖出计划
T+1模拟买入价
T+2实际最高价
实际可兑现收益
```

则必须先改：

```text
src/top10decision/premium/predict.py
```

原因：

```text
这些不是简单展示字段，而是需要先进入 premium_top30 / premium_full / premium_verify CSV。
```

推荐下一步：

```text
优先改 predict.py，补 T+1 买入价保护和 T+2 卖出计划字段。
然后再改 report_md.py 展示这些新增字段。
```

---

## 十三、下一次直接使用的开发口令

下次继续时，直接输入：

```text
继续 top10-decision Premium 主线。

当前阶段是 Premium V3 / E_ret_plus / EHX model_v1 接线完成后的实盘执行层增强阶段。

上一轮已经完成：
1）EHX 从 coldstart_v1 升级到 model_v1；
2）outputs/premium/models/ehx_delta.joblib 与 ehx_meta.json 已生成；
3）run_premium.yml 提交链路已修复；
4）report_md.py 已改为 V3.2 人类操作表头版；
5）预测表与验证表当前只保留：
操作排名｜代码｜名称｜收盘价｜T+2预期收益｜预期价格区间｜T+2上涨概率｜模型置信度；
6）明确不要再显示：
操作结论、风险提示。

当前用户真实交易流程是：
T：分析基准日；
T+1：早上竞价买入；
T+2：盘中择时卖出。

下一阶段不要回头改 Decision，不要再修 coldstart，不要先动 train.py/run_premium.yml。

下一步优先检查并修改：
src/top10decision/premium/predict.py

目标是补实盘执行字段：
T+1建议买入方式
T+1可接受买入价
T+2卖出计划

后续验证表再补：
T+1模拟买入价
T+2实际最高价
T+2实际收盘价
实际可兑现收益
预测是否有效
误差原因

注意：新增字段必须先进入 premium_top30 / premium_full / premium_verify CSV，再由 report_md.py 展示。
```

---

## 十四、当前阶段判断

当前 Premium 主线已经从：

```text
模型接线 / 报错修复
```

切换为：

```text
实盘执行层增强
```

当前系统方向正确。

下一步重点不是“模型能不能跑”，而是：

```text
模型结果如何指导 T+1 买入和 T+2 卖出。
```
