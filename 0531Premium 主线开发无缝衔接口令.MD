# 0530 Premium 主线开发无缝衔接口令

## 一、进入主线后的第一句话

继续 `top10-decision` 的 **Premium 主线**。

当前阶段是：

```text
Premium V3 / E_ret_plus / EHX 残差增强接线验收阶段
```

当前最高目标不是扩展新功能，也不是优化 Markdown 报告样式，而是保证：

```text
Decision 输出之后，Premium 能稳定计算、输出、验证 E_ret_plus。
```

---

## 二、当前系统定位

`top10-decision` 下面当前有两层核心系统：

### 1. Decision 主线

Decision 负责判断：

```text
能不能买
买不买得到
成本是否可接受
风险是否可接受
EV 是否值得做
```

Decision 当前已阶段性收口，不要在 Premium 主线里回头深挖 Decision。

### 2. Premium 主线

Premium 负责判断：

```text
买到之后，从 T+1 到 T+2 是否仍有可兑现溢价
```

Premium 当前核心变量已经升级为：

```text
E_ret_plus
```

其本质是：

```text
E_ret_plus = 原始 E_ret + EHX 残差增强修正
```

当前不是推翻原始 E_ret，而是在原始 E_ret 上做残差增强。

---

## 三、刚完成的关键修复

本轮已经修复：

```text
src/top10decision/premium/predict.py
```

### 修复 1：解决 Run Premium 报错

原报错：

```text
ValueError: Length of values (1) does not match length of index (47)
```

失败位置：

```text
predict.py 第 866 行附近
```

旧错误代码核心问题：

```python
pd.Series(["ehx:unknown"], index=df.index)
```

当 `df.index` 有 47 行时，只给 1 个值，会触发 Pandas 长度不匹配。

已改为安全取值逻辑：

```python
if len(df) > 0 and "eret_plus_src" in df.columns and df["eret_plus_src"].notna().any():
    eret_plus_src = str(df["eret_plus_src"].dropna().iloc[0])
else:
    eret_plus_src = "ehx:unknown"
```

### 修复 2：处理 `_zscore()` 空值 warning

旧逻辑在全空列上会触发：

```text
RuntimeWarning: Mean of empty slice
RuntimeWarning: Degrees of freedom <= 0 for slice
```

已增加空列保护：

```python
x = pd.to_numeric(s, errors="coerce")
if len(x) == 0 or x.notna().sum() == 0:
    return pd.Series(np.zeros(len(x)), index=x.index)
```

---

## 四、当前验收状态

本轮已重新运行：

```text
.github/workflows/run_premium.yml
```

运行方式：

```text
workflow_dispatch
command: all
branch: main
```

最新结果：

```text
Run Premium (Follow Decision) #233
Status: Success
commit: 3b8d618
```

结论：

```text
predict.py 第 866 行长度不匹配问题已经解除。
Premium 主线 workflow 已跑通。
```

---

## 五、当前必须验收的输出文件

后续继续时，优先验收以下 5 类文件。

### 1. `_last_run.txt`

路径：

```text
outputs/premium/_last_run.txt
```

重点看：

```text
ok:
target_date:
pending:
pending_reason:
verify_pending:
verify_reason:
eret_plus_src:
ehx_mode:
ehx_reason:
ehx_load_reason:
out_top30:
out_full:
out_verify:
report_md:
```

关键判断：

```text
ok 必须为 True
eret_plus_src 不能空
out_top30 / out_full / out_verify / report_md 必须有路径
```

当前允许出现：

```text
eret_plus_src: ehx:coldstart_v1
```

这说明 EHX 模型不足时，冷启动链路已接上。

### 2. `premium_top30_*.csv`

路径：

```text
outputs/premium/premium_top30_*.csv
```

重点验字段：

```text
eret_pred_raw
eret_plus_value
eret_plus_delta
eret_plus_direction
eret_plus_conf
eret_plus_conf_score
eret_plus_src
rank_eret_plus
```

判断标准：

```text
文件非空
Top30 有股票行
eret_plus_value 不能整列空
eret_plus_src 不能整列空
rank_eret_plus 有排序
```

### 3. `premium_full_*.csv`

路径：

```text
outputs/premium/premium_full_*.csv
```

重点验字段：

```text
eret_pred_raw
eret_plus_value
eret_plus_delta
eret_plus_src
rank_eret_plus
```

判断标准：

```text
文件非空
全量候选池正常
E_ret_plus 字段保留
```

### 4. `premium_verify_*.csv`

路径：

```text
outputs/premium/premium_verify_*.csv
```

重点验字段：

```text
raw_abs_err
plus_abs_err
improve_flag
hit_up
```

说明：

```text
如果 T+2 真实行情还没成熟，verify 可能 pending 或部分为空。
但文件必须生成，字段必须存在。
```

### 5. `premium_latest.md`

路径：

```text
docs/reports/premium_latest.md
```

重点看报告是否展示：

```text
Premium V3
E_ret_plus
EHX修正值
EHX方向
EHX置信度
EHX来源
验证摘要
Raw MAE
Plus MAE
Plus 优于 Raw 比例
```

---

## 六、当前不要做的事

后续进入 Premium 主线时，不要先做这些：

```text
不要继续扩展 EHX 新功能
不要新增 artifacts.py
不要先改 train.py
不要先改 report_md.py
不要先改 schemas.py
不要先改 io.py
不要回头深挖 Decision 主线
不要把训练样本不足误判为主报错
不要忽略 GitHub 线上文件实际内容
```

当前主线已经从“修报错”进入“验收输出质量与字段完整性”。

---

## 七、下一步建议顺序

### 第一步：确认线上 `predict.py` 已彻底清掉旧错误字符串

在 GitHub 文件页面搜索：

```text
pd.Series(["ehx:unknown"], index=df.index)
```

要求：

```text
必须搜不到
```

### 第二步：验收 `_last_run.txt`

确认：

```text
ok: True
eret_plus_src 非空
out_top30 / out_full / out_verify / report_md 非空
```

### 第三步：验收 CSV 字段

打开最新日期对应文件：

```text
outputs/premium/premium_top30_最新日期.csv
outputs/premium/premium_full_最新日期.csv
outputs/premium/premium_verify_最新日期.csv
```

确认 E_ret_plus / EHX 字段完整。

### 第四步：验收报告层

打开：

```text
docs/reports/premium_latest.md
```

确认报告层展示 EHX 修正、方向、置信度、来源和验证表。

### 第五步：再判断是否需要进入下一阶段

只有在以上验收都通过后，才进入下一阶段。

---

## 八、下一阶段可能的主线

如果当前验收全部通过，下一阶段优先级如下：

### 优先级 1：清理 pending / 交易日历状态

如果 `_last_run.txt` 里出现：

```text
pending_reason: missing_TUSHARE_TOKEN|fallback_business_day
```

则下一步应检查：

```text
GitHub Actions Secret 是否有 TUSHARE_TOKEN
run_premium.yml 是否正确注入 TUSHARE_TOKEN
predict.py 是否正确读取 os.getenv("TUSHARE_TOKEN")
```

注意：

```text
不要盲目改预测逻辑。
先确认环境变量链路。
```

### 优先级 2：确认 EHX 是否进入 model_v1

如果当前是：

```text
eret_plus_src: ehx:coldstart_v1
```

说明预测链可运行，但还没有真正使用训练模型。

后续可检查：

```text
outputs/premium/models/ehx_delta.joblib
outputs/premium/models/ehx_meta.json
```

如果模型不存在或训练样本不足，这是允许状态，但需要记录：

```text
EHX 当前处于冷启动增强模式
不是致命错误
```

### 优先级 3：回看 E_ret_plus 是否真的产生横向分化

重点看：

```text
eret_pred_raw
eret_plus_value
eret_plus_delta
rank_eret_plus
```

判断：

```text
eret_plus_delta 是否全为 0
eret_plus_value 是否只是复制 eret_pred_raw
rank_eret_plus 是否有实际排序差异
```

如果没有横向分化，才进入特征和残差建模优化。

---

## 九、下一次直接使用的口令

下次进入该主线时，直接输入：

```text
继续 top10-decision Premium 主线。

当前阶段是 Premium V3 / E_ret_plus / EHX 残差增强接线验收阶段。

上一轮已修复 predict.py 第 866 行 eret_plus_src 长度不匹配错误：
ValueError: Length of values (1) does not match length of index (47)

旧错误字符串：
pd.Series(["ehx:unknown"], index=df.index)

已经替换为安全取值逻辑，并顺手修复 _zscore() 全空列 warning。

Run Premium (Follow Decision) #233 已经跑通，Status: Success，commit: 3b8d618。

当前不要扩展新功能，不要回头改 Decision，不要先动 train.py/report_md.py/schemas.py/io.py。

第一优先任务是验收输出文件：
outputs/premium/_last_run.txt
outputs/premium/premium_top30_*.csv
outputs/premium/premium_full_*.csv
outputs/premium/premium_verify_*.csv
docs/reports/premium_latest.md

重点确认：
ok=True
eret_plus_src 非空
eret_pred_raw / eret_plus_value / eret_plus_delta / eret_plus_src / rank_eret_plus 字段完整
raw_abs_err / plus_abs_err / improve_flag / hit_up 字段完整
报告层展示 E_ret_plus / EHX 修正 / EHX方向 / EHX置信度 / EHX来源

如果验收都通过，再进入下一阶段：
1）清理 pending_reason / TUSHARE_TOKEN 环境链路；
2）确认 EHX 是否从 coldstart_v1 进入 model_v1；
3）检查 E_ret_plus 是否产生真实横向分化。
```

---

## 十、当前系统判断

当前 Premium 主线方向正确。

本轮问题本质已经从：

```text
代码执行报错
```

切换为：

```text
输出质量验收
环境变量/交易日历链路确认
EHX 冷启动到模型化升级
```

后续必须继续按“线上 GitHub 实际内容”为准，不以本地记忆或旧下载文件为准。
