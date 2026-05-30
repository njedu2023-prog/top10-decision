# Premium 主线开发衔接口令_0530

## 一、当前主线定位

当前仓库：`njedu2023-prog/top10-decision`

当前主线：`Premium 主线`

当前阶段：`E_ret_plus / EHX 残差增强接线与验收阶段`

当前最高价值不是生成好看的 Markdown 报告，而是让 Premium 成为 Decision 之后的第二层价值判断系统：

- Decision 负责判断：能不能买、买不买得到、成本和风险是否可接受。
- Premium 负责判断：买到之后，从 T+1 到 T+2 是否仍有可兑现溢价。
- 当前 Premium 的核心目标是生成、展示、验证、回灌 `E_ret_plus`。
- `E_ret_plus` 的本质是对原始 `E_ret` 做残差增强，而不是推翻原有预测链。

当前技术路线：

```text
delta_ret = real_premium_ret - eret_pred_raw
eret_plus_value = eret_pred_raw + eret_plus_delta
```

---

## 二、已经完成的文件修改

本轮已围绕 Premium V3 / E_ret_plus 主线修改过以下文件：

```text
src/top10decision/premium/train.py
src/top10decision/premium/predict.py
src/top10decision/premium/report_md.py
src/top10decision/premium/schemas.py
src/top10decision/premium/io.py
```

### 1. train.py

目标：训练 EHX 残差增强层。

已完成内容：

- 新增 EHX 残差学习逻辑。
- 训练输出：
  - `outputs/premium/models/ehx_delta.joblib`
  - `outputs/premium/models/ehx_meta.json`
- 增加 EHX 训练追溯字段：
  - `ehx_trained`
  - `ehx_reason`
  - `ehx_n_samples`
  - `ehx_min_samples`
  - `delta_mae`
  - `delta_rmse`
  - `plus_improve_rate`
- 修复旧 `append_eval_history()` 截断 EHX 新字段的问题，至少在 `train.py` 内部已绕开旧 schema 截断风险。
- 当前运行结果显示：
  - `trained=False`
  - `n_days=0`
  - `n_samples=0`
  - 原因：没有可用训练样本。
- 这个不是当前致命报错。训练样本不足时应允许 Premium 预测链继续走冷启动 EHX。

### 2. predict.py

目标：让 Premium 在线推理真正计算并输出 `E_ret_plus`。

已完成设计目标：

- 读取 `data/pred/pred_source_latest.csv`。
- 合并 Decision 输出。
- 构建 factor packs。
- 识别原始 `E_ret`。
- 优先加载 `outputs/premium/models/ehx_delta.joblib`。
- 如果模型不存在，使用 `ehx:coldstart_v1` 兜底。
- 输出字段包括：
  - `eret_pred_raw`
  - `eret_plus_value`
  - `eret_plus_delta`
  - `eret_plus_direction`
  - `eret_plus_conf`
  - `eret_plus_conf_score`
  - `eret_plus_src`
  - `rank_eret_plus`

当前卡点也在这个文件。

### 3. report_md.py

目标：让报告层展示 E_ret_plus / EHX。

已完成内容：

- 报告标题升级为 Premium V3。
- 预测表展示：
  - `E_ret原始值`
  - `E_ret_plus`
  - `EHX修正值`
  - `EHX方向`
  - `EHX置信度`
  - `EHX来源`
- 验证表展示：
  - `Raw绝对误差`
  - `Plus绝对误差`
  - `Plus优于Raw`
- 增加 EHX 验证摘要：
  - Raw MAE
  - Plus MAE
  - MAE 改善
  - Plus 优于 Raw 比例

### 4. schemas.py

目标：字段契约层正式纳入 EHX 字段。

已完成内容：

- `DecisionInputSchema` 增加原始 E_ret 与 E_ret_plus 字段别名。
- `PremiumRankOutputSchema` 增加 EHX 输出字段。
- 新增 `PremiumVerifyOutputSchema`。
- `PremiumEvalHistorySchema` 增加 EHX 训练与评估字段。
- 目的是防止新字段被旧 schema 截断。

### 5. io.py

目标：IO 层不再丢失 EHX 字段。

已完成内容：

- `append_eval_history()` 改为保留历史列、schema列和新 row 列。
- `write_premium_top30()` / `write_premium_full()` 优先按 schema 排序，但保留额外字段。
- `write_premium_verify()` 接入 `PremiumVerifyOutputSchema`。
- `_ensure_columns()` 默认 `keep_extra=True`。
- `load_decision_merge()` 增强 `dec_p_fill` 字段兼容。

---

## 三、当前真实卡点

当前 GitHub Actions 仍然失败。

失败 workflow：

```text
.github/workflows/run_premium.yml
```

失败步骤：

```text
Run Premium
```

失败文件：

```text
src/top10decision/premium/predict.py
```

失败位置：

```text
predict.py line 866
```

当前错误：

```text
ValueError: Length of values (1) does not match length of index (47)
```

报错行仍然是旧代码：

```python
eret_plus_src = str(df.get("eret_plus_src", pd.Series(["ehx:unknown"], index=df.index)).iloc[0]) if len(df) > 0 else "ehx:unknown"
```

根因：

- `df` 有 47 行。
- `pd.Series(["ehx:unknown"], index=df.index)` 只有 1 个值。
- Pandas 要求 values 长度和 index 长度一致。
- 因此必然报 `Length of values (1) does not match length of index (47)`。

关键判断：

- 这个 bug 根因已经明确。
- 之前多次下载完整 `predict.py` 覆盖没有真正改掉线上文件。
- 线上文件仍然保留旧代码。
- 下一步必须先确保 GitHub 线上 `predict.py` 中彻底搜不到这句：

```text
pd.Series(["ehx:unknown"], index=df.index)
```

---

## 四、下一次进入后第一优先任务

下一次不要先扩展新功能，不要继续改其它文件。

第一优先任务：只修 `predict.py` 第 866 行附近的 `eret_plus_src` 取值逻辑。

### 必须替换掉的旧代码

```python
eret_plus_src = str(df.get("eret_plus_src", pd.Series(["ehx:unknown"], index=df.index)).iloc[0]) if len(df) > 0 else "ehx:unknown"
```

### 推荐替换代码

```python
if len(df) > 0 and "eret_plus_src" in df.columns and df["eret_plus_src"].notna().any():
    eret_plus_src = str(df["eret_plus_src"].dropna().iloc[0])
else:
    eret_plus_src = "ehx:unknown"
```

### 替换后必须检查

在 GitHub 文件页面搜索：

```text
pd.Series(["ehx:unknown"], index=df.index)
```

要求：

```text
必须搜不到。
```

如果还能搜到，说明没有修复成功，不要跑 workflow。

---

## 五、可选顺手修复

`predict.py` 日志里还有 warning：

```text
RuntimeWarning: Mean of empty slice
RuntimeWarning: Degrees of freedom <= 0 for slice.
```

位置：

```text
predict.py line 165
_zscore()
```

这不是当前致命错误，但建议顺手修。

### 当前旧逻辑大致为

```python
def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = np.nanmean(x.values)
    sd = np.nanstd(x.values)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd
```

### 建议修为

```python
def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if len(x) == 0 or x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    mu = np.nanmean(x.values)
    sd = np.nanstd(x.values)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd
```

---

## 六、修复后验收步骤

### 第一步：提交 predict.py

提交后确认最新 commit 确实更新了：

```text
src/top10decision/premium/predict.py
```

### 第二步：运行 workflow

手动运行：

```text
.github/workflows/run_premium.yml
```

### 第三步：看是否通过 Run Premium

必须确认不再出现：

```text
Length of values (1) does not match length of index (47)
```

### 第四步：确认进入后续步骤

应该至少能进入：

```text
Sanity check outputs
Commit outputs + market cache
```

如果这两个步骤开始运行，说明 `predict.py` 主流程至少已经继续往后走。

---

## 七、跑通后需要验收的输出

重点看以下文件是否生成或更新：

```text
outputs/premium/premium_top30_*.csv
outputs/premium/premium_full_*.csv
outputs/premium/premium_verify_*.csv
outputs/premium/_last_run.txt
docs/reports/premium_latest.md
```

如果 EHX 模型已训练成功，还要看：

```text
outputs/premium/models/ehx_delta.joblib
outputs/premium/models/ehx_meta.json
```

但当前训练样本不足，所以模型文件可能暂时没有，这是允许的。预测链应使用：

```text
ehx:coldstart_v1
```

---

## 八、跑通后重点验收字段

在 `premium_top30_*.csv` 或 `premium_full_*.csv` 中检查字段：

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

在 `premium_verify_*.csv` 中检查字段：

```text
raw_abs_err
plus_abs_err
improve_flag
hit_up
```

在 `_last_run.txt` 中检查：

```text
eret_plus_src
ehx_mode
ehx_reason
ehx_load_reason
out_top30
out_full
out_verify
report_md
```

---

## 九、当前不要做的事

下一次不要先做这些事：

- 不要继续改 `train.py`。
- 不要继续改 `report_md.py`。
- 不要继续改 `schemas.py`。
- 不要继续改 `io.py`。
- 不要新增不存在的 `artifacts.py`。
- 不要继续扩展 EHX 功能。
- 不要试图解释训练样本不足为主报错。
- 不要忽略 GitHub 线上文件是否真的覆盖成功。

当前最小闭环就是：

```text
修 predict.py 第866行
↓
确认旧字符串搜不到
↓
跑 run_premium.yml
↓
验收输出文件和字段
```

---

## 十、下一次对话启动口令

下次进入该主线时，直接使用以下口令：

```text
继续 top10-decision Premium 主线。
当前阶段是 E_ret_plus / EHX 接线验收。
不要扩展新功能，先修 predict.py 第866行 eret_plus_src 长度不匹配错误。
线上仍有旧代码：
eret_plus_src = str(df.get("eret_plus_src", pd.Series(["ehx:unknown"], index=df.index)).iloc[0]) if len(df) > 0 else "ehx:unknown"
必须替换为安全取值逻辑，并确认 GitHub 线上搜不到：
pd.Series(["ehx:unknown"], index=df.index)
然后重新跑 run_premium.yml，验收 premium_top30、premium_full、premium_verify、_last_run 和 premium_latest.md。
```

---

## 十一、当前系统判断

当前 Premium 主线方向没有错。

真正问题不是 EHX 设计，而是：

```text
predict.py 修复没有真正落到 GitHub 线上文件。
```

这属于执行覆盖问题，不是模型逻辑问题。

下一次必须用“线上文件实际内容”为准，不以本地下载文件为准。
