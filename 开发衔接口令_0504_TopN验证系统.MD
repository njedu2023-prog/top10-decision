# 开发衔接口令_0504_TopN验证系统.MD

## 一、当前主线定位

当前主线是 `top10-decision` 仓库中的 **TopN Targets / 最终总排序 Top10 后验验证系统**。

目标不是修改原预测主链路，而是在原系统旁边新增独立验证闭环：

- D 日读取系统预测名单；
- T 日收盘后验证 D 日名单真实涨跌；
- 长期累计上涨率、涨停率、下跌率、平盘率；
- 进一步统计 Top1、Top2、Top4、Top10 的有效性；
- 验证 EV、RiskPenalty 与真实收益/上涨之间的解释力；
- 为后续实盘只买 Top1、Top2、Top4 还是 Top10 提供统计证据。

原需求核心口径已确认：

> 如果 `TopN Targets` 表里没有数据，就从总表选前 10 名。

工程解释已经校准为：

> “总表”不应依赖 Markdown 报告展示表解析，而应直接读取真实数据源 `data/decision/decision_candidates_YYYYMMDD.csv`，按最终排序取前 10。

---

## 二、当前已完成内容

已新增/修改验证系统相关文件：

1. `scripts/validate_topn_targets.py`
2. `scripts/backfill_topn_targets_validation.py`
3. `.github/workflows/validate_topn_targets.yml`
4. `outputs/validation/*`
5. `docs/topn_targets_validation.html`

当前 GitHub Actions 中存在独立 workflow：

- `Validate TopN Targets`
- workflow 文件：`.github/workflows/validate_topn_targets.yml`
- 当前运行模式中使用的是：
  - `MODE=backfill`
  - `START_DATE=20260302`
  - `END_DATE=20260428`
  - `TOPN=10`
  - 实际执行命令：`python scripts/backfill_topn_targets_validation.py --topn 10 --start-date 20260302 --end-date 20260428`

---

## 三、已定位的关键问题

此前页面一直显示：

- 历史累计样本总数：0
- 上涨率：N/A
- 最新一期验证明细：暂无数据

Actions 日志显示所有报告被跳过：

```text
[SKIP] decision_report_20260302.md: 缺 signal_date 或 exec_date
...
[DONE] source=decision_report_final_rank_top10 history_rows=0
```

根因不是页面问题，也不是 GitHub Actions 没跑，而是：

1. workflow 实际运行的是 `scripts/backfill_topn_targets_validation.py`；
2. 该脚本解析 report 元信息失败；
3. report 中日期格式是 Markdown 列表格式，例如：

```text
- signal_date: **20260430**
- exec_date: **20260506**
```

旧脚本没有正确兼容这种格式，所以所有 report 都被判定为缺日期并跳过。

---

## 四、最新修复方向

已经生成了完善版文件：

- 本地下载文件名：`backfill_topn_targets_validation_final.py`
- 应覆盖到仓库路径：

```text
scripts/backfill_topn_targets_validation.py
```

该完善版修复点：

1. `parse_report_meta()` 支持：
   - `signal_date: 20260430`
   - `- signal_date: **20260430**`
   - `- signal_date: \`20260430\``

2. `exec_date` 兜底：
   - 如果 report 内没解析到 `exec_date`；
   - 则从文件名 `decision_report_YYYYMMDD.md` 提取。

3. `signal_date` 兜底：
   - 如果 report 内没解析到 `signal_date`；
   - 则从 report 里的 `decision_candidates_YYYYMMDD.csv` 提取；
   - 若仍无法提取，再用 `exec_date` 兜底。

4. Top10 数据源改为：
   - `data/decision/decision_candidates_YYYYMMDD.csv`
   - 优先按 `rank` 升序；
   - 若无 `rank`，按 `EV` 降序；
   - 取前 10；
   - 标记 `validation_source = final_rank_top10_from_candidates`。

5. 输出仍保持：
   - `outputs/validation/topn_targets_validation_history.csv`
   - `outputs/validation/topn_targets_validation_latest.csv`
   - `outputs/validation/topn_targets_validation_summary.json`
   - `outputs/validation/topn_targets_validation_summary.md`
   - `outputs/validation/topn_targets_validation_by_date.csv`
   - `outputs/validation/topn_targets_validation_by_rank.csv`
   - `outputs/validation/topn_targets_validation_by_ev_risk.csv`
   - `docs/topn_targets_validation.html`

---

## 五、下一次进入后的第一步

不要回退讨论旧方案，不要再纠结 `TopN Targets` 展示表本身。

下一次直接做以下验收：

### 1. 确认文件是否已覆盖

检查：

```text
scripts/backfill_topn_targets_validation.py
```

确认其内容是否包含：

```python
def parse_report_meta(text: str, report_path: Path) -> Dict[str, str]:
```

并且内部存在：

```python
CANDIDATE_IN_TEXT_PATTERN = re.compile(r"decision_candidates_(\d{8})\.csv")
```

以及：

```python
meta["exec_date"] = m.group(1)
```

和：

```python
meta["signal_date"] = m.group(1)
```

若不存在，说明 GitHub 上还没有覆盖成功，需要重新覆盖 `scripts/backfill_topn_targets_validation.py`。

---

## 六、覆盖后立即执行的验收步骤

在 GitHub Actions 页面手动运行：

```text
Validate TopN Targets
```

运行完成后查看日志中 `Run TopN Targets validation` 步骤。

预期日志不应再出现：

```text
缺 signal_date 或 exec_date
```

应该出现类似：

```text
[OK] report=decision_report_20260302.md D=20260302 T=20260302 rows=10 source=final_rank_top10_from_candidates
...
[DONE] source=final_rank_top10_from_candidates history_rows=xxx verified_rows=yyy
```

判断标准：

- `history_rows > 0`：说明最终总排序 Top10 已经提取成功；
- `verified_rows > 0`：说明行情数据也成功匹配，已经能计算真实涨跌；
- `history_rows > 0` 但 `verified_rows = 0`：说明名单提取成功，但行情数据匹配仍需继续排查。

---

## 七、如果仍然没有数据，下一步排查顺序

### 情况 A：仍出现 `缺 signal_date 或 exec_date`

说明 workflow 运行的不是新覆盖文件，或者覆盖到了错误文件。

处理：

1. 检查 workflow 真实执行命令；
2. 当前 workflow 执行的是：

```bash
python scripts/backfill_topn_targets_validation.py --topn 10 --start-date 20260302 --end-date 20260428
```

3. 所以必须覆盖的是：

```text
scripts/backfill_topn_targets_validation.py
```

不是只覆盖 `scripts/validate_topn_targets.py`。

---

### 情况 B：日志显示 `missing candidates csv`

说明日期解析已成功，但没有找到对应 candidates 文件。

处理：

检查 `data/decision/` 下是否存在：

```text
decision_candidates_YYYYMMDD.csv
```

尤其要看 report 中的 `signal_date` 对应文件是否存在。

如果 `signal_date` 是 20260430，则应存在：

```text
data/decision/decision_candidates_20260430.csv
```

---

### 情况 C：`history_rows > 0`，但页面累计样本仍是 0

说明 history 有候选名单，但没有已验证行情。

处理：

检查输出文件：

```text
outputs/validation/topn_targets_validation_history.csv
```

重点看字段：

```text
validation_status
validation_note
D_close
T_close
T_return_pct
```

如果大量是：

```text
未找到 T 日行情
```

则下一步主线应转为行情文件匹配逻辑，而不是名单提取逻辑。

---

## 八、当前最重要的工程判断

本模块的核心数据源必须固定为：

```text
data/decision/decision_candidates_YYYYMMDD.csv
```

而不是：

```text
outputs/decision/decision_report_YYYYMMDD.md
```

原因：

1. report 是展示层；
2. candidates 是真实计算层；
3. 展示表可能为空或格式变化；
4. candidates 才能稳定代表最终总排序；
5. 原需求“TopN Targets 为空则总表前10”在工程上应解释为“从 candidates 最终排序中取前10”。

---

## 九、下一阶段目标

当前先完成第一阶段：

> 让 `history_rows > 0`，确保验证名单能稳定生成。

完成后进入第二阶段：

> 让 `verified_rows > 0`，确保 T 日行情能匹配并计算真实涨跌幅。

第二阶段重点可能涉及：

1. 行情数据实际存放路径；
2. D 日 close 与 T 日 close 的字段名称；
3. 是否应从 `data/market/`、`data/raw/`、`data/pred/archive/` 或其它数据快照读取行情；
4. 是否应接入上游 `a-share-top3-data` 快照；
5. 是否需要增加明确的行情源优先级。

---

## 十、下一次无缝衔接口令

下次进入本任务时，请直接按以下口令继续：

```text
继续 top10-decision 的 TopN Targets / 最终总排序 Top10 后验验证系统主线，不回退旧方案。

当前已确认原需求：TopN Targets 为空时，从总表取前10；工程实现上总表应读取 data/decision/decision_candidates_YYYYMMDD.csv，而不是解析 decision_report 的 Markdown 展示表。

当前关键问题已定位：GitHub Actions 实际执行 scripts/backfill_topn_targets_validation.py，旧脚本因无法解析 “- signal_date: **YYYYMMDD**” 和 “- exec_date: **YYYYMMDD**” 导致所有 decision_report 被 SKIP，history_rows=0。

已生成完善版 backfill_topn_targets_validation_final.py，必须覆盖到 scripts/backfill_topn_targets_validation.py。覆盖后手动运行 Validate TopN Targets workflow，先验收日志是否出现 [OK] report=... rows=10，以及 [DONE] history_rows>0。

若仍为0，按顺序排查：
1）workflow 是否真的运行新 backfill 脚本；
2）是否缺 decision_candidates_YYYYMMDD.csv；
3）若 history_rows>0 但 verified_rows=0，则进入行情匹配排查，不再回到 TopN 表解析问题。

最高目标：建立独立、非侵入式、可累计统计的 TopN / final_rank_top10 后验验证闭环，为后续 Top1/Top2/Top4/Top10 实盘组合选择提供真实统计依据。
```
