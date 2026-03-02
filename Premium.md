# Premium 子系统 — T+2 收盘价分布预测（手工交易版 V2）需求契约（Draft）

> **核心目标（唯一主线）**：在交易日 **T 收盘后**，对每只候选股票给出 **T+2 收盘价的概率分布预测**（分位数），并通过真值回流实现 **可校准、自学习、逐级增强** 的准闭环迭代。  
> **辅助信息**（如 P_fill）只做标注，不参与过滤与排序。

---

## 0. 术语与时间轴（强约束）
- **T**：本次预测的基准交易日（收盘后生成预测）。
- **T+1**：下一交易日（执行交易，Premium 不负责交易逻辑）。
- **T+2**：第二个后续交易日（以交易日历顺延，非自然日）。
- **Close[T]**：T 日收盘价（真值）。
- **Close[T+2]**：T+2 日收盘价（真值，用于验收与学习）。

> 交易日推进规则：必须以“可获取真值的交易日”推进，周末/节假日顺延；禁止用自然日硬推。

---

## 1. 输入（Inputs）
### 1.1 主输入（候选与先验）
- **路径**：`data/pred/pred_source_latest.csv`
- **性质**：来自 a-top10 的强池/强度排序产物（全市场强势候选，不要求包含价格真值）。
- **最低必需字段（必须存在）**：
  - `trade_date`（YYYYMMDD）
  - `ts_code`
  - `name`（若缺可为空）
- **可选先验字段（用于分桶/特征，但不强制）**：
  - `p_premium` / `probability` / `prob`（上涨概率或强势概率）
  - 强度/题材/热度/资金等任意字段（不锁死字段名；内部需做字段别名映射）

> 备注：`pred_source_latest.csv` **不要求**包含任何价格字段。价格由真值层提供。

### 1.2 真值缓存输入（必须可得，用于预测还原与学习打标）
- **路径目录**：`data/market/`
- **文件命名**（必须锁死）：
  - `data/market/daily_{T}.csv`
  - `data/market/daily_{T2}.csv`（T2 = T+2）
- **最低必需字段（必须存在）**：
  - `ts_code`
  - `close`

> 没有真值缓存时：系统必须进入 pending/降级逻辑，但**不得报错卡死**。

### 1.3 decision 产物（可选合并，仅标注）
- **来源**：`outputs/decision/*.csv`
- **规则**：
  - 只允许“字段合并与标签标注”，不得用于过滤候选
  - 合并后的字段统一 `dec_` 前缀

---

## 2. 输出（Outputs）—— 文件与路径契约（锁死）
系统每次运行（每个 trade_date=T）必须产出：

1) **Top30 预测表**  
- `outputs/premium/premium_top30_{T}.csv`

2) **Full 预测表（全量候选）**  
- `outputs/premium/premium_full_{T}.csv`

3) **验证表（真值到达后自动填充）**  
- `outputs/premium/premium_verify_{T}.csv`

4) **报告（Markdown）**  
- `docs/reports/premium_{T}.md`  
- `docs/reports/premium_latest.md`（每次覆盖）

5) **运行追溯（每次覆盖）**  
- `outputs/premium/_last_run.txt`

> 运行要求：即使真值未到，Top30/Full/Report 仍必须产出（pending 模式），不得断更。

---

## 3. 输出字段契约（核心）
### 3.1 预测核心字段（必须存在）
在 `premium_top30_{T}.csv` 与 `premium_full_{T}.csv` 中，必须包含：

- `rank`（从 1 开始，连续；必须为第一列）
- `trade_date`（T）
- `target_date`（T2 = T+2，若无法确定可为空，但需在报告说明）
- `ts_code`
- `name`

**T+2 收盘价分布预测（核心价值字段）**：
- `close_T`（真值 Close[T]，来自 `daily_{T}.csv`；若缺则为空，并触发 pending）
- `r_p05` / `r_p25` / `r_p50` / `r_p75` / `r_p95`  
  - 定义：`r = ln(Close[T+2] / Close[T])` 的分位数预测  
- `close_T2_p05` / `close_T2_p25` / `close_T2_p50` / `close_T2_p75` / `close_T2_p95`  
  - 由上面的 r 分位数还原：`close_T2_pXX = close_T * exp(r_pXX)`

> 若你只想输出 P10/P50/P90 也可以，但建议与截图一致，默认采用 P05/P25/P50/P75/P95。

### 3.2 先验与标注字段（必须保留但不强制有值）
- `p_premium`（若主输入存在则透传；否则可为空）
- `dec_*`（来自 decision 合并，允许为空）
- `p_fill` / `fill_hint`（可成交概率/提示）  
  - **规则**：只标注，不参与排序、不参与过滤

### 3.3 验证字段（verify 表）
`premium_verify_{T}.csv` 必须与 Top30 **顺序完全一致（不可重新排序）**，包含：

- `rank`, `trade_date`, `target_date`, `ts_code`, `name`
- `close_T`（真值）
- `close_T2_actual`（真值 Close[T+2]，若未到则为空）
- `r_actual = ln(close_T2_actual / close_T)`
- `err_p50 = close_T2_actual - close_T2_p50`
- `hit_p90`（是否落在预测区间内：`close_T2_actual <= close_T2_p95`）
- `hit_p10`（是否高于下界：`close_T2_actual >= close_T2_p05`）

> 真值未到：verify 允许为空壳，但必须在 report 与 _last_run 里写清 pending 原因。

---

## 4. 排序规则（锁死）
Premium 的排序以 **预测中位数收益**为主（可交易性更强）：

- 默认排序键：`r_p50` 降序  
- 若 `r_p50` 缺失，则退化为：
  - `p_premium` 降序（若存在）
  - 否则保持主输入原顺序（不得随机）

> `P_fill` 不参与排序；`dec_*` 不参与排序。

---

## 5. Pending 与降级策略（必须实现）
### 5.1 预测时缺 close_T（daily_T 缺失）
- `close_T` 为空
- `r_*` 与 `close_T2_*` 允许为空
- 必须输出 Top30/Full/Report
- `_last_run.txt` 标注：`pending=True`，`reason=truth_not_ready_close_T`

### 5.2 真值未到（daily_T2 缺失）
- 预测仍可产出（因为预测不依赖 Close[T+2]）
- verify 表标注 pending
- `_last_run.txt` 标注：`verify_pending=True`，`verify_reason=truth_not_ready_T2`

### 5.3 目标交易日无法推进（target_date 无法确定）
- `target_date` 可为空
- 预测可产出，但报告必须提示“无法推进到 T+2 的原因”

---

## 6. 自学习闭环（Learning Loop）—— 阶梯策略（30/60/90/150）
### 6.1 训练样本定义（强约束）
- 一个有效样本 = 在同一 ts_code 上同时拥有：
  - 特征（来自 pred_source_latest + 可选真值切片）
  - 标签 `r_actual = ln(C2/C0)`（需要 close_T 与 close_T2_actual）

### 6.2 阶梯模型启用规则（建议默认）
- **S0：N < 30**  
  - 仅用全局历史分布（强收缩）
- **S1：30 ≤ N < 60**  
  - 单因子分桶（p_premium 或强度）+ 收缩
- **S2：60 ≤ N < 90**  
  - 双因子分桶 + 收缩
- **S3：90 ≤ N < 150**  
  - 稳健线性/岭回归预测 r_p50 + 残差分位估计区间
- **S4：N ≥ 150**  
  - LGBM 分位数回归（直接学习 P05/P50/P95 等）+ 校准

> N 以“可训练样本数”为准（不是日历天数）。

### 6.3 校准指标（必须落盘）
系统必须维护并输出校准指标（至少在 md 报告中）：
- P95 覆盖率：`Pr(actual <= p95)` 应接近 0.95
- P05 覆盖率：`Pr(actual >= p05)` 应接近 0.95
- P50 偏差：median(err_p50) 接近 0
- 近 10/30 天滚动校准曲线（可先表格化）

---

## 7. 报告（Markdown）结构要求（最低）
报告必须包含：
1) 本次运行摘要：T、T2、model_stage、pending/verify_pending、原因
2) Top30 价格分布预测表（P05/P25/P50/P75/P95）
3) 校准与误差诊断（滚动覆盖率、偏差）
4) 验证表（若真值未到则显示 pending 状态与原因）

---

## 8. 工程约束（必须遵守）
- 不得引入“拉取 TOP3 全库”的全量同步行为
- 只允许按日期拉取最小真值切片并缓存
- 输出路径与命名必须稳定，不得随意改动
- 任何异常不得导致 workflow 失败：应转为 pending 并写清 reason
- 所有新增字段必须向后兼容（不删旧字段）

---

## 9. 版本与验收标准
### V2 验收（必须满足）
- `target_date` 能在真值可得时正确推到 T+2
- `premium_top30/full` 中 `close_T2_p50` 不再全空/全 0
- 真值到达后 `premium_verify` 自动填充 `close_T2_actual` 与误差字段
- 校准指标可输出且可持续更新（准闭环成立）

---
