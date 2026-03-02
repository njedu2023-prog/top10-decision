# ANCHOR — Premium 主线（T+2 收盘价分布预测 V2）无缝对接锚点

> 用途：你关机退出后，下次开新对话把本 MD 直接贴给我，我会以此作为**唯一工程主线锚点**继续推进（不跑偏、不重问已确认内容）。

---

## 0) 当前日期与状态
- 当前日期：2026-03-02
- 仓库：`njedu2023-prog/top10-decision`（GitHub 在线部署）
- 当前状态：Premium 工作流已跑通、产物能落盘（Top30/full/verify/md/_last_run），此前报错 `rank already exists` 已修复；目前输出内容“不全”是数据链路/真值链路未接通导致（非报错）。

---

## 1) 最高优先级主线目标（已锁死）
**Premium 的唯一主线目标：在交易日 T 收盘后，预测每只候选股的 T+2 收盘价分布，并可校准自学习。**

- 目标预测：`Close[T+2]` 分布（分位数）
- 默认分位点：`P05 / P25 / P50 / P75 / P95`
- `P_fill`：0 权重，仅标注，不参与过滤与排序

---

## 2) 已确认的核心逻辑（不可改）
### 2.1 时间轴
- T：收盘后生成预测报告
- T+1：人类执行交易（Premium 不负责交易）
- T+2：用于验证与学习（按交易日历推进，非自然日）

### 2.2 内部建模变量（建议锁死）
- 用对数收益：`r = ln(Close[T+2] / Close[T])`
- 输出：`r_p05/r_p25/r_p50/r_p75/r_p95`
- 还原价格：`close_T2_pXX = close_T * exp(r_pXX)`

### 2.3 排序（锁死）
- 默认排序：`r_p50` 降序
- 若缺：退化到 `p_premium`（若存在）
- 再缺：保持源表顺序（不得随机）

---

## 3) V2 需求契约（文档层）
- 已决定：**覆盖重写 `Premium.md` 为 V2 契约**（V1 推翻）
- V2 契约核心点：
  - 输入：`data/pred/pred_source_latest.csv`（无价格也允许）
  - 真值缓存：`data/market/daily_{T}.csv` & `data/market/daily_{T2}.csv`（至少 ts_code/close）
  - 输出文件（锁死）：
    - `outputs/premium/premium_top30_{T}.csv`
    - `outputs/premium/premium_full_{T}.csv`
    - `outputs/premium/premium_verify_{T}.csv`
    - `docs/reports/premium_{T}.md`
    - `docs/reports/premium_latest.md`
    - `outputs/premium/_last_run.txt`（覆盖）

---

## 4) 当前痛点（必须先解决）
### 4.1 真值链路未接通（导致内容不全）
现象：
- `target_date` 为空
- `premium_verify_{T}.csv` 极小/空壳（pending）
- Top30/full 中 `e_premium/score_ev` 之前为 0（旧逻辑）；未来将由价格分布字段取代核心地位

根因方向：
- top10-decision 仓库当前可见 `data/` 下无 `data/market/` 真值缓存目录（或未被 workflow commit 回仓库）
- Premium 的 `ensure_daily_cached()` 未能把 `daily_{T}` / `daily_{T2}` 持久落盘到仓库（可能只在 runner 里存在，未提交）

---

## 5) 已知可用证据（用于快速定位）
### 5.1 最近一次 Premium Run 日志关键行（示例）
- `[premium][predict] ok=True trade_date=20260227 reason=pending`
- 产物路径存在：
  - `outputs/premium/premium_top30_20260227.csv`
  - `outputs/premium/premium_full_20260227.csv`
  - `outputs/premium/premium_verify_20260227.csv`
  - `docs/reports/premium_20260227.md`

### 5.2 仓库产物现状（示例）
- `premium_top30_20260227.csv`：`p_premium` 有值，但 `target_date` 空，`e_premium/score_ev` 为 0（旧方案遗留）
- 目标：切换到 V2 价格分布字段输出（见第 2 节）

---

## 6) 下一步工程执行清单（下次开工直接做）
> 目标：先把“真值闭环”跑通，再做“分布预测冷启动”，再接“阶梯自学习”。

### Step A（必做）：真值缓存落盘 + 提交
- 确保 `data/market/` 目录在仓库中存在并可持续更新
- 每次 Premium run（或独立 workflow）要把新增的：
  - `data/market/daily_{T}.csv`
  - `data/market/daily_{T2}.csv`（真值到达时）
  一并 `git add` + `commit` 回仓库
- 验收：仓库可直接看到 `data/market/daily_YYYYMMDD.csv`

### Step B（必做）：Premium 输出字段切换到价格分布
- 修改 `src/top10decision/premium/predict.py`：
  - 读取 `close_T`
  - 产出 `r_p05/p25/p50/p75/p95`
  - 产出 `close_T2_p05/p25/p50/p75/p95`
  - verify 写入 `close_T2_actual/err_p50/in_p10/in_p90`
- 修改 `src/top10decision/premium/report_md.py`：展示分位数表 + 校准表

### Step C（必做）：冷启动 + 阶梯学习框架
- 阶梯：30/60/90/150（N=有效可训练样本数）
- S0：全局分布强收缩
- S1/S2：分桶+收缩
- S3：稳健线性/岭回归
- S4：分位数回归（LGBM Quantile）+ 校准
- 校准指标：P95 覆盖率 / P05 覆盖率 / P50 偏差（滚动 10/30 天）

---

## 7) 关键文件清单（下次我将按“全码覆盖”交付）
- `Premium.md`（文档已决定覆盖为 V2）
- `.github/workflows/run_premium.yml`（需要确保 commit 包含 `data/market/**`）
- `scripts/run_premium.py`
- `src/top10decision/premium/`
  - `config.py`
  - `market_truth.py`（ensure_daily_cached / load_daily）
  - `predict.py`（核心：分布预测与 verify）
  - `report_md.py`（核心：两表 + 校准）
  - （后续新增）`calibration.py` / `learn.py`（阶梯学习）

---

## 8) 下次开工口令（复制这段发我即可）
**口令：继续 Premium V2 主线。目标：先打通 data/market 真值缓存落盘与提交；再把 predict 输出切换为 Close[T+2] 分布（P05/P25/P50/P75/P95）；再接入阶梯自学习（30/60/90/150）与校准指标。按全文件覆盖方式给我改动。**
