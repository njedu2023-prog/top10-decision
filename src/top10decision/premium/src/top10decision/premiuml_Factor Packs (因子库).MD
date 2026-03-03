# Premium — Factor Packs（因子库契约）

目标：为 Premium 的 T→T+2 收盘价分布预测提供“可回放、可扩展、可降级”的因子库分层。
说明：本文件是【契约与审计规范】；实际计算逻辑需由代码实现。

---

## 0. 总体规则（系统级锁死）

### 0.1 目标变量（锁死）
- 内部预测变量：r = ln(Close[T+2] / Close[T])
- 输出：r 的分位数（P05/P25/P50/P75/P95）→ 还原 Close[T+2] 的分位数

### 0.2 Factor Packs 分层原则（锁死）
- Pack0_base：永远可用（只依赖本仓库已落盘数据），必须永久保底。
- Pack1_tushare_basic：可选增强（依赖 Tushare/daily_basic 等），可用则启用，不可用则自动跳过。
- Pack2_limit_micro：可选护城河（依赖涨停结构字段），可用则启用，不可用则自动跳过。

### 0.3 自动启用与降级（必须）
运行时必须输出并记录：
- factor_packs_used：本次实际启用的包列表
- factor_packs_missing：因缺数据/缺字段而跳过的包列表
- degrade_mode：是否发生降级（true/false）
- missing_fields：关键缺失字段列表（便于排查）

---

## 1. 数据源契约（已验证）

### 1.1 候选与先验（pred_source）
文件：data/pred/pred_source_latest.csv
已知字段（至少存在）：
- trade_date, verify_date, rank, ts_code, name
- prob, StrengthScore, ThemeBoost, board
- run_id, run_attempt, commit_sha, generated_at_utc

### 1.2 行情真值（market daily）
文件：data/market/daily_YYYYMMDD.csv
已知字段（至少存在）：
- ts_code, trade_date, open, high, low, close, vol, amount

---

## 2. Pack0_base（必选｜永久保底）

### 2.1 设计目的
- 在不依赖任何外部接口（仅依赖本仓库落盘数据）情况下，保证 Premium 能稳定产出分位分布并可自学习闭环。

### 2.2 输入依赖
- pred_source_latest.csv（先验信号）
- market daily_{T}.csv（Close[T] + OHLCV）
- market daily_{T-k..T}.csv（近 N 天 OHLCV，用于衍生因子；建议 N=10/20）

### 2.3 核心因子（中心位置：r_p50 优先）
来自 pred_source：
- f_prob = prob
- f_strength = StrengthScore
- f_theme = ThemeBoost
- f_rank = rank（可选）
- f_board = board（类别特征：建议先做“board_id/编码”，不要强依赖外部热度）

### 2.4 核心因子（区间宽度与尾部：r_p05/r_p95 优先）
来自 market OHLCV（近 N 日派生，建议 N=10/20）：
动量：
- ret_1d = ln(close_T / close_T-1)
- ret_3d = ln(close_T / close_T-3)
- ret_5d = ln(close_T / close_T-5)

波动：
- vol_5d = std(ret_1d over last 5 trading days)
- vol_10d = std(ret_1d over last 10 trading days)

振幅/影线（需要 open/high/low/close）：
- range_1d = (high_T - low_T) / close_T
- upper_shadow_1d = (high_T - max(open_T, close_T)) / close_T
- lower_shadow_1d = (min(open_T, close_T) - low_T) / close_T
- range_5d = max(high over 5d) / min(low over 5d) - 1

量能（vol/amount）：
- amount_z_5d = amount_T / mean(amount over 5d)
- vol_z_5d = vol_T / mean(vol over 5d)

### 2.5 缺失策略（锁死）
- 任何单因子缺失不得导致流程中断：用中位数/0/unknown 填充，并增加 is_missing_* 标记（或写入 missing_fields）。
- 若近 N 日窗口不足：自动缩短窗口（例如 10d→5d→1d），并记录 degrade_mode=true。

---

## 3. Pack1_tushare_basic（可选增强｜流动性/市值硬因子）

### 3.1 设计目的
- 引入更稳健的流动性与规模因子，提升分位区间的稳定性与可校准性。

### 3.2 依赖条件
- 仓库已配置 Tushare Token（TUSHARE_TOKEN）
- 能获得 daily_basic（或等价数据）

### 3.3 建议字段（优先级从高到低）
- turnover_rate（真实换手率）
- circ_mv（流通市值）/ total_mv（总市值）
- volume_ratio（量比，如可得）

### 3.4 落盘建议（强烈推荐：避免污染 market daily）
- 建议落盘到：data/market_basic/daily_basic_YYYYMMDD.csv
- 不建议直接合并进 data/market/daily_YYYYMMDD.csv（避免路径与口径污染）

### 3.5 缺失与降级
- 任一关键字段缺失 → 本包整体跳过（仍保留 Pack0）
- 记录 factor_packs_missing 中注明缺失原因

---

## 4. Pack2_limit_micro（可选护城河｜涨停结构微观因子）

### 4.1 设计目的
- 通过涨停结构刻画“右尾肥度/风险宽度”，形成 Premium 的护城河因子体系。

### 4.2 依赖字段（示例）
- open_times（开板次数）
- seal_amount（封单金额）
- first_limit_time / last_limit_time（封板时间）

### 4.3 缺失与降级
- 只要上述核心字段不能稳定获取 → 本包自动跳过
- 禁止因本包缺失导致 Premium 预测中断

---

## 5. 审计输出要求（必须写入报告或 _last_run）

每次运行必须记录：
- factor_packs_used
- factor_packs_missing
- degrade_mode
- missing_fields（若有）
- 主要窗口参数（N_window=10/20 等）

---

## 6. 备注：版本与演进
- Factor Packs 是“能力分层”，不强制按时间顺序启用。
- 但 Pack0_base 必须永久保底；Pack1/Pack2 只能“可插拔增强”，不得破坏 Pack0 的产出与闭环。
