Premium（T+2 收盘价分布预测）— 手工交易版 V2 需求契约（锁死版）

一句话定位：Premium 是“强池候选（来自 a-top10）→ 预测 T+2 收盘价分布（分位数）→ 真值回流校准 → 阶梯自学习”的稳定闭环系统。

唯一主线目标：在交易日 T 收盘后，对候选股票给出 T+2 收盘价的分布预测（P05/P25/P50/P75/P95），并可被真值校准、逐日自我迭代变准。

明确边界（锁死）：
- p_fill 等“能否买到/成交性”信息默认只做标注，不作为过滤与排序条件（0 权重影响排序）。
- ✅ 专业分层：价格分布预测主线保持纯粹；p_fill 作为独立“执行成功概率线”并列输出，避免污染标签与误导人类。

0. 术语与时间轴（锁死）

T：本次预测基准交易日（收盘后生成报告，T 当日已无法交易）

T+1：下一交易日（人类手工交易执行日，Premium 不负责交易执行）

T+2：第二个后续交易日（按交易日历顺延，不是自然日）

Close[T]：T 日收盘价（真值）

Close[T+2]：T+2 日收盘价（真值，用于验证/学习）

交易日历推进规则（锁死）

T+1、T+2 必须按“可获取行情真值的交易日”推进；遇周末/节假日顺延。

若 T+2 真值尚未到达：必须进入 pending，不得报错卡死，预测仍需稳定产出。

1. 输入数据（锁死）

1.1 主输入：强池候选与先验（必须）

路径：data/pred/pred_source_latest.csv

来源：a-top10 输出的强势候选池（可包含强度/概率/题材等先验信号；不要求包含价格）

最低必需字段（必须能解析）：

trade_date（YYYYMMDD）

ts_code

name（可缺/可空）

可选字段（用于分桶/特征；不锁字段名，允许别名映射）：

概率类：p_premium / probability / prob / p

强度/题材/热度/资金等任意字段（存在则用，不存在则跳过）

注意：pred_source_latest.csv 不需要包含价格。价格由真值层提供。

1.2 行情真值（必须，用于预测还原与学习打标）

目录：data/market/

命名（锁死）：

data/market/daily_{T}.csv

data/market/daily_{T2}.csv（其中 T2 = T+2）

最低必需字段（必须）：

ts_code

close

真值未到/缺失：允许 pending，但不得影响预测产出与落盘。

1.3 decision 产物（可选，仅合并标注，不得过滤）

来源 glob：outputs/decision/*.csv

规则（锁死）：

仅用于字段合并/标签；不得用于过滤候选池

合并键：trade_date + ts_code（name 仅用于补全/校验）

合并字段统一使用 dec_ 前缀，避免冲突

1.4 执行线输入（可选）：p_fill / 执行成功概率

来源：可来自 pred_source_latest 或 decision 合并字段（或未来独立执行模型）

规则（锁死）：
- 只作为“执行成功概率线”并列输出与提示
- 不得用于过滤候选池
- 不得用于排序（0 权重影响排序）

2. 输出文件（锁死）

每次运行（trade_date=T）必须产出以下文件（即使 pending 也不得断更）：

Top30 预测表

outputs/premium/premium_top30_{T}.csv

Full 预测表（全量候选展开）

outputs/premium/premium_full_{T}.csv

验证表（顺序与 Top30 完全一致）

outputs/premium/premium_verify_{T}.csv

报告（Markdown）

docs/reports/premium_{T}.md

docs/reports/premium_latest.md（每次覆盖）

运行追溯（每次覆盖）

outputs/premium/_last_run.txt

3. 核心预测定义（锁死）

Premium 的预测目标不是收益率本身，而是 T+2 收盘价分布。

3.1 内部建模变量：对数收益 r（锁死定义）

定义：r = ln(Close[T+2] / Close[T])

预测输出：r 的分位数（P05/P25/P50/P75/P95）

价格还原：

Close[T+2]_pXX = Close[T] * exp(r_pXX)

这样可避免不同股票价格尺度差异导致的训练不稳。

4. 输出字段契约（锁死）

4.1 Top30 / Full 表字段（必须包含）

基础字段（必须）

rank（从 1 连续编号；必须为第一列；不允许重复插入）

trade_date（T）

target_date（T2=T+2，若无法确定可为空，但必须在报告与 _last_run 说明原因）

ts_code

name

价格真值与分布预测（核心价值字段，必须）

close_T（Close[T]，来自 daily_{T}.csv；缺失则为空并触发 pending）

r_p05 r_p25 r_p50 r_p75 r_p95

close_T2_p05 close_T2_p25 close_T2_p50 close_T2_p75 close_T2_p95

先验/标签字段（允许为空，但建议保留）

p_premium（若主输入存在则透传；否则可为空）

risk_flags / confidence / data_quality（只提示，不过滤）

p_fill / fill_hint（执行成功概率线：只标注，不过滤、不排序，允许为空）

dec_*（决策层合并字段：只标注）

4.2 验证表字段（锁死）

premium_verify_{T}.csv 必须与 Top30 行顺序完全一致（不可重新排序），必须包含：

rank, trade_date, target_date, ts_code, name

close_T

close_T2_actual（Close[T+2] 真值；未到则为空）

r_actual = ln(close_T2_actual / close_T)（未到则为空）

err_p50 = close_T2_actual - close_T2_p50（未到则为空）

in_p90（是否落在区间内：close_T2_actual <= close_T2_p95，未到则为空）

in_p10（是否高于下界：close_T2_actual >= close_T2_p05，未到则为空）

5. 排序规则（锁死）

Premium 的排序以“预测中位数收益”作为主排序依据：

默认排序：r_p50 降序

若 r_p50 缺失：退化为 p_premium 降序（若存在）

若 p_premium 也缺失：保持主输入原顺序（不得随机排序）

明确（锁死）：
- p_fill 不参与排序；dec_* 不参与排序；任何风险提示字段不参与硬过滤。
- p_fill 在报告中仅作为“能买到概率/执行风险提示”并列展示，避免误导为价格方向信号。

6. Pending 机制（不得报错卡死）

6.1 预测时缺 close_T（daily_T 缺失）

允许输出 close_T 为空，分布预测字段可为空

必须仍产出 Top30/Full/Report

_last_run.txt 必须写明 pending=True、reason=truth_not_ready_close_T

6.2 验证时缺 close_T2_actual（daily_T2 缺失）

预测仍可产出（预测不依赖真值）

verify 表写空壳（或部分字段为空）

_last_run.txt 必须写明 verify_pending=True、verify_reason=truth_not_ready_T2

6.3 无法推进 target_date

target_date 可为空

报告必须明确：无法推进到 T+2 的原因（例如真值源缺失/交易日历不可用）

7. 自学习闭环（锁死：30/60/90/150 阶梯）

7.1 有效训练样本定义（锁死）

一个有效样本需要同一 ts_code 同时具备：

特征（来自 pred_source_latest 及可选合并/真值切片）

标签：r_actual = ln(C2/C0)（必须有 close_T 与 close_T2_actual）

训练样本数 N 指“有效可训练样本数”，不是日历天数。

7.2 阶梯启用规则（锁死）

S0：N < 30

仅用全局历史分布（强收缩基线），保证稳定不断更

S1：30 ≤ N < 60

单因子分桶（优先 p_premium 分位桶），桶内统计 + 收缩

S2：60 ≤ N < 90

双因子分桶（p_premium × 强度/热度 任一稳定字段），桶内统计 + 收缩

S3：90 ≤ N < 150

稳健线性/岭回归预测 r_p50，分位用残差分布估计

S4：N ≥ 150

分位数回归模型（如 LGBM Quantile）直接输出 r_p05/p50/p95 等，并持续校准

阶梯升级必须自动化；必要时允许“校准恶化”触发降级（避免 regime 切换导致自信爆炸）。

7.3 校准指标（必须输出）

系统必须输出并持续更新（至少写入 md 报告）：

P95 覆盖率：Pr(actual <= p95) 接近 0.95

P05 覆盖率：Pr(actual >= p05) 接近 0.95

P50 偏差：median(err_p50) 接近 0

近 10/30 天滚动校准统计（表格即可）

8. 报告页面要求（锁死：必须两张核心表）

8.1 预测表（T 收盘后生成）

标题示例：{T} → {T+2} 价格分布预测（Top30）

必须包含：

基础字段：rank / trade_date / target_date / ts_code / name / close_T

分位数预测：P05/P25/P50/P75/P95（价格）

可选：p_premium、风险提示、p_fill（仅标注）

8.2 验证表（T+2 真值到达后自动填充）

标题示例：{T} → {T+2} 价格分布预测验证（Top30）

必须包含：

与预测表相同顺序（不可重新排序）

close_T2_actual、err_p50

覆盖命中：in_p10/in_p90（区间覆盖）

9. 工程约束（锁死）

禁止在 top10-decision 侧“同步 TOP3 全库”；只允许按日期拉取最小真值切片并缓存

输出路径与命名稳定：任何外部对接以此为准，不得随意改动

任何异常不得导致 workflow 失败：必须 pending 并记录 reason

所有新增字段必须向后兼容：允许新增，不允许删除/改名造成断链

10. 专业解冻条款（系统级，优先级高于锁死条款）

说明：锁定的目的是防止开发飘逸；若为了系统整体价值更高、更专业，可按本条款“有条件解冻”。

10.1 允许解冻的范围（默认关闭，需显式开启）

A) ✅ 允许：p_fill 参与 risk_flags / confidence / data_quality 的生成（仅提示）

B) ✅ 允许（推荐的专业路径）：p_fill 仅用于“区间宽度/置信度调节”（影响 r_p05/p95 或等价的区间宽度），不得影响 r_p50，不得影响排序，不得过滤

C) ⚠️ 不建议：p_fill 进入中心预测（r_p50）或进入排序/过滤。若必须启用，需满足 10.2 的严格条件并明确升级目标为“可执行收益分布”。

10.2 解冻的必要条件（满足其一不足，需同时满足）

1) 口径统一：p_fill 定义固定、可复现、跨天一致（非临时手工/非随意变更）

2) 真值可回流：能稳定获得执行真值（是否买到/成交价/滑点之一或组合），覆盖率足够，能形成有效样本 N（否则学习会变慢且偏）

3) 目标明确升级：若 p_fill 进入中心预测或排序，必须将 Premium 的主目标从“价格分布预测”明确升级为“可执行收益/成交后收益分布预测”，并接受标签与校准口径变化

10.3 解冻后的审计要求（必须）

- 报告中必须显式声明：p_fill 当前影响范围（仅提示 / 仅区间宽度 / 进入中心预测等）
- 校准指标必须同时输出：价格分布覆盖率（仍以 Close[T+2] 为真值）以及（若升级目标）执行相关指标
- 若近 10/30 天校准显著恶化，允许自动降级/回退（恢复到“纯粹价格分布主线”）

11. V2 验收标准（锁死）

必须满足：

premium_top30/full 中 close_T2_p50 不再全空/全 0（在 close_T 可得时）

target_date 能在真值可得时正确推进到 T+2

真值到达后 premium_verify 自动填充 close_T2_actual 与误差/覆盖字段

校准指标能持续更新（准闭环成立）

p_fill 默认仅标注，不影响排序与过滤（除非按第 10 条款显式解冻）
