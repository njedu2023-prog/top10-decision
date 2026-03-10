继续 top10-decision 主线开发。

当前主线：
Premium 主线，且已从普通报告层升级为 E_ret_plus 主线。

一、当前大状态

Decision 当前已基本收口，暂不回头深挖：

- P_fill 学习链已打通
- E_ret 学习链已打通
- Decision 主引擎接线已完成
- Cost / RiskPenalty / EV 已基本做实
- decision_report 已可输出 TopN Targets + Full Candidate Pool

Decision 当前定性：
主体开发已基本完成，阶段性收口完成。

当前不要再回头深挖：

- Decision 内部小修小补
- 历史补跑问题
- ST / tail risk 个别样本触发问题
- Premium 之外的展示层问题

二、Premium 当前已锁定的新主线

Premium 不再以“人类看报告更方便、更好看”为主目标。
那个重要度只有约 10%。

Premium 当前最高目标已锁定为：

以 E_ret 为第一核心核，
继续构建更强、更准、可闭环增强的第二核心核：

E_ret_plus

当前锁定主链：

Raw / T_Close / 上下文
→ E_ret
→ X（EHX：E_ret 二级增强层）
→ E_ret_plus

当前锁定理解：

- E_ret 是第一核心核
- E_ret_plus 是 Premium 的最终核心核
- P_fill / Cost / RiskPenalty / EV 不是第一核心，它们要围绕 E_ret_plus 服务
- P10 / P50 / P90 不是当前主线起点，只能在 E_ret_plus 稳定后作为下游表达层

三、当前已经完成到哪一步

1. Premium 契约文档已重写升级
已形成 Premium Contract V3 主线思路：
不再是普通报告层，而是 E_ret_plus 二级预测核开发。

2. predict.py 已接入 E_ret_plus 主链
当前已能产出这些字段：

- eret_pred_raw
- eret_plus_value
- eret_plus_delta
- eret_plus_direction
- eret_plus_conf
- eret_plus_conf_score
- eret_plus_src

3. 当前实际跑通的是 coldstart 版 EHX
已确认：
outputs/premium/_last_run.txt 中显示：

eret_plus_src: ehx:coldstart_v1

说明：
当前 E_ret_plus 主链已接通，
但还没有真正吃到训练后的 EHX 模型，
当前仍是冷启动增强器，不是学习版增强器。

4. outputs/premium 已成功产出
已看到 20260309 这批产物成功落盘：

- premium_top30_20260309.csv
- premium_full_20260309.csv
- premium_verify_20260309.csv
- _last_run.txt

说明：
Premium 预测链已跑通，不是断链问题。

四、当前不要误判的问题

当前 verify_pending = True 是正常的，不是系统坏了。

原因：
现在这批是 20260309 的预测，
当前业务时间机制仍然是：

- T = 20260309
- T+1 = 20260310
- T+2 = 20260311

而当前 T+2 真实收盘价还没到来，
所以：

- premium_verify 暂时 pending
- plus_abs_err / raw_abs_err / improve_flag 还不能正式验收

这是正常业务时间纪律，不是错误。

五、之前已锁定、这次必须继续遵守的时间机制

必须继续遵守：

预测-成熟-训练 三阶段时间纪律

核心规则：

1. 预测链跑当前
当前交易日 T 收盘后，可以正常生成 Premium 预测产物。

2. 未成熟样本只允许 pending 落库
若 T+2 真值未到，则只能先落库，不允许强行验真，不允许拿去正式训练。

3. 训练链只吃成熟历史样本
只有已经等到 T+2 真值的历史样本，才允许进入训练与评估。

一句话锁死：

预测系统跑当前，
学习系统只吃过去已成熟样本，
未到 T+2 的样本只允许 pending 落库，不允许进入正式训练与正式验真。

六、当前已经定位出的两个关键问题

下一次不要散，直接查这两个点：

问题 1：
为什么当前还是 ehx:coldstart_v1，
而不是训练后的模型版 EHX 接管？

高概率只会有两种可能：

- predict.py 还没有真正优先加载 ehx_delta.joblib
- 或者 outputs/premium/models/ 下根本没有真正产出 ehx_delta.joblib / ehx_meta.json
- 或者虽然产出了，但 workflow 运行态没拿到 / 路径不对 / 加载失败后自动回退了冷启动

问题 2：
为什么 _last_run.txt 里出现：

pending_reason: missing_TUSHARE_TOKEN|fallback_business_day

说明：
当前 Premium 运行环境里，没有正确拿到 TUSHARE_TOKEN，
导致 target_date 只能走工作日兜底，不是严格交易日历推进。

这个也要修。

七、下一次最合理的开发顺序

下一次继续时，不要再泛谈价值，不要回头碰 Decision 主线。
直接按下面顺序推进：

第一步：
先验收 / 排查 outputs/premium/models/ 目录
看是否已经存在：

- ehx_delta.joblib
- ehx_meta.json

如果没有，说明训练链没真正产出 EHX 模型。

第二步：
检查 premium/train.py 当前是否已经真正训练并落盘 EHX 模型
重点不是旧 LR / LGBM，
重点是：

- delta_ret 是否真的定义并训练
- ehx_delta.joblib 是否真的保存
- ehx_meta.json 是否真的保存
- _last_run / eval_history 是否有 ehx_trained 相关痕迹

第三步：
检查 premium/predict.py 当前是否真的做到：

优先加载 ehx_delta.joblib
加载成功则走模型版 EHX
加载失败才回退 coldstart_v1

当前最关键的目标就是把：

eret_plus_src: ehx:coldstart_v1

推进成：

eret_plus_src: ehx:model_v1
或同等语义的学习版来源标识。

第四步：
检查 Premium workflow / 运行环境里是否正确注入了 TUSHARE_TOKEN

因为当前 _last_run.txt 已证明：
Premium 运行态没有正确拿到它。

第五步：
等 T+2 真值真正到来后，再正式验收：

- raw_abs_err
- plus_abs_err
- improve_flag
- E_ret_plus 是否比 E_ret 更准

在 T+2 未到前，不要误判 verify_pending 是错误。

八、当前文件优先级

下一次优先看这些文件：

1. src/top10decision/premium/train.py
2. src/top10decision/premium/predict.py
3. outputs/premium/models/（看模型产物有没有）
4. outputs/premium/_last_run.txt
5. .github/workflows/ 里 Premium 相关 workflow（检查 TUSHARE_TOKEN 注入）

九、当前总判断

当前状态不是坏状态，而是：

- Premium 的 E_ret_plus 主链已接通
- 预测落盘已成功
- 时间纪律正常
- 但学习版 EHX 尚未真正接管
- 当前仍停留在 coldstart 增强阶段
- 下一刀必须聚焦在“让训练版 EHX 真正落盘并被 predict.py 接管”

十、下一次开干时的第一句话

先不要回头动 Decision。
先查 Premium 的 EHX 模型有没有真正落盘，以及 predict.py 是否真的优先加载模型。
当前要解决的是：把 eret_plus_src 从 ehx:coldstart_v1 推进到学习版模型接管。
