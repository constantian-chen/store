# react_syn_persistent vs interpreter 性能提升计划

## 结论先行

从当前两份日志看，`eval_react_syn_persistent_150.out` 还不能证明已经比 `eval_interpreter_27b.out` 有稳定提升：

| 文件 | 实际进度 | 正确 | 错误 | 异常 | 未完成 | 答题准确率 |
|---|---:|---:|---:|---:|---:|---:|
| `eval_react_syn_persistent_150.out` | 1-50，其中第 50 未完成 | 41 | 8 | 0 | 1 | 41/49 = 83.7% |
| `eval_interpreter_27b.out` | 1-219，其中第 219 未完成 | 193 | 20 | 5 | 1 | 193/213 = 90.6% |
| `eval_interpreter_27b.out` 前 49 题 | 1-49 | 43 | 6 | 0 | 0 | 87.8% |

所以优先级应该从“继续堆 GEPA/prompt”调整为：

1. 先让 persistent 评测能稳定跑完同一段，避免第 50 题附近早停。
2. 让 synthesized-tools 路径只在有收益时启用，否则退回 `python_eval` interpreter。
3. 修掉答案等价判断和抽取误判，避免“数学等价但判错”。
4. 对真正难题再加自洽投票、验证 pass、领域化求解策略。

---

## 当前差距

### 1. persistent 跑不完整

`eval_react_syn_persistent_150.out` 标头写的是 `Evaluating test problems 1..150 of 500`，但日志只到第 50 题开头，第 50 题没有 `Predicted` / `Result`。这说明当前最大问题不是准确率，而是稳定性和恢复能力。

推荐先补：

- 每题级 checkpoint：每完成一题立刻落 JSONL，重启后从最后完成题继续。
- LM 调用异常重试：`InternalServerError` / connection error / timeout 至少重试 2 次，指数退避。
- 单题硬超时：超过总时限后记录 `TIMEOUT`，释放 sandbox/interpreter，继续下一题。
- 评测 summary：结束时打印 `completed/correct/wrong/exception/incomplete`，避免只靠 grep 日志判断。

### 2. 前 49 题 persistent 低于 interpreter

前 49 题对比：

- persistent：41/49。
- interpreter：43/49。

persistent 额外输在第 35、38、43 题；第 43 题尤其明显，interpreter 给出 `2^k`，persistent 输出 `0`。这类问题不像是“没有工具”，更像是 synthesized tool 或 reasoner 把泛化公式题当成数值题处理。

### 3. 有些判错其实是等价表达式

两边都出现等价表达式被判错的情况：

- 第 11 题：`3*pi/13 - 4*log(3/2)/13` 与 `-4*log(3)/13 + 4*log(2)/13 + 3*pi/13` 等价。
- 第 38 题：`(4035^(1/1024)-1)/2` 与 `-1/2 + 4035**(1/1024)/2` 等价。
- 第 92 题：`(-11-4i)/3` 与 `-11/3 - 4i/3` 等价，interpreter 被判错。

这部分不是模型能力问题，是 verifier/normalizer 问题，属于低成本高收益。

---

## Tier 0：评测稳定性和可比性

### 0.1 增加断点续跑

目标：评测任何时候断掉，都能从最后完成题继续，而不是丢掉整段。

改动：

- 将 `evaluate_with_logging` 改成 JSONL 或每题 flush JSON。
- 启动时读取已有 log，跳过已完成题。
- 支持 `--resume-log path`。
- 每题记录：`idx/status/predicted/expected/result/error/duration/num_tool_calls/synthesized_tools_called`。

验收：

- 手动中断后重启，能从下一题继续。
- 第 50 题卡住不会阻塞后续题。

### 0.2 单题超时和资源清理

目标：避免 synthesized tool 或 LM 响应拖垮整轮 eval。

改动：

- 给单题总耗时加硬限制，例如 900 秒。
- 超时后调用 sandbox shutdown、GC，并记录 `TIMEOUT`。
- 将 `litellm.InternalServerError`、`APITimeoutError`、JSON parse failure 统一记录为可恢复异常。

验收：

- 跑 1-150 时不会因单题异常停整个进程。
- summary 中能区分 `WRONG`、`EXCEPTION`、`TIMEOUT`、`INCOMPLETE`。

---

## Tier 1：先吃掉低成本准确率

### 1.1 强化答案规范化和等价验证

目标：修掉数学等价但判错。

优先处理：

- `log(2/3)` vs `-log(3/2)`。
- `a/b + ci/d` vs `(a+ci)/b`。
- `sqrt[n]{x}` vs `x**(1/n)`。
- 小数答案转分数或根式。
- `\boxed{}`、`\left...\right`、中文/英文解释中的答案抽取。

建议实现：

- 在 `metric_fnv4` 或 verifier 前加 `normalize_answer(expr)`。
- 对 parse 成功的表达式使用 `sympy.simplify(pred - gold) == 0`。
- 对复数表达式单独替换 `i -> I`。
- 对概率/根式/组合数保留符号表达式，不急着 float 化。

预期收益：

- 当前 interpreter 至少第 11、92 题可恢复。
- persistent 至少第 11、38 题可恢复。

### 1.2 extractor 二次抽取

目标：正确答案已经在 trajectory 中时，不要抽错。

改动：

- 如果最终 `answer` verifier 失败，从最近 3 个 observation/thought 中重新抽取候选。
- 对候选逐个 verify，取第一个等价 gold 的候选。
- 评测模式可以用 gold 做诊断；正式模式用 verifier/consistency score 选择。

---

## Tier 2：让 synthesized-tools 只在有收益时启用

### 2.1 增加 tool-use gating

当前 persistent 每题都先合成工具，成本高，而且可能把简单题带偏。建议加一个轻量决策器：

- `DIRECT_PYTHON`：组合、枚举、数值、DP、概率积分等，直接走 interpreter。
- `SYNTHESIZE_TOOL`：题目需要复用复杂结构、几何计算、专门 checker 时再合成。
- `NO_TOOL_REASONING`：纯代数恒等式、通项推导，少写无意义代码。

验收：

- 前 49 题中，第 35、38、43 这类公式/泛化题不再被 synthesized tool 输出 `0`。
- 平均每题 LM 调用次数下降。

### 2.2 synthesized tool 失败时 fallback 到 interpreter

流程建议：

1. 先运行 synthesized tool。
2. 如果返回 `0`、`INSUFFICIENT_DATA`、异常、空值、明显与问题类型不符，触发 fallback。
3. fallback 使用 `interpreter.py` 的 `python_eval` ReAct 路径。
4. 两个候选答案都存在时，用 verifier 或独立 checker 选答案。

这项应该是 persistent 相对 interpreter 的核心路线：不是替代 interpreter，而是在 interpreter 基线之上加工具合成。

---

## Tier 3：针对难题的质量提升

### 3.1 Self-consistency 投票

对高风险题跑 N 次，N 建议从 3 或 5 开始：

- temperature > 0。
- 每次独立得到答案。
- 用 `math_verify`/SymPy 将等价答案聚类。
- 取最大簇；若无多数，进入 verification pass。

只对这些题启用：

- 第一次答案 verifier 信心低。
- synthesized 和 interpreter 答案冲突。
- 输出为估算题、几何题、复杂组合题。

### 3.2 Verification pass

生成答案后再让模型或代码检查：

- 代回原条件。
- 小规模枚举验证通项。
- 数值采样验证闭式。
- 几何题用坐标/符号双算。

当前错题里第 10、12、45、79、83、86、105、118、140、168、209 都适合用 verification pass 抓错。

---

## Tier 4：GEPA 反馈继续做，但放在稳定性之后

原计划里的 pred_name 分组反馈仍然有价值，不过优先级应排在稳定 eval 和 fallback 之后。

建议保留：

- `_fb_tool_creator`：工具是否被调用、是否算错、是否本题真的需要工具。
- `_fb_tool_repair`：repair 前后错误和成功率。
- `_fb_reasoner`：是否过早 finish、是否忽略 observation。
- `_fb_extractor`：trajectory 中已有正确值却抽错的情况。

新增反馈重点：

- 如果 interpreter fallback 正确而 synthesized 错，反馈给 tool_creator/reasoner：不要过度合成或过度相信工具。
- 如果 synthesized 正确而 interpreter 错，反馈保留该题型的工具策略。

---

## 推荐执行顺序

1. 实现 Tier 0：断点续跑、单题超时、summary。
2. 实现 Tier 1.1：答案 normalizer/verifier，先重算已有日志能恢复多少。
3. 重跑 persistent 1-150，拿到完整可比结果。
4. 实现 Tier 2：tool-use gating + interpreter fallback。
5. 再跑 1-150，对比 persistent、interpreter、hybrid 三条曲线。
6. 对剩余错题加 Tier 3：self-consistency + verification pass。
7. 最后再跑 GEPA，并使用 pred_name 分组反馈优化四个 predictor。

---

## 成功标准

短期：

- persistent/hybrid 能完整跑完 1-150。
- 断点续跑可用。
- 前 49 题至少超过 interpreter 的 43/49。

中期：

- 1-150 答题准确率超过 interpreter baseline。
- 异常率低于 2%。
- 平均单题耗时不超过当前 persistent 的 70%。

长期：

- full test 500 题完成率接近 100%。
- hybrid 相比 pure interpreter 有稳定净提升，而不是只在少数题上波动。
