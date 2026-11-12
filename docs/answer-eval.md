# 答案评测方案（Ragas 风格）

本文说明 Mnemo 的 violet 答案评测如何组织数据、计算指标和解读结果。实现位于 `evals/answer_eval.py`、`evals/answer_metrics.py` 和 `evals/judges.py`。

## 1. 目标

课程 Topic 5 要求同时报告检索命中率和答案正确性。检索层已有 `Hit@5` / `MRR`；答案评测补齐：

- 答案是否与参考答案一致；
- 答案是否被检索上下文支持；
- 引用是否有效、是否支持对应 claim；
- 不可回答问题时是否明确拒答；
- 检索命中/未命中时答案正确率如何变化。

## 2. 数据模型

评测记录按 Ragas 风格组织，便于后续接入 Ragas 或做工具间对照：

| 字段 | 含义 |
|---|---|
| `sample_id` | 题目 ID |
| `user_input` | 用户问题 |
| `response` | 系统回答 |
| `retrieved_contexts` | 本次实际召回的父块全文 |
| `retrieved_context_ids` | 父块 ID |
| `reference` | 参考答案 |
| `reference_context_ids` | `expected_files` 或 `expected_parent_ids` |
| `cited_sources` | 回答中实际引用到的来源及全文 |
| `trace` | 检索 trace（模式、轮次、查询、耗时） |
| `judge_a` | 正确性 judge 输出 |
| `judge_b` | 忠实度/引用 judge 输出 |
| `retrieval_metrics` | 确定性检索指标 |
| `citation_metrics` | 确定性引用指标 |
| `errors` | 生成/检索/judge 的降级信息 |

完整逐样本结果写入 `evals/results/answer_violet_<run_id>.jsonl`。

## 3. 指标

### 3.1 答案正确性

Judge A 对照 `reference_answer` 和 `required_facts` 打 0/1/2 分：

- 2：关键事实全部正确，无矛盾；
- 1：主结论正确，漏次要事实或数字/单位不够精确；
- 0：关键事实错误、矛盾、答非所问，或该答时拒答。

报告：

```text
Fully Correct Rate = score=2 / 可回答且已判分样本数
Mean Correctness   = mean(score / 2)
```

### 3.2 事实覆盖率

```text
Fact Coverage = 命中的 required_facts / required_facts 总数
All-Facts Rate = coverage=1 的题数 / 总题数
```

规则匹配只做规范化子串匹配，同义改写交给 Judge A。

### 3.3 忠实度

Judge B 将答案拆成原子 claim，逐条判断 `supported` / `partial` / `unsupported`：

```text
Faithfulness = supported claims / total claims
Faithfulness Weighted = (supported + 0.5 * partial) / total
```

忠实度只统计可回答题，且排除“可回答问题被误拒”的样本；误拒由 `False Refusal Rate` 单独统计，避免拒答话术把忠实度刷高。

### 3.4 引用指标

```text
Citation Validity  = 有效 [n] / 全部 [n]
Citation Precision = 被引来源确实支持 claim 的 [n] / 全部 [n]
Citation Coverage  = 有至少一个有效引用的 claim / 需要引用的 claim
Chunk Support Rate = 被引父块全文包含 quote 的 [n] / 支持 claim 的 [n]
```

引用来源的全文按 `parent_id` 从 PostgreSQL 读取，不使用 240 字 snippet 判断。

### 3.5 检索指标

```text
Hit@5
MRR
ExpectedContextCoverage = 命中的 expected_files / expected_files 总数
AllEvidenceHit = coverage == 1 ? 1 : 0
ContextPrecision（文档级标签下按文件去重）
```

`ExpectedContextCoverage` 用于暴露多跳题“只命中一个文档也算 Hit@5=1”的高估。

另外报告：

```text
Correctness | hit  = hit@5=1 的样本中 score=2 的比例
Correctness | miss = hit@5=0 的样本中 score=2 的比例
```

这能区分“检索错了”和“检索对了但生成错了”。

### 3.6 拒答

```text
Refusal Accuracy   = score=2 的不可回答样本 / 不可回答样本数
False Refusal Rate = judge 标记 refused 的可回答样本 / 可回答样本数
```

## 4. 两个 Judge

### Judge A：Correctness

输入：

```text
question + query_type + reference_answer + required_facts + answer
```

不提供 sources，避免“有来源 = 正确”的偏差。参考答案之外的额外细节不进入 `wrong_facts`；它们是否被上下文支持由 Judge B 判断。后处理有一致性保护：`missing_facts` 和 `wrong_facts` 都为空、且不是误拒时，0 分会纠正为 2 分。输出：

```json
{
  "score": 0,
  "refused": false,
  "missing_facts": [],
  "wrong_facts": [],
  "reason": ""
}
```

### Judge B：Faithfulness + Citation

输入：

```text
question + answer + 引用来源 [n] + 全部检索上下文 C1...C5
```

不提供 `reference_answer`，避免“和参考答案一致 = 有据”的偏差。输出：

```json
{
  "claims": [
    {
      "claim": "",
      "verdict": "supported",
      "needs_citation": true,
      "cited_marks": [1],
      "evidence_source": "C1",
      "quote": ""
    }
  ],
  "citations": [
    {
      "mark": 1,
      "valid": true,
      "supports_claim": true,
      "claim_index": 0,
      "quote": ""
    }
  ]
}
```

## 5. 运行方式

先启动后端，并确保 `.env` 中有 `DASHSCOPE_API_KEY`、`BASE_URL`、`AGENT_BASE_MODEL`。

```bash
# 3 条 pilot
python evals/answer_eval.py --limit 3 --run-id pilot

# 完整 50 条 + 6 条补充拒答
python evals/answer_eval.py --run-id full

# 只跑生成和确定性指标，不调用 judge
python evals/answer_eval.py --no-judge --limit 10

# 断点续跑
python evals/answer_eval.py --run-id full --resume

# 只重跑 judge，不重新生成答案
python evals/answer_eval.py --run-id full --rejudge
```

默认参数：

```text
kb_id = violet-kb
retrieval_mode = fast
top_k = 5
judge_model = qwen-plus
```

`--top-k` 会作为 `top_k` 查询参数传给 `/chat_with_agent/stream`；后端缺省时回落到 `AGENT_TOP_K`。报告里打印的是实际请求值。

### 5.1 运行身份

每次 run 还会写 `_meta.json`，把数字绑定到产生它的语料和运行时：

```text
kb.fingerprint          # (file_hash, file_name, status) 的排序 SHA-256
kb.files                # 本次参与评测的文件清单
runtime.git             # 代码版本 + dirty 标记
runtime.chunk_config    # 父子块大小 / overlap / separators
runtime.models          # agent / summarization / embedding / rerank
runtime.generation      # temperature / default_top_k
eval.golden_sha256      # 评测集内容哈希
eval.retrieval_mode     # fast / auto / deep / baseline
eval.top_k              # 实际请求的召回数
eval.judge_model        # Judge A/B 使用的模型
eval.sample_count       # 本次实际评测的样本数
```

`--resume` 会比较这份身份；KB、代码、模型或 chunk 配置变化时拒绝续跑。要做可复现的答案评测，建议把 `AGENT_TEMPERATURE` 设为 `0`，并在报告里同时记录 `top_k` 和样本量。

`--keep-conversations` 可保留评测会话；默认在结束后删除，避免污染数据库。

## 6. 输出

```text
evals/results/answer_violet_<run_id>.jsonl          # 逐样本完整记录
evals/results/answer_violet_<run_id>_summary.json   # 聚合指标
evals/results/answer_violet_<run_id>_report.md      # 人读报告
evals/results/answer_violet_<run_id>_human_review.csv
evals/results/answer_violet_<run_id>_meta.json      # 运行身份（KB 指纹 + runtime + eval 配置）
```

`human_review.csv` 默认抽 12 条，按低分和题型分层，供人工复核 Judge A/B。

## 7. 当前结果

本次运行（2026-11，`violet-kb` / `fast` / `top_k=5`）共 56 条：46 可回答 + 10 不可回答。

| 指标 | 结果 |
|---|---:|
| Fully Correct Rate | 93.5%（43/46） |
| Faithfulness | 97.6% |
| Citation Validity / Precision / Coverage | 98.8% / 98.0% / 94.3% |
| Chunk Support Rate | 78.1% |
| Refusal Accuracy | 100%（10/10） |
| False Refusal Rate | 4.3%（2/46） |
| Hit@5（重建上下文） | 95.7% |
| ExpectedContextCoverage | 86.2% |

完整报告见 [`answer-eval-report.md`](answer-eval-report.md)，聚合指标见 [`answer-eval-summary.json`](answer-eval-summary.json)，人工抽检表见 [`answer-eval-human-review.csv`](answer-eval-human-review.csv)。

## 8. 已知边界

- 参考答案来自外部 violet benchmark，正式结论前应人工抽检；
- 检索上下文按 trace 查询重建，会多一次检索调用；`Hit@5` / `ExpectedContextCoverage` 与主回答调用的实际 top5 可能不完全一致，官方检索指标仍以 `run_eval.py` 为准；
- `hit=0` 不代表一定检索失败：答案可能由其他文档或参数知识支持，需结合 cited sources 和 expected_files 人工复核；
- violet 标签是文档级，`ExpectedContextCoverage` 仍是文档级；
- Judge 与生成模型默认同为 `qwen-plus`，存在自偏好风险，建议人工抽检并用 `--judge-model` 做交叉验证；
- 56 条样本的 95% 置信区间约 ±10pp，分题型样本更小，报告应同时给出数量。
