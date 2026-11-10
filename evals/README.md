# 检索评测（violet 50）

目的：用固定问题集证明检索策略有效，而不是只写“我用了混合检索”。

评测集只有一套：`evals/golden_violet.jsonl`，共 50 条问题，对应 `violet-kb`。
题型分布：31 条单跳、7 条多跳、4 条数值、4 条不可回答、2 条版本冲突、2 条别名。
每条同时带有 `expected_files`、`required_facts` 和 `reference_answer`，检索评测先用 `expected_files`，后续答案评测可直接用参考答案。

## 1. 评测语料

语料默认放在项目本地目录：

```text
data/violet-evergarden-kb/
```

该目录被 `.gitignore` 忽略，不提交到公开仓库。共 12 份 DOCX 参与评测；`README.docx` 是语料说明，不上传，避免成为检索噪声。
`evals/golden_violet.jsonl` 由 `evals/build_violet_golden.py` 从 `data/violet-evergarden-kb/data/qa-benchmark.zh.jsonl` 转换而来。

## 2. 上传到知识库

先启动后端，然后：

```bash
python evals/seed_violet_kb.py --base-url http://127.0.0.1:8010
```

默认知识库 ID 是 `violet-kb`，上传后会等待 12 份文件全部入库。
如果已经有可用的 `violet-kb`，可以跳过这一步。

语料不在默认位置时：

```bash
python evals/seed_violet_kb.py --corpus-dir /path/to/violet-evergarden-kb
# 或
VIOLET_KB_DIR=/path/to/violet-evergarden-kb python evals/seed_violet_kb.py
```

## 3. 跑评测

```bash
# 只校验评测集格式
python evals/run_eval.py --check-only

# 默认走 violet-kb + golden_violet.jsonl
python evals/run_eval.py --retrieval-mode fast
python evals/run_eval.py --retrieval-mode deep
python evals/run_eval.py --retrieval-mode auto

# baseline：旧链路 hybrid + rerank
python evals/run_eval.py --retrieval-mode baseline --mode hybrid+rerank
```

输出 Recall@5、MRR、P50/P95 延迟、分题型指标和未命中样本。

## 4. 实测结果（50 条问题，k=5）

| 模式 | Recall@5 | MRR | P50 | P95 |
|---|---:|---:|---:|---:|
| baseline（hybrid+rerank） | 1.000 | 0.832 | 2208 ms | 2457 ms |
| fast | 1.000 | 0.849 | 2261 ms | 2620 ms |
| deep | 1.000 | 0.845 | 7783 ms | 12193 ms |
| auto | 1.000 | 0.849 | 2302 ms | 8371 ms |

结论：

- 这套语料上 Recall@5 已经饱和，差异主要体现在 MRR；
- `fast` 的 MRR 略高于 baseline，但 deep / auto 没有继续提升；
- deep 的 P95 约 12 秒，auto 约 8 秒，约为 fast 的 4–5 倍；
- 当前默认检索模式是 `fast`；需要多轮反思时再显式选择 `auto` / `deep`。

## 5. 已知边界

- **标签是文档级**：命中 `expected_files` 中任意一个文件的任意父块即判对，MRR/Recall 偏乐观。父块级标签需要后续补 `expected_parent_ids`。
- **检索评测只测检索层**：答案正确性、忠实度和引用支持率见第 6 节答案评测。
- **检索层的 4 条不可回答样本**只统计“是否零召回”；答案评测额外补了 6 条，共 10 条用于拒答指标。
- **violet 语料不提交**：默认在 `data/violet-evergarden-kb/`，公开仓库没有完整语料；课程提交包可以自带，或用 `--corpus-dir` / `VIOLET_KB_DIR` 指定。
- **分块消融另开实验**：修改 chunk size / overlap 需要重建索引，不能直接复用当前 `violet-kb` 的向量。

## 6. 答案评测

`evals/answer_eval.py` 调用真实 `/chat_with_agent/stream`，在检索评测之外补：

- Judge A：对照 `reference_answer` / `required_facts` 的 0/1/2 正确性评分；
- Judge B：claim 级忠实度 + `[n]` 引用支持；
- 确定性 `Hit@5`、`ExpectedContextCoverage`、`Citation Precision`、`Refusal Accuracy`。

```bash
# 3 条 pilot
python evals/answer_eval.py --limit 3

# 50 条 + 6 条补充拒答
python evals/answer_eval.py --run-id full

# 断点续跑 / 只重跑 judge / 只跑生成
python evals/answer_eval.py --run-id full --resume
python evals/answer_eval.py --run-id full --rejudge
python evals/answer_eval.py --no-judge --limit 10
```

默认 `violet-kb`、`fast`、`top_k=5`、`judge_model=qwen-plus`。结果写入 `evals/results/answer_violet_<run_id>.jsonl`、`_summary.json`、`_report.md` 和 `_human_review.csv`。方法与指标定义见 [`docs/answer-eval.md`](../docs/answer-eval.md)。

本次 56 条（46 可回答 + 10 不可回答）结果：

| 指标 | 结果 |
|---|---:|
| 完全正确率 | 93.5%（43/46） |
| 忠实度（strict） | 97.6% |
| 引用有效率 / 精确率 / 覆盖率 | 98.8% / 98.0% / 94.3% |
| Chunk 支持率 | 78.1% |
| 拒答准确率 | 100%（10/10） |
| 误拒率 | 4.3%（2/46） |
| Hit@5（重建上下文） | 95.7% |
| ExpectedContextCoverage | 86.2% |

完整报告、摘要和人工抽检表见 [`docs/answer-eval-report.md`](../docs/answer-eval-report.md)、[`docs/answer-eval-summary.json`](../docs/answer-eval-summary.json)、[`docs/answer-eval-human-review.csv`](../docs/answer-eval-human-review.csv)。

边界：

- 上下文按 trace 查询重建，会多一次检索调用；`Hit@5` 仅作答案评测诊断，官方检索指标以 `run_eval.py` 为准；
- 文档级标签仍未升级为父块级；
- judge 与生成模型默认同为 `qwen-plus`，需要人工抽检和 `--judge-model` 交叉验证。
