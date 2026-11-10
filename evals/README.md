# 检索评测（violet 50）

目的：用固定问题集证明检索策略有效，而不是只写“我用了混合检索”。

评测集只有一套：`evals/golden_violet.jsonl`，共 50 条问题，对应 `violet-kb`。
题型分布：31 条单跳、7 条多跳、4 条数值、4 条不可回答、2 条版本冲突、2 条别名。
每条同时带有 `expected_files`、`required_facts` 和 `reference_answer`，检索评测先用 `expected_files`，后续答案评测可直接用参考答案。

## 1. 评测语料

语料来自外部目录：

```text
rag data/violet-evergarden-kb/docx
```

共 12 份 DOCX 参与评测；`README.docx` 是语料说明，不上传，避免成为检索噪声。
`evals/golden_violet.jsonl` 由 `evals/build_violet_golden.py` 从该语料自带的 `qa-benchmark.zh.jsonl` 转换而来。

## 2. 上传到知识库

先启动后端，然后：

```bash
python evals/seed_violet_kb.py --base-url http://127.0.0.1:8010
```

默认知识库 ID 是 `violet-kb`，上传后会等待 12 份文件全部入库。
如果已经有可用的 `violet-kb`，可以跳过这一步。

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
- **只测检索层**：不测最终回答是否忠实；答案正确性、忠实度和引用支持率需要 LLM judge 或人工抽检。
- **4 条不可回答样本**目前只统计“是否零召回”，不能替代最终拒答评测。
- **violet 语料在仓库外**：没有 `rag data/violet-evergarden-kb` 时无法重新入库；仓库内保留的是转换后的 `golden_violet.jsonl` 和实测结果。
- **分块消融另开实验**：修改 chunk size / overlap 需要重建索引，不能直接复用当前 `violet-kb` 的向量。
