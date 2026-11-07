# 检索评测

目的：用固定问题集证明检索策略有效，而不是只写“我用了混合检索”。

评测集：10 份 PDF（5 份现行制度 + 5 份过时/区域/承包商版本干扰文档）× 30 条问题，包含单跳事实和版本区分问题。

## 1. 生成语料

```bash
python evals/build_corpus.py
```

会在 `evals/corpus/` 生成 10 份 PDF（5 份现行制度 + 5 份过时/区域/承包商版本干扰文档）。语料由脚本生成，保证任何人都能复现。

## 2. 上传到知识库

先启动后端，然后：

```bash
python evals/seed_eval_kb.py --base-url http://127.0.0.1:8010
```

默认知识库 ID 是 `eval-kb`，上传后会等待 10 份文件全部入库。

## 3. 跑评测

```bash
# 只校验格式
python evals/run_eval.py --check-only

# 稠密 / 混合 / 混合+重排
python evals/run_eval.py --kb-id eval-kb --mode dense
python evals/run_eval.py --kb-id eval-kb --mode hybrid
python evals/run_eval.py --kb-id eval-kb --mode hybrid+rerank
```

输出 Recall@5、MRR、P50/P95 延迟，以及未命中的样本。

## 4. 实测结果（30 条问题，k=5）

| 检索策略 | Recall@5 | MRR | P95 延迟 |
|---|---|---|---|
| 仅稠密向量 | 1.000 | 0.867 | 1212 ms |
| 稠密 + BM25 + RRF | 1.000 | 0.867 | 1292 ms |
| 再加 qwen3-rerank | 1.000 | 0.933 | 2525 ms |

剩余错误集中在「现行 vs 过时版本」的区分题：两类文档都包含查询里的数字，rerank 无法稳定判断哪份是现行版本。下一步应给文档加 `status=current/archived` 元数据，在检索前过滤，而不是继续调 rerank。

## 5. Agentic-lite 评测（33 条）

`golden_agentic.jsonl` 覆盖 10 条多跳、8 条关系、6 条时间/版本、4 条不可回答、5 条单跳回归：

```bash
python evals/run_eval.py --golden evals/golden_agentic.jsonl --mode hybrid+rerank
python evals/run_eval.py --golden evals/golden_agentic.jsonl --retrieval-mode fast
python evals/run_eval.py --golden evals/golden_agentic.jsonl --retrieval-mode deep
python evals/run_eval.py --golden evals/golden_agentic.jsonl --retrieval-mode auto
```

本地实测（2026-11，k=5）：

| 模式 | Recall@5 | MRR | P95 |
|---|---:|---:|---:|
| baseline | 1.000 | 0.948 | 3134 ms |
| fast | 1.000 | 0.948 | 2235 ms |
| deep | 1.000 | 0.948 | 11110 ms |
| auto | 1.000 | 0.948 | 7564 ms |

结论：当前语料上 Recall 已饱和，deep / auto 没有质量增益，只增加延迟。该结果不满足“多跳 Recall +10pp”的门槛，需要更难的多跳集或更强的 critic。

`golden_violet.jsonl`（50 条真实语料，需外部 `rag data/violet-evergarden-kb`）是补充对照，不在默认复现路径；它的标签是文档级，命中文件任意父块即判对，MRR/Recall 偏乐观。

## 6. 已知边界

- 只测检索层，不测最终回答是否忠实；答案质量建议加 LLM judge 或人工抽检 20 条。
- 评测语料是合成/小规模数据，真实业务文档需要重新标 golden set。
- Agentic-lite 当前没有证明质量增益；默认已改为 `fast`，需要多轮时显式选择 `auto` / `deep`。
- 修改分块参数需要重建索引，分块消融请另开 collection 或用独立脚本。
