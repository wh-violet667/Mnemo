# Mnemo 实验总览

> 本文把项目的实验压成三组核心问题：检索模式、答案质量、分块策略。
>
> 原始报告仍保留在 `docs/`，本文只负责统一口径和结论。

## 0. 先区分两套评测口径

| 口径 | 判定方式 | 典型数字 |
|---|---|---|
| 旧文档级标签 | 命中 `expected_files` 中任意父块 | Recall@5 可以到 1.000 |
| 新证据级标签 | `gold_evidence.quote` 与 chunk 文本的字符 bigram 覆盖率 ≥ 0.55 | Recall@5 约 0.85–0.98 |

因此：

- 旧 Agentic 表的 `Recall@5=1.000` **不能**直接和新分块实验的 `0.848/0.957` 比较。
- 新分块实验统一使用 `gold_evidence`，结果之间才可比。
- 答案评测是另一套：它看生成答案、引用和拒答，不只看检索。

---

## 1. 检索模式 A/B：baseline / fast / auto / deep

**问题**：多轮检索是否真的比单轮更好？

**设置**：50 题，`violet-kb`，12 份 DOCX，`top_k=5`；`baseline` 为 hybrid + rerank，`fast` 一次检索，`auto` 规则判断 + 零召回升级，`deep` 最多 2 轮 Plan/Retrieve/Critique/Refine。

| 模式 | Recall@5 | MRR | P50 | P95 |
|---|---:|---:|---:|---:|
| baseline | 1.000 | 0.832 | 2208 ms | 2457 ms |
| fast | 1.000 | **0.849** | 2261 ms | 2620 ms |
| deep | 1.000 | 0.845 | 7783 ms | 12193 ms |
| auto | 1.000 | **0.849** | 2302 ms | 8371 ms |

**结论**：

- 旧标签下 Recall 已经饱和，差异只在 MRR。
- `deep`/`auto` 没有继续提升 MRR，P95 却高 4–5 倍。
- 默认改成 `fast`；`deep` 保留的价值是可控多轮轨迹和可观测性，不是已经被数据证明的质量提升。
- 这组是文档级旧标签，**没有**配对 bootstrap CI，属于方向性结论。

原始设计：[`phase1-agentic-lite.md`](phase1-agentic-lite.md)。

---

## 2. 答案评测：56 题、Judge A/B

**问题**：答案是否正确、是否有证据支持、引用是否有效、该拒答时是否拒答。

**设置**：56 题（46 可回答 + 10 不可回答），`violet-kb` / `fast` / `top_k=5`；生成模型 `qwen-plus`（历史），Judge A 打 0/1/2 正确性，Judge B 判断 claim 级忠实度和引用支持。

| 指标 | 结果 |
|---|---:|
| 完全正确率 | 93.5%（43/46） |
| 平均正确性 | 0.935 |
| 事实覆盖率 | 75.7% |
| 全事实命中率 | 56.5% |
| 忠实度（strict） | 97.6% |
| 引用有效率 / 精确率 / 覆盖率 | 98.8% / 98.0% / 94.3% |
| Chunk 支持率 | 78.1% |
| 拒答准确率 | 100%（10/10） |
| 误拒率 | 4.3%（2/46） |
| Hit@5（重建上下文） | 95.7% |
| ExpectedContextCoverage | 86.2% |
| P50 / P95 | 8.9s / 16.5s |

**失败样本**：`v-q007`、`v-q008` 误拒；`v-q033` 把“外传在剧场版之前”答成“无直接先后顺序”。

**结论**：

- 答案正确率、忠实度、引用和拒答都不错，但 `Chunk 支持率 78.1%` 说明“引用来源”与“真正支持 claim 的 chunk”之间仍有差距。
- 这是 **qwen-plus 历史基线**；当前生成模型已换成 `deepseek-v4.1-flash`，这组数字不能冒充当前结果。
- 生成和 Judge 同源，可能有自偏好；正式结论仍需人工抽检。
- 运行中 `memory_usage=0.0%`，所以这**不是**记忆系统实验，只是普通 RAG 答案评测。

原始报告：[`answer-eval-report.md`](answer-eval-report.md)。

---

## 3. 分块实验

分块实验分成两个不同问题：

1. **DOCX 尺寸**：child size/overlap 怎么选？
2. **Markdown 策略**：结构感知是否比 recursive 更好？

### 3.1 DOCX 尺寸：200/0 vs 200/50 vs 400/0

**设置**：12 份 DOCX，50 题（46 条可回答参与指标），child 级，`top_k=5`，`hybrid+rerank`，每个配置独立 KB，配对 bootstrap 95% CI。

| 策略 | Recall@5 | MRR | Precision@5 | nDCG@5 | P50 / P95 | 子块数 |
|---|---:|---:|---:|---:|---:|---:|
| 200/0 | 0.848 | 0.670 | 0.300 | 0.700 | 2870 / 5777 ms | 321 |
| 200/50 | 0.913 | 0.715 | 0.330 | 0.749 | 3146 / 5577 ms | 352 |
| 400/0 | **0.957** | **0.765** | **0.357** | **0.811** | **2683 / 3625 ms** | **164** |

相对 200/0：

- `400/0`：ΔRecall **+0.109** [+0.022, +0.217]、ΔPrecision **+0.057** [+0.017, +0.100]、ΔnDCG **+0.111** [+0.028, +0.196]；MRR 的 CI 跨 0。
- `200/50`：ΔRecall +0.065 [-0.022, +0.153]、ΔnDCG +0.049 [-0.027, +0.133]，不显著。

**结论**：本次语料上 `400/0` 在 Recall、Precision、nDCG 和延迟上都最好；但这是 50 题、child 检索层的结果，不能直接说明答案质量最好。生产默认仍是 `fine-overlap`（200/50），`coarse-no-overlap` profile 对齐到实际验证过的 `400/0`。

原始报告：[`chunking-comparison.md`](chunking-comparison.md)。

### 3.2 Markdown 策略：v1 按标题硬切 → v2 短章节合并

**设置**：同一份 12 篇 Markdown，50 题，child 级，`top_k=5`，`hybrid+rerank`；三组为 `recursive-overlap`（1000/200 + 400/0）、`recursive-no-parent-overlap`（1000/0 + 400/0）、`markdown-heading`（1000/0 + 400/0）。

v1 按每个 `##` 硬切，产生大量小 chunk：

| 策略 | Recall@5 | MRR | Precision@5 | nDCG@5 | 子块数 |
|---|---:|---:|---:|---:|---:|
| recursive-overlap | 0.935 | 0.804 | **0.357** | 0.831 | 190 |
| recursive-no-parent-overlap | 0.935 | 0.759 | 0.326 | 0.792 | 168 |
| markdown-heading v1 | 0.935 | **0.828** | 0.317 | **0.840** | 243 |

结构完整性：表头保留率 `0.794 → 1.000`，但 `markdown-heading` 的 Precision 下降 **-0.039 [-0.074, -0.009]**，CI 不跨 0；中位 chunk 长度只有 192。

v2 合并同一一级标题下的相邻短章节：

| 策略 | Recall@5 | MRR | Precision@5 | nDCG@5 | 子块数 |
|---|---:|---:|---:|---:|---:|
| recursive-overlap | 0.935 | 0.815 | **0.357** | 0.838 | 190 |
| recursive-no-parent-overlap | 0.935 | 0.770 | 0.326 | 0.798 | 168 |
| markdown-heading v2 | 0.935 | **0.835** | 0.339 | **0.849** | 209 |

相对 `recursive-overlap`：

- `markdown-heading v2`：ΔRecall 0、ΔMRR +0.020 [-0.056, +0.098]、ΔnDCG +0.011 [-0.056, +0.077]、ΔPrecision **-0.017 [-0.052, +0.017]**。

Precision 的下降不再显著；v1 → v2 的子块数 243→209、中位长度 192→297。

**Parent 级（生产回表路径）**：

| 策略 | Recall@5 | MRR | Precision@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| recursive-overlap | 0.935 | 0.855 | **0.383** | 0.862 |
| recursive-no-parent-overlap | 0.957 | 0.797 | 0.361 | 0.827 |
| markdown-heading v2 | **0.978** | **0.874** | 0.374 | **0.880** |

相对 parent 基线：

- `markdown-heading v2`：ΔRecall +0.043 [0.000, +0.109]、ΔMRR +0.019 [-0.043, +0.084]、ΔnDCG +0.018 [-0.030, +0.073]、ΔPrecision -0.009 [-0.039, +0.022]。
- 方向正向，但 CI 触 0 / 跨 0，**不能宣称显著**。

**答案级确定性指标（no-judge）**：只放在 v2 报告附录 A。方向同 parent 级：ExpectedContextCoverage 0.895→**0.917**、AllEvidenceHit 0.783→**0.826**、FactCoverage 0.795→**0.806**，ContextPrecision 0.497→0.485；完整 LLM Judge 未跑，不能当作完整答案质量结论。

**结论**：

- 结构切分确实修好了表格：表头保留率 `0.79 → 1.00`。
- v1 的主要问题是短章节产生小 chunk，导致 Precision 显著下降；v2 合并短章节后，Precision 降幅收窄到不显著。
- Child 级 Recall 仍是 0.935；multi-hop 0.857→1.000，single-hop 0.935→0.903，净效果持平。
- Parent 级和答案级都略偏正向，但样本小、CI 触 0 或跨 0。
- 因此默认仍是 `fine-overlap`；`markdown-heading` 作为 Markdown 知识库的可选策略。

原始报告：[`chunking-markdown-strategy-v2.md`](chunking-markdown-strategy-v2.md)；v1 归档：[`archive/chunking-markdown-strategy.md`](archive/chunking-markdown-strategy.md)。

---

## 4. 三组实验共同说明了什么

1. **检索链路是有效的**：证据级 Recall 在 0.85–0.98；答案历史正确率 93.5%。
2. **分块主要影响 Recall/Precision，Agentic 多轮主要影响延迟**：在这套 50 题语料上，`400/0` 的分块收益比 `deep/auto` 的多轮收益更明显。
3. **结构感知不是天然更优**：它先修好了表格完整性，但硬切会产生小 chunk；短章节合并后才把 Precision 拉回不显著。
4. **默认策略保持保守**：`fast` + `fine-overlap` 没有被一次 50 题实验替换，避免过拟合。
5. **实验结果有明确边界**：50/56 题、单语料、单次运行；分块实验有配对 bootstrap，其他多是点估计。

---

## 5. 边界与未做

- **完整 LLM Judge 答案评测**：当前 `deepseek-v4.1-flash` + `markdown-heading` 的正确率/忠实度/拒答还没有跑。
- **记忆系统 A/B**：历史答案评测 memory usage 0%，没有验证记忆对回答的增益。
- **更大/多格式语料**：目前只有 violet 一套；DOCX 和 Markdown 是同一内容的不同格式。
- **Token 级分块**：当前仍是字符预算，不是 token 预算。
- **MQE / HyDE / 图记忆**：教程里提到的这些方向还没有做受控实验。
- **DOCX 的 parent/答案级对比**：DOCX 实验只有 child 级。

---

## 6. 复现入口

| 内容 | 文件 |
|---|---|
| 检索评测 | `evals/run_eval.py`、`evals/golden_violet.jsonl` |
| 答案评测 | `evals/answer_eval.py`、`evals/answer_metrics.py`、`evals/judges.py` |
| 分块实验 | `evals/chunk_experiment.py`、`evals/seed_violet_kb.py` |
| 运行身份 | 每次评测写 `*_meta.json`（KB 指纹、代码版本、模型、chunk 配置） |
