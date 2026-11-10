# Violet 答案评测报告

- 生成时间：2026-11-11T03:32:44
- 知识库：`violet-kb`
- 生成模型：`qwen-plus`
- Judge：`qwen-plus`
- 检索模式：`fast` / top_k=5
- 样本：56 条（可回答 46，不可回答 10）
- 上下文：按 trace 查询重建的 top-k 父块全文

## 总体指标

| 指标 | 值 |
|---|---:|
| 完全正确率 | 93.5% |
| 平均正确性 | 0.935 |
| 事实覆盖率 | 75.7% |
| 全事实命中率 | 56.5% |
| 忠实度（strict） | 97.6% |
| 忠实度（partial=0.5） | 98.2% |
| 引用有效率 | 98.8% |
| 引用精确率 | 98.0% |
| 引用覆盖率 | 94.3% |
| Chunk 支持率 | 78.1% |
| 拒答准确率 | 100.0% |
| 误拒率 | 4.3% |
| Hit@5 | 95.7% |
| Correctness \| hit（n=44） | 93.2% |
| Correctness \| miss（n=2） | 100.0% |
| ExpectedContextCoverage | 86.2% |
| AllEvidenceHit | 76.1% |
| ContextPrecision | 46.8% |
| P50 / P95 | 8913 ms / 16532 ms |
| 首 token P50 / P95 | 5592 ms / 6726 ms |
| 记忆使用率 | 0.0% |

## 分题型

| 题型 | n | 完全正确 | 忠实度 | 引用精确率 | Hit@5 | ExpectedCoverage |
|---|---:|---:|---:|---:|---:|---:|
| alias | 2 | 100.0% | 100.0% | 100.0% | 50.0% | 50.0% |
| multi_hop | 7 | 85.7% | 97.5% | 100.0% | 85.7% | 73.8% |
| numeric | 4 | 100.0% | 95.0% | 94.7% | 100.0% | 75.0% |
| single_hop | 31 | 93.5% | 97.4% | 99.2% | 100.0% | 93.5% |
| unanswerable | 10 | - | - | 84.2% | - | - |
| version_conflict | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 75.0% |

## 低分样本

- `v-q007` [single_hop] score=0 hit=1.0 refused=True missing=['Sincerely', 'TRUE'] wrong=[]
- `v-q008` [single_hop] score=0 hit=1.0 refused=True missing=['みちしるべ', '路标', '茅原实里'] wrong=[]
- `v-q033` [multi_hop] score=0 hit=1.0 refused=False missing=['完结篇'] wrong=['二者无直接先后顺序', '它们是并行于TV之后的不同叙事分支', '2020剧场版是TV之后的主线完结篇']

## 失败与限制

- 参考答案来自外部 violet benchmark；正式结论前应人工抽检参考答案。
- 检索上下文为按 trace 查询重建的 top-k 父块；与主回答调用存在一次重复检索，Hit@5 仅作答案评测诊断，官方检索指标以 `run_eval.py` 为准。
- violet 标签是文档级；`ExpectedContextCoverage` 仍是文档级，chunk 级需补 `expected_parent_ids`。
- `hit=0` 不代表一定检索失败：答案可能由其他文档或参数知识支持，需结合 cited sources 和 expected_files 人工复核。
- 本次运行没有记录到生成/检索/judge 错误。
