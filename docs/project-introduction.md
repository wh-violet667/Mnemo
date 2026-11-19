# Mnemo 项目介绍与答辩问答

> 项目：Mnemo —— 基于知识库与长期记忆的个性化问答助手
> 技术栈：FastAPI + LangGraph + Milvus + PostgreSQL + Redis + Streamlit
> 文档定位：课程项目介绍、答辩讲稿底稿、老师追问准备

---

## 0. 一页速览

### 一句话介绍

Mnemo 是一个面向个人/小团队知识库的 RAG 问答系统：文档经过父子块切分和向量化后进入 Milvus，用户提问时通过 hybrid 检索、rerank 和 LangGraph Agent 生成带证据引用的回答；同时把历史对话压缩成可版本化、可追溯来源的长期记忆，保存在 PostgreSQL，并投影到 Milvus 供语义召回。

### 三个主要贡献

1. **可核验的 RAG 链路**：父子块、dense + BM25、RRF、rerank、fast/auto/deep 检索、`[n]` 引用、claim 对齐 quote 和段落 locator，不是只把文档丢给大模型。
2. **以 PostgreSQL 为事实源的长期记忆**：摘要、语义/情节/程序记忆、版本历史、来源链、supersede 和索引状态都有明确数据模型；Milvus 只是可重建投影。
3. **可复现的评测工程**：50 题 golden set、`gold_evidence` 证据标签、Recall/MRR/Precision/nDCG、答案 Judge A/B、运行身份 fingerprint，以及独立 KB 的分块策略对比和配对 bootstrap。

### 三个最需要诚实说明的边界

1. **当前是单用户演示**：没有登录，`user_id` 由客户端传入，不能直接公开部署。
2. **标签和评测仍有噪声**：`gold_evidence` 主要是自动生成，需要人工抽检；答案评测的 93.5% 是切换 DeepSeek 前的 `qwen-plus` 历史基线。
3. **跨存储仍是最终一致**：文档入库和删除已经做了顺序/补偿，但还没有完整的 outbox worker；Milvus 投影失败仍可能需要 reindex。

### 三个下一步

1. 加认证与租户隔离，默认只监听本机。
2. 结构感知已完成“短章节合并”，并补了 56 条答案级确定性指标；下一步跑完整 LLM Judge 和更大语料，再决定是否调整默认 child 或 strategy。
3. 补 10–15 条人工证据复核、CI、Alembic 迁移和 outbox。

---

## 1. 选题背景与目标

### 1.1 要解决的问题

普通聊天模型有三个明显问题：

- **不知道私有资料**：课程笔记、产品手册、政策文件不在训练数据里。
- **容易编造**：模型可能给出看似合理但没有依据的答案。
- **没有持续记忆**：每次对话都从零开始，无法记住用户偏好、历史决定和已确认事实。

Mnemo 的目标不是训练一个新的大模型，而是围绕一个已有模型构建三层能力：

1. **知识层**：把文档变成可检索、可引用的证据。
2. **记忆层**：把对话变成可管理、可追溯的长期事实。
3. **评测层**：用固定问题集和证据标签证明检索/回答到底有没有变好。

### 1.2 为什么是 RAG，而不是微调

- 课程/产品资料会更新，RAG 可以重新入库，不需要重新训练。
- 微调更适合改变“表达风格/任务格式”，不适合注入大量事实。
- RAG 能返回来源，满足“可核验引用”的课程要求。
- 本项目的数据规模只有 12 份 DOCX、50 道题，微调既容易过拟合，也无法证明引用能力。

### 1.3 项目的边界

它是一个**课程级、单用户、可复现的工程原型**，不是生产级 SaaS：

- 单用户，无登录。
- 语料规模小，评测集 50 题。
- 依赖 OpenCode Go、DashScope、Zilliz 等外部服务。
- 重点在架构、证据链和评测，不在模型训练或算法创新。

---

## 2. 总体架构

### 2.1 分层结构

```text
客户端
  Streamlit :8501
      │ HTTP / JSONL 流
      ▼
应用层
  FastAPI :8010
      │
  LangGraph Agent
      ├── search_knowledge_base
      ├── get_memory
      └── get_raw_conversation_by_summary_id
      │
数据层
  PostgreSQL：事实源（用户、知识库、文件、父块、会话、记忆、版本、来源）
  Milvus：检索投影（知识库子块、记忆）
  Redis：LangGraph checkpoint、用户画像缓存
      │
外部服务
  OpenCode Go / DeepSeek V4.1 Flash：对话生成
  DashScope：text-embedding-v4、qwen3.7-text-rerank、Judge
```

### 2.2 数据层分工

| 存储 | 存什么 | 为什么 |
|---|---|---|
| PostgreSQL | 用户、知识库、文件、父块、原始对话、摘要、记忆、版本、来源链 | 事务、约束、可审计、可重建 |
| Milvus | 知识库子块向量 + BM25 稀疏向量、记忆检索投影 | 高性能 dense/sparse 检索 |
| Redis | LangGraph checkpoint、用户画像缓存 | 会话上下文、低延迟读取 |
| 本地文件系统 | 上传临时文件 | 解析前暂存，成功/失败后清理 |

核心原则：**PostgreSQL 是事实源，Milvus 是可重建的检索投影，Redis 是可过期的会话状态。**

---

## 3. 核心组件详解

### 3.1 Streamlit 前端

主要文件：`frontend/app.py`、`frontend/streaming.py`

职责：

- 登录后的主界面：对话区、知识库面板、文件状态、记忆抽屉。
- 通过 `requests` 读取 `/chat_with_agent/stream` 的逐行 JSON 事件。
- 解析 `status`、`answer`、`trace`、`sources`、`error` 五类事件。
- 来源卡片展示 `[n]`、文件名、quote、locator，而不是只显示文件名。
- 对消息和来源内容做 `html.escape`，Markdown 使用 bleach 白名单，降低 XSS 风险。
- 支持创建知识库、上传文件、查看 `processing / ready / failed`、删除文件/知识库、重命名会话、删除记忆。

边界：

- `MNEMO_USER_ID` 默认 1，仅适合单用户演示。
- 前端是单文件为主，UI/CSS/状态耦合较重，后续应拆组件。
- 流式请求失败时只能展示错误，不会自动重试整轮问答。

### 3.2 FastAPI 接口层

主要文件：`backend/api.py`

主要端点：

| 端点 | 作用 |
|---|---|
| `POST /document_upload` | 上传文档，流式写临时文件、计算 SHA-256、后台处理 |
| `POST /knowledge-bases` | 创建知识库 |
| `GET /knowledge-bases` | 列出用户知识库 |
| `GET /knowledge-bases/{id}/files` | 查看文件及状态 |
| `DELETE /knowledge-bases/{id}` | 删除知识库 |
| `DELETE /knowledge-bases/{id}/documents/{hash}` | 删除文件 |
| `POST /conversations` | 创建会话 |
| `PATCH /conversations/{id}/knowledge-base` | 切换会话知识库 |
| `GET /chat_with_agent/stream` | 流式问答 |
| `GET /memories` | 查看长期记忆 |
| `DELETE /memories/{id}` | 软删除记忆 |
| `GET /runtime` | 返回评测所需的运行时身份 |

工程细节：

- 上传有 200MB 限制、10 并发 semaphore、流式大小校验。
- 支持按内容 SHA-256 去重；失败文件可以清理后重传。
- `identifiers.py` 对知识库 ID 和文件哈希做白名单校验，Milvus filter 二次转义，避免表达式注入。
- 文档入库的 PostgreSQL 写入是单事务；Milvus 在事务后写，失败会清理半成品并标记 `failed`。
- 删除知识库/文件改为先删 Milvus、再删 PostgreSQL，避免“PG 已删、向量还在”的不可重试状态。

### 3.3 LangGraph Agent

主要文件：`backend/agent.py`

Agent 的职责不是“直接回答问题”，而是决定：

- 要不要查知识库；
- 要不要查长期记忆；
- 要不要查原始对话；
- 如何把工具结果组织成最终回答。

三个工具：

1. `search_knowledge_base(query)`
   调用检索管线，返回带编号的证据片段。
2. `get_memory(query, semantic_k, episodic_k, procedural_k)`
   检索摘要/语义/情节/程序记忆。
3. `get_raw_conversation_by_summary_id(summary_id)`
   在摘要不够详细时回查原始对话。

中间件：

- `ToolCallLimitMiddleware` 限制 `get_memory` 和 `search_knowledge_base` 的调用次数。
- `SummarizationMiddleware` 处理长对话压缩。
- `dynamic_prompt` 根据知识库类型注入不同系统提示词，并注入用户画像。

会话状态：

- `thread_id = user_id:conversation_id`。
- Redis checkpoint 默认 TTL 60 分钟，读取时刷新。
- 不同会话互不共享上下文。

为什么用 LangGraph：

- 需要显式的工具调用循环，而不是一次 RAG chain。
- 需要 checkpoint、会话恢复、流式事件和工具状态。
- 需要限制工具调用次数，避免 Agent 无限检索。

### 3.4 检索管线

主要文件：`backend/retrieval.py`、`backend/routing.py`、`backend/knowledge_base_manager.py`

#### fast / auto / deep

| 模式 | 行为 | 适用 |
|---|---|---|
| `fast` | 一次 hybrid 召回 + 回表 + rerank | 默认，延迟低 |
| `auto` | 规则判断是否需要 deep；fast 零召回时升级一次 | 不确定问题复杂度 |
| `deep` | Planner → Retrieve → Critic → Refine，最多 2–3 轮 | 多跳、版本冲突、证据不足 |

实际结果：在 violet 50 题上，deep/auto 没有提升 Recall/MRR，P95 反而增加约 4–5 倍；因此默认改成 `fast`。Agentic-lite 的价值主要是**可控多轮轨迹和可观测性**，不是已经被数据证明的质量提升。

#### hybrid 检索

- dense：`text-embedding-v4` 生成查询向量，Milvus COSINE。
- sparse：Milvus 内置 BM25，中文 analyzer。
- 融合：`RRFRanker`，避免 dense/sparse 分数量纲不一致。
- 失败降级：hybrid 失败时退回 dense。

#### rerank

- 使用 `qwen3.7-text-rerank` 对候选父块重新排序。
- 失败时保持 RRF 顺序，不阻断回答。
- 代价是多一次远程调用，P95 会增加；这也是为什么 `fast` 默认仍保留 rerank，但不再默认 deep。

#### 检索粒度

- **child 200/50** 是检索单元，存 Milvus。
- **parent 1000/200** 是上下文单元，存 PostgreSQL。
- 先召回 child，得到 `parent_id`，再回表取父块，最后统一 rerank。

### 3.5 文档入库与分块

主要文件：`backend/documents_process.py`、`backend/chunking.py`、`backend/markdown_chunking.py`

流程：

1. 上传临时文件，计算 SHA-256。
2. 检查同用户、同知识库下是否已有文件。
3. 写 `file_metadata`，状态 `processing`。
4. 后台任务加载 PDF/DOCX/Markdown。
5. 按策略执行父块切分 → 子块切分。
6. 每个父块生成 UUID；每个子块记录 `parent_id`、`file_hash`、`file_name`。
7. PostgreSQL 单事务写入：文件状态、父块、块哈希。
8. Milvus 写入子块向量。
9. 成功写 `ready`；失败写 `failed`，并清理 PostgreSQL 半成品。
10. 重传失败文件时，先按 `file_hash` 清理旧 Milvus 向量和旧 PG 记录。

默认配置：

- parent：1000 字符，overlap 200
- child：200 字符，overlap 50
- separators：`\n\n`、`\n`、`。`、`！`、`？`、`，`、`、`、空格、空串

上传界面提供命名策略：

- `fine-overlap`：recursive，1000/200 + 200/50
- `coarse-no-overlap`：recursive，1000/200 + 400/0
- `markdown-heading`：结构感知，1000/0 + 400/0，只接受 `.md`

`markdown-heading` 先去掉 YAML front matter，再按 ATX 标题建立 `heading_path`；同一一级标题下的相邻短章节会合并到同一父/子块，标题行保留；表格与代码块尽量整体保留，必须切表时每个子块重复表头；每个父/子块记录 `heading_path`、`heading_paths`、`start`、`end`、`block_type`。它用标题边界替代 overlap，因此父/子 overlap 必须为 0。

第一次上传会把 profile/config 绑定到知识库；后续上传必须匹配，否则返回 409。旧知识库若已有文件但没有记录配置，会标记为 `legacy-unknown`，禁止继续追加，避免混合索引。

为什么父子块：

- 子块小，向量表达更集中，召回更准。
- 父块大，给模型足够上下文，避免“召回准但上下文碎”。
- 代价是存储和回表成本更高。

已知问题：

- 默认 `recursive` 策略仍可能切断表格，`q002`、`q004` 这类题会受影响；`markdown-heading` 已把表格完整率作为专门目标。
- 仍是字符预算，不是 token 预算；中英文混合时长度估计偏保守。
- `markdown-heading` 只覆盖 Markdown；PDF/DOCX 尚未统一转 Markdown。
- 失败文件如果进程崩溃卡在 `processing`，还没有 watchdog。

### 3.6 存储模型

#### PostgreSQL 主要表

| 表 | 作用 |
|---|---|
| `users` | 用户与画像 |
| `conversations` | 会话 |
| `knowledge_bases` | 知识库、绑定的 chunk profile/config |
| `file_metadata` | 文件哈希、文件名、状态、错误、实际 chunk profile/config |
| `parent_chunks` | 父块原文 |
| `chunk_hashes` | 子块哈希 |
| `raw_conversations` | 原始对话消息 |
| `summaries` | 一次记忆压缩批次的摘要 |
| `memories` | 当前记忆事实 |
| `memory_versions` | 记忆版本历史 |
| `memory_sources` | 记忆到原始消息的来源链 |

关键约束：

- `file_metadata` 主键 `(file_hash, knowledge_base_id, user_id)`。
- `parent_chunks`、`chunk_hashes` 外键指向 `file_metadata`，删除文件时级联。
- `memories.source_summary_id` 外键指向 `summaries`。
- `memory_versions` 外键指向 `memories`，只追加不覆盖。
- `memory_sources` 记录记忆由哪些消息派生。

#### Milvus 集合

知识库集合字段：

- `id`：自增主键
- `user_id`：partition key
- `knowledge_base_id`
- `file_hash`
- `parent_id`
- `text`
- `metadata`：JSON，含 `file_name`、`parent_index` 等
- `dense_vector`
- `sparse_bm25`

记忆集合额外有：

- `memory_type`：summary / semantic / episodic / procedural
- `status`：active / superseded / deleted
- `summary_id`
- `importance`
- `last_access_at`

#### Redis

- LangGraph checkpoint：保存对话状态，支持多轮和恢复。
- 用户画像缓存：减少每次 prompt 构造时的 PostgreSQL 查询。

### 3.7 长期记忆

主要文件：`backend/auto_store_memory_from_psql.py`、`backend/memory_manager.py`、`backend/reindex_memories.py`

#### 记忆类型

| 类型 | 内容 | 例子 |
|---|---|---|
| summary | 一次对话压缩后的摘要 | “用户在准备 RAG 课程项目” |
| semantic | 事实/概念 | “用户偏好 Python” |
| episodic | 历史事件 | “上次讨论过 Milvus 过滤” |
| procedural | 步骤/方法 | “用户习惯先写测试再改代码” |

#### 触发方式

- 后台任务每 180 秒运行一次。
- 只处理未摘要消息。
- 估算 token 超过 400 才调用 LLM 提取，避免每轮对话都产生记忆。
- 一次提取写一个 summary、若干 memory、memory_version、memory_source。

#### 冲突与版本

- 新记忆先和旧记忆做向量候选召回。
- 相似度 ≥ 0.9 时，旧记忆标记 `superseded`，新记忆 `active`，不删除旧内容。
- PostgreSQL 单事务写 summary、memory、version、source、消息 `summary_id` 和用户画像。
- 事务提交后再投影 Milvus；失败标记 `index_status=failed`，用 `reindex_memories.py` 修复。

#### 删除记忆

- PostgreSQL 先软删除，`memory_versions.valid_to` 关闭。
- Milvus 更新 `status=deleted`。
- 检索时再用 PostgreSQL 的 `status='active'` 过滤 Milvus 返回的 ID；即使 Milvus 更新失败，已删除记忆也不会被返回。

#### 已知边界

- 0.9 阈值是工程 baseline，不是语义判断；相似但不同的事实可能被误 supersede。
- 记忆评测基本为 0：当前 50 题没有覆盖“跨会话个性化/记忆冲突/记忆删除后检索”。
- 后台任务没有跨进程锁，多 worker 可能重复抽取。
- 被 LLM 过滤掉的消息不会写 `summary_id`，可能每个周期重复进入提取。

### 3.8 引用与可核验性

主要文件：`backend/citations.py`

流程：

1. 检索结果按 `[1]`、`[2]` 编号。
2. 模型回答里要求使用 `[n]` 标注来源。
3. 流式结束后，用回答中的 claim 和来源文本做字符 bigram 对齐，选出支持该 claim 的 quote。
4. 生成段落 locator（第几段）。
5. 前端来源卡片展示 `[n]`、文件名、quote、locator。
6. 只下发被引用到的来源，避免把 top-k 无关文档全塞给用户。

边界：

- quote 选择是字符重叠启发式，不是严格 NLI/蕴含判断。
- 引用精确率 98% 是历史 Judge 结果，不代表每个 quote 都人工核验。
- Chunk 支持率 78.1% 说明仍有约 1/5 的引用在 chunk 级支持不足。
- 表格/跨段证据容易选错 quote。

### 3.9 评测体系

主要文件：`evals/` 目录

#### 检索评测

- `golden_violet.jsonl`：50 题，46 可回答 + 4 不可回答。
- `gold_evidence`：每条证据带 `file / quote / start / end / fact`。
- `run_eval.py`：
  - 有 `gold_evidence` 时按 quote 覆盖率判定相关；
  - 支持 `--granularity parent|child`；
  - 输出 Recall@k、MRR、Precision@k、nDCG@k、P50/P95、`best_evidence_overlap`。
- `build_gold_evidence.py`：从 `required_facts` + `reference_answer` + 段落重叠生成证据标签。
- `evidence.py`：quote 覆盖率、段落匹配、文件匹配。

#### 答案评测

- `answer_eval.py` 调用真实 `/chat_with_agent/stream`。
- Judge A：对照 `reference_answer` / `required_facts` 的 0/1/2 正确性。
- Judge B：claim 级忠实度 + `[n]` 引用支持。
- 确定性指标：Hit@5、ExpectedContextCoverage、Citation Precision、Refusal Accuracy。
- 历史结果：56 条，完全正确率 93.5%，忠实度 97.6%，拒答准确率 100%。
- **注意**：93.5% 是切换 OpenCode Go 前的 `qwen-plus` 生成 + `qwen-plus` judge 历史基线，不是当前 DeepSeek 的结果；需要重跑。

#### 分块实验

- `chunk_experiment.py` 为每个配置建独立 KB、重新上传、跑 `run_eval`。
- DOCX 尺寸实验三组：child 200/0、200/50、400/0。
- Markdown 策略实验三组：recursive-overlap、recursive-no-parent-overlap、markdown-heading。
- 相对基线做配对 bootstrap 95% CI，并统计子块数、表格完整性和分题型 Recall。
- `--parent-eval` 额外跑生产回表路径（parent 级），避免只看 child 检索。
- 生成 `docs/chunking-comparison.md`、`docs/chunking-markdown-strategy.md`（初版）和 `docs/chunking-markdown-strategy-v2.md`（短章节合并版），包括指标、CI、hit/miss 翻转、基线未命中。

#### 运行身份

每次评测写 `_meta.json`：

- KB 文件 fingerprint
- git sha + dirty
- chunk 配置
- 模型名
- top_k
- golden sha256
- sample_count

这样不同 run 不会在不知情的情况下被拼在一起。

---

## 4. 关键数据流

### 4.1 文档入库

```text
Streamlit 上传
  → FastAPI 校验类型/大小/知识库 ID
  → 流式写临时文件 + SHA-256
  → 查 file_metadata 状态
      ├─ ready/processing：按重复处理
      └─ failed：清旧向量 + 清旧 PG 记录，重新入库
  → 写 processing
  → BackgroundTask
  → 解析 DOCX/PDF
  → parent/child split
  → PostgreSQL 单事务（file_metadata + parent_chunks + chunk_hashes）
  → Milvus 写 child vectors
  → ready / failed
```

### 4.2 问答

```text
Streamlit
  → GET /chat_with_agent/stream
  → FastAPI ensure_conversation / 标题 / KB
  → LangGraph Agent
      ├─ 需要知识库：search_knowledge_base
      │     → hybrid child 召回
      │     → PostgreSQL 回表 parent
      │     → rerank
      │     → 编号 + sources
      ├─ 需要记忆：get_memory
      │     → Milvus 记忆召回
      │     → PostgreSQL active 校验
      └─ 需要原文：get_raw_conversation_by_summary_id
  → LLM 生成
  → claim/quote 对齐
  → sources 事件
  → BackgroundTask 保存 raw_conversations
```

### 4.3 记忆

```text
raw_conversations
  → 180s 后台任务
  → 未摘要消息 + token > 400
  → LLM 提取 summary / semantic / episodic / procedural / profile
  → Milvus 找 supersede 候选
  → PostgreSQL 单事务写 summary、memory、version、source、消息 summary_id、画像
  → Milvus 投影
      ├─ 成功：index_status=indexed
      └─ 失败：index_status=failed → reindex_memories.py
```

### 4.4 评测

```text
seed_violet_kb.py
  → 上传 12 份 DOCX 到指定 KB
run_eval.py
  → 固定 golden + top_k + 检索模式
  → 逐题检索
  → evidence 覆盖率判定
  → Recall/MRR/Precision/nDCG + 延迟
chunk_experiment.py
  → 多配置独立 KB
  → run_eval × N
  → 配对 bootstrap + 翻转样本
  → docs/chunking-comparison.md
```

---

## 5. 评测结果

### 5.1 旧的检索结果

在旧的文档级标签下，50 题、top_k=5 时 Recall@5 达到 1.000，MRR 约 0.83–0.85。问题是：

- 标签只要求命中“正确文件”的任意父块；
- Recall 饱和，无法区分 chunking；
- 不能说明证据位置是否正确。

因此后来补了 `gold_evidence`，把评测从“文档级”升级为“证据级”。

### 5.2 分块策略真实结果

#### 5.2.1 DOCX 尺寸对比

实验设置：

- 12 份 DOCX
- 同一 `text-embedding-v4` / `qwen3.7-text-rerank`
- 50 题，child 级检索，`top_k=5`，`hybrid+rerank`
- 每个配置独立 KB
- 46 条可回答样本参与 Recall/MRR/Precision/nDCG

| 策略 | Recall@5 | MRR | Precision@5 | nDCG@5 | P50 / P95 | 子块数 |
|---|---:|---:|---:|---:|---:|---:|
| child 200/0 | 0.848 | 0.670 | 0.300 | 0.700 | 2870 / 5777 ms | 321 |
| child 200/50 | 0.913 | 0.715 | 0.330 | 0.749 | 3146 / 5577 ms | 352 |
| child 400/0 | **0.957** | **0.765** | **0.357** | **0.811** | **2683 / 3625 ms** | **164** |

相对 200/0 的配对 bootstrap 95% CI：

- 400/0：ΔRecall **+0.109**（+0.022~+0.217）、ΔPrecision **+0.057**（+0.017~+0.100）、ΔnDCG **+0.111**（+0.028~+0.196）；MRR 区间跨 0。
- 200/50：ΔRecall +0.065（-0.022~+0.153）、ΔnDCG +0.049（-0.027~+0.133），不显著。

结论：本次语料上 **400 字非重叠 child** 在检索指标和延迟上都最好；但这是 50 题、检索层的结论，不能直接说明答案质量最好。生产默认仍是 200/50，下一步应该用答案评测和更大语料复核，而不是只凭一次 50 题实验改默认。

#### 5.2.2 结构感知 Markdown 对比

为了避免“DOCX vs Markdown”的源格式混淆，第二组实验把同一份 12 篇 Markdown 语料分别用两种边界策略切分：

- `recursive-overlap`：当前 `coarse-no-overlap` profile，1000/200 + 400/0。
- `recursive-no-parent-overlap`：1000/0 + 400/0，用来隔离父块 overlap。
- `markdown-heading`：结构感知，1000/0 + 400/0，按标题、段落、表格切分。
- 同样的 `text-embedding-v4` / `qwen3.7-text-rerank`、50 题、child 级检索、`top_k=5`、`hybrid+rerank`。

初版 `markdown-heading` 按每个 `##` 硬切，产生 243 个子块、中位长度 192，child 级 Precision 显著下降（-0.039）。当前实现加入**短章节合并**：同一一级标题下的相邻章节打包到同一父/子块，标题行保留，子块数降到 209、中位长度升到 297。

| 策略 | Recall@5 | MRR | Precision@5 | nDCG@5 | P50 / P95 | 子块数 |
|---|---:|---:|---:|---:|---:|---:|
| recursive-overlap | 0.935 | 0.815 | **0.357** | 0.838 | **1228 / 2056 ms** | 190 |
| recursive-no-parent-overlap | 0.935 | 0.770 | 0.326 | 0.798 | 1244 / 1859 ms | 168 |
| markdown-heading | 0.935 | **0.835** | 0.339 | **0.849** | 1147 / 2101 ms | 209 |

生产路径（parent 级回表，`hybrid_retrieval_knowledge_base` → PostgreSQL → rerank）：

| 策略 | Recall@5 | MRR | Precision@5 | nDCG@5 | P50 / P95 |
|---|---:|---:|---:|---:|---:|
| recursive-overlap | 0.935 | 0.855 | **0.383** | 0.862 | 1362 / 2441 ms |
| recursive-no-parent-overlap | 0.957 | 0.797 | 0.361 | 0.827 | 1264 / 2645 ms |
| markdown-heading | **0.978** | **0.874** | 0.374 | **0.880** | 1459 / 2924 ms |

结构完整性（直接扫描 Milvus 子块）：

| 策略 | 中位长度 | 含表格子块 | 表头保留率 | 列数一致率 |
|---|---:|---:|---:|---:|
| recursive-overlap | 312 | 68 | 0.794 | 0.794 |
| recursive-no-parent-overlap | 307 | 66 | 0.788 | 0.788 |
| markdown-heading | 297 | 73 | **1.000** | **1.000** |

相对 `recursive-overlap` 的配对 bootstrap 95% CI：

- child 级 `markdown-heading`：ΔRecall +0.000（-0.065~+0.065）、ΔMRR +0.020（-0.056~+0.098）、ΔnDCG +0.011（-0.056~+0.077）、ΔPrecision -0.017（-0.052~+0.017）；Precision 降幅不再显著。
- parent 级 `markdown-heading`：ΔRecall +0.043（0.000~+0.109）、ΔMRR +0.019（-0.043~+0.084）、ΔnDCG +0.018（-0.030~+0.073）、ΔPrecision -0.009（-0.039~+0.022）；方向正向，但 CI 触 0 / 跨 0，不能宣称显著。
- 答案级 `no-judge` 确定性指标（56 条，46 可回答）：ExpectedContextCoverage 0.895→**0.917**、AllEvidenceHit 0.783→**0.826**、FactCoverage 0.795→**0.806**、ContextPrecision 0.497→0.485；完整 LLM Judge（正确率/忠实度/拒答）尚未跑，不能把这组数字当成完整答案质量结论。

怎么读：

- 结构切分**确实修好了表格**：表头保留率从约 0.79 提升到 1.00。
- 合并短章节后，child 级 Precision 不再显著下降；报告把“更多小 chunk”这个 v1 的主要副作用消掉了。
- child 级 Recall 仍是 0.935；multi-hop 从 0.857 升到 1.000，但 single-hop 从 0.935 降到 0.903，净效果持平。稳定 miss（`v-q011`、`v-q026`）是实体别名/关系题，不是 chunk 边界问题。
- parent 级生产路径上 `markdown-heading` 的 Recall/MRR/nDCG 都更高，但样本只有 50 题，CI 仍触 0，不能当作已证实的收益。
- 因此**不替换默认 `fine-overlap`**；`markdown-heading` 可作为表格/章节结构重要的 Markdown 知识库的可选策略。下一步是答案级评测和更大语料复核。

完整报告：[`docs/chunking-markdown-strategy-v2.md`](chunking-markdown-strategy-v2.md)；初版硬切对比保留在 [`docs/chunking-markdown-strategy.md`](chunking-markdown-strategy.md)。

### 5.3 答案评测

历史答案评测：56 条（46 可回答 + 10 不可回答），`qwen-plus` 生成 + `qwen-plus` Judge：

- 完全正确率 93.5%
- 忠实度（strict）97.6%
- 引用有效率 / 精确率 / 覆盖率 98.8% / 98.0% / 94.3%
- Chunk 支持率 78.1%
- 拒答准确率 100%
- 误拒率 4.3%

这组数字的边界：

- 是旧模型/旧配置的历史基线，不是当前 DeepSeek 的结果；
- 生成和 Judge 同源，可能有自偏好；
- 56 条样本的 95% CI 约 ±10pp；
- 人工复核表仍需要填写。

### 5.4 结果怎么读

- Recall 高不等于上下文干净：Precision@5 只有 0.30–0.36。
- 400/0 的收益主要来自“更少的块 + 更完整的证据”，但大块也可能带入噪声。
- 200/50 的 overlap 有正向趋势，但区间跨 0，不能过度解读。
- 不可回答样本 clean rate 为 0%，因为 child 检索没有拒答阈值；这不是 chunking 差异。
- 真正影响答辩可信度的是：证据标签质量、答案评测是否重跑、是否承认 50 题样本的波动。

---

## 6. 工程边界与已知问题

### 6.1 安全

- 没有登录/鉴权，`user_id` 由客户端传入。
- 默认监听 `0.0.0.0`，CORS 允许任意来源。
- 只适合本机单用户演示，不能直接公开部署。
- 这是当前最大的 P0。

### 6.2 数据一致性

- 文档入库已做 PostgreSQL 单事务 + Milvus 后写 + 失败清理。
- 删除知识库/文件已改为先删 Milvus、再删 PG，失败可重试。
- 记忆删除用 PostgreSQL active 过滤兜底。
- 但还没有完整 outbox：Milvus 投影失败仍需要 reindex。
- 后台任务没有跨进程锁，多 worker 可能重复抽取。

### 6.3 评测

- `gold_evidence` 主要是自动生成，需要人工抽检。
- 50 题、12 份文档，规模小，bootstrap 区间可能较宽。
- 答案评测 93.5% 是历史基线，需要按当前模型重跑。
- 记忆能力没有专门评测集。
- 没有 CI 和数据库迁移系统。

### 6.4 功能

- 表格/跨段证据容易切碎。
- 版本冲突只靠提示词和检索，没有文档 `status=current/archived` 元数据。
- 前端仍是单文件为主。
- 单用户，没有多租户。

---

## 7. 老师可能问的问题与回答 / Anticipated Questions & Answers

### 7.1 选题与创新 / Topic & Contributions

**Q1：这个项目和普通 ChatGPT 有什么区别？**

**Q1 (EN): How is this different from ordinary ChatGPT?**

普通 ChatGPT 不知道私有资料，也不能保证引用。Mnemo 把文档切块、向量化、检索、rerank、引用、长期记忆和评测串成一条完整链路；回答必须尽量落在检索证据上，并且能展开来源。

**EN:** Ordinary ChatGPT does not know private documents and cannot guarantee citations. Mnemo connects ingestion, chunking, vectorization, retrieval, rerank, citations, long-term memory and evaluation into one pipeline. Answers are grounded in retrieved evidence and the sources can be expanded.

**Q2：创新点是什么？是不是只是套 RAG？**

**Q2 (EN): What is the contribution? Is it just RAG?**

不是算法创新，而是工程和证据链创新：

1. 父子块 + hybrid + rerank + Agentic-lite 的完整检索链路；
2. PostgreSQL 作为记忆事实源，Milvus 只是可重建投影；
3. claim 对齐 quote + locator 的可核验引用；
4. 证据级评测 + 独立 KB 分块对比 + 配对 bootstrap。

如果老师追问“有没有新算法”，要诚实回答：没有提出新模型或新检索算法，主要贡献是把多个工程组件做成可运行、可评测、可解释的系统。

**EN:** The contribution is engineering and evidence-chain work, not a new algorithm:

1. a complete retrieval pipeline with parent-child chunks, hybrid search, rerank and Agentic-lite;
2. PostgreSQL as the memory source of truth, with Milvus as a rebuildable projection;
3. claim-aligned quotes and locators for verifiable citations;
4. evidence-level evaluation, independent-KB chunking comparison and paired bootstrap.

If asked whether there is a new algorithm, answer honestly: there is no new model or retrieval algorithm. The contribution is turning multiple components into a runnable, measurable and explainable system.

**Q3：为什么不做微调？**

**Q3 (EN): Why not fine-tune a model?**

资料会更新，微调成本高且不能返回来源；50 题/12 份文档也不足以支撑微调。RAG 更适合“私有事实 + 可引用 + 可更新”的场景。

**EN:** Documents change, fine-tuning is expensive, and it does not provide citations. A 50-question/12-document corpus is also too small for fine-tuning. RAG fits the “private facts + citations + updates” setting much better.

**Q4：目标用户是谁？**

**Q4 (EN): Who is the target user?**

课程/个人知识库、产品手册问答、政策文件查询等“资料私有、需要引用、需要记住用户偏好”的场景。当前实现是单用户原型。

**EN:** Course or personal knowledge bases, product manuals, policy-document search and similar settings where documents are private, citations matter and user preferences should be remembered. The current implementation is a single-user prototype.

### 7.2 架构与选型 / Architecture & Choices

**Q5：为什么用 FastAPI + Streamlit？**

**Q5 (EN): Why FastAPI + Streamlit?**

FastAPI 适合异步 IO、流式响应和依赖注入；Streamlit 能用最短时间做出可交互 Demo。生产环境可以把前端换成 React，但课程项目更看重端到端可运行。

**EN:** FastAPI fits async I/O, streaming and dependency injection. Streamlit makes an interactive demo possible in the shortest time. A production system could replace the frontend with React, but this course project prioritizes an end-to-end runnable system.

**Q6：为什么用 LangGraph，不用普通 LangChain chain？**

**Q6 (EN): Why LangGraph instead of a plain LangChain chain?**

因为需要工具调用循环、会话 checkpoint、流式事件和工具次数限制。普通 chain 更适合固定流水线，Agent 需要状态和分支。

**EN:** The system needs a tool-calling loop, conversation checkpoints, streaming events and tool-call limits. A plain chain fits a fixed pipeline better; an agent needs state and branches.

**Q7：为什么同时用 PostgreSQL、Milvus、Redis，是否过度设计？**

**Q7 (EN): Why PostgreSQL, Milvus and Redis together? Is that over-engineered?**

三者职责不同：

- PostgreSQL 负责事实、事务、版本和来源；
- Milvus 负责向量/BM25 检索；
- Redis 负责会话 checkpoint 和缓存。

如果只用向量库，记忆版本、来源链、事务和审计很难做；如果只用 PostgreSQL，向量检索性能不够。代价是跨存储一致性，需要额外设计。

**EN:** They have different responsibilities:

- PostgreSQL stores facts, transactions, versions and provenance;
- Milvus handles vector/BM25 retrieval;
- Redis stores conversation checkpoints and cache.

A vector-only store makes memory versions, provenance, transactions and auditing hard. PostgreSQL-only storage is not enough for vector retrieval performance. The cost is cross-store consistency, which needs explicit design.

**Q8：为什么用父子块，不直接用一种块？**

**Q8 (EN): Why parent-child chunks instead of a single chunk type?**

子块小，召回准；父块大，给模型足够上下文。只用子块会导致上下文碎片化，只用父块会让向量表达被稀释。代价是存储和回表成本。

**EN:** Small child chunks retrieve more precisely; large parent chunks give the model enough context. Child-only retrieval fragments context, while parent-only retrieval dilutes the embedding. The cost is extra storage and parent lookup.

**Q9：为什么 hybrid，不只用向量？**

**Q9 (EN): Why hybrid retrieval instead of dense-only?**

dense 擅长语义相似，BM25 擅长专有名词、数字、别名和精确匹配。RRF 融合两者的排名，不需要调分数纲。失败时还能降级到 dense。

**EN:** Dense retrieval is good at semantic similarity; BM25 is good at names, numbers, aliases and exact matches. RRF merges the rankings without score normalization. If hybrid search fails, the system falls back to dense retrieval.

**Q10：为什么还要 rerank？**

**Q10 (EN): Why add rerank?**

召回阶段追求 recall，排序阶段追求 precision。rerank 能用更贵的交叉编码模型重排 top 候选，提高 MRR/证据位置；代价是延迟。

**EN:** Retrieval prioritizes recall; reranking prioritizes precision. A more expensive cross-encoder reranks the top candidates and improves MRR/evidence position, at the cost of latency.

**Q11：为什么对话、embedding、rerank 用不同服务？**

**Q11 (EN): Why use different services for chat, embedding and rerank?**

对话模型需要生成质量和成本平衡；embedding 需要中文语义能力；rerank 需要 Cross-Encoder 式的相关性判断。分开选型更灵活，也避免一个服务绑定所有能力。

**EN:** The chat model balances generation quality and cost; the embedding model needs Chinese semantic ability; rerank needs cross-encoder-style relevance judgment. Selecting them separately is more flexible and avoids binding all capabilities to one provider.

### 7.3 数据与一致性 / Data & Consistency

**Q12：文档上传后到底经历了什么？**

**Q12 (EN): What happens after a document is uploaded?**

校验 → 临时文件 + SHA-256 → 查重/状态 → 写 processing → 后台解析 → 父子块 → PostgreSQL 单事务 → Milvus 子块向量 → ready/failed。失败会清理半成品，重传会先清理旧投影。

**EN:** Validation → temp file + SHA-256 → duplicate/status check → write processing → background parsing → parent/child split → one PostgreSQL transaction → Milvus child vectors → ready/failed. Failures clean up partial data; a retry first removes the old projection.

**Q13：PostgreSQL 和 Milvus 怎么保持一致？**

**Q13 (EN): How are PostgreSQL and Milvus kept consistent?**

PostgreSQL 是事实源；Milvus 是投影。文档入库先提交 PostgreSQL，再写 Milvus；Milvus 失败会清理 PG 半成品并标记 failed。记忆写入也是先 PG 事务，再投影 Milvus，失败标记 `index_status=failed`，用 reindex 修复。当前没有完整 outbox，这是下一步。

**EN:** PostgreSQL is the source of truth; Milvus is a projection. Document ingestion commits PostgreSQL first, then writes Milvus; a Milvus failure cleans partial PostgreSQL rows and marks the file failed. Memory writes follow the same pattern: PostgreSQL transaction first, then Milvus projection; failures set `index_status=failed` and are repaired by reindex. There is no full outbox yet; that is the next step.

**Q14：删除知识库/文件为什么先删 Milvus？**

**Q14 (EN): Why delete Milvus before PostgreSQL for a KB or file?**

如果先删 PostgreSQL，Milvus 失败后会留下孤儿向量，而且 API 重试会 404，无法清理。先删 Milvus，失败时 PostgreSQL 还在，用户可以重试；成功后再删 PG。代价是 PG 删除失败时元数据还在，但可重试。

**EN:** Deleting PostgreSQL first can leave orphan vectors if Milvus fails, and an API retry then gets 404 and cannot clean them. Deleting Milvus first keeps PostgreSQL metadata available for a retry; once Milvus succeeds, PostgreSQL is deleted. The trade-off is that a PostgreSQL deletion failure leaves metadata behind, but it remains retryable.

**Q15：删除记忆为什么用 PostgreSQL 过滤？**

**Q15 (EN): Why does memory deletion filter through PostgreSQL?**

Milvus 的 `status=deleted` 更新可能失败。检索时用 PostgreSQL 的 `status='active'` 校验 Milvus 返回的 ID，失败就按空结果处理，避免已删除记忆继续进入回答。

**EN:** The Milvus `status=deleted` update can fail. Retrieval validates returned IDs against PostgreSQL `status='active'` and fails closed, so a deleted memory cannot continue to enter answers.

**Q16：重复上传怎么判断？**

**Q16 (EN): How is duplicate upload detected?**

用文件内容 SHA-256 + 知识库 + 用户判断。`ready/processing` 视为重复；`failed` 允许清理旧记录后重传。

**EN:** By file content SHA-256 plus knowledge base and user. `ready/processing` is treated as a duplicate; `failed` is allowed to clean the old record and retry.

**Q17：多用户隔离怎么做的？**

**Q17 (EN): How is multi-user isolation implemented?**

数据模型和 Milvus filter 都有 `user_id`，但当前 `user_id` 由客户端传入，没有认证，所以只是“逻辑隔离”，不是安全隔离。公开部署前必须加登录和 token。

**EN:** The data model and Milvus filters include `user_id`, but the current `user_id` comes from the client and there is no authentication. It is logical isolation, not security isolation. Login and tokens are required before public deployment.

### 7.4 检索与生成 / Retrieval & Generation

**Q18：fast/auto/deep 的区别？为什么 deep 没提升还保留？**

**Q18 (EN): What are fast/auto/deep? Why keep deep if it did not improve results?**

fast 一次检索；auto 规则路由/零召回升级；deep 做 Plan/Retrieve/Critique/Refine。violet 50 题上 deep/auto 没提升 Recall/MRR，P95 反而高 4–5 倍，所以默认 fast。保留 deep 是为了多跳/版本冲突场景和可观测性，不是为了刷分。

**EN:** Fast performs one retrieval; auto applies rule-based routing and escalates on zero recall; deep runs Plan/Retrieve/Critique/Refine. On the violet 50 questions, deep/auto did not improve Recall/MRR and P95 was 4–5× higher, so fast is the default. Deep remains for multi-hop/version-conflict cases and observability, not for chasing a benchmark number.

**Q19：为什么旧的 Recall@5 是 1.000，现在只有 0.848？**

**Q19 (EN): Why was the old Recall@5 1.000 but the new one 0.848?**

旧标签是文档级：命中正确文件的任意父块就算对。新标签是证据级：要覆盖 gold evidence quote。后者更严格，所以数字下降是预期，不是系统变差。

**EN:** The old label was document-level: any parent chunk from the correct file counted. The new label is evidence-level: the retrieved chunk must cover the gold evidence quote. The stricter definition lowers the number; it does not mean the system got worse.

**Q20：为什么 Precision 只有 0.30–0.36？**

**Q20 (EN): Why is Precision only 0.30–0.36?**

top_k=5 里通常只有 1–2 个 chunk 覆盖证据，其他是相关但不等价的上下文；Precision@5 分母固定为 5，所以天然偏低。它反映“上下文干净程度”，不等于答案正确率。

**EN:** In top-5, usually only 1–2 chunks cover the evidence; the rest are related but non-equivalent context. Precision@5 uses a fixed denominator of 5, so it is naturally low. It measures context cleanliness, not answer correctness.

**Q21：400/0 最好，为什么默认还是 200/50？**

**Q21 (EN): 400/0 performed best; why is the default still 200/50?**

400/0 是 50 题、child 检索层的实验结果，回答质量和更大语料还没验证；直接改默认有过拟合风险。正确做法是补答案评测和多语料复核，再决定是否切换。这个回答能体现你知道“实验结论的适用范围”。

**EN:** 400/0 is a 50-question, child-retrieval result; answer quality and a larger corpus are not yet verified. Changing the default immediately risks overfitting. The correct next step is answer evaluation and multi-corpus validation before switching. This answer shows that you understand the scope of an experimental conclusion.

**Q22：为什么 200/50 的提升不显著？**

**Q22 (EN): Why is the 200/50 improvement not significant?**

ΔRecall +0.065，95% CI -0.022~+0.153，跨 0；50 题样本量小，overlap 的收益可能被题目差异淹没。不能只看均值，要看区间和效应量。

**EN:** ΔRecall is +0.065 with a 95% CI of -0.022 to +0.153, crossing zero. With only 50 questions, the overlap benefit can be swamped by question-level variance. Look at intervals and effect size, not only the mean.

**Q23：不可回答样本 clean rate 0% 怎么解释？**

**Q23 (EN): Why is the clean rate 0% on unanswerable questions?**

child 检索没有拒答阈值，top-k 总会返回 chunk，所以“零召回”不适用。真正的拒答在答案层由 prompt 和 Judge 评估；检索层这项只能说明缺少阈值，不是 chunking 的锅。

**EN:** Child retrieval has no refusal threshold, so top-k always returns chunks; “zero recall” does not apply. Actual refusal is evaluated at the answer layer by the prompt and judge. This metric only shows that a threshold is missing, not that chunking is wrong.

**Q24：引用是怎么保证不是编的？**

**Q24 (EN): How do you ensure citations are not fabricated?**

模型被要求只使用检索编号；流式结束后用回答 claim 和来源文本做字符 bigram 对齐，生成 quote 和段落 locator；前端只展示被引用来源。它不是严格 NLI，仍可能错，所以有 Citation Precision/Chunk Support 和人工抽检。

**EN:** The model is required to cite retrieval indices. After streaming, claims are aligned with source text using character-bigram overlap to produce quotes and paragraph locators; the frontend shows only cited sources. This is not strict NLI and can still be wrong, hence Citation Precision/Chunk Support and human review.

**Q25：表格/跨段证据为什么容易失败？**

**Q25 (EN): Why do tables and cross-paragraph evidence fail more often?**

字符切分可能把表格行、表头、跨段句子切断；child 200 尤其容易。实际失败样本 q002/q004 就与表格/日期有关。下一步可以做表格感知切分、按标题切分或加大 child。

**EN:** Character-based splitting can cut table rows, headers and cross-paragraph sentences; child size 200 is especially vulnerable. Real failures q002/q004 are related to tables/dates. Future work includes table-aware splitting, heading-based splitting or a larger child size.

### 7.5 记忆 / Memory

**Q26：长期记忆怎么抽取？**

**Q26 (EN): How is long-term memory extracted?**

180 秒后台任务扫描未摘要消息，token 超过 400 才调用 LLM，一次提取 summary、semantic、episodic、procedural 和 user_profile；PostgreSQL 单事务写事实、版本、来源和消息标记，然后投影 Milvus。

**EN:** A 180-second background job scans unsummarized messages. Only when the token count exceeds 400 does it call the LLM to extract summary, semantic, episodic and procedural memories plus a user profile. One PostgreSQL transaction writes facts, versions, provenance and message markers, then projects to Milvus.

**Q27：记忆冲突怎么处理？**

**Q27 (EN): How are memory conflicts handled?**

新记忆与旧记忆做向量召回，相似度 ≥0.9 时旧记忆标 `superseded`，新记忆 `active`，不物理删除；版本表保留 `valid_from/valid_to/superseded_by`。0.9 是工程阈值，不是语义判断，可能误替换，后续可以接 LLM ADD/UPDATE/NOOP。

**EN:** New memories are matched against old memories by vector recall. At similarity ≥0.9, the old memory becomes `superseded`, the new one becomes `active`, and nothing is physically deleted. The version table keeps `valid_from/valid_to/superseded_by`. 0.9 is an engineering threshold, not semantic judgment, so false supersede is possible; a future version can use LLM ADD/UPDATE/NOOP decisions.

**Q28：删除记忆后还能被检索吗？**

**Q28 (EN): Can a deleted memory still be retrieved?**

正常情况下不能：PostgreSQL 软删除 + Milvus status 更新；即使 Milvus 更新失败，检索也会用 PostgreSQL active 集合过滤。极端情况下 reindex 前投影仍可能短暂不一致，所以要做回归测试。

**EN:** Normally no: PostgreSQL soft-deletes and Milvus updates the status. Even if the Milvus update fails, retrieval filters against the PostgreSQL active set. In extreme cases the projection can be briefly inconsistent before reindex, so regression tests are needed.

**Q29：记忆评测为什么是 0%？**

**Q29 (EN): Why is the memory evaluation 0%?**

当前 answer_eval 的问题集主要考知识库事实，没有触发跨会话记忆/个性化；`memory_used_rate` 记录为 0，不代表记忆功能没实现，而是评测没有覆盖。下一步要单独构造记忆召回、冲突、删除后检索的测试集。

**EN:** The current answer_eval set mainly tests knowledge-base facts and does not trigger cross-session memory or personalization. A `memory_used_rate` of 0 means the evaluation did not cover memory, not that memory is unimplemented. A separate test set is needed for memory recall, conflicts and post-deletion retrieval.

**Q30：多进程会不会重复提取记忆？**

**Q30 (EN): Can multiple processes duplicate memory extraction?**

会。每个进程都会启动 180 秒任务，当前没有 advisory lock / SKIP LOCKED。单进程演示没问题，多 worker 需要加锁或独立 worker。

**EN:** Yes. Every process starts the 180-second job, and there is no advisory lock or SKIP LOCKED. A single-process demo is fine; multi-worker deployment needs a lock or a dedicated worker.

### 7.6 评测与实验 / Evaluation & Experiments

**Q31：50 题够吗？**

**Q31 (EN): Are 50 questions enough?**

不够支撑很强的统计结论。我们用配对 bootstrap 报告 95% CI，并明确说明样本量限制；它适合做课程级对比和失败分析，不适合宣称“生产级最优”。

**EN:** Not enough for strong statistical conclusions. We report paired-bootstrap 95% CIs and state the sample-size limitation. It is suitable for course-level comparison and failure analysis, not for claiming a production optimum.

**Q32：gold_evidence 怎么生成？可靠吗？**

**Q32 (EN): How is gold_evidence generated? Is it reliable?**

用 `required_facts` 找源文档句子，找不到再用 `reference_answer` 做段落重叠匹配；`build_gold_evidence.py --check-only` 校验 quote 是否在源文档中。别名题 q047 曾误匹配到剧集标题，已修正为译名对照表；但自动标签仍建议人工抽检 10–15 条。

**EN:** `required_facts` finds a sentence in the source document; if that fails, `reference_answer` is matched against paragraphs. `build_gold_evidence.py --check-only` validates that quotes exist in the source. The alias question q047 was once matched to an episode-title paragraph and has been corrected to the translation table, but auto labels still need 10–15 human spot checks.

**Q33：93.5% 是当前模型的吗？**

**Q33 (EN): Is 93.5% the current model's result?**

不是。它是切换 OpenCode Go 前的 `qwen-plus` 生成 + `qwen-plus` Judge 历史基线；当前生成模型是 `deepseek-v4.1-flash`，必须重跑答案评测后才能对比。这个回答要主动说，不能把历史数字冒充当前结果。

**EN:** No. It is a historical baseline generated and judged by `qwen-plus` before the OpenCode Go switch. The current generator is `deepseek-v4.1-flash`, and the answer evaluation must be rerun before comparison. State this proactively; do not present a historical number as the current result.

**Q34：怎么复现实验？**

**Q34 (EN): How can the experiment be reproduced?**

`_meta.json` 记录 KB fingerprint、git sha、chunk 配置、模型、top_k、golden sha256、sample_count；`chunk_experiment.py` 为每个配置建独立 KB；`run_eval.py` 和 `answer_eval.py` 都写 meta。结果目录默认在 `evals/results/`，报告在 `docs/`。

**EN:** `_meta.json` records the KB fingerprint, git SHA, chunk configuration, models, top_k, golden SHA-256 and sample count. `chunk_experiment.py` creates an independent KB per configuration, and both `run_eval.py` and `answer_eval.py` write meta. Results live under `evals/results/`, reports under `docs/`.

**Q35：为什么答案评测用 LLM Judge？**

**Q35 (EN): Why use an LLM judge for answer evaluation?**

人工成本高，Judge 可以做规模化初筛；但 Judge 有偏差，所以同时保留确定性指标、Judge A/B 和人工抽检 CSV。历史 baseline 还是同源模型 judge，需要交叉验证。

**EN:** Human review is expensive; a judge scales the initial screening. Judges are biased, so we keep deterministic metrics, Judge A/B and a human-review CSV. The historical baseline used a same-family judge and still needs cross-validation.

### 7.7 工程与安全 / Engineering & Security

**Q36：没有登录，怎么上线？**

**Q36 (EN): There is no login; how would you deploy this?**

当前不能直接上线。最小改造是 JWT/OAuth 登录、从 token 推导 `user_id`、默认监听 127.0.0.1、CORS 白名单、per-user 限流；再做租户隔离和审计日志。

**EN:** It cannot be deployed as-is. The minimum change is JWT/OAuth login, deriving `user_id` from the token, listening on 127.0.0.1 by default, a CORS allowlist and per-user rate limits; then add tenant isolation and audit logs.

**Q37：最大的技术债是什么？**

**Q37 (EN): What is the biggest technical debt?**

按优先级：

1. 认证与租户隔离；
2. 记忆 outbox / 多 worker 锁；
3. gold_evidence 人工复核和答案评测重跑；
4. Alembic 迁移 + CI；
5. 表格/标题感知切分。

**EN:** By priority:

1. authentication and tenant isolation;
2. memory outbox / multi-worker lock;
3. human review of gold_evidence and an answer-eval rerun;
4. Alembic migrations + CI;
5. table/heading-aware chunking.

**Q38：如果给你两周，你先做什么？**

**Q38 (EN): With two weeks, what would you do first?**

先做认证 + 默认本机监听 + 分块默认值复核；再做 outbox 和记忆任务锁；最后补人工标签和答案评测。不会先加新功能。

**EN:** Authentication, localhost-by-default and a review of the chunking default; then the outbox and memory-task lock; finally human labels and answer evaluation. I would not start by adding new features.

**Q39：如果给你两个月呢？**

**Q39 (EN): With two months?**

加多租户、文档版本 `current/archived`、表格感知切分、记忆 LLM 冲突裁决、自动评测 CI、前端组件化，以及更大规模多领域语料。

**EN:** Multi-tenancy, document versions `current/archived`, table-aware chunking, LLM memory-conflict resolution, automated evaluation CI, frontend componentization and a larger multi-domain corpus.

**Q40：这个项目你最自豪的是什么？**

**Q40 (EN): What are you most proud of in this project?**

不是某个模型，而是把“文档入库—检索—引用—记忆—评测”做成了可运行、可解释、可复现的闭环，并且知道它的边界在哪里。

**EN:** Not a particular model, but turning “ingestion—retrieval—citation—memory—evaluation” into a runnable, explainable and reproducible loop, while knowing where its boundaries are.

**Q41：最遗憾的是什么？**

**Q41 (EN): What do you regret most?**

记忆能力没有专门评测，答案评测还没用当前 DeepSeek 重跑；另外安全边界和 outbox 还没做到生产级。

**EN:** Memory has no dedicated evaluation, the answer evaluation has not been rerun with the current DeepSeek model, and the security boundary and outbox are not production-grade.

### 7.8 容易被追问的“陷阱题” / Likely Follow-up Traps

**Q42：“你说 400/0 最好，那为什么不直接改默认？”**

**Q42 (EN): “You say 400/0 is best; why not change the default?”**

因为结论来自 50 题、child 检索层实验，不是答案质量实验；直接改默认可能过拟合。正确做法是先补答案评测和多语料，再决定。

**EN:** Because the result comes from a 50-question, child-retrieval experiment, not an answer-quality experiment. Changing the default directly risks overfitting. The right move is answer evaluation and a larger corpus first.

**Q43：“Recall 1.0 是不是说明检索没问题？”**

**Q43 (EN): “Does Recall 1.0 mean retrieval has no problem?”**

不是。旧 Recall 是文档级，饱和说明指标太粗；升级到证据级后 400/0 也只有 0.957，说明仍有边界失败。

**EN:** No. The old Recall was document-level, and saturation means the metric was too coarse. At evidence level, even 400/0 reaches only 0.957, so boundary failures remain.

**Q44：“Precision 0.3 是不是系统很差？”**

**Q44 (EN): “Is Precision 0.3 a sign that the system is bad?”**

Precision@5 分母固定为 5，而每题 gold evidence 通常只有 1–2 个 chunk；0.3–0.36 反映 top-5 里必然有相关但不等价的上下文，不等于答案正确率。要结合 Recall、nDCG 和答案评测看。

**EN:** Precision@5 has a fixed denominator of 5, while each question usually has only 1–2 gold evidence chunks. A value of 0.3–0.36 reflects related but non-equivalent context in top-5; it is not answer correctness. Read it together with Recall, nDCG and answer evaluation.

**Q45：“你的引用是模型自己编的吗？”**

**Q45 (EN): “Are the citations fabricated by the model?”**

编号是模型生成的，但 quote/locator 是从检索文本里对齐出来的；模型不能凭空产生来源文本。仍然可能选错 quote，所以有 Citation Precision/Chunk Support 和人工抽检。

**EN:** The citation numbers are generated by the model, but quotes/locators are aligned from retrieved text; the model cannot invent source text. It can still choose the wrong quote, hence Citation Precision/Chunk Support and human review.

**Q46：“为什么不用更大的模型？”**

**Q46 (EN): “Why not use a larger model?”**

课程环境更重视成本、延迟和可复现；当前 DeepSeek 已能支撑生成，瓶颈更多在检索、证据和评测。模型可以替换，架构不应绑死。

**EN:** The course setting prioritizes cost, latency and reproducibility. The current DeepSeek model is sufficient for generation; the bottleneck is retrieval, evidence and evaluation. Models should be replaceable, and the architecture should not be tied to one.

**Q47：“这个系统能处理多少文档？”**

**Q47 (EN): “How many documents can this system handle?”**

当前是 12 份 DOCX、50 题的课程规模；Milvus 分区和异步 IO 可以扩展，但上传/embedding/rerank 是主要瓶颈，后台任务也没有水平扩展锁。

**EN:** The current scale is 12 DOCX files and 50 questions. Milvus partitions and async I/O can scale, but upload/embedding/rerank are the main bottlenecks, and background tasks have no horizontal-scaling lock.

**Q48：“如果 Milvus 挂了怎么办？”**

**Q48 (EN): “What happens if Milvus goes down?”**

应用启动会失败；hybrid 检索会降级 dense，但 Milvus 整体不可用时无法检索。记忆检索在 PostgreSQL 校验失败时 fail-closed。生产需要重试、降级和健康检查。

**EN:** Application startup fails. Hybrid retrieval can fall back to dense, but when Milvus is entirely unavailable there is no retrieval. Memory retrieval fails closed when the PostgreSQL check fails. Production needs retries, degradation and health checks.

---

## 8. 演示建议

### 8.0 30 秒答辩稿

Mnemo 是一个带长期记忆的可核验 RAG 助手。文档经过父子块切分和 hybrid 检索后，用 rerank 选出证据，再由 LangGraph Agent 生成带 `[n]` 引用和 quote 的回答；对话会定时压缩成 PostgreSQL 中的版本化记忆，Milvus 只做可重建投影。我们用 50 题证据级评测、答案 Judge A/B 和独立 KB 的分块实验证明链路有效，同时也明确它目前是单用户课程原型，安全和评测仍有边界。

### 8.1 5 分钟演示顺序

1. 打开 Streamlit，选择 `violet-kb`。
2. 问一个单跳事实题，展示 `[n]` 和来源 quote/locator。
3. 问一个多跳/版本冲突题，展示 trace 和 deep/auto 的差异。
4. 打开记忆抽屉，展示摘要/语义/情节/程序记忆和删除。
5. 打开 `docs/chunking-comparison.md`，讲 400/0 的真实结果和 CI。
6. 最后主动讲边界：单用户、标签自动、93.5% 是历史基线。

### 8.2 启动命令

```powershell
# 基础设施
docker compose up -d

# 后端
cd backend
$env:PORT="8010"
..\.venv\Scripts\python.exe main.py

# 前端
cd ..
$env:MNEMO_API_BASE_URL="http://127.0.0.1:8010"
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py --server.port 8501
```

### 8.3 测试与评测

```bash
python -m pytest -q
python evals/run_eval.py --check-only
python evals/build_gold_evidence.py --check-only
python evals/chunk_experiment.py --dry-run
```

---

## 9. 未来工作

### 近期

- 认证、租户隔离、默认本机监听、CORS 白名单。
- 记忆 outbox / 多 worker 锁。
- 人工复核 `gold_evidence` 10–15 条。
- 用当前 DeepSeek 重跑答案评测，区分生成模型和 Judge。
- 把 400/0 放进答案评测，再决定默认配置。

### 中期

- 文档版本 `current/archived` 和版本冲突提示。
- 表格/标题感知切分，减少 q002/q004 类失败。
- 记忆 LLM 冲突裁决（ADD/UPDATE/NOOP）。
- Alembic + CI + 集成测试。

### 长期

- 多租户 SaaS。
- 多领域语料和更大评测集。
- 前端组件化/React。
- 成本、延迟和质量的自动回归。

---

## 10. 总结

Mnemo 的核心不是“接了一个大模型”，而是把 RAG 和长期记忆做成了一条可运行、可解释、可评测的工程链路：

- 文档有父子块、状态和来源；
- 检索有 hybrid、rerank 和 trace；
- 回答有引用和 quote；
- 记忆有事实源、版本、来源和重建；
- 评测有证据标签、指标、meta 和分块实验。

它仍然是单用户课程原型，安全、评测和一致性都有明确边界；但这些边界被写进了文档和实验，而不是藏在 Demo 里。这是这个项目最值得讲的部分。
