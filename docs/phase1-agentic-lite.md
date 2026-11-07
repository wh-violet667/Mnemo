# Phase 1：Agentic-lite 实现文档

- 状态：草案 v0.1，待审核
- 日期：2026-11-08
- 前置：Phase 0 审计完成、Phase 0.5 数据一致性修复完成（`42fa209`）
- 范围：只做 Agentic-lite，不包含 KG、多 Agent、Web 搜索、完整 GraphRAG

---

## 1. 目标与非目标

### 1.1 目标

在不破坏现有 fast 路径的前提下，为知识库检索增加可控的多轮反思能力：

- 新增 `retrieval_mode`：`fast / auto / deep`，默认 `fast`；
- `fast`：一次 hybrid + rerank，不升级；
- `deep`：Plan → Retrieve → Critique → Refine，最多 2 轮；
- `auto`：规则判断 + fast 无结果时升级一次；
- 多轮检索只做一次统一 rerank；
- 新增 `trace` 流式事件，前端可展示“本次：深入 · 2 轮 · 8 条证据”；
- 建立多跳/关系/时间评测集，用数据验证增益；
- 保持现有 `search_knowledge_base` 的引用编号、`ContextSchema.sources` 和前端来源卡片兼容。

### 1.2 非目标

- 不引入 Neo4j / 图谱查询；
- 不做多 Agent 协作；
- 不接 Web 搜索；
- 不做无限 ReAct 循环；
- 不改变现有记忆系统；
- 不在本阶段增加 `use_kg` 参数；
- 不强制拆前端 `state.py` / `components.py`。

---

## 2. 关键设计决策

### 2.1 单工具内部模式（推荐）

V1 不新增 `deep_search_knowledge_base` 工具，而是让现有工具根据上下文切换：

```text
search_knowledge_base(query)
  -> runtime.context.retrieval_mode
     -> fast: 一次 hybrid + rerank
     -> deep: 内部 Plan/Retrieve/Critique/Refine
     -> auto: 规则判断，必要时升级
```

理由：

- `fast` / `deep` 由代码强制，不依赖模型“选对工具”；
- 现有 `ToolCallLimitMiddleware(run_limit=1)` 不变；
- 多轮逻辑封装在工具内部，容易做预算、停止条件和测试；
- 不改变 Agent 的工具列表和提示词结构。

备选方案（若坚持双工具）：用 `wrap_model_call` 按 `retrieval_mode` 过滤可用工具，或准备两套编译好的 Agent。不要只靠提示词约束。

### 2.2 `auto` 第一版只做规则 + 无结果升级

- 不引入独立分类模型；
- 规则命中则 `deep`，否则先 `fast`；
- `fast` 返回 0 个父块时，在本次请求内升级一次 `deep`；
- 不根据 rerank 分数做复杂阈值升级，留到 V1.1。

### 2.3 模型分工

- 最终回答：`AGENT_BASE_MODEL`（当前 qwen-plus）；
- planner / critic：`SUMMARIZATION_MODEL`（当前 qwen-turbo，成本更低；JSON 不稳定时可升级 qwen-plus）；
- 不让主模型同时承担 planner、critic、generator 三次调用；
- deep 模式预算：1 次 planner + 每轮 1 次 critic + 1 次生成。

### 2.4 预算

| 项 | 默认值 | 硬上限 |
|---|---:|---:|
| 最大轮次 | 2 | 3 |
| 每轮最多查询数 | 3 | 3 |
| 总超时 | 12000 ms | 20000 ms |
| 最终证据条数 | 8 | 10 |
| critic 输入文档数 | 6 | 8 |

超时或达到轮次上限时，仍返回已有证据并进入生成，不允许直接返回空答案。

---

## 3. 请求流程

```text
前端选择 retrieval_mode
  -> GET /chat_with_agent/stream?retrieval_mode=...
  -> FastAPI 枚举校验
  -> stream_agent_response(..., retrieval_mode)
  -> ContextSchema(retrieval_mode=...)
  -> Agent 调用 search_knowledge_base
  -> run_retrieval(mode, ...)
       fast: 1 次检索
       deep: 2 轮反思检索
       auto: 规则 + 无结果升级
  -> 统一 rerank / 编号
  -> 工具返回带编号的上下文
  -> Agent 生成回答
  -> 流式输出 answer / status / trace / sources
```

---

## 4. 模块设计

### 4.1 `backend/routing.py`（新增）

```python
from typing import Literal

RetrievalMode = Literal["auto", "fast", "deep"]

DEEP_HINTS = (
    "比较", "区别", "差异", "分别", "各自", "同时", "以及", "还是",
    "版本", "适用", "生效", "过期", "之前", "之后", "冲突",
    "关系",
)

def needs_deep(query: str) -> bool:
    """规则判断是否需要深度检索。"""

def build_initial_queries(query: str) -> list[str]:
    """生成第一轮查询，最多 3 条。"""

def build_refined_query(query: str, missing: list[str]) -> str:
    """根据 critic 的 missing 维度改写第二轮查询。"""

def merge_parent_ids(rounds: list[list[str]]) -> list[str]:
    """按轮次合并 parent_id，保序去重。"""

def should_stop(
    *,
    iteration: int,
    max_iterations: int,
    new_parent_count: int,
    critic: dict | None,
    elapsed_ms: int,
    budget_ms: int,
) -> bool:
    """统一停止条件。"""
```

第一版规则建议：

- `needs_deep = True`：
  - 命中 `DEEP_HINTS`；
  - 查询长度 ≥ 40 字；
  - 包含两个及以上明显实体（书名号《...》、≥3 字符的大写开头拉丁词粗略判断；
    TV/OP/ED 等两字符媒介缩写不算内容实体）。
- 否则 `False`。

注（V1 实测修正）：violet 语料 A/B 中"双问号"规则 4/4 全误判（同文档复合问句）、
"哪些"提示词误判枚举型问题，均已从规则中移除；"分别"保留（多跳比较的核心信号）。

`build_initial_queries`：

- 总是包含原问题；
- 含“和/与/、/以及/还是/区别/差异”时，按分隔符拆成两个子查询；
- 去重、去空、截断到 3 条；
- 不做同义词改写。

`build_refined_query`：

- 使用 critic 返回的 `missing` 维度；
- 生成一条“原问题 + 缺失维度”的查询；
- 如果 `missing` 为空，返回空字符串，触发停止。

### 4.2 `backend/retrieval.py`（新增）

```python
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

@dataclass
class RetrievalTrace:
    mode: str
    rounds: int
    queries: list[str]
    parent_count: int
    critic: str
    elapsed_ms: int
    escalated: bool = False
    degraded: bool = False

@dataclass
class RetrievalResult:
    documents: list[dict]
    snippets: str
    sources: list[dict]
    trace: RetrievalTrace
    sufficient: bool | None

Retriever = Callable[..., Awaitable[list[str]]]  # 实现时一律用关键字参数调用
Critic = Callable[..., Awaitable[dict]]

async def run_retrieval(
    query: str,
    *,
    user_id: int,
    knowledge_base_id: str,
    top_k: int,
    mode: str = "fast",
    start_index: int = 1,
    max_rounds: int = 2,
    budget_ms: int = 12000,
    retriever: Retriever | None = None,
    critic: Critic | None = None,
) -> RetrievalResult:
    ...
```

默认依赖：

- `retriever`：`MilvusClient.hybrid_retrieval_knowledge_base(query, knowledge_base_id, user_id, top_k=top_k*3, mode="hybrid")`；
- `load_documents`：`PostgreSQLClient.get_parents(knowledge_base_id=..., parent_ids=..., user_id=...)`；
- `critic`：调用 `SUMMARIZATION_MODEL`，返回结构化 JSON；
- `rerank`：复用 `documents_process.rerank_documents`；
- `order_by_rerank` / `dedupe_documents` / `build_numbered_context`：复用 `citations.py`。

`run_retrieval` 不直接依赖 `ContextSchema`，只接收参数；工具层负责写入 `runtime.context.sources`。

### 4.3 `backend/config.py`（修改）

新增提示词：

```python
AGENTIC_PLAN_PROMPT = """..."""
AGENTIC_CRITIC_PROMPT = """..."""
```

规划提示词输出：

```json
{
  "queries": ["查询1", "查询2"],
  "reason": "为什么这样拆"
}
```

Critic 提示词输出：

```json
{
  "sufficient": false,
  "missing": ["角色适用域", "生效时间"],
  "rewrite": "针对缺失维度的改写查询",
  "reason": "证据缺少..."
}
```

要求：

- 只输出 JSON；
- 不允许输出 SQL/Cypher；
- critic 输入只放截断后的证据摘要，避免 token 膨胀；
- JSON 解析失败先尝试 `json_repair`，再失败一次就停止本轮循环。

### 4.4 `backend/tools.py`（修改）

`search_knowledge_base` 改为：

```python
@tool("search_knowledge_base")
async def search_knowledge_base(query: str, runtime: ToolRuntime[ContextSchema]) -> str:
    mode = runtime.context.retrieval_mode or "fast"
    result = await run_retrieval(
        query,
        user_id=runtime.context.user_id,
        knowledge_base_id=runtime.context.knowledge_base_id,
        top_k=runtime.context.top_k,
        mode=mode,
        start_index=len(runtime.context.sources) + 1,
    )
    if result.sources:
        runtime.context.sources.extend(result.sources)
    writer = get_stream_writer()
    writer({"type": "trace", "content": asdict(result.trace)})
    if not result.snippets:
        return "未检索到相关文档"
    if result.sufficient is False:
        return (
            "【检索提示】证据可能不足。以下内容仅供回答参考；"
            "如果仍然无法回答，请明确说明未在知识库中找到相关信息。\n\n"
            + result.snippets
        )
    return f"检索到的相关文档如下（回答时请用方括号编号标注来源）：\n{result.snippets}"
```

注意：

- 工具名保持不变；
- `ToolCallLimitMiddleware` 仍是 1 次；
- 多轮检索在工具内部完成；
- `trace` 与 `status` 分开。

### 4.5 `backend/agent.py`（修改）

- `stream_agent_response` 增加 `retrieval_mode: str = "fast"`；
- 构造 `ContextSchema` 时传入；
- custom 流处理：

```python
elif stream_mode == "custom":
    custom_info = chunk_data
    if isinstance(custom_info, dict) and custom_info.get("type") == "trace":
        yield json.dumps(
            {"type": "trace", "content": custom_info.get("content", {})},
            ensure_ascii=False,
        ) + "\n"
    else:
        yield json.dumps(
            {"type": "status", "content": custom_info},
            ensure_ascii=False,
        ) + "\n"
```

- 现有 `answer` / `sources` / `error` 行为不变；
- 最终 `sources` 仍只发一次。

### 4.6 `backend/api.py` / `backend/schemas.py`（修改）

`api.py`：

```python
from typing import Literal

@router.get("/chat_with_agent/stream")
async def chat_with_agent(
    query: str,
    knowledge_base_id: str,
    user_id: int,
    conversation_id: str = "default",
    retrieval_mode: Literal["auto", "fast", "deep"] = "fast",
    ...
):
    ...
    return StreamingResponse(
        stream_agent_response(
            ...,
            retrieval_mode=retrieval_mode,
        ),
        ...
    )
```

`schemas.py`：

```python
@dataclass
class ContextSchema:
    user_id: int
    knowledge_base_id: str
    top_k: int = 5
    conversation_id: str = "default"
    retrieval_mode: str = "fast"
    sources: List[Dict[str, Any]] = field(default_factory=list)
```

### 4.7 `frontend/streaming.py`（修改）

`StreamSession` 增加：

- `retrieval_mode` 构造参数；
- 请求参数 `retrieval_mode`；
- `trace: dict | None` 字段；
- `snapshot()` 返回 `trace`；
- 处理 `event_type == "trace"`。

### 4.8 `frontend/app.py`（修改）

- `session_state` 增加 `retrieval_mode`，默认 `"fast"`；
- 发送框上方或下方增加三档选择；
- `_start_stream` 把 `st.session_state.retrieval_mode` 传给 `StreamSession`；
- 回答结束后，如果 `trace` 存在，展示：

```text
本次：深入 · 2 轮 · 8 条证据 · 4.2s
```

第一版不拆 `state.py` / `components.py`。

---

## 5. 接口契约

### 5.1 HTTP

```http
GET /chat_with_agent/stream
  ?query=...
  &knowledge_base_id=...
  &user_id=1
  &conversation_id=...
  &retrieval_mode=auto
```

`retrieval_mode` 只接受：

- `auto`
- `fast`
- `deep`

非法值由 FastAPI 返回 422。

### 5.2 流式事件

```json
{"type":"status","content":"正在规划检索..."}
{"type":"answer","content":"..."}
{"type":"trace","content":{"mode":"deep","rounds":2,"queries":["..."],"parent_count":8,"critic":"sufficient","elapsed_ms":4200,"escalated":false}}
{"type":"sources","content":[{"index":1,"file_name":"...","parent_id":"...","snippet":"..."}]}
{"type":"error","content":"..."}
```

兼容性：

- 旧前端忽略 `trace` 也能正常工作；
- 旧后端不返回 `trace` 时，新前端不显示执行摘要。

### 5.3 `RetrievalResult`

```python
@dataclass
class RetrievalResult:
    documents: list[dict]   # rerank 后的父块，按相关性排序
    snippets: str           # 带 [n] 编号的上下文
    sources: list[dict]     # 与 snippets 编号一致
    trace: RetrievalTrace
    sufficient: bool | None # 有 critic 结果时为 True/False；fast 及 auto 未升级为 None
```

---

## 6. 算法细节

### 6.1 fast

```text
1. queries = [query]
2. parent_ids = retriever(query, top_k * 3)
3. documents = load_documents(parent_ids)
4. rerank(query, documents, top_k)
5. build_numbered_context(documents, start_index)
6. trace = {mode:"fast", rounds:1, queries:[query], critic:"skipped"}
```

要求：

- 只调用一次 retriever；
- 不做 planner / critic；
- 不升级；
- 无结果时返回空 snippets，由 Agent 说明未找到。

### 6.2 deep

```text
1. queries = planner(query) 或 build_initial_queries(query)
2. for round in 1..max_rounds:
     a. 并行 retriever 每个 query
     b. 合并 parent_ids，保序去重
     c. 如果本轮没有新增 parent_id，停止
     d. documents = load_documents(合并后的累计 parent_ids)（每轮加载，见 6.6）
     e. critic = evaluate(query, documents)
     f. 如果 critic.sufficient 或达到预算/轮次，停止
     g. queries = [build_refined_query(query, critic.missing)]
3. 统一 rerank 一次
4. build_numbered_context
5. trace = {mode:"deep", rounds, queries, parent_count, critic, elapsed_ms}
```

约束：

- `max_rounds` 默认 2，硬上限 3；
- 每轮最多 3 个 query；
- critic 输入最多 6 个父块，单个父块截断到 800 字；
- 如果 critic JSON 解析失败，尝试修复一次；仍失败则停止循环，返回已有证据；
- 达到上限时 `sufficient=False`，但必须返回已有证据，不能返回空。

### 6.3 auto

```text
if needs_deep(query):
    -> deep
else:
    -> fast
    if documents == []:
        -> deep（只再跑一轮，escalated=True）
```

要求：

- `fast` 用户选择时绝不升级；
- `auto` 的升级只发生在无结果时；
- trace 中 `escalated=true` 表示发生过升级；
- 第一版不根据 rerank 分数升级。

### 6.4 planner

- 输入：原始问题；
- 输出：最多 3 条查询；
- 失败降级：`build_initial_queries`；
- 不把知识库内容放进 planner prompt，避免泄漏和 token 膨胀。

### 6.5 critic

- 输入：原始问题 + 当前证据摘要；
- 输出：`sufficient / missing / rewrite / reason`；
- 判断标准：
  - 是否覆盖问题中的所有子问题；
  - 是否有角色/地区/时间等约束；
  - 是否存在版本冲突未解决；
  - 是否有足够证据支撑回答。
- 不允许 critic 直接生成最终答案。

### 6.6 合并、去重、rerank、编号

- 每轮检索只收集 `parent_id`；随后立即对**累计** parent_ids 调一次 `get_parents`：
  - 加载结果按 `parent_id` 去重后缓存在内存中，既供本轮 critic 使用（截断后），
    也供最后一轮的统一 rerank 使用，避免重复回表；
  - 即"每轮一次 get_parents（累计集合）"，而不是"每轮各取各的"或"全程只在结尾取一次"——
    后者无法支撑 critic 对证据内容的判断；
- 按 `parent_id` 保序去重；
- 全部轮次结束后，用原始问题对缓存文档做一次 `rerank_documents`；
- rerank 失败时保持 RRF 顺序；
- `build_numbered_context` 的 `start_index = len(context.sources) + 1`；
- 一次请求内来源编号连续，不重复。

### 6.7 停止条件

满足任一条件即停止：

- `iteration >= max_iterations`；
- 本轮没有新增 `parent_id`；
- `critic.sufficient == True`；
- `elapsed_ms >= budget_ms`；
- planner/critic 连续失败；
- 查询为空或全部重复。

停止后：

- 如果已有证据，继续生成；
- 如果无证据，返回“未检索到相关文档”；
- trace 记录停止原因。

---

## 7. 配置

`.env.example` 新增：

```env
# Agentic-lite
AGENTIC_ENABLED=true
AGENTIC_DEFAULT_MODE=fast
AGENTIC_MAX_ROUNDS=2
AGENTIC_TIMEOUT_MS=12000
AGENTIC_MAX_QUERIES=3
```

代码中做边界裁剪：

- `max_rounds = min(max(int(env), 1), 3)`；
- `budget_ms = min(max(int(env), 3000), 20000)`；
- `max_queries = min(max(int(env), 1), 3)`。

`AGENTIC_ENABLED=false` 时：

- `fast` 正常；
- `deep` / `auto` 降级为 fast；
- trace 标记 `degraded=true`。

`AGENTIC_DEFAULT_MODE` 的消费点：**只在 `api.py`**。端点的 `retrieval_mode` 默认值不设字面量，
而是在请求未显式传参时回落到该环境变量；`agent.py` / `schemas.py` 一律以调用方传入值为准，
不再各自硬编码 `"fast"`，避免多处默认值不一致。

---

## 8. 文件改动清单

| 文件 | 类型 | 改动 |
|---|---|---|
| `backend/routing.py` | 新增 | 规则、查询生成、停止条件 |
| `backend/retrieval.py` | 新增 | fast/deep/auto 检索管线 |
| `backend/config.py` | 修改 | planner/critic 提示词 |
| `backend/tools.py` | 修改 | `search_knowledge_base` 接管线、trace、sources |
| `backend/agent.py` | 修改 | `retrieval_mode`、trace 事件 |
| `backend/api.py` | 修改 | 枚举参数、透传 |
| `backend/schemas.py` | 修改 | `ContextSchema.retrieval_mode` |
| `frontend/streaming.py` | 修改 | 模式参数、trace 事件 |
| `frontend/app.py` | 修改 | 三档选择、执行摘要 |
| `evals/golden_agentic.jsonl` | 新增 | 多跳/关系/时间/不可回答集 |
| `evals/run_eval.py` | 修改 | retrieval_mode、parent_id 标签 |
| `evals/run_agent_eval.py` | 新增（可选） | 端到端答案评测 |
| `tests/test_routing.py` | 新增 | 纯函数测试 |
| `tests/test_retrieval.py` | 新增 | fake retriever/critic 测试 |
| `.env.example` | 修改 | Agentic 配置 |
| `README.md` | 修改 | 简短说明三档模式与评测命令 |

---

## 9. 测试计划

### 9.1 单元测试

`tests/test_routing.py`：

- 单跳问题 `needs_deep=False`；
- 比较/版本/多实体问题 `needs_deep=True`；
- `build_initial_queries` 去重、最多 3 条；
- `merge_parent_ids` 保序去重；
- `should_stop` 的轮次、无新增、预算、critic 充分四种情况。

`tests/test_retrieval.py`：

- `fast` 只调用 retriever 一次；
- `deep` 在 critic 不足时调用两轮；
- `deep` 达到 max_rounds 停止；
- `deep` 无新增 parent_id 停止；
- `auto` 在 fast 无结果时升级；
- `auto` 在 fast 有结果时不升级；
- source 编号从 `start_index` 开始且连续。

### 9.2 集成测试

- 启动后端，`retrieval_mode=fast` 对话正常；
- `retrieval_mode=deep` 返回 trace，rounds ≤ 2；
- 非法 mode 返回 422；
- 旧前端不带 mode 时默认 fast；
- `sources` 仍只发一次，编号与 answer 引用一致。

### 9.3 回归

- `python -m pytest -q` 全部通过；
- 现有 16 个测试不退化；
- 现有单跳问题答案不劣化。

---

## 10. 评测计划

### 10.1 数据集

新增 `evals/golden_agentic.jsonl`：

- 10 条跨文档多跳；
- 8 条关系/适用域；
- 6 条时间/版本；
- 4 条不可回答；
- 5 条单跳回归。

每条格式：

```json
{
  "id": "a001",
  "query": "...",
  "query_type": "multi_hop|relational|temporal|unanswerable|single_hop",
  "expected_parent_ids": ["..."],
  "required_facts": ["..."],
  "expected_path": ["PolicyA", "SUPERSEDES", "PolicyB"]
}
```

### 10.2 检索评测

改造 `evals/run_eval.py`：

- `--retrieval-mode baseline|fast|deep|auto`；
- `baseline` 保持现有路径；
- 其他模式调用 `run_retrieval`；
- `is_relevant` 优先使用 `expected_parent_ids`，没有时回退到 `expected_phrase` / `expected_file`；
- 输出分类型指标。

指标：

- Evidence Recall@5；
- MRR；
- P50 / P95；
- 平均检索轮次；
- auto 升级率（fast 无结果触发 deep 的比例）；
- auto 规则误判率（单跳回归子集被 `needs_deep` 判为 deep 的比例，目标 ≤10%）。

注：fast 永不升级由代码构造保证，不设“fast 误升级率”指标。

### 10.3 答案评测（可选）

`evals/run_agent_eval.py`：

- 调真实 `/chat_with_agent/stream`；
- 记录 answer / sources / trace；
- 20 条样本人工评分或 LLM judge；
- 输出答案正确率、引用支持率、平均 token、P95。

### 10.4 上线门槛

- 多跳子集 Evidence Recall 提升 ≥10pp；
- 20 条答案抽样正确率提升 ≥5pp；
- P95 ≤ 现有 fast 的 2 倍；
- auto 规则误判率 ≤10%（单跳不被错误路由到 deep）；
- `auto` 相比 fast 有可测量增益，且不过度升级。

不达标时：

- 保留 `deep` 作为可选能力；
- 默认使用 `fast`；需要多轮时显式选择 `auto` / `deep`。

---

## 11. 实施顺序与验收

| 步骤 | 内容 | 验收 |
|---|---|---|
| 1 | `routing.py` + `test_routing.py` | 纯函数测试全过 |
| 2 | `retrieval.py` + `test_retrieval.py` | fake 测试全过，不依赖 DB/LLM |
| 3 | planner/critic 提示词 + 真实调用 | JSON 解析失败可降级 |
| 4 | `tools.py` / `agent.py` / `api.py` / `schemas.py` | fast/deep/auto API 可用 |
| 5 | `frontend/streaming.py` / `app.py` | 三档选择与 trace 展示可用 |
| 6 | 评测集 + `run_eval.py` | 分类型指标与 A/B 报告 |
| 7 | 评审 + 灰度 | 门槛通过，可回滚 |

每个步骤完成后提供：

- 改动文件列表；
- 测试命令与结果；
- 一个可复现的请求示例；
- 已知问题。

---

## 12. 风险与回滚

| 风险 | 影响 | 缓解 |
|---|---|---|
| planner/critic JSON 不稳定 | 多轮失败 | `json_repair` + 一次重试 + 降级 |
| deep 延迟过高 | 体验变差 | 2 轮上限、12s 预算、统一 rerank |
| critic 误判充分 | 答案缺证据 | 评测集、trace 记录、人工抽检 |
| 模型选错路径 | fast 被升级 | 单工具内部模式，代码强制 |
| source 编号重复 | 引用错位 | 统一 rerank 后一次性编号 |
| 旧前端不兼容 | 页面异常 | trace 可选，旧字段不变 |
| 评测标签不准 | 指标失真 | parent_id 标签 + 人工抽检 |

回滚：

- `AGENTIC_ENABLED=false` 直接回 fast；
- 前端隐藏三档选择即可；
- API 保留 `retrieval_mode`，默认 fast，不影响旧客户端。

---

## 13. 待确认问题

1. 是否接受“单工具内部模式”，不新增 `deep_search_knowledge_base`？
2. `auto` 第一版只做规则 + 无结果升级，是否接受？
3. planner/critic 使用 `SUMMARIZATION_MODEL`，是否接受？
4. 默认模式已改为 `fast`；需要多轮时显式选择 `auto` / `deep`。
5. 评测集先做 24-30 条，还是直接 40-60 条？
6. 是否接受第一版只改 `app.py` / `streaming.py`，不拆 `state.py` / `components.py`？
7. `deep` 的 max_rounds 默认 2，是否接受？
8. 是否需要在阶段 1 就暴露 `use_kg` 占位参数？（建议不暴露，阶段 5 再加）

---

## 14. 参考实现顺序（建议）

```text
1. routing.py
2. test_routing.py
3. retrieval.py（注入 retriever / critic）
4. test_retrieval.py
5. config.py 提示词
6. tools.py 接入
7. schemas.py / api.py
8. agent.py trace
9. frontend/streaming.py
10. frontend/app.py
11. golden_agentic.jsonl
12. run_eval.py
13. README / .env.example
14. 评审与门禁
```

---

## 15. 完成定义（Definition of Done）

- `fast` / `auto` / `deep` 三种模式均可通过 API 调用；
- `fast` 严格一次检索，不升级；
- `deep` 最多 2 轮，预算内停止；
- trace 事件可被前端展示；
- sources 编号连续且只发一次；
- 现有单跳问答不退化；
- 新增多跳评测集并产出 A/B 报告；
- 所有新增纯函数有测试；
- 文档、`.env.example`、README 更新；
- 可通过 `AGENTIC_ENABLED=false` 回滚。
