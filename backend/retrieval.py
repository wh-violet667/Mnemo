"""fast / deep / auto 检索管线（Agentic-lite）。

设计约束：
- 不依赖 ContextSchema，只接收显式参数；工具层负责写入 runtime.context.sources；
- 检索依赖（retriever / load_documents / critic / planner / rerank）全部可注入，
  默认实现 lazy 加载 Milvus / PostgreSQL / LLM，单元测试用 fake 注入；
- deep 模式的多轮反思在管线内部完成，对 Agent 而言仍是一次工具调用。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from citations import build_numbered_context, dedupe_documents, order_by_rerank
from routing import (
    build_initial_queries,
    build_refined_query,
    merge_parent_ids,
    needs_deep,
    stop_reason,
)

logger = logging.getLogger(__name__)

# 注入签名（一律用关键字参数调用）：
# retriever(query=) -> list[str]（parent_id 列表）
# load_documents(parent_ids=) -> list[dict]
# critic(query=, documents=) -> dict | None（None 表示失败，已内部降级）
# planner(query=) -> list[str]（失败返回 []，调用方降级 build_initial_queries）
# rerank(query=, documents=) -> list[dict]
Retriever = Callable[..., Awaitable[list[str]]]
LoadDocuments = Callable[..., Awaitable[list[dict]]]
Critic = Callable[..., Awaitable[Optional[dict]]]
Planner = Callable[..., Awaitable[list[str]]]
Rerank = Callable[..., Awaitable[list[dict]]]

_VALID_MODES = ("auto", "fast", "deep")
_CRITIC_MAX_DOCS = 6
_CRITIC_MAX_CHARS = 800


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
    sufficient: Optional[bool]


def load_agentic_config() -> dict:
    """读取 Agentic 环境变量并做边界裁剪（见文档 §7）。"""

    def _int_env(name: str, default: int, low: int, high: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return min(max(value, low), high)

    enabled = os.getenv("AGENTIC_ENABLED", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )
    default_mode = os.getenv("AGENTIC_DEFAULT_MODE", "fast").strip().lower()
    if default_mode not in _VALID_MODES:
        default_mode = "fast"
    return {
        "enabled": enabled,
        "default_mode": default_mode,
        "max_rounds": _int_env("AGENTIC_MAX_ROUNDS", 2, 1, 3),
        "budget_ms": _int_env("AGENTIC_TIMEOUT_MS", 12000, 3000, 20000),
        "max_queries": _int_env("AGENTIC_MAX_QUERIES", 3, 1, 3),
    }


# ---------------------------------------------------------------- 默认依赖


def _default_retriever(user_id: int, knowledge_base_id: str, top_k: int) -> Retriever:
    async def _retrieve(query: str) -> list[str]:
        from milvus_client import get_milvus_client

        client = await get_milvus_client()
        return await client.hybrid_retrieval_knowledge_base(
            query=query,
            knowledge_base_id=knowledge_base_id,
            top_k=top_k * 3,
            user_id=user_id,
        )

    return _retrieve


def _default_load_documents(user_id: int, knowledge_base_id: str) -> LoadDocuments:
    async def _load(parent_ids: list[str]) -> list[dict]:
        from postgresql_client import get_postgresql_client

        client = await get_postgresql_client()
        return await client.get_parents(
            knowledge_base_id=knowledge_base_id,
            parent_ids=parent_ids,
            user_id=user_id,
        )

    return _load


def _default_rerank(top_k: int) -> Rerank:
    async def _rerank(query: str, documents: list[dict]) -> list[dict]:
        from documents_process import rerank_documents

        try:
            result = await rerank_documents(
                query, [str(doc.get("text") or "") for doc in documents], top_k
            )
        except Exception as exc:  # rerank 失败保持 RRF 顺序，不阻断回答
            logger.warning(f"rerank 失败，保持召回顺序: {exc}")
            return [dict(doc) for doc in documents]
        return order_by_rerank(documents, result)

    return _rerank


_critic_model = None


def _get_structured_model(cache_name: str):
    """planner / critic 共用的低成本模型（SUMMARIZATION_MODEL）。"""
    existing = globals()[cache_name]
    if existing is not None:
        return existing
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=os.getenv("SUMMARIZATION_MODEL", "qwen-turbo"),
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        temperature=0.1,
    )
    globals()[cache_name] = model
    return model


def _truncate_evidence(documents: list[dict]) -> list[dict]:
    """critic 输入只放截断后的证据摘要，避免 token 膨胀。"""
    evidence = []
    for doc in documents[:_CRITIC_MAX_DOCS]:
        evidence.append(
            {
                "file_name": doc.get("file_name") or "未知文件",
                "text": str(doc.get("text") or "")[:_CRITIC_MAX_CHARS],
            }
        )
    return evidence


def _parse_json_object(raw: str) -> Optional[dict]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        from json_repair import repair_json

        value = repair_json(raw, return_objects=True)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


async def _call_structured_model(prompt: str, payload: dict, timeout_s: float) -> Optional[dict]:
    """调用结构化模型；JSON 解析失败先 json_repair，再重试一次，仍失败返回 None。"""
    import config

    model = _get_structured_model("_critic_model")
    for attempt in range(2):
        suffix = "" if attempt == 0 else "\n请只输出合法 JSON 对象，不要输出任何其他内容。"
        try:
            response = await asyncio.wait_for(
                model.ainvoke(
                    [
                        {"role": "system", "content": prompt + suffix},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ]
                ),
                timeout=timeout_s,
            )
        except Exception as exc:
            logger.warning(f"结构化模型调用失败（第 {attempt + 1} 次）: {exc}")
            continue
        content = getattr(response, "content", "")
        parsed = _parse_json_object(content if isinstance(content, str) else str(content))
        if parsed is not None:
            return parsed
        logger.warning(f"结构化输出 JSON 解析失败（第 {attempt + 1} 次）")
    return None


def _default_critic() -> Critic:
    async def _critic(query: str, documents: list[dict]) -> Optional[dict]:
        import config

        prompt = getattr(config, "AGENTIC_CRITIC_PROMPT")
        result = await _call_structured_model(
            prompt,
            {"question": query, "evidence": _truncate_evidence(documents)},
            timeout_s=8.0,
        )
        if result is None:
            return None
        # 只保留约定字段，sufficient 强制布尔化
        return {
            "sufficient": bool(result.get("sufficient")),
            "missing": [str(m) for m in (result.get("missing") or []) if str(m).strip()],
            "rewrite": str(result.get("rewrite") or ""),
            "reason": str(result.get("reason") or ""),
        }

    return _critic


def _default_planner() -> Planner:
    async def _plan(query: str) -> list[str]:
        import config

        prompt = getattr(config, "AGENTIC_PLAN_PROMPT")
        result = await _call_structured_model(
            prompt, {"question": query}, timeout_s=8.0
        )
        if not result:
            return []
        queries = [
            str(q).strip()
            for q in (result.get("queries") or [])
            if str(q).strip()
        ]
        return queries

    return _plan


# ---------------------------------------------------------------- 主流程


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _critic_status(critic_result: Optional[dict]) -> str:
    if critic_result is None:
        return "failed"
    return "sufficient" if critic_result.get("sufficient") else "insufficient"


async def _finalize(
    query: str,
    documents: list[dict],
    rerank: Rerank,
    top_k: int,
    start_index: int,
) -> tuple[list[dict], str, list[dict]]:
    """统一 rerank 一次、去重、截取 top_k、编号。"""
    if not documents:
        return [], "", []
    ranked = await rerank(query=query, documents=documents)
    ranked = dedupe_documents(ranked)[:top_k]
    snippets, sources = build_numbered_context(ranked, start_index=start_index)
    return ranked, snippets, sources


async def _run_fast(
    query: str,
    *,
    retriever: Retriever,
    load_documents: LoadDocuments,
) -> tuple[list[dict], list[str], int]:
    """一次 hybrid 召回 + 回表。返回 (documents, parent_ids, 实际检索轮数)。"""
    parent_ids = await retriever(query=query) or []
    if not parent_ids:
        return [], [], 1
    documents = await load_documents(parent_ids=parent_ids)
    return documents, parent_ids, 1


async def _run_deep(
    query: str,
    *,
    max_rounds: int,
    max_queries: int,
    budget_ms: int,
    started: float,
    retriever: Retriever,
    load_documents: LoadDocuments,
    critic: Critic,
    planner: Planner,
) -> dict:
    """Plan → Retrieve → Critique → Refine，最多 max_rounds 轮。"""
    try:
        planned = await planner(query=query)
    except Exception as exc:
        logger.warning(f"planner 失败，降级规则生成查询: {exc}")
        planned = []
    queries = [q for q in (planned or []) if q][:max_queries]
    if not queries:
        queries = build_initial_queries(query)[:max_queries]
    if not queries:
        queries = [query]

    queries_used: list[str] = list(queries)
    collected: list[str] = []
    doc_cache: dict[str, dict] = {}
    critic_result: Optional[dict] = None
    rounds = 0
    stop = None

    for iteration in range(1, max_rounds + 1):
        # 预算检查只对第 2 轮及以后生效：无论 planner 多慢，
        # 第 1 轮检索必须执行，否则 deep 会退回零证据（比 fast 还差）。
        if iteration > 1 and _elapsed_ms(started) >= budget_ms:
            stop = "budget_exhausted"
            break

        results = await asyncio.gather(
            *(retriever(query=q) for q in queries), return_exceptions=True
        )
        round_id_lists: list[list[str]] = []
        for q, result in zip(queries, results):
            if isinstance(result, Exception):
                logger.warning(f"检索子查询失败 {q!r}: {result}")
                continue
            round_id_lists.append(result or [])

        merged = merge_parent_ids([collected] + round_id_lists)
        new_count = len(merged) - len(collected)
        rounds = iteration

        if new_count == 0:
            stop = stop_reason(
                iteration=iteration,
                max_iterations=max_rounds,
                new_parent_count=0,
                critic=None,
                elapsed_ms=_elapsed_ms(started),
                budget_ms=budget_ms,
            )
            break

        collected = merged
        documents = await load_documents(parent_ids=collected)
        for doc in documents:
            pid = doc.get("parent_id")
            if pid is not None:
                doc_cache[pid] = doc
        ordered_documents = [doc_cache[pid] for pid in collected if pid in doc_cache]

        critic_result = await critic(query=query, documents=ordered_documents)

        reason = stop_reason(
            iteration=iteration,
            max_iterations=max_rounds,
            new_parent_count=new_count,
            critic=critic_result,
            elapsed_ms=_elapsed_ms(started),
            budget_ms=budget_ms,
        )
        if reason:
            stop = reason
            break

        refined = build_refined_query(query, (critic_result or {}).get("missing") or [])
        if not refined:
            stop = "empty_refined_query"
            break
        queries = [refined]
        queries_used.append(refined)

    final_documents = [doc_cache[pid] for pid in collected if pid in doc_cache]
    return {
        "documents": final_documents,
        "rounds": rounds,
        "queries": queries_used,
        "critic_result": critic_result,
        "critic_status": _critic_status(critic_result) if rounds else "skipped",
        "stop_reason": stop,
    }


async def run_retrieval(
    query: str,
    *,
    user_id: int,
    knowledge_base_id: str,
    top_k: int,
    mode: str = "fast",
    start_index: int = 1,
    max_rounds: int = 2,
    max_queries: int = 3,
    budget_ms: int = 12000,
    enabled: bool = True,
    retriever: Optional[Retriever] = None,
    load_documents: Optional[LoadDocuments] = None,
    critic: Optional[Critic] = None,
    planner: Optional[Planner] = None,
    rerank: Optional[Rerank] = None,
) -> RetrievalResult:
    started = time.monotonic()

    retriever = retriever or _default_retriever(user_id, knowledge_base_id, top_k)
    load_documents = load_documents or _default_load_documents(user_id, knowledge_base_id)
    rerank = rerank or _default_rerank(top_k)

    requested = mode if mode in _VALID_MODES else "fast"
    degraded = False
    if not enabled and requested in ("deep", "auto"):
        executed = "fast"
        degraded = True
    elif requested == "auto":
        executed = "deep" if needs_deep(query) else "fast"
    else:
        executed = requested

    escalated = False
    sufficient: Optional[bool] = None
    critic_status = "skipped"

    if executed == "fast":
        documents, parent_ids, rounds = await _run_fast(
            query, retriever=retriever, load_documents=load_documents
        )
        queries_used = [query]
        # auto 且 fast 返回 0 个父块时，本次请求内升级一次 deep
        if requested == "auto" and enabled and not parent_ids:
            escalated = True
            executed = "deep"
            critic = critic or _default_critic()
            planner = planner or _default_planner()
            deep_result = await _run_deep(
                query,
                max_rounds=max_rounds,
                max_queries=max_queries,
                budget_ms=budget_ms,
                started=started,
                retriever=retriever,
                load_documents=load_documents,
                critic=critic,
                planner=planner,
            )
            documents = deep_result["documents"]
            rounds = deep_result["rounds"]
            queries_used = deep_result["queries"]
            critic_status = deep_result["critic_status"]
            critic_result = deep_result["critic_result"]
            sufficient = (
                bool(critic_result.get("sufficient"))
                if isinstance(critic_result, dict)
                else None
            )
    else:
        critic = critic or _default_critic()
        planner = planner or _default_planner()
        deep_result = await _run_deep(
            query,
            max_rounds=max_rounds,
            max_queries=max_queries,
            budget_ms=budget_ms,
            started=started,
            retriever=retriever,
            load_documents=load_documents,
            critic=critic,
            planner=planner,
        )
        documents = deep_result["documents"]
        rounds = deep_result["rounds"]
        queries_used = deep_result["queries"]
        critic_status = deep_result["critic_status"]
        critic_result = deep_result["critic_result"]
        sufficient = (
            bool(critic_result.get("sufficient"))
            if isinstance(critic_result, dict)
            else None
        )

    final_documents, snippets, sources = await _finalize(
        query, documents, rerank, top_k, start_index
    )

    trace = RetrievalTrace(
        mode=executed,
        rounds=rounds,
        queries=queries_used,
        parent_count=len(final_documents),
        critic=critic_status,
        elapsed_ms=_elapsed_ms(started),
        escalated=escalated,
        degraded=degraded,
    )
    return RetrievalResult(
        documents=final_documents,
        snippets=snippets,
        sources=sources,
        trace=trace,
        sufficient=sufficient,
    )
