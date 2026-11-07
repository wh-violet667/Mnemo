"""引用编号与来源格式化。

这里只放纯函数，不依赖数据库或 Web 框架，方便单元测试与复用。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def dedupe_documents(
    documents: Iterable[Dict[str, Any]], key: str = "parent_id"
) -> List[Dict[str, Any]]:
    """按 key 保序去重。

    检索链路里用 ``list(set(...))`` 去重会把 rerank/RRF 的顺序打乱，
    这里统一改成保序去重，保证“第 1 条”就是最相关的那条。
    """

    seen = set()
    result: List[Dict[str, Any]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        identity = doc.get(key)
        if identity is None:
            identity = doc.get("text")
        if identity is None:
            identity = id(doc)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(doc)
    return result


def order_by_rerank(
    documents: Iterable[Dict[str, Any]], rerank_result: Any
) -> List[Dict[str, Any]]:
    """按 rerank 返回的 index 重排父块；失败时保持原顺序。

    rerank 的 index 是相对输入列表的下标，这里映射回带 file_name 的原始父块。
    """

    documents = list(documents)
    if not rerank_result:
        return [dict(document) for document in documents]

    ordered: List[Dict[str, Any]] = []
    for item in rerank_result.get("output", {}).get("results", []):
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(documents):
            document = dict(documents[index])
        else:
            text = (item.get("document") or {}).get("text")
            matched = next(
                (doc for doc in documents if doc.get("text") == text), None
            )
            if matched is None:
                continue
            document = dict(matched)
        document["score"] = item.get("relevance_score")
        ordered.append(document)

    return ordered or [dict(document) for document in documents]


def build_numbered_context(
    documents: Iterable[Dict[str, Any]], start_index: int = 1
) -> Tuple[str, List[Dict[str, Any]]]:
    """把父块整理成带编号的上下文，并返回前端可渲染的来源列表。

    Args:
        documents: 每项至少包含 ``text``，可选 ``file_name`` / ``parent_id`` / ``score``。
        start_index: 起始编号。一次对话里多次检索时，编号连续递增，避免 [1] 重复。

    Returns:
        (给模型看的文本, 来源列表)。没有有效文本时返回 ("", [])。
    """

    blocks: List[str] = []
    sources: List[Dict[str, Any]] = []
    index = start_index

    for doc in documents:
        text = str(doc.get("text") or "").strip()
        if not text:
            continue

        file_name = str(doc.get("file_name") or "未知文件")
        source = {
            "index": index,
            "file_name": file_name,
            "parent_id": doc.get("parent_id"),
            "snippet": text[:240],
        }
        if doc.get("score") is not None:
            source["score"] = doc["score"]

        blocks.append(f"[{index}] 来源：{file_name}\n{text}")
        sources.append(source)
        index += 1

    if not blocks:
        return "", []

    return "\n\n---\n\n".join(blocks), sources
