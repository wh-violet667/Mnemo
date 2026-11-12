"""引用编号与来源格式化。

这里只放纯函数，不依赖数据库或 Web 框架，方便单元测试与复用。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

_CITATION_MARK = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)\s+|\n+")


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


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _overlap_score(claim: str, sentence: str) -> float:
    """Character-bigram overlap; cheap and language-agnostic for zh/ja/en."""
    claim_grams = _char_ngrams(claim)
    sentence_grams = _char_ngrams(sentence)
    if not claim_grams or not sentence_grams:
        return 0.0
    return len(claim_grams & sentence_grams) / len(claim_grams)


def split_sentences(text: str) -> List[str]:
    """Split into sentences, keeping the terminator on the left sentence."""
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(str(text or "")) if part.strip()]
    return parts or ([str(text).strip()] if str(text or "").strip() else [])


def claims_by_marker(answer: str) -> Dict[int, List[str]]:
    """Map each ``[n]`` marker to the text immediately before it.

    Consecutive markers such as ``[1][2]`` share the preceding claim; a marker
    at the start of the answer has no claim and is ignored.
    """
    text = str(answer or "")
    matches = list(_CITATION_MARK.finditer(text))
    result: Dict[int, List[str]] = {}
    previous_end = 0
    last_segment = ""
    for match in matches:
        segment = text[previous_end : match.start()].strip().lstrip("。！？!?；;，,、")
        if segment:
            last_segment = segment
        if last_segment:
            result.setdefault(int(match.group(1)), []).append(last_segment)
        previous_end = match.end()
    return result


def select_supporting_quote(
    claim: str, source: str, max_chars: int = 200
) -> str:
    """Pick the source sentence with the highest overlap with the claim.

    This is deliberately a deterministic lexical match, not an LLM call: the
    quote is shown as evidence for a claim, so it must be reproducible and
    exactly present in the source. Falls back to the first sentence when the
    claim has no lexical overlap.
    """
    source_text = " ".join(str(source or "").split())
    if not source_text:
        return ""
    sentences = split_sentences(source_text)
    if not sentences:
        sentences = [source_text]
    best = max(sentences, key=lambda sentence: _overlap_score(claim, sentence))
    if _overlap_score(claim, best) <= 0.0:
        best = sentences[0]
    quote = best.strip()
    if len(quote) > max_chars:
        quote = quote[: max_chars - 1].rstrip() + "…"
    return quote


def locate_quote(source: str, quote: str) -> str:
    """Return a human locator (paragraph number) for a quote in a source."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", str(source or ""))
        if paragraph.strip()
    ]
    if not paragraphs:
        return ""
    target = _normalize_text(quote.rstrip("…"))
    if target:
        for index, paragraph in enumerate(paragraphs, 1):
            if target in _normalize_text(paragraph):
                return f"第 {index} 段"
    return "第 1 段"


def enrich_sources_with_quotes(
    answer: str,
    sources: Iterable[Dict[str, Any]],
    source_texts: Dict[str, str] | None = None,
    max_chars: int = 200,
    fallback_claim: str = "",
) -> List[Dict[str, Any]]:
    """Attach ``quote`` and ``locator`` to each source for the wire payload.

    Full source text stays in ``source_texts`` and is never copied into the
    returned dictionaries; only the supporting quote is exposed to the client.
    """
    claims = claims_by_marker(answer)
    texts = source_texts or {}
    enriched: List[Dict[str, Any]] = []
    for source in sources:
        item = dict(source)
        parent_id = str(item.get("parent_id") or "")
        source_text = str(texts.get(parent_id) or item.get("text") or item.get("snippet") or "")
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            index = -1

        best_quote = ""
        best_score = -1.0
        for claim in claims.get(index, []):
            quote = select_supporting_quote(claim, source_text, max_chars)
            score = _overlap_score(claim, quote)
            if score > best_score:
                best_quote, best_score = quote, score
        if not best_quote:
            best_quote = select_supporting_quote(fallback_claim, source_text, max_chars)

        item["quote"] = best_quote
        locator = locate_quote(source_text, best_quote)
        if locator:
            item["locator"] = locator
        item.pop("text", None)
        enriched.append(item)
    return enriched
