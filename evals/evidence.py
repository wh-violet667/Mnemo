"""Evidence-span helpers shared by gold-label building and retrieval metrics.

The evaluation set is labelled at the *evidence* level, not the file level.
A retrieval hit is relevant when its text covers one of the gold evidence
quotes.  Character-bigram overlap makes the check tolerant of a chunk boundary
cutting through a sentence, while exact containment remains the fast path.

These helpers are dependency-free so they can be tested without PostgreSQL,
Milvus or an embedding API.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    """Case-fold and remove whitespace for stable comparison."""

    return _WHITESPACE.sub("", str(value or "")).casefold()


def char_ngrams(value: str, n: int = 2) -> set[str]:
    normalized = normalize_text(value)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def quote_overlap(text: str, quote: str) -> float:
    """Return how much of ``quote`` is covered by ``text`` (0..1).

    Exact containment is 1.0.  Otherwise we use the fraction of the quote's
    character bigrams that also occur in the chunk.  This is intentionally
    asymmetric: a long chunk containing one short quote should still be
    relevant, while a chunk with only a few common bigrams should not be.
    """

    normalized_text = normalize_text(text)
    normalized_quote = normalize_text(quote)
    if not normalized_text or not normalized_quote:
        return 0.0
    if normalized_quote in normalized_text:
        return 1.0
    if len(normalized_quote) < 6:
        return 0.0

    quote_grams = char_ngrams(normalized_quote)
    text_grams = char_ngrams(normalized_text)
    if not quote_grams:
        return 0.0
    return len(quote_grams & text_grams) / len(quote_grams)


def file_matches(document_file: Any, evidence_file: Any) -> bool:
    """Match ``01-overview.docx`` against a stored file name."""

    document_name = str(document_file or "")
    evidence_name = Path(str(evidence_file or "")).name
    if not document_name or not evidence_name:
        return False
    if document_name == evidence_name:
        return True
    return Path(evidence_name).stem in Path(document_name).stem


def document_evidence_score(
    document: Mapping[str, Any],
    row: Mapping[str, Any],
) -> float:
    """Best evidence overlap for one retrieved document."""

    evidence = row.get("gold_evidence") or []
    if not evidence:
        return 0.0

    text = str(document.get("text") or "")
    file_name = document.get("file_name")
    best = 0.0
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        if not file_matches(file_name, item.get("file")):
            continue
        score = quote_overlap(text, str(item.get("quote") or ""))
        best = max(best, score)
    return best


def is_document_relevant(
    document: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    threshold: float = 0.55,
) -> bool:
    """Evidence-level relevance, with the old document-level labels as fallback."""

    if row.get("gold_evidence"):
        return document_evidence_score(document, row) >= threshold

    expected_ids = [str(pid) for pid in (row.get("expected_parent_ids") or [])]
    if expected_ids:
        return str(document.get("parent_id")) in set(expected_ids)

    document_file = str(document.get("file_name") or "")
    expected_files = [
        str(name).strip()
        for name in (row.get("expected_files") or [])
        if str(name).strip()
    ]
    if expected_files:
        return any(Path(name).stem in Path(document_file).stem for name in expected_files)

    phrase = " ".join(str(row.get("expected_phrase") or "").split())
    file_name = str(row.get("expected_file") or "").strip()
    text = " ".join(str(document.get("text") or "").split())
    if phrase and phrase in text:
        return True
    if file_name and file_name in document_file:
        return True
    return False


def best_evidence_overlap(
    documents: Sequence[Mapping[str, Any]], row: Mapping[str, Any]
) -> float:
    return max(
        (document_evidence_score(document, row) for document in documents),
        default=0.0,
    )


def _sentence_start(text: str, position: int, lower_bound: int) -> int:
    candidates = [
        text.rfind(mark, lower_bound, position)
        for mark in ("\n", "。", "！", "？", "；", ".", "!", "?")
    ]
    best = max(candidates)
    return best + 1 if best >= lower_bound else lower_bound


def _sentence_end(text: str, position: int, upper_bound: int) -> int:
    candidates = [
        text.find(mark, position, upper_bound)
        for mark in ("\n", "。", "！", "？", "；", ".", "!", "?")
    ]
    present = [value for value in candidates if value >= 0]
    return min(present) + 1 if present else upper_bound


def find_quote_span(
    text: str,
    needle: str,
    *,
    window: int = 90,
    max_quote_chars: int = 320,
) -> tuple[str, int, int] | None:
    """Find ``needle`` and return a sentence-like quote plus its offsets.

    Offsets refer to the original ``text`` string.  ``None`` means the needle
    was not found.
    """

    if not text or not needle:
        return None

    start = text.casefold().find(needle.casefold())
    if start < 0:
        normalized_text = normalize_text(text)
        normalized_needle = normalize_text(needle)
        compact_start = normalized_text.find(normalized_needle)
        if compact_start < 0:
            return None
        # Best-effort mapping for whitespace-only differences: walk the original
        # text and count non-whitespace characters.
        seen = 0
        start = 0
        for index, character in enumerate(text):
            if not character.isspace():
                if seen == compact_start:
                    start = index
                    break
                seen += 1
        else:
            return None

    end = start + len(needle)
    lower = max(0, start - window)
    upper = min(len(text), end + window)
    quote_start = _sentence_start(text, start, lower)
    quote_end = _sentence_end(text, end, upper)

    # Table cells and headings often end immediately after a short value
    # ("石立太一").  Expand those to a readable context window.
    if quote_end - quote_start < 40:
        quote_start = max(0, start - window)
        quote_end = min(len(text), end + window)

    if quote_end - quote_start > max_quote_chars:
        quote_start = max(quote_start, start - window // 2)
        quote_end = min(quote_end, end + window // 2)
    quote = text[quote_start:quote_end].strip()
    if not quote:
        return None
    # Keep offsets aligned with the stripped quote.
    leading = len(text[quote_start:quote_end]) - len(
        text[quote_start:quote_end].lstrip()
    )
    return quote, quote_start + leading, quote_end


def best_paragraph_match(
    text: str,
    query: str,
    *,
    min_score: float = 0.42,
    max_quote_chars: int = 420,
) -> tuple[str, int, int, float] | None:
    """Return the paragraph most overlapping with ``query``.

    Exact-keyword lookup fails when the answer is a paraphrase; paragraph
    overlap still finds the source passage that supports it.
    """

    if not text or not query:
        return None

    best: tuple[str, int, int, float] | None = None
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    for paragraph in paragraphs:
        score = quote_overlap(paragraph, query)
        if score < min_score:
            continue
        if best is None or score > best[3]:
            start = text.find(paragraph)
            if start < 0:
                continue
            quote = paragraph[:max_quote_chars].strip()
            best = (quote, start, start + len(quote), score)
    return best


def dedupe_evidence(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop evidence entries whose quotes substantially overlap."""

    result: list[dict[str, Any]] = []
    for item in items:
        quote = str(item.get("quote") or "").strip()
        if not quote:
            continue
        if any(
            str(existing.get("file")) == str(item.get("file"))
            and (
                quote_overlap(str(existing.get("quote") or ""), quote) >= 0.85
                or quote_overlap(quote, str(existing.get("quote") or "")) >= 0.85
            )
            for existing in result
        ):
            continue
        result.append(dict(item))
    return result
