"""Deterministic metrics for the violet answer evaluation.

The LLM judges live in ``evals/judges.py``.  This module only contains
pure functions so the scoring logic can be tested without network access.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


_PUNCTUATION = "，。；：、（）()【】[]“”\"'‘’!?！？,;:"


def normalize_text(text: str) -> str:
    """Normalise Chinese/English text for cheap substring comparisons."""

    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = value.translate({ord(ch): None for ch in _PUNCTUATION})
    return re.sub(r"\s+", "", value)


def unique(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def match_required_facts(
    answer: str, required_facts: Optional[Sequence[str]]
) -> Dict[str, Any]:
    """Match facts as normalised substrings; the judge handles paraphrases."""

    facts = [str(fact).strip() for fact in (required_facts or []) if str(fact).strip()]
    if not facts:
        return {"hits": [], "missing": [], "coverage": None}

    normalised_answer = normalize_text(answer)
    hits: List[str] = []
    missing: List[str] = []
    for fact in facts:
        if normalize_text(fact) in normalised_answer:
            hits.append(fact)
        else:
            missing.append(fact)

    return {
        "hits": hits,
        "missing": missing,
        "coverage": len(hits) / len(facts),
    }


def dedupe_documents(documents: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for document in documents or []:
        key = str(
            document.get("parent_id")
            or document.get("file_name")
            or document.get("text")
            or id(document)
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(document)
    return result


def _expected_files(row: Dict[str, Any]) -> List[str]:
    return [
        str(name).strip()
        for name in (row.get("expected_files") or [])
        if str(name).strip()
    ]


def _expected_parent_ids(row: Dict[str, Any]) -> List[str]:
    return [
        str(parent_id).strip()
        for parent_id in (row.get("expected_parent_ids") or [])
        if str(parent_id).strip()
    ]


def document_is_relevant(document: Dict[str, Any], row: Dict[str, Any]) -> bool:
    """Same document-level relevance rule used by ``run_eval.py``."""

    expected_ids = _expected_parent_ids(row)
    if expected_ids:
        return str(document.get("parent_id")) in set(expected_ids)

    doc_file = str(document.get("file_name") or "")
    expected_files = _expected_files(row)
    if expected_files:
        return any(
            Path(name).stem.lower() in doc_file.lower() for name in expected_files
        )

    phrase = " ".join(str(row.get("expected_phrase") or "").split())
    file_name = str(row.get("expected_file") or "").strip()
    text = " ".join(str(document.get("text") or "").split())
    if phrase and phrase in text:
        return True
    if file_name and file_name in doc_file:
        return True
    return False


def expected_context_coverage(
    row: Dict[str, Any], documents: Sequence[Dict[str, Any]]
) -> Optional[float]:
    """Coverage of expected parent IDs or expected files in retrieved docs."""

    expected_ids = _expected_parent_ids(row)
    if expected_ids:
        retrieved = {str(document.get("parent_id")) for document in documents}
        hits = sum(1 for parent_id in expected_ids if parent_id in retrieved)
        return hits / len(expected_ids)

    expected_files = unique(_expected_files(row))
    if not expected_files:
        return None

    retrieved_files = [
        str(document.get("file_name") or "").lower() for document in documents
    ]
    hits = [
        name
        for name in expected_files
        if any(Path(name).stem.lower() in file_name for file_name in retrieved_files)
    ]
    return len(hits) / len(expected_files)


def compute_retrieval_metrics(
    row: Dict[str, Any], documents: Optional[Sequence[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Deterministic retrieval metrics for one question.

    ``hit_at_5`` and MRR keep the existing document-level definition.  The
    additional coverage metric exposes multi-hop questions where only one of
    several expected documents was retrieved.
    """

    docs = dedupe_documents(documents)
    expected_ids = _expected_parent_ids(row)
    expected_files = unique(_expected_files(row))
    has_expected = bool(expected_ids or expected_files)
    relevances = [document_is_relevant(doc, row) for doc in docs] if has_expected else []

    hit_at_5: Optional[float] = None
    mrr: Optional[float] = None
    context_precision_parent: Optional[float] = None
    context_precision_file: Optional[float] = None
    if has_expected:
        hit_at_5 = 1.0 if any(relevances) else 0.0
        mrr = 0.0
        for rank, relevant in enumerate(relevances, start=1):
            if relevant:
                mrr = 1.0 / rank
                break
        if docs:
            context_precision_parent = sum(1 for value in relevances if value) / len(docs)

        retrieved_files = unique(
            str(document.get("file_name") or "") for document in docs
        )
        if retrieved_files:
            relevant_files = 0
            for file_name in retrieved_files:
                synthetic_doc = {"file_name": file_name}
                if document_is_relevant(synthetic_doc, row):
                    relevant_files += 1
            context_precision_file = relevant_files / len(retrieved_files)

    context_precision = (
        context_precision_parent
        if expected_ids
        else (context_precision_file if expected_files else context_precision_parent)
    )

    coverage = expected_context_coverage(row, docs)
    if coverage is None:
        all_evidence_hit = None
    else:
        all_evidence_hit = 1.0 if coverage >= 1.0 else 0.0

    return {
        "hit_at_5": hit_at_5,
        "mrr": mrr,
        "expected_context_coverage": coverage,
        "all_evidence_hit": all_evidence_hit,
        "context_precision": context_precision,
        "context_precision_parent": context_precision_parent,
        "context_precision_file": context_precision_file,
        "retrieved_files": [doc.get("file_name") for doc in docs],
        "retrieved_parent_ids": [doc.get("parent_id") for doc in docs],
    }


def extract_citation_marks(answer: str) -> List[int]:
    """Extract ``[1]``, ``[1][3]`` and ``[1,3]`` style marks."""

    marks: List[int] = []
    for group in re.findall(r"\[([0-9,\s]+)\]", str(answer or "")):
        for number in re.findall(r"\d+", group):
            marks.append(int(number))
    return marks


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def citation_metrics(
    answer: str,
    cited_sources: Optional[Sequence[Dict[str, Any]]],
    judge_b: Optional[Dict[str, Any]],
    source_texts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Compute citation validity, precision, coverage and chunk support."""

    marks = extract_citation_marks(answer)
    source_by_index: Dict[int, Dict[str, Any]] = {}
    for source in cited_sources or []:
        index = _as_int(source.get("index"))
        if index >= 0:
            source_by_index[index] = source

    judge_citations: Dict[int, Dict[str, Any]] = {}
    for citation in ((judge_b or {}).get("citations") or []):
        mark = _as_int(citation.get("mark"))
        if mark >= 0:
            judge_citations[mark] = citation

    valid_marks = sum(1 for mark in marks if mark in source_by_index)
    supporting_marks = 0
    chunk_supported_marks = 0

    for mark in marks:
        citation = judge_citations.get(mark)
        if not citation or citation.get("supports_claim") is not True:
            continue
        if citation.get("valid") is False:
            continue
        supporting_marks += 1

        source = source_by_index.get(mark) or {}
        parent_id = str(source.get("parent_id") or "")
        text = str((source_texts or {}).get(parent_id) or source.get("snippet") or "")
        quote = normalize_text(str(citation.get("quote") or ""))
        if quote and quote in normalize_text(text):
            chunk_supported_marks += 1

    claims = (judge_b or {}).get("claims") or []
    claims_needing = [claim for claim in claims if claim.get("needs_citation")]
    claims_covered = 0
    for claim in claims_needing:
        cited_marks = [_as_int(mark) for mark in (claim.get("cited_marks") or [])]
        if any(
            mark in judge_citations
            and judge_citations[mark].get("supports_claim") is True
            for mark in cited_marks
            if mark >= 0
        ):
            claims_covered += 1

    return {
        "marks": marks,
        "total_marks": len(marks),
        "valid_marks": valid_marks,
        "supporting_marks": supporting_marks,
        "claims_needing_citation": len(claims_needing),
        "claims_covered": claims_covered,
        "chunk_supported_marks": chunk_supported_marks,
        "citation_validity": valid_marks / len(marks) if marks else None,
        "citation_precision": (
            supporting_marks / len(marks)
            if marks and judge_b is not None
            else None
        ),
        "citation_coverage": (
            claims_covered / len(claims_needing)
            if claims_needing and judge_b is not None
            else None
        ),
        "chunk_support_rate": (
            chunk_supported_marks / supporting_marks
            if supporting_marks and judge_b is not None
            else None
        ),
    }


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _percentile(values: Sequence[float], p: float) -> Optional[float]:
    numbers = sorted(float(value) for value in values if value is not None)
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0]
    position = (len(numbers) - 1) * (p / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(numbers) - 1)
    weight = position - lower
    return numbers[lower] * (1 - weight) + numbers[upper] * weight


def _judge_score(record: Dict[str, Any]) -> Optional[int]:
    judge = record.get("judge_a") or {}
    score = judge.get("score")
    return _as_int(score) if score is not None else None


def _group_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    answerable = [r for r in records if r.get("query_type") != "unanswerable"]
    unanswerable = [r for r in records if r.get("query_type") == "unanswerable"]
    scored_answerable = [r for r in answerable if _judge_score(r) is not None]
    scored_unanswerable = [r for r in unanswerable if _judge_score(r) is not None]

    scores = [_judge_score(r) for r in scored_answerable]
    fully_correct = sum(1 for score in scores if score == 2)
    mean_correctness = _mean(score / 2 for score in scores if score is not None)

    fact_coverages = [
        r["fact_check"]["coverage"]
        for r in scored_answerable
        if (r.get("fact_check") or {}).get("coverage") is not None
    ]
    all_facts = sum(1 for coverage in fact_coverages if coverage >= 1.0)

    claims = [
        claim
        for record in answerable
        if not bool((record.get("judge_a") or {}).get("refused"))
        for claim in ((record.get("judge_b") or {}).get("claims") or [])
    ]
    supported_claims = sum(1 for claim in claims if claim.get("verdict") == "supported")
    partial_claims = sum(1 for claim in claims if claim.get("verdict") == "partial")
    total_claims = len(claims)

    citation_records = [
        record.get("citation_metrics") or {}
        for record in records
        if record.get("judge_b") is not None
    ]
    total_marks = sum(int(item.get("total_marks") or 0) for item in citation_records)
    valid_marks = sum(int(item.get("valid_marks") or 0) for item in citation_records)
    supporting_marks = sum(
        int(item.get("supporting_marks") or 0) for item in citation_records
    )
    claims_needing = sum(
        int(item.get("claims_needing_citation") or 0) for item in citation_records
    )
    claims_covered = sum(
        int(item.get("claims_covered") or 0) for item in citation_records
    )
    chunk_supported = sum(
        int(item.get("chunk_supported_marks") or 0) for item in citation_records
    )

    correct_refusals = sum(
        1 for record in scored_unanswerable if _judge_score(record) == 2
    )
    false_refusals = sum(
        1
        for record in scored_answerable
        if bool((record.get("judge_a") or {}).get("refused"))
    )

    retrieval_metrics = [
        record.get("retrieval_metrics") or {} for record in scored_answerable
    ]
    hit_records = [
        record
        for record in scored_answerable
        if (record.get("retrieval_metrics") or {}).get("hit_at_5") == 1.0
    ]
    miss_records = [
        record
        for record in scored_answerable
        if (record.get("retrieval_metrics") or {}).get("hit_at_5") == 0.0
    ]
    correctness_given_hit = (
        sum(1 for record in hit_records if _judge_score(record) == 2)
        / len(hit_records)
        if hit_records
        else None
    )
    correctness_given_miss = (
        sum(1 for record in miss_records if _judge_score(record) == 2)
        / len(miss_records)
        if miss_records
        else None
    )
    latencies = [
        record.get("latency_ms")
        for record in records
        if record.get("latency_ms") is not None
    ]
    first_tokens = [
        record.get("first_token_ms")
        for record in records
        if record.get("first_token_ms") is not None
    ]

    return {
        "n_total": len(records),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_answerable_judged": len(scored_answerable),
        "n_unanswerable_judged": len(scored_unanswerable),
        "n_errors": sum(1 for record in records if record.get("errors")),
        "fully_correct_rate": (
            fully_correct / len(scored_answerable) if scored_answerable else None
        ),
        "mean_correctness": mean_correctness,
        "fact_coverage_mean": _mean(fact_coverages),
        "all_facts_rate": all_facts / len(fact_coverages) if fact_coverages else None,
        "faithfulness": supported_claims / total_claims if total_claims else None,
        "faithfulness_weighted": (
            (supported_claims + 0.5 * partial_claims) / total_claims
            if total_claims
            else None
        ),
        "unsupported_claim_rate": (
            1 - (supported_claims / total_claims) if total_claims else None
        ),
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "partial_claims": partial_claims,
        "citation_validity": valid_marks / total_marks if total_marks else None,
        "citation_precision": (
            supporting_marks / total_marks if total_marks else None
        ),
        "citation_coverage": (
            claims_covered / claims_needing if claims_needing else None
        ),
        "chunk_support_rate": (
            chunk_supported / supporting_marks if supporting_marks else None
        ),
        "total_citation_marks": total_marks,
        "supporting_citation_marks": supporting_marks,
        "claims_needing_citation": claims_needing,
        "claims_covered": claims_covered,
        "refusal_accuracy": (
            correct_refusals / len(scored_unanswerable)
            if scored_unanswerable
            else None
        ),
        "false_refusal_rate": (
            false_refusals / len(scored_answerable) if scored_answerable else None
        ),
        "hit_at_5": _mean(item.get("hit_at_5") for item in retrieval_metrics),
        "n_hit": len(hit_records),
        "n_miss": len(miss_records),
        "correctness_given_hit": correctness_given_hit,
        "correctness_given_miss": correctness_given_miss,
        "mrr": _mean(item.get("mrr") for item in retrieval_metrics),
        "expected_context_coverage": _mean(
            item.get("expected_context_coverage") for item in retrieval_metrics
        ),
        "all_evidence_hit": _mean(
            item.get("all_evidence_hit") for item in retrieval_metrics
        ),
        "context_precision": _mean(
            item.get("context_precision") for item in retrieval_metrics
        ),
        "context_precision_file": _mean(
            item.get("context_precision_file") for item in retrieval_metrics
        ),
        "context_precision_parent": _mean(
            item.get("context_precision_parent") for item in retrieval_metrics
        ),
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "p50_first_token_ms": _percentile(first_tokens, 50),
        "p95_first_token_ms": _percentile(first_tokens, 95),
        "memory_used_rate": _mean(
            1.0 if record.get("memory_used") else 0.0 for record in records
        ),
    }


def aggregate_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate overall and per-query-type metrics."""

    by_type: Dict[str, Dict[str, Any]] = {}
    types = sorted({str(record.get("query_type") or "unknown") for record in records})
    for query_type in types:
        by_type[query_type] = _group_metrics(
            [record for record in records if record.get("query_type") == query_type]
        )
    return {"overall": _group_metrics(records), "by_query_type": by_type}


def record_faithfulness(judge_b: Optional[Dict[str, Any]]) -> Optional[float]:
    claims = (judge_b or {}).get("claims") or []
    if not claims:
        return None
    supported = sum(1 for claim in claims if claim.get("verdict") == "supported")
    return supported / len(claims)
