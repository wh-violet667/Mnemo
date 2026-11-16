"""Build evidence-level labels for the violet retrieval set.

The original golden file only knows which *documents* contain an answer
(``expected_files``).  That is too coarse to compare chunking strategies: two
different chunk layouts can both hit the same document while retrieving
different evidence.  This script finds the source passage for each required
fact and attaches it as ``gold_evidence``:

    {"file": "01-overview.docx", "quote": "...", "start": 512, "end": 688}

Offsets refer to the text produced by ``docx2txt`` (the same loader used by the
ingestion pipeline).  The quote is the primary label; offsets are kept for
debugging and future chunk-boundary analysis.

Usage:
    python evals/build_gold_evidence.py
    python evals/build_gold_evidence.py --corpus-dir data/violet-evergarden-kb
    python evals/build_gold_evidence.py --check-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import docx2txt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.evidence import (  # noqa: E402
    best_paragraph_match,
    dedupe_evidence,
    find_quote_span,
    quote_overlap,
)

DEFAULT_CORPUS = ROOT / "data" / "violet-evergarden-kb"
DEFAULT_GOLDEN = ROOT / "evals" / "golden_violet.jsonl"
DEFAULT_REPORT = ROOT / "evals" / "gold_evidence_report.json"
_SENTENCE_RE = re.compile(r"[。！？!?；;\n]+")


def resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} JSON 解析失败：{exc}") from exc
    return rows


def write_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_docx_texts(corpus_dir: Path) -> Dict[str, str]:
    docx_dir = corpus_dir / "docx"
    if not docx_dir.is_dir():
        docx_dir = corpus_dir
    if not docx_dir.is_dir():
        raise SystemExit(f"找不到语料目录：{docx_dir}")

    texts: Dict[str, str] = {}
    for path in sorted(docx_dir.glob("*.docx")):
        if path.name.lower() == "readme.docx":
            continue
        texts[path.name] = docx2txt.process(str(path))
    if not texts:
        raise SystemExit(f"{docx_dir} 中没有可用的 docx")
    return texts


def _answer_segments(answer: str) -> List[str]:
    segments = [
        segment.strip()
        for segment in _SENTENCE_RE.split(str(answer or ""))
        if len(segment.strip()) >= 4
    ]
    return segments


def _query_terms(query: str) -> List[str]:
    """Extract quoted aliases such as “维尔莉特·埃弗加登” from the question."""

    return [
        match.group(1).strip()
        for match in re.finditer(r"[“\"]([^”\"]+)[”\"]", str(query or ""))
        if len(match.group(1).strip()) >= 2
    ]


def evidence_for_row(
    row: Dict[str, Any], texts: Dict[str, str]
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Return ``(gold_evidence, missing_facts)`` for one golden row."""

    if row.get("query_type") == "unanswerable":
        return [], []

    expected_files = [
        str(name).strip()
        for name in (row.get("expected_files") or [])
        if str(name).strip()
    ]
    facts = [
        str(fact).strip()
        for fact in (row.get("required_facts") or [])
        if str(fact).strip()
    ]
    answer = str(row.get("reference_answer") or "").strip()

    evidence: List[Dict[str, Any]] = []
    missing: List[str] = []

    for file_name in expected_files:
        text = texts.get(file_name)
        if not text:
            missing.append(f"{file_name}:missing_file")
            continue

        file_evidence: List[Dict[str, Any]] = []
        for term in _query_terms(row.get("query", "")):
            span = find_quote_span(text, term)
            if span is None:
                continue
            quote, start, end = span
            file_evidence.append(
                {
                    "file": file_name,
                    "quote": quote,
                    "start": start,
                    "end": end,
                    "fact": term,
                    "match_type": "query",
                    "match_score": 0.95,
                }
            )

        for fact in facts:
            span = find_quote_span(text, fact)
            if span is None:
                continue
            quote, start, end = span
            file_evidence.append(
                {
                    "file": file_name,
                    "quote": quote,
                    "start": start,
                    "end": end,
                    "fact": fact,
                    "match_type": "fact",
                    "match_score": 0.75,
                }
            )

        # Keyword lookup fails for paraphrased answers; paragraph overlap still
        # finds the passage that supports the reference answer.
        answer_candidates = [answer, *_answer_segments(answer)] if answer else []
        for candidate in answer_candidates:
            match = best_paragraph_match(text, candidate)
            if match is not None and len(match[0]) >= 40:
                quote, start, end, score = match
            else:
                span = find_quote_span(text, candidate)
                if span is None:
                    continue
                quote, start, end = span
                score = quote_overlap(quote, candidate)
            file_evidence.append(
                {
                    "file": file_name,
                    "quote": quote,
                    "start": start,
                    "end": end,
                    "fact": candidate[:80],
                    "match_type": "answer",
                    "match_score": round(score, 4),
                }
            )

        if not file_evidence:
            missing.extend(f"{file_name}:{fact}" for fact in facts)
            continue

        # A quoted alias in the question is the most precise label; do not mix
        # in a lower-quality fact/answer paragraph from the same file.
        query_evidence = [
            item for item in file_evidence if item.get("match_type") == "query"
        ]
        if query_evidence:
            file_evidence = query_evidence

        # Keep labels compact but retain both precise facts and a supporting
        # paragraph when they are different quotes.
        ranked = sorted(
            file_evidence,
            key=lambda item: (
                item.get("match_type") == "answer",
                float(item.get("match_score", 0)),
                len(item["quote"]),
            ),
            reverse=True,
        )
        evidence.extend(dedupe_evidence(ranked)[:2])

    if evidence:
        return dedupe_evidence(evidence), missing
    return [], missing


def validate_evidence(row: Dict[str, Any], texts: Dict[str, str]) -> List[str]:
    errors: List[str] = []
    for item in row.get("gold_evidence") or []:
        if not isinstance(item, dict):
            errors.append(f"{row.get('id')}: evidence is not an object")
            continue
        file_name = str(item.get("file") or "")
        quote = str(item.get("quote") or "").strip()
        if not file_name or not quote:
            errors.append(f"{row.get('id')}: missing file or quote")
            continue
        text = texts.get(file_name)
        if text is None:
            errors.append(f"{row.get('id')}: unknown file {file_name}")
            continue
        if quote not in text:
            errors.append(f"{row.get('id')}: quote not found in {file_name}")
        start = item.get("start")
        end = item.get("end")
        if isinstance(start, int) and isinstance(end, int):
            if not (0 <= start <= end <= len(text)):
                errors.append(f"{row.get('id')}: invalid offsets for {file_name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="为 violet 评测集生成 chunk 级证据标签")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="语料目录，默认 data/violet-evergarden-kb 或 VIOLET_KB_DIR",
    )
    parser.add_argument("--output", default="", help="输出 JSONL，默认覆盖 --golden")
    parser.add_argument(
        "--report", default=str(DEFAULT_REPORT), help="标签质量报告 JSON"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="只校验已有 gold_evidence"
    )
    args = parser.parse_args()

    golden_path = resolve_path(args.golden, DEFAULT_GOLDEN)
    corpus_dir = resolve_path(
        args.corpus_dir or os.getenv("VIOLET_KB_DIR"), DEFAULT_CORPUS
    )
    texts = load_docx_texts(corpus_dir)
    rows = load_rows(golden_path)

    if args.check_only:
        errors = [
            error
            for row in rows
            for error in validate_evidence(row, texts)
        ]
        if errors:
            raise SystemExit("gold_evidence 校验失败：\n" + "\n".join(errors[:20]))
        print(f"gold_evidence OK：{len(rows)} 条样本")
        return

    report = {
        "golden": _display_path(golden_path),
        "corpus": _display_path(corpus_dir),
        "rows": len(rows),
        "rows_with_evidence": 0,
        "rows_without_evidence": [],
        "evidence_count": 0,
        "missing_facts": [],
        "validation_errors": [],
    }

    for row in rows:
        evidence, missing = evidence_for_row(row, texts)
        row["gold_evidence"] = evidence
        if evidence:
            report["rows_with_evidence"] += 1
            report["evidence_count"] += len(evidence)
        elif row.get("query_type") != "unanswerable":
            report["rows_without_evidence"].append(row.get("id"))
        if missing:
            report["missing_facts"].append(
                {"id": row.get("id"), "missing": missing}
            )
        report["validation_errors"].extend(validate_evidence(row, texts))

    output_path = resolve_path(args.output, golden_path)
    write_rows(output_path, rows)
    report_path = resolve_path(args.report, DEFAULT_REPORT)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"写入 {output_path}")
    print(
        f"有证据的样本：{report['rows_with_evidence']}/{len(rows)}；"
        f"证据条数：{report['evidence_count']}"
    )
    if report["rows_without_evidence"]:
        print(f"仍无可回答证据：{report['rows_without_evidence']}")
    if report["validation_errors"]:
        print(f"校验错误：{report['validation_errors'][:5]}")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
