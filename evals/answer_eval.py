#!/usr/bin/env python
"""End-to-end answer evaluation for the violet knowledge base.

The runner calls the real streaming chat endpoint, reconstructs the full
retrieved contexts from the trace query, and then runs two independent
LLM judges:

* Judge A: factual correctness against ``reference_answer``.
* Judge B: claim-level faithfulness and citation support.

Deterministic metrics (retrieval coverage, citation validity, fact matching)
are computed without an LLM.

Usage:
    python evals/answer_eval.py --limit 3
    python evals/answer_eval.py
    python evals/answer_eval.py --no-judge --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from evals.answer_metrics import (  # noqa: E402
    aggregate_records,
    citation_metrics,
    compute_retrieval_metrics,
    dedupe_documents,
    extract_citation_marks,
    match_required_facts,
    record_faithfulness,
)
from evals.judges import (  # noqa: E402
    build_judge,
    judge_correctness,
    judge_grounding,
)


DEFAULT_GOLDEN = ROOT / "evals" / "golden_violet.jsonl"
DEFAULT_UNANSWERABLE = ROOT / "evals" / "unanswerable_violet.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "evals" / "results"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"找不到 {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} JSON 解析失败: {exc}") from exc
            if not row.get("query"):
                raise SystemExit(f"{path}:{line_no} 缺少 query 字段")
            rows.append(row)
    return rows


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _judge_score(record: Dict[str, Any]) -> Optional[int]:
    judge = record.get("judge_a") or {}
    score = judge.get("score")
    return _as_int(score) if score is not None else None


def _sanitize_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", value or "")
    return (cleaned or datetime.now().strftime("%Y%m%d_%H%M%S"))[:20]


async def fetch_answer(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    row: Dict[str, Any],
    conversation_id: str,
) -> Dict[str, Any]:
    params = {
        "query": row["query"],
        "knowledge_base_id": args.kb_id,
        "user_id": args.user_id,
        "conversation_id": conversation_id,
        "retrieval_mode": args.retrieval_mode,
    }
    answer_parts: List[str] = []
    sources: List[Dict[str, Any]] = []
    trace: Dict[str, Any] = {}
    statuses: List[str] = []
    errors: List[str] = []
    first_token_ms: Optional[float] = None
    started = time.perf_counter()

    async with client.stream(
        "GET",
        f"{args.base_url.rstrip('/')}/chat_with_agent/stream",
        params=params,
    ) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {response.status_code}: {body[:300]}")

        async for line in response.aiter_lines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            content = event.get("content")
            if event_type == "answer":
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                answer_parts.append(str(content or ""))
            elif event_type == "sources":
                sources = content if isinstance(content, list) else []
            elif event_type == "trace":
                trace = content if isinstance(content, dict) else {}
            elif event_type == "status":
                statuses.append(str(content or ""))
            elif event_type == "error":
                errors.append(str(content or ""))

    return {
        "answer": "".join(answer_parts).strip(),
        "sources": sources,
        "trace": trace,
        "status_events": statuses,
        "errors": errors,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "first_token_ms": first_token_ms,
        "memory_used": any("记忆" in status for status in statuses),
    }


async def prepare_cited_sources(
    sources: Sequence[Dict[str, Any]],
    answer: str,
    pg_client: Any,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    marks = set(extract_citation_marks(answer))
    cited = [
        dict(source)
        for source in sources
        if _as_int(source.get("index")) in marks
    ]
    parent_ids = [
        str(source.get("parent_id"))
        for source in cited
        if source.get("parent_id")
    ]
    source_texts: Dict[str, str] = {}
    errors: List[str] = []
    if parent_ids:
        try:
            parents = await pg_client.get_parents(
                parent_ids=parent_ids,
                knowledge_base_id=args.kb_id,
                user_id=args.user_id,
            )
            source_texts = {
                str(parent.get("parent_id")): str(parent.get("text") or "")
                for parent in parents
            }
        except Exception as exc:  # pragma: no cover - DB dependent
            errors.append(f"load cited parent text: {exc}")

    for source in cited:
        parent_id = str(source.get("parent_id") or "")
        source["text"] = (
            source_texts.get(parent_id)
            or str(source.get("snippet") or "")
        )

    return cited, source_texts, errors


async def reconstruct_contexts(
    row: Dict[str, Any],
    trace: Dict[str, Any],
    sources: Sequence[Dict[str, Any]],
    pg_client: Any,
    args: argparse.Namespace,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Rebuild the full top-k contexts used by the agent.

    The stream only sends cited sources to the frontend.  The trace contains
    the actual (possibly rewritten) retrieval query, so we run the same fast
    retrieval once more in this process to recover full parent texts.
    """

    queries = trace.get("queries") if isinstance(trace, dict) else None
    actual_query = (
        str(queries[0])
        if isinstance(queries, list) and queries
        else str(row["query"])
    )
    errors: List[str] = []
    documents: List[Dict[str, Any]] = []

    try:
        from retrieval import run_retrieval

        result = await run_retrieval(
            query=actual_query,
            user_id=args.user_id,
            knowledge_base_id=args.kb_id,
            top_k=args.top_k,
            mode=args.retrieval_mode,
        )
        documents = list(result.documents or [])
    except Exception as exc:  # pragma: no cover - Milvus/DB dependent
        errors.append(f"reconstruct retrieval: {exc}")

    if not documents and sources:
        parent_ids = [
            str(source.get("parent_id"))
            for source in sources
            if source.get("parent_id")
        ]
        if parent_ids:
            try:
                documents = await pg_client.get_parents(
                    parent_ids=parent_ids,
                    knowledge_base_id=args.kb_id,
                    user_id=args.user_id,
                )
            except Exception as exc:  # pragma: no cover - DB dependent
                errors.append(f"load cited parents as fallback: {exc}")
        if not documents:
            documents = [
                {
                    "parent_id": source.get("parent_id"),
                    "file_name": source.get("file_name"),
                    "text": source.get("snippet") or "",
                }
                for source in sources
            ]

    documents = dedupe_documents(documents)
    contexts = [
        {
            "context_id": f"C{index}",
            "parent_id": document.get("parent_id"),
            "file_name": document.get("file_name"),
            "text": document.get("text") or "",
            "score": document.get("score"),
        }
        for index, document in enumerate(documents, start=1)
    ]
    return actual_query, contexts, documents, errors


async def evaluate_one(
    client: httpx.AsyncClient,
    row: Dict[str, Any],
    index: int,
    args: argparse.Namespace,
    judge: Optional[Tuple[Any, Any]],
    pg_client: Any,
) -> Dict[str, Any]:
    conversation_id = f"eval-{args.run_id}-{index:03d}"
    record: Dict[str, Any] = {
        "sample_id": row.get("id"),
        "user_input": row.get("query"),
        "query_type": row.get("query_type") or "unknown",
        "difficulty": row.get("difficulty"),
        "reference": row.get("reference_answer"),
        "required_facts": row.get("required_facts") or [],
        "reference_context_ids": row.get("expected_parent_ids")
        or row.get("expected_files")
        or [],
        "conversation_id": conversation_id,
        "response": "",
        "retrieved_contexts": [],
        "retrieved_context_ids": [],
        "cited_sources": [],
        "trace": {},
        "status_events": [],
        "memory_used": False,
        "latency_ms": None,
        "first_token_ms": None,
        "retrieval_metrics": {},
        "fact_check": None,
        "judge_a": None,
        "judge_b": None,
        "citation_metrics": {},
        "faithfulness": None,
        "errors": [],
    }

    try:
        generation = await fetch_answer(client, args, row, conversation_id)
    except Exception as exc:
        record["errors"].append(f"generation: {exc}")
        return record

    record.update(
        {
            "response": generation["answer"],
            "trace": generation["trace"],
            "status_events": generation["status_events"],
            "memory_used": generation["memory_used"],
            "latency_ms": generation["latency_ms"],
            "first_token_ms": generation["first_token_ms"],
        }
    )
    record["errors"].extend(generation["errors"])

    if not record["response"]:
        record["errors"].append("empty answer")
        return record

    context_task = asyncio.create_task(
        reconstruct_contexts(
            row, generation["trace"], generation["sources"], pg_client, args
        )
    )
    cited_task = asyncio.create_task(
        prepare_cited_sources(
            generation["sources"], record["response"], pg_client, args
        )
    )
    (
        (actual_query, contexts, documents, retrieval_errors),
        (cited_sources, source_texts, citation_errors),
    ) = await asyncio.gather(context_task, cited_task)

    record["actual_query"] = actual_query
    record["retrieved_contexts"] = contexts
    record["retrieved_context_ids"] = [
        context.get("parent_id") for context in contexts
    ]
    record["retrieval_metrics"] = compute_retrieval_metrics(row, documents)
    record["errors"].extend(retrieval_errors)
    record["cited_sources"] = cited_sources
    record["errors"].extend(citation_errors)

    if record["query_type"] != "unanswerable":
        record["fact_check"] = match_required_facts(
            record["response"], row.get("required_facts")
        )

    if judge is not None:
        primary, fallback = judge
        judge_a_task = asyncio.create_task(
            judge_correctness(primary, fallback, row, record["response"])
        )
        judge_b_task = asyncio.create_task(
            judge_grounding(
                primary,
                fallback,
                row,
                record["response"],
                cited_sources,
                contexts,
            )
        )
        judge_a, judge_b = await asyncio.gather(
            judge_a_task, judge_b_task, return_exceptions=True
        )
        if isinstance(judge_a, Exception):
            record["errors"].append(f"judge_a: {judge_a}")
        else:
            record["judge_a"] = judge_a
        if isinstance(judge_b, Exception):
            record["errors"].append(f"judge_b: {judge_b}")
        else:
            record["judge_b"] = judge_b

    record["citation_metrics"] = citation_metrics(
        record["response"],
        cited_sources,
        record["judge_b"],
        source_texts,
    )
    record["faithfulness"] = record_faithfulness(record["judge_b"])
    return record


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _num(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _ms(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.0f} ms"


def select_human_sample(
    records: Sequence[Dict[str, Any]], size: int
) -> List[Dict[str, Any]]:
    if size <= 0:
        return []
    scored = [record for record in records if _judge_score(record) is not None]
    if not scored:
        return list(records[:size])
    low = sorted(
        scored,
        key=lambda record: (_judge_score(record) or 0, record.get("query_type") or ""),
    )[: max(1, size // 2)]
    remaining = [record for record in scored if record not in low]
    random.Random(42).shuffle(remaining)
    return (low + remaining)[:size]


def write_human_review(
    path: Path, records: Sequence[Dict[str, Any]], size: int
) -> None:
    sample = select_human_sample(records, size)
    columns = [
        "sample_id",
        "query_type",
        "user_input",
        "response",
        "reference",
        "judge_score",
        "refused",
        "missing_facts",
        "wrong_facts",
        "faithfulness",
        "citation_precision",
        "human_score",
        "human_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in sample:
            judge_a = record.get("judge_a") or {}
            faithfulness = record.get("faithfulness")
            if (
                record.get("query_type") != "unanswerable"
                and judge_a.get("refused")
            ):
                faithfulness = None
            writer.writerow(
                {
                    "sample_id": record.get("sample_id"),
                    "query_type": record.get("query_type"),
                    "user_input": record.get("user_input"),
                    "response": record.get("response"),
                    "reference": record.get("reference"),
                    "judge_score": judge_a.get("score"),
                    "refused": judge_a.get("refused"),
                    "missing_facts": "; ".join(judge_a.get("missing_facts") or []),
                    "wrong_facts": "; ".join(judge_a.get("wrong_facts") or []),
                    "faithfulness": _num(faithfulness),
                    "citation_precision": _num(
                        (record.get("citation_metrics") or {}).get(
                            "citation_precision"
                        )
                    ),
                    "human_score": "",
                    "human_notes": "",
                }
            )


def write_report(
    path: Path,
    summary: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    overall = summary["overall"]
    by_type = summary.get("by_query_type") or {}
    lines = [
        "# Violet 答案评测报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 知识库：`{args.kb_id}`",
        f"- 生成模型：`{os.getenv('AGENT_BASE_MODEL', 'unknown')}`",
        f"- Judge：`{'disabled' if args.no_judge else args.judge_model}`",
        f"- 检索模式：`{args.retrieval_mode}` / top_k={args.top_k}",
        f"- 样本：{overall['n_total']} 条（可回答 {overall['n_answerable']}，不可回答 {overall['n_unanswerable']}）",
        "- 上下文：按 trace 查询重建的 top-k 父块全文",
        "",
        "## 总体指标",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 完全正确率 | {_pct(overall['fully_correct_rate'])} |",
        f"| 平均正确性 | {_num(overall['mean_correctness'])} |",
        f"| 事实覆盖率 | {_pct(overall['fact_coverage_mean'])} |",
        f"| 全事实命中率 | {_pct(overall['all_facts_rate'])} |",
        f"| 忠实度（strict） | {_pct(overall['faithfulness'])} |",
        f"| 忠实度（partial=0.5） | {_pct(overall['faithfulness_weighted'])} |",
        f"| 引用有效率 | {_pct(overall['citation_validity'])} |",
        f"| 引用精确率 | {_pct(overall['citation_precision'])} |",
        f"| 引用覆盖率 | {_pct(overall['citation_coverage'])} |",
        f"| Chunk 支持率 | {_pct(overall['chunk_support_rate'])} |",
        f"| 拒答准确率 | {_pct(overall['refusal_accuracy'])} |",
        f"| 误拒率 | {_pct(overall['false_refusal_rate'])} |",
        f"| Hit@5 | {_pct(overall['hit_at_5'])} |",
        f"| Correctness \\| hit（n={overall['n_hit']}） | {_pct(overall['correctness_given_hit'])} |",
        f"| Correctness \\| miss（n={overall['n_miss']}） | {_pct(overall['correctness_given_miss'])} |",
        f"| ExpectedContextCoverage | {_pct(overall['expected_context_coverage'])} |",
        f"| AllEvidenceHit | {_pct(overall['all_evidence_hit'])} |",
        f"| ContextPrecision | {_pct(overall['context_precision'])} |",
        f"| P50 / P95 | {_ms(overall['p50_ms'])} / {_ms(overall['p95_ms'])} |",
        f"| 首 token P50 / P95 | {_ms(overall['p50_first_token_ms'])} / {_ms(overall['p95_first_token_ms'])} |",
        f"| 记忆使用率 | {_pct(overall['memory_used_rate'])} |",
        "",
        "## 分题型",
        "",
        "| 题型 | n | 完全正确 | 忠实度 | 引用精确率 | Hit@5 | ExpectedCoverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for query_type, metrics in sorted(by_type.items()):
        lines.append(
            f"| {query_type} | {metrics['n_total']} | "
            f"{_pct(metrics['fully_correct_rate'])} | "
            f"{_pct(metrics['faithfulness'])} | "
            f"{_pct(metrics['citation_precision'])} | "
            f"{_pct(metrics['hit_at_5'])} | "
            f"{_pct(metrics['expected_context_coverage'])} |"
        )

    low_scores = [
        record
        for record in records
        if _judge_score(record) is not None and _judge_score(record) < 2
    ]
    lines.extend(["", "## 低分样本", ""])
    if not low_scores:
        lines.append("没有 score < 2 的样本。")
    else:
        for record in low_scores[:20]:
            judge_a = record.get("judge_a") or {}
            hit = (record.get("retrieval_metrics") or {}).get("hit_at_5")
            lines.append(
                f"- `{record.get('sample_id')}` [{record.get('query_type')}] "
                f"score={judge_a.get('score')} hit={hit} "
                f"refused={judge_a.get('refused')} "
                f"missing={judge_a.get('missing_facts')} "
                f"wrong={judge_a.get('wrong_facts')}"
            )

    errors = [record for record in records if record.get("errors")]
    lines.extend(["", "## 失败与限制", ""])
    lines.append(
        "- 参考答案来自外部 violet benchmark；正式结论前应人工抽检参考答案。"
    )
    lines.append(
        "- 检索上下文为按 trace 查询重建的 top-k 父块；与主回答调用存在一次重复检索，Hit@5 仅作答案评测诊断，官方检索指标以 `run_eval.py` 为准。"
    )
    lines.append(
        "- violet 标签是文档级；`ExpectedContextCoverage` 仍是文档级，chunk 级需补 `expected_parent_ids`。"
    )
    lines.append(
        "- `hit=0` 不代表一定检索失败：答案可能由其他文档或参数知识支持，需结合 cited sources 和 expected_files 人工复核。"
    )
    if errors:
        lines.append(f"- 有 {len(errors)} 条样本带错误/降级信息：")
        for record in errors[:10]:
            lines.append(
                f"  - `{record.get('sample_id')}`: "
                + "; ".join(record.get("errors") or [])[:300]
            )
    else:
        lines.append("- 本次运行没有记录到生成/检索/judge 错误。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def rejudge_existing(
    args: argparse.Namespace,
    rows_by_id: Dict[str, Dict[str, Any]],
    output_path: Path,
    output_dir: Path,
    stem: str,
    judge: Tuple[Any, Any],
) -> None:
    """Re-run Judge A/B on existing records without regenerating answers."""

    if not output_path.exists():
        raise SystemExit(f"找不到 {output_path}，无法 rejudge")
    records: List[Dict[str, Any]] = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)

    primary, fallback = judge
    only_ids = {
        value.strip()
        for value in (getattr(args, "only_id", "") or "").split(",")
        if value.strip()
    }
    for index, record in enumerate(records, start=1):
        sample_id = str(record.get("sample_id"))
        if only_ids and sample_id not in only_ids:
            continue
        row = rows_by_id.get(sample_id)
        if not row:
            continue
        record["errors"] = [
            error
            for error in (record.get("errors") or [])
            if not str(error).startswith(("judge_a:", "judge_b:"))
        ]
        if record.get("response"):
            judge_a_task = asyncio.create_task(
                judge_correctness(primary, fallback, row, record["response"])
            )
            judge_b_task = asyncio.create_task(
                judge_grounding(
                    primary,
                    fallback,
                    row,
                    record["response"],
                    record.get("cited_sources") or [],
                    record.get("retrieved_contexts") or [],
                )
            )
            judge_a, judge_b = await asyncio.gather(
                judge_a_task, judge_b_task, return_exceptions=True
            )
            if isinstance(judge_a, Exception):
                record["errors"].append(f"judge_a: {judge_a}")
            else:
                record["judge_a"] = judge_a
            if isinstance(judge_b, Exception):
                record["errors"].append(f"judge_b: {judge_b}")
            else:
                record["judge_b"] = judge_b

        # 保留已有有效 judge 时，不把一次超时误报成最终错误。
        if record.get("judge_a") is not None:
            record["errors"] = [
                error
                for error in record.get("errors") or []
                if not str(error).startswith("judge_a:")
            ]
        if record.get("judge_b") is not None:
            record["errors"] = [
                error
                for error in record.get("errors") or []
                if not str(error).startswith("judge_b:")
            ]

        if record.get("query_type") != "unanswerable":
            record["fact_check"] = match_required_facts(
                record.get("response") or "", row.get("required_facts")
            )
        else:
            record["fact_check"] = None
        source_texts = {
            str(source.get("parent_id")): str(
                source.get("text") or source.get("snippet") or ""
            )
            for source in (record.get("cited_sources") or [])
        }
        record["citation_metrics"] = citation_metrics(
            record.get("response") or "",
            record.get("cited_sources") or [],
            record.get("judge_b"),
            source_texts,
        )
        record["faithfulness"] = record_faithfulness(record.get("judge_b"))
        print(
            f"[rejudge {index}/{len(records)}] {record.get('sample_id')} "
            f"score={(record.get('judge_a') or {}).get('score')}",
            flush=True,
        )

    write_jsonl(output_path, records)
    summary = aggregate_records(records)
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_human_review(
        output_dir / f"{stem}_human_review.csv", records, args.human_sample
    )
    write_report(output_dir / f"{stem}_report.md", summary, records, args)
    print(f"rejudged: {output_path}", flush=True)


async def run(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "backend" / ".env")

    rows = load_jsonl(Path(args.golden))
    unanswerable_path = Path(args.unanswerable)
    if unanswerable_path.exists():
        rows.extend(load_jsonl(unanswerable_path))
    else:
        print(f"warning: {unanswerable_path} 不存在，跳过补充拒答题")

    seen = set()
    unique_rows = []
    for row in rows:
        key = str(row.get("id") or row.get("query"))
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    rows = unique_rows
    if args.limit and not args.rejudge:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("没有可评测样本")
    rows_by_id = {
        str(row.get("id") or row.get("query")): row for row in rows
    }

    args.run_id = _sanitize_run_id(args.run_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"answer_violet_{args.run_id}"
    output_path = output_dir / f"{stem}.jsonl"
    meta_path = output_dir / f"{stem}_meta.json"

    meta = {
        "golden": str(Path(args.golden)),
        "unanswerable": str(Path(args.unanswerable)),
        "kb_id": args.kb_id,
        "user_id": args.user_id,
        "retrieval_mode": args.retrieval_mode,
        "top_k": args.top_k,
        "judge_model": None if args.no_judge else args.judge_model,
    }
    if args.resume and meta_path.exists():
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta != meta:
            raise SystemExit(
                "resume 配置与已有 meta 不一致，换一个 --run-id：\n"
                f"old={old_meta}\nnew={meta}"
            )
    else:
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    judge = None
    if not args.no_judge:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("BASE_URL")
        if not api_key or not base_url:
            raise SystemExit("缺少 DASHSCOPE_API_KEY 或 BASE_URL，无法调用 judge")
        judge = build_judge(args.judge_model, base_url, api_key)

    if args.rejudge:
        if judge is None:
            raise SystemExit("--rejudge 不能与 --no-judge 同时使用")
        await rejudge_existing(
            args, rows_by_id, output_path, output_dir, stem, judge
        )
        return

    from postgresql_client import get_postgresql_client

    pg_client = await get_postgresql_client()
    records: List[Dict[str, Any]] = []
    existing_ids = set()
    if args.resume and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # 进程中断时最后一行可能只写了一半；忽略并在下面重写干净文件。
                continue
            if not isinstance(record, dict):
                continue
            records.append(record)
            existing_ids.add(str(record.get("sample_id")))
        output_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    timeout = httpx.Timeout(args.request_timeout, connect=15.0)
    jsonl_handle = output_path.open(
        "a" if args.resume and output_path.exists() else "w",
        encoding="utf-8",
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        for index, row in enumerate(rows, start=1):
            sample_id = str(row.get("id") or row.get("query"))
            if sample_id in existing_ids:
                print(
                    f"[{index}/{len(rows)}] {sample_id} already done, skip",
                    flush=True,
                )
                continue
            record = await evaluate_one(client, row, index, args, judge, pg_client)
            records.append(record)
            existing_ids.add(sample_id)
            jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_handle.flush()
            score = _judge_score(record)
            print(
                f"[{index}/{len(rows)}] {row.get('id')} "
                f"{row.get('query_type')} score={score} "
                f"errors={len(record.get('errors') or [])}",
                flush=True,
            )

        jsonl_handle.close()
        summary = aggregate_records(records)
        (output_dir / f"{stem}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_human_review(
            output_dir / f"{stem}_human_review.csv",
            records,
            args.human_sample,
        )
        write_report(output_dir / f"{stem}_report.md", summary, records, args)

        if not args.keep_conversations:
            for record in records:
                conversation_id = record.get("conversation_id")
                if not conversation_id:
                    continue
                try:
                    await client.delete(
                        f"{args.base_url.rstrip('/')}/conversations/{conversation_id}",
                        params={"user_id": args.user_id},
                    )
                except Exception:
                    pass

    await pg_client.close()
    print(f"\nwritten: {output_path}")
    print(f"written: {output_dir / (stem + '_summary.json')}")
    print(f"written: {output_dir / (stem + '_report.md')}")
    print(f"written: {output_dir / (stem + '_human_review.csv')}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "backend" / ".env")
    parser = argparse.ArgumentParser(description="Violet 答案评测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--unanswerable", default=str(DEFAULT_UNANSWERABLE))
    parser.add_argument("--kb-id", default="violet-kb")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--retrieval-mode", default="fast", choices=["fast"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "qwen-plus"))
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--human-sample", type=int, default=12)
    parser.add_argument(
        "--resume", action="store_true", help="从同 run-id 的 JSONL 继续，跳过已完成样本"
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="只对已有 JSONL 重跑 Judge A/B，不重新生成答案",
    )
    parser.add_argument(
        "--only-id",
        default="",
        help="配合 --rejudge，只重跑指定 sample id，多个用逗号分隔",
    )
    parser.add_argument("--keep-conversations", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
