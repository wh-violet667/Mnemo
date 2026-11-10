#!/usr/bin/env python
"""检索评测脚本：Recall@k / MRR@k / 延迟。

只测「检索层」：给定问题，看正确父块是否出现在前 k 条。
答案质量（是否胡说）建议另加 LLM judge 或人工抽检。

用法：
    python evals/run_eval.py --check-only
    python evals/run_eval.py --retrieval-mode fast
    python evals/run_eval.py --retrieval-mode deep
    python evals/run_eval.py --retrieval-mode auto
    python evals/run_eval.py --retrieval-mode baseline --mode hybrid+rerank
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from evals.metrics import recall_at_k, reciprocal_rank, summarize  # noqa: E402
from backend.citations import order_by_rerank  # noqa: E402


def load_golden(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"找不到 {path}。请检查 evals/golden_violet.jsonl 是否存在。"
        )

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} JSON 解析失败：{exc}") from exc
            if not row.get("query"):
                raise SystemExit(f"{path}:{line_no} 缺少 query 字段")
            rows.append(row)
    return rows


def is_relevant(document: Dict[str, Any], row: Dict[str, Any]) -> bool:
    """命中判定：优先 expected_parent_ids，其次 expected_files（多文件），最后 expected_phrase / expected_file。"""

    expected_ids = [str(pid) for pid in (row.get("expected_parent_ids") or [])]
    if expected_ids:
        return str(document.get("parent_id")) in set(expected_ids)

    doc_file = str(document.get("file_name") or "")
    expected_files = [str(f).strip() for f in (row.get("expected_files") or []) if str(f).strip()]
    if expected_files:
        stem_matches = [
            f for f in expected_files if Path(f).stem in doc_file
        ]
        return bool(stem_matches)

    phrase = " ".join(str(row.get("expected_phrase") or "").split())
    file_name = str(row.get("expected_file") or "").strip()
    text = " ".join(str(document.get("text") or "").split())
    if phrase and phrase in text:
        return True
    if file_name and file_name in doc_file:
        return True
    return False


def _is_unanswerable(row: Dict[str, Any]) -> bool:
    """不可回答样本：不参与 Recall/MRR，单独统计噪声召回。"""
    if row.get("query_type") == "unanswerable":
        return True
    return (
        "expected_parent_ids" in row
        and not row.get("expected_parent_ids")
        and not row.get("expected_phrase")
        and not row.get("expected_file")
    )


def _print_report(summary: Dict[str, float], results: List[Dict[str, Any]], args) -> None:
    print(f"\ngolden: {args.golden}")
    print(f"mode: {args.mode} | retrieval_mode: {args.retrieval_mode} | n: {len(results)} | k: {args.k}")
    print("\n| 指标 | 值 |")
    print("|---|---|")
    for key, value in summary.items():
        if key.endswith("_ms"):
            print(f"| {key} | {value:.0f} ms |")
        else:
            print(f"| {key} | {value:.3f} |")

    # 分类型指标（多跳/关系/时间/单跳/不可回答分开看才有意义）
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        by_type.setdefault(row.get("query_type") or "unknown", []).append(row)
    if len(by_type) > 1:
        print("\n| query_type | n | recall@k | MRR | 平均轮次 |")
        print("|---|---|---|---|---|")
        for qtype, group in sorted(by_type.items()):
            scored = [g for g in group if g.get("recall") is not None]
            recall = (
                sum(g["recall"] for g in scored) / len(scored) if scored else None
            )
            rr_values = [g["rr"] for g in scored if g.get("rr")]
            mrr = sum(rr_values) / len(rr_values) if rr_values else None
            rounds = [g["rounds"] for g in group if g.get("rounds") is not None]
            avg_rounds = sum(rounds) / len(rounds) if rounds else None
            print(
                f"| {qtype} | {len(group)} | "
                f"{f'{recall:.3f}' if recall is not None else '-'} | "
                f"{f'{mrr:.3f}' if mrr is not None else '-'} | "
                f"{f'{avg_rounds:.2f}' if avg_rounds is not None else '-'} |"
            )

    unanswerable = [r for r in results if r.get("clean") is not None]
    if unanswerable:
        clean_rate = sum(1 for r in unanswerable if r["clean"]) / len(unanswerable)
        print(f"\n不可回答噪声控制：{sum(1 for r in unanswerable if r['clean'])}/{len(unanswerable)} 条零召回（clean rate {clean_rate:.2f}）")

    agentic_rows = [r for r in results if r.get("trace_mode")]
    if agentic_rows:
        escalated = [r for r in agentic_rows if r.get("escalated")]
        single_hop = [r for r in agentic_rows if r.get("query_type") == "single_hop"]
        misrouted = [r for r in single_hop if r.get("trace_mode") == "deep"]
        rounds = [r["rounds"] for r in agentic_rows if r.get("rounds") is not None]
        print("\nAgentic 指标：")
        if rounds:
            print(f"- 平均检索轮次：{sum(rounds) / len(rounds):.2f}")
        if args.retrieval_mode == "auto":
            print(f"- auto 升级率：{len(escalated)}/{len(agentic_rows)}")
            if single_hop:
                print(
                    f"- auto 规则误判率（单跳被路由 deep）："
                    f"{len(misrouted)}/{len(single_hop)}"
                )

    misses = [row for row in results if row["recall"] == 0.0]
    if misses:
        print(f"\n未命中 {len(misses)} 条（前 10 条）：")
        for row in misses[:10]:
            print(f"- [{row.get('id')}] {row['query']} -> {row['top_files']}")
    else:
        print("\n全部命中。")


async def evaluate(args) -> None:
    from dotenv import load_dotenv

    for candidate in (ROOT / ".env", ROOT / "backend" / ".env"):
        if candidate.exists():
            load_dotenv(candidate)

    from documents_process import rerank_documents
    from milvus_client import get_milvus_client
    from postgresql_client import get_postgresql_client

    if args.retrieval_mode != "baseline":
        from retrieval import run_retrieval

    rows = load_golden(Path(args.golden))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("golden 为空，没有可评测的样本")

    milvus_client = await get_milvus_client()
    postgresql_client = await get_postgresql_client()

    results: List[Dict[str, Any]] = []
    for row in rows:
        kb_id = row.get("kb_id") or args.kb_id
        user_id = int(row.get("user_id") or args.user_id)

        started = time.perf_counter()
        trace_info: Dict[str, Any] = {}

        if args.retrieval_mode == "baseline":
            mode = "dense" if args.mode == "dense" else "hybrid"
            parent_ids = await milvus_client.hybrid_retrieval_knowledge_base(
                query=row["query"],
                knowledge_base_id=kb_id,
                top_k=args.k,
                user_id=user_id,
                mode=mode,
            )
            documents = await postgresql_client.get_parents(
                parent_ids=parent_ids,
                knowledge_base_id=kb_id,
                user_id=user_id,
            )

            if args.mode == "hybrid+rerank" and documents:
                reranked = await rerank_documents(
                    row["query"], [doc["text"] for doc in documents], args.k
                )
                documents = order_by_rerank(documents, reranked)

            documents = documents[: args.k]
            effective_mode = args.mode
        else:
            retrieval_result = await run_retrieval(
                row["query"],
                user_id=user_id,
                knowledge_base_id=kb_id,
                top_k=args.k,
                mode=args.retrieval_mode,
            )
            documents = retrieval_result.documents
            effective_mode = args.retrieval_mode
            trace_info = {
                "trace_mode": retrieval_result.trace.mode,
                "rounds": retrieval_result.trace.rounds,
                "escalated": retrieval_result.trace.escalated,
                "degraded": retrieval_result.trace.degraded,
                "critic": retrieval_result.trace.critic,
            }

        latency_ms = (time.perf_counter() - started) * 1000
        relevances = [is_relevant(document, row) for document in documents]
        unanswerable = _is_unanswerable(row)

        results.append(
            {
                "id": row.get("id"),
                "query": row["query"],
                "query_type": row.get("query_type") or "unknown",
                "mode": effective_mode,
                "recall": None if unanswerable else recall_at_k(relevances, args.k),
                "rr": None if unanswerable else reciprocal_rank(relevances),
                "clean": (len(documents) == 0) if unanswerable else None,
                "latency_ms": latency_ms,
                "top_files": [doc.get("file_name") for doc in documents],
                "relevant": relevances,
                **trace_info,
            }
        )

    scored_results = [r for r in results if r.get("recall") is not None]
    summary = summarize(scored_results, args.k) if scored_results else {}
    _print_report(summary, results, args)

    if args.output:
        payload = {
            "golden": str(args.golden),
            "mode": args.mode,
            "k": args.k,
            "summary": summary,
            "results": results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n已写入 {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mnemo 检索评测")
    parser.add_argument(
        "--golden",
        default=str(ROOT / "evals" / "golden_violet.jsonl"),
        help="评测集路径，默认 evals/golden_violet.jsonl",
    )
    parser.add_argument(
        "--kb-id", default="violet-kb", help="默认知识库 ID（violet 评测集）"
    )
    parser.add_argument("--user-id", type=int, default=1, help="默认用户 ID")
    parser.add_argument("--k", type=int, default=5, help="Recall@k 的 k")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "hybrid+rerank"],
        default="hybrid",
        help="baseline 检索模式，用于消融对比",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["baseline", "fast", "deep", "auto"],
        default="baseline",
        help="baseline 走 --mode 的旧链路；fast/deep/auto 走 Agentic-lite run_retrieval",
    )
    parser.add_argument("--output", default="", help="把结果写成 JSON")
    parser.add_argument(
        "--check-only", action="store_true", help="只校验评测集格式，不连接数据库"
    )
    args = parser.parse_args()

    if args.check_only:
        rows = load_golden(Path(args.golden))
        print(f"OK：{len(rows)} 条样本，格式正确。")
        return

    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
