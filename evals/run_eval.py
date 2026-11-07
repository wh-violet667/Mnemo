#!/usr/bin/env python
"""检索评测脚本：Recall@k / MRR@k / 延迟。

只测「检索层」：给定问题，看正确父块是否出现在前 k 条。
答案质量（是否胡说）建议另加 LLM judge 或人工抽检。

用法：
    python evals/run_eval.py --check-only
    python evals/run_eval.py --kb-id kb-1 --user-id 1
    python evals/run_eval.py --mode dense
    python evals/run_eval.py --mode hybrid+rerank --k 5 --limit 20
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
            f"找不到 {path}。先复制 evals/golden.example.jsonl 为 evals/golden.jsonl，再按你的知识库改。"
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
    """命中判定：原文包含 expected_phrase，或文件名包含 expected_file。"""

    phrase = " ".join(str(row.get("expected_phrase") or "").split())
    file_name = str(row.get("expected_file") or "").strip()
    text = " ".join(str(document.get("text") or "").split())
    if phrase and phrase in text:
        return True
    if file_name and file_name in str(document.get("file_name") or ""):
        return True
    return False


def _print_report(summary: Dict[str, float], results: List[Dict[str, Any]], args) -> None:
    print(f"\ngolden: {args.golden}")
    print(f"mode: {args.mode} | n: {len(results)} | k: {args.k}")
    print("\n| 指标 | 值 |")
    print("|---|---|")
    for key, value in summary.items():
        if key.endswith("_ms"):
            print(f"| {key} | {value:.0f} ms |")
        else:
            print(f"| {key} | {value:.3f} |")

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
        mode = "dense" if args.mode == "dense" else "hybrid"

        started = time.perf_counter()
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
        latency_ms = (time.perf_counter() - started) * 1000
        relevances = [is_relevant(document, row) for document in documents]

        results.append(
            {
                "id": row.get("id"),
                "query": row["query"],
                "mode": args.mode,
                "recall": recall_at_k(relevances, args.k),
                "rr": reciprocal_rank(relevances),
                "latency_ms": latency_ms,
                "top_files": [doc.get("file_name") for doc in documents],
                "relevant": relevances,
            }
        )

    summary = summarize(results, args.k)
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
        "--golden", default=str(ROOT / "evals" / "golden.jsonl"), help="评测集路径"
    )
    parser.add_argument("--kb-id", default="默认知识库", help="默认知识库 ID")
    parser.add_argument("--user-id", type=int, default=1, help="默认用户 ID")
    parser.add_argument("--k", type=int, default=5, help="Recall@k 的 k")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "hybrid+rerank"],
        default="hybrid",
        help="检索模式，用于消融对比",
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
