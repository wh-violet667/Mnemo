"""把 violet-evergarden-kb 的 qa-benchmark.zh.jsonl 转成 run_eval 可用的 golden 格式。

字段映射：
    question     -> query
    type         -> query_type（保留原值：single_hop/multi_hop/numeric/
                   version_conflict/unanswerable/alias）
    source_docs  -> expected_files（docs/06-films.md -> 06-films.docx）
    answer_keywords -> required_facts
    unanswerable -> expected_parent_ids = []（检索层判噪声，不参与 Recall）

注意：标签是文档级粒度（docx 入库后会被切成多个父块），命中该文件任意
父块即判对，指标比父块级判定偏乐观，解读报告时需注明。

用法：
    python evals/build_violet_golden.py
    python evals/build_violet_golden.py --kb-dir data/violet-evergarden-kb

语料默认放在 `data/violet-evergarden-kb/`（不提交到 Git），也可通过
`VIOLET_KB_DIR`、`--kb-dir` 或 `--benchmark` 指定。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_DIR = ROOT / "data" / "violet-evergarden-kb"
OUTPUT = ROOT / "evals" / "golden_violet.jsonl"


def resolve_kb_dir(value: str | None) -> Path:
    """解析语料根目录；相对路径按项目根目录处理。"""
    if not value:
        return DEFAULT_KB_DIR
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def convert_row(row: dict, kb_id: str, user_id: int) -> dict:
    source_docs = row.get("source_docs") or []
    expected_files = [
        Path(doc).with_suffix(".docx").name for doc in source_docs
    ]
    converted = {
        "id": f"v-{row['id']}",
        "query": row["question"],
        "kb_id": kb_id,
        "user_id": user_id,
        "query_type": row.get("type") or "unknown",
        "difficulty": row.get("difficulty"),
        "required_facts": row.get("answer_keywords") or [],
        "reference_answer": row.get("answer"),
    }
    if row.get("unanswerable"):
        converted["query_type"] = "unanswerable"
        converted["expected_parent_ids"] = []
    else:
        converted["expected_files"] = expected_files
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="转换 violet benchmark 为 golden 格式")
    parser.add_argument("--kb-id", default="violet-kb")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument(
        "--kb-dir",
        default=None,
        help="语料根目录，默认 data/violet-evergarden-kb 或 VIOLET_KB_DIR",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="qa-benchmark.zh.jsonl 路径；默认 <kb-dir>/data/qa-benchmark.zh.jsonl",
    )
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    kb_dir = resolve_kb_dir(args.kb_dir or os.getenv("VIOLET_KB_DIR"))
    benchmark = (
        Path(args.benchmark).expanduser()
        if args.benchmark
        else kb_dir / "data" / "qa-benchmark.zh.jsonl"
    )
    if not benchmark.is_absolute():
        benchmark = ROOT / benchmark
    if not benchmark.exists():
        raise SystemExit(
            f"找不到 {benchmark}。请把 violet-evergarden-kb 放到 "
            "data/violet-evergarden-kb/，或用 --kb-dir / --benchmark 指定。"
        )

    rows = []
    with benchmark.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("question"):
                raise SystemExit(f"{benchmark}:{line_no} 缺少 question 字段")
            rows.append(convert_row(row, args.kb_id, args.user_id))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    type_counts: dict[str, int] = {}
    for row in rows:
        type_counts[row["query_type"]] = type_counts.get(row["query_type"], 0) + 1
    print(f"已写出 {len(rows)} 条到 {output_path}")
    print(f"题型分布: {dict(sorted(type_counts.items()))}")


if __name__ == "__main__":
    main()
