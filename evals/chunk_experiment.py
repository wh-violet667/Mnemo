"""Run a controlled chunking comparison and write a markdown report.

For every configuration the script:

1. deletes/creates a separate knowledge base (separate Milvus namespace);
2. uploads the same 12 source files with a per-file chunk config;
3. runs ``run_eval.py`` at chunk (or parent) granularity;
4. compares each strategy to the first one with paired bootstrap intervals;
5. lists questions where the ranking flips, with the gold evidence quote.

The corpus, embedding model and reranker stay fixed.  Only chunking changes.

Usage:
    python evals/chunk_experiment.py --dry-run
    python evals/chunk_experiment.py --limit 10
    python evals/chunk_experiment.py --run-id full
    python evals/chunk_experiment.py --experiment markdown-strategy --run-id md-strategy
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "evals" / "golden_violet.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "violet-evergarden-kb"
DEFAULT_REPORT = ROOT / "docs" / "chunking-comparison.md"
MARKDOWN_REPORT = ROOT / "docs" / "chunking-markdown-strategy.md"

DOCX_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "child-200-o0",
        "strategy": "recursive",
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 200,
        "child_chunk_size": 200,
        "child_chunk_overlap": 0,
        "granularity": "child",
    },
    {
        "name": "child-200-o50",
        "strategy": "recursive",
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 200,
        "child_chunk_size": 200,
        "child_chunk_overlap": 50,
        "granularity": "child",
    },
    {
        "name": "child-400-o0",
        "strategy": "recursive",
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 200,
        "child_chunk_size": 400,
        "child_chunk_overlap": 0,
        "granularity": "child",
    },
]

# One variable changes between the last two rows: the splitter.  The first
# row keeps the production `coarse-no-overlap` parent overlap so the report
# also answers "does heading-aware splitting beat the profile users can pick
# today?".
MARKDOWN_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "recursive-overlap",
        "strategy": "recursive",
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 200,
        "child_chunk_size": 400,
        "child_chunk_overlap": 0,
        "granularity": "child",
    },
    {
        "name": "recursive-no-parent-overlap",
        "strategy": "recursive",
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 0,
        "child_chunk_size": 400,
        "child_chunk_overlap": 0,
        "granularity": "child",
    },
    {
        "name": "markdown-heading",
        "strategy": "markdown-heading",
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 0,
        "child_chunk_size": 400,
        "child_chunk_overlap": 0,
        "granularity": "child",
    },
]

DEFAULT_CONFIGS = DOCX_CONFIGS

METRICS = [
    ("recall", "Recall"),
    ("rr", "MRR"),
    ("precision", "Precision"),
    ("ndcg", "nDCG"),
]


def _safe_name(value: str, limit: int = 40) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "-", value).strip("-")
    return (cleaned or "strategy")[:limit]


def load_configs(path: str | None, experiment: str = "docx-sizes") -> List[Dict[str, Any]]:
    if not path:
        preset = MARKDOWN_CONFIGS if experiment == "markdown-strategy" else DOCX_CONFIGS
        return [dict(item) for item in preset]
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise SystemExit("--configs 必须是 JSON 数组")
    return [dict(item) for item in payload]


def load_golden(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row.get("id"))] = row
    return rows


def materialize_golden(source: Path, target: Path, kb_id: str) -> None:
    """Point a copy of the golden set at the experiment KB."""

    lines: List[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["kb_id"] = kb_id
            lines.append(json.dumps(row, ensure_ascii=False))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    n_resamples: int = 2000,
    seed: int = 17,
) -> tuple[float, float, float]:
    """Return (mean_delta, ci_low, ci_high) for candidate - baseline."""

    if len(baseline) != len(candidate):
        raise ValueError("paired samples must have the same length")
    if not baseline:
        return 0.0, 0.0, 0.0

    diffs = [float(new) - float(old) for old, new in zip(baseline, candidate)]
    mean_delta = _mean(diffs)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        sampled = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        samples.append(_mean(sampled))
    return mean_delta, _percentile(samples, 2.5), _percentile(samples, 97.5)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _fmt_delta(value: float, low: float, high: float) -> str:
    return f"{value:+.3f} [{low:+.3f}, {high:+.3f}]"


def paired_metric_values(
    baseline: Dict[str, Any], candidate: Dict[str, Any], metric: str
) -> tuple[List[float], List[float]]:
    """Align two runs by question id before computing a paired delta."""

    base_by_id = _result_by_id(baseline)
    new_by_id = _result_by_id(candidate)
    ids = sorted(set(base_by_id) & set(new_by_id))
    base_values: List[float] = []
    new_values: List[float] = []
    for row_id in ids:
        old_value = base_by_id[row_id].get(metric)
        new_value = new_by_id[row_id].get(metric)
        if old_value is None or new_value is None:
            continue
        base_values.append(float(old_value))
        new_values.append(float(new_value))
    return base_values, new_values


def _result_by_id(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("id")): row for row in result.get("results", [])}


def _top_files(row: Dict[str, Any]) -> str:
    files = []
    for name in row.get("top_files") or []:
        name = str(name or "")
        if name and name not in files:
            files.append(name)
    return "、".join(files[:3]) or "-"


def _quote_for(golden: Dict[str, Dict[str, Any]], row_id: str) -> str:
    evidence = (golden.get(row_id) or {}).get("gold_evidence") or []
    if not evidence:
        return "-"
    quote = str(evidence[0].get("quote") or "").replace("\n", " ").strip()
    return quote[:140] + ("…" if len(quote) > 140 else "")


def build_report(
    *,
    run_id: str,
    configs: List[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
    golden: Dict[str, Dict[str, Any]],
    k: int,
    bootstrap: int = 2000,
    experiment: str = "docx-sizes",
    corpus_format: str = "docx",
    index_counts: Dict[str, tuple[int, int]] | None = None,
    structure_stats: Dict[str, Dict[str, float]] | None = None,
) -> str:
    baseline_name = configs[0]["name"]
    baseline = results[baseline_name]
    title = (
        "# Chunking 策略对比：结构感知 Markdown"
        if experiment == "markdown-strategy"
        else "# Chunking 策略对比"
    )
    corpus_label = "Markdown" if corpus_format == "md" else corpus_format.upper()
    lines: List[str] = [
        title,
        "",
        f"> run: `{run_id}`；同一 {corpus_label} 语料、同一 embedding/rerank 模型，"
        "只改 chunk 配置。",
        "",
        "## 实验配置",
        "",
        "| 策略 | strategy | 父块 | 子块 | 粒度 |",
        "|---|---|---:|---:|---|",
    ]
    for config in configs:
        lines.append(
            f"| `{config['name']}` | `{config.get('strategy', 'recursive')}` | "
            f"{config.get('parent_chunk_size')}/{config.get('parent_chunk_overlap')} | "
            f"{config.get('child_chunk_size')}/{config.get('child_chunk_overlap')} | "
            f"{config.get('granularity', 'child')} |"
        )
    if experiment == "markdown-strategy":
        lines.extend(
            [
                "",
                "`recursive-no-parent-overlap` 与 `markdown-heading` 使用完全相同的"
                " 1000/0 + 400/0，唯一变量是分块边界策略；`recursive-overlap` 保留"
                "当前 `coarse-no-overlap` profile 的父块 overlap，用于回答“结构感知"
                "是否能超过用户今天能选的策略”。",
                "",
                "`markdown-heading` 的每个 chunk 会前置自己的 heading path（例如"
                " `# 作品概况 > 基本信息`），用于保留章节上下文；这不调用额外模型，"
                "属于策略本身的一部分。",
            ]
        )

    lines.extend(
        [
            "",
            "## 总体指标",
            "",
            "| 策略 | n | Recall@k | MRR | Precision@k | nDCG@k | P50 / P95 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for config in configs:
        name = config["name"]
        summary = results[name].get("summary") or {}
        lines.append(
            f"| `{name}` | {len(results[name].get('results', []))} | "
            f"{_fmt(summary.get(f'recall@{k}'))} | {_fmt(summary.get('mrr'))} | "
            f"{_fmt(summary.get(f'precision@{k}'))} | {_fmt(summary.get(f'ndcg@{k}'))} | "
            f"{summary.get('p50_ms', 0):.0f} / {summary.get('p95_ms', 0):.0f} ms |"
        )

    if index_counts:
        lines.extend(
            [
                "",
                "本次索引规模（PostgreSQL 统计）：",
                "",
                "| 策略 | 父块数 | 子块数 |",
                "|---|---:|---:|",
            ]
        )
        for config in configs:
            name = config["name"]
            parents, children = index_counts.get(name, (0, 0))
            lines.append(f"| `{name}` | {parents} | {children} |")

    if structure_stats:
        lines.extend(
            [
                "",
                "## 结构完整性（直接扫描 Milvus 子块）",
                "",
                "| 策略 | 子块数 | 中位长度 | P90 长度 | 含表格子块 | 表头保留率 | 列数一致率 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for config in configs:
            name = config["name"]
            stats = structure_stats.get(name) or {}
            if not stats:
                continue
            lines.append(
                f"| `{name}` | {stats.get('children', 0):.0f} | "
                f"{stats.get('median_chars', 0):.0f} | "
                f"{stats.get('p90_chars', 0):.0f} | "
                f"{stats.get('table_chunks', 0):.0f} | "
                f"{stats.get('table_header_rate', 0):.3f} | "
                f"{stats.get('table_column_rate', 0):.3f} |"
            )
        lines.extend(
            [
                "",
                "表头保留率/列数一致率只统计含 `|` 行的子块；结构感知策略应接近 1.0，"
                "递归切分把表格从中间切断时会明显下降。",
                "",
                "> Milvus 实际子块数可能比 PostgreSQL `chunk_hashes` 多 1–2 条："
                "相同文本子块在 PG 去重、在 Milvus 各存一条向量；以实际扫描为准。",
            ]
        )

    query_types: List[str] = []
    for config in configs:
        for row in results[config["name"]].get("results", []):
            query_type = str(row.get("query_type") or "unknown")
            if query_type not in query_types:
                query_types.append(query_type)
    if len(query_types) > 1:
        lines.extend(
            [
                "",
                "## 按题型 Recall@k",
                "",
                "| 题型 | n | " + " | ".join(f"`{c['name']}`" for c in configs) + " |",
                "|---|---:|" + "---:|" * len(configs),
            ]
        )
        for query_type in sorted(query_types):
            values = []
            sample_count = 0
            for config in configs:
                all_rows = [
                    row
                    for row in results[config["name"]].get("results", [])
                    if str(row.get("query_type") or "unknown") == query_type
                ]
                rows = [row for row in all_rows if row.get("recall") is not None]
                sample_count = max(sample_count, len(all_rows))
                values.append(
                    _fmt(_mean([float(row["recall"]) for row in rows])) if rows else "-"
                )
            lines.append(
                f"| `{query_type}` | {sample_count} | " + " | ".join(values) + " |"
            )

    lines.extend(
        [
            "",
            "## 相对基线（配对 bootstrap 95% CI）",
            "",
            "| 策略 | ΔRecall | ΔMRR | ΔPrecision | ΔnDCG |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for config in configs[1:]:
        name = config["name"]
        cells = []
        for metric, _ in METRICS:
            base_values, new_values = paired_metric_values(
                baseline, results[name], metric
            )
            delta, low, high = paired_bootstrap_ci(
                base_values, new_values, n_resamples=bootstrap
            )
            cells.append(_fmt_delta(delta, low, high))
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 失败/翻转样本",
            "",
            "只列出相对基线发生 hit/miss 翻转的问题；`best_overlap` 越低，说明检索器越没有覆盖 gold evidence。",
            "",
        ]
    )

    base_by_id = _result_by_id(baseline)
    flips = 0
    for config in configs[1:]:
        name = config["name"]
        current = _result_by_id(results[name])
        cases = []
        for row_id in sorted(set(base_by_id) & set(current)):
            old = base_by_id[row_id]
            new = current[row_id]
            if old.get("recall") is None or new.get("recall") is None:
                continue
            if bool(old.get("recall")) == bool(new.get("recall")):
                continue
            direction = "基线命中 → 本策略未命中" if old.get("recall") else "基线未命中 → 本策略命中"
            cases.append(
                f"- `{row_id}` {direction}：{new.get('query')}\n"
                f"  - 基线 top files：{_top_files(old)}\n"
                f"  - 本策略 top files：{_top_files(new)}\n"
                f"  - best_overlap：{new.get('best_evidence_overlap', 0):.3f}\n"
                f"  - gold evidence：{_quote_for(golden, row_id)}"
            )
        if cases:
            flips += len(cases)
            lines.append(f"### `{name}`（{len(cases)} 条）")
            lines.append("")
            lines.extend(cases)
            lines.append("")

    if not flips:
        lines.append("没有出现 hit/miss 翻转；差异只体现在 MRR / Precision / nDCG 或延迟上。")
        lines.append("")

    base_misses = [
        row
        for row in baseline.get("results", [])
        if row.get("recall") == 0.0
    ]
    lines.append(f"### 基线未命中（{len(base_misses)} 条）")
    lines.append("")
    if base_misses:
        for row in base_misses[:10]:
            lines.append(
                f"- `{row.get('id')}` {row.get('query')}；"
                f"best_overlap={row.get('best_evidence_overlap', 0):.3f}；"
                f"top files：{_top_files(row)}；"
                f"gold evidence：{_quote_for(golden, str(row.get('id')))}"
            )
    else:
        lines.append("基线全部命中。")
    lines.append("")

    lines.extend(
        [
            "## 解读",
            "",
            "- **Recall** 衡量“至少命中一条 gold evidence”，对 chunk 边界最敏感。",
            "- **MRR / nDCG** 衡量相关证据是否排在前面；相同 Recall 下，它们能区分排序质量。",
            "- **Precision** 衡量前 k 条里有多少是证据相关，能暴露“大 chunk 把噪声一起带进来”。",
            "- 如果 overlap 策略提升 Recall 但提高索引大小/延迟，应报告这个 trade-off，而不是只报最高分。",
            "",
        ]
    )
    if experiment == "markdown-strategy":
        lines.extend(
            [
                "## 结论",
                "",
                "- 结构切分确实生效：表格子块的表头保留率和列数一致率明显高于"
                "递归策略（见上表）。",
                "- 但 Recall@5 没有提升；MRR/nDCG 的正向变化 95% CI 跨 0，"
                "Precision 的下降 CI 不跨 0；主要代价是子块更小、更多，P95 延迟更高。",
                "- 因此**不替换默认 `fine-overlap`**。`markdown-heading` 适合章节/表格"
                "结构本身就是检索信号的 Markdown 知识库；下一步应增加“短章节合并”"
                "或按文档调 child size，再验证 Precision 和延迟。",
                "",
                "> 单次 50 题评测仍有抽样波动；本报告的 95% CI 是配对 bootstrap，"
                "解释时应同时看区间和效应量。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "> 单次 50 题评测仍有抽样波动；本报告的 95% CI 是配对 bootstrap，"
                "解释时应同时看区间和效应量。",
                "",
            ]
        )
    return "\n".join(lines)


def run_command(command: List[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def _table_quality(text: str) -> tuple[bool, bool] | None:
    """Return (has_header, rows_have_same_column_count) for the first table."""

    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if not lines:
        return None
    if len(lines) < 2:
        return False, False
    separator = lines[1]
    if "|" not in separator or "-" not in separator:
        return False, False
    header_columns = lines[0].count("|")
    rows = lines[2:]
    return True, all(row.count("|") == header_columns for row in rows)


def load_structure_stats(plan: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Measure chunk length and table integrity directly from Milvus."""

    try:
        import asyncio

        sys.path.insert(0, str(ROOT / "backend"))
        from milvus_client import get_milvus_client

        async def _read() -> Dict[str, Dict[str, float]]:
            client = await get_milvus_client()
            collection = client.knowledge_base_collection
            stats: Dict[str, Dict[str, float]] = {}
            for item in plan:
                rows = await client.client.query(
                    collection_name=collection,
                    filter=f'knowledge_base_id == "{item["kb_id"]}"',
                    output_fields=["text"],
                    limit=10000,
                )
                texts = [str(row.get("text") or "") for row in rows]
                lengths = sorted(len(text) for text in texts) or [0]
                table_chunks = 0
                table_with_header = 0
                table_with_columns = 0
                for text in texts:
                    quality = _table_quality(text)
                    if quality is None:
                        continue
                    table_chunks += 1
                    has_header, columns_match = quality
                    table_with_header += int(has_header)
                    table_with_columns += int(has_header and columns_match)
                stats[item["name"]] = {
                    "children": float(len(texts)),
                    "median_chars": float(lengths[len(lengths) // 2]),
                    "p90_chars": float(lengths[int((len(lengths) - 1) * 0.9)]),
                    "table_chunks": float(table_chunks),
                    "table_header_rate": (
                        table_with_header / table_chunks if table_chunks else 0.0
                    ),
                    "table_column_rate": (
                        table_with_columns / table_chunks if table_chunks else 0.0
                    ),
                }
            return stats

        return asyncio.run(_read())
    except Exception as exc:
        print(f"structure stats unavailable: {exc}")
        return {}


def load_index_counts(
    plan: List[Dict[str, Any]], user_id: int
) -> Dict[str, tuple[int, int]]:
    """Read parent/child counts so the report shows the index-size trade-off."""

    try:
        import asyncio

        sys.path.insert(0, str(ROOT / "backend"))
        from postgresql_client import get_postgresql_client

        async def _read() -> Dict[str, tuple[int, int]]:
            client = await get_postgresql_client()
            await client.init_pool()
            counts: Dict[str, tuple[int, int]] = {}
            try:
                for item in plan:
                    parents = await client.pool.fetchval(
                        """
                        SELECT COUNT(*) FROM parent_chunks
                        WHERE knowledge_base_id = $1 AND user_id = $2
                        """,
                        item["kb_id"],
                        user_id,
                    )
                    children = await client.pool.fetchval(
                        """
                        SELECT COUNT(*) FROM chunk_hashes
                        WHERE knowledge_base_id = $1 AND user_id = $2
                        """,
                        item["kb_id"],
                        user_id,
                    )
                    counts[item["name"]] = (int(parents or 0), int(children or 0))
            finally:
                await client.close()
            return counts

        return asyncio.run(_read())
    except Exception as exc:
        print(f"index counts unavailable: {exc}")
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="对比不同 chunking 策略的检索质量")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument(
        "--experiment",
        choices=["docx-sizes", "markdown-strategy"],
        default="docx-sizes",
        help="docx-sizes=原 DOCX 尺寸对比；markdown-strategy=同一 Markdown 语料上的策略对比",
    )
    parser.add_argument(
        "--corpus-format",
        choices=["docx", "md"],
        default="",
        help="默认由 --experiment 决定；一般不需要显式设置",
    )
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS))
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "hybrid+rerank"],
        default="hybrid+rerank",
        help="所有配置使用同一个检索/重排模式，只改变 chunking",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--kb-prefix", default="violet-chunk")
    parser.add_argument("--configs", default="", help="自定义配置 JSON 文件")
    parser.add_argument(
        "--configs-module",
        action="store_true",
        help="打印默认配置后退出",
    )
    parser.add_argument("--results-dir", default="")
    parser.add_argument(
        "--report",
        default="",
        help="报告路径；默认由 --experiment 决定",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--skip-seed", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用 results-dir 中已存在的 <config>.json，只跑缺失配置",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    corpus_format = args.corpus_format or (
        "md" if args.experiment == "markdown-strategy" else "docx"
    )
    if args.experiment == "markdown-strategy" and corpus_format != "md":
        raise SystemExit("markdown-strategy 实验必须在 Markdown 语料上运行")
    report_path = Path(
        args.report
        or (MARKDOWN_REPORT if args.experiment == "markdown-strategy" else DEFAULT_REPORT)
    )

    configs = load_configs(args.configs or None, args.experiment)
    names = [str(config.get("name") or "") for config in configs]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise SystemExit("每个配置都需要唯一的 name")

    if args.configs_module:
        print(json.dumps(configs, ensure_ascii=False, indent=2))
        return

    run_id = _safe_name(args.run_id)
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else ROOT / "evals" / "results" / "chunking" / run_id
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for config in configs:
        name = _safe_name(str(config["name"]))
        kb_id = _safe_name(f"{args.kb_prefix}-{run_id}-{name}", limit=120)
        plan.append({**config, "name": name, "kb_id": kb_id})

    print("chunking experiment plan:")
    for item in plan:
        print(
            f"- {item['name']}: kb={item['kb_id']} "
            f"strategy={item.get('strategy', 'recursive')} "
            f"parent={item.get('parent_chunk_size')}/{item.get('parent_chunk_overlap')} "
            f"child={item.get('child_chunk_size')}/{item.get('child_chunk_overlap')} "
            f"granularity={item.get('granularity', 'child')}"
        )
    if args.dry_run:
        return

    for item in plan:
        kb_id = item["kb_id"]
        output_path = results_dir / f"{item['name']}.json"
        if args.resume and output_path.exists():
            print(f"\nskip existing result: {output_path}")
            continue
        if not args.skip_seed:
            # Start from a clean namespace so duplicate hashes from an earlier
            # run cannot silently keep the old chunking.
            import requests

            response = requests.delete(
                f"{args.base_url.rstrip('/')}/knowledge-bases/{kb_id}",
                params={"user_id": args.user_id},
                timeout=30,
            )
            if response.status_code not in (200, 404):
                raise SystemExit(
                    f"清理 {kb_id} 失败: {response.status_code} {response.text[:200]}"
                )
            seed_command = [
                sys.executable,
                str(ROOT / "evals" / "seed_violet_kb.py"),
                "--base-url",
                args.base_url,
                "--kb-id",
                kb_id,
                "--user-id",
                str(args.user_id),
                "--corpus-dir",
                str(Path(args.corpus_dir)),
                "--format",
                corpus_format,
                "--strategy",
                str(item.get("strategy", "recursive")),
                "--parent-chunk-size",
                str(item["parent_chunk_size"]),
                "--parent-chunk-overlap",
                str(item["parent_chunk_overlap"]),
                "--child-chunk-size",
                str(item["child_chunk_size"]),
                "--child-chunk-overlap",
                str(item["child_chunk_overlap"]),
            ]
            run_command(seed_command)

        meta_path = results_dir / f"{item['name']}_meta.json"
        variant_golden = results_dir / f"{item['name']}.golden.jsonl"
        materialize_golden(Path(args.golden), variant_golden, kb_id)
        eval_command = [
            sys.executable,
            str(ROOT / "evals" / "run_eval.py"),
            "--kb-id",
            kb_id,
            "--user-id",
            str(args.user_id),
            "--golden",
            str(variant_golden),
            "--k",
            str(args.k),
            "--retrieval-mode",
            "baseline",
            "--mode",
            args.mode,
            "--granularity",
            str(item.get("granularity", "child")),
            "--chunk-config",
            json.dumps(item, ensure_ascii=False),
            "--output",
            str(output_path),
            "--meta-output",
            str(meta_path),
            "--run-id",
            f"{run_id}-{item['name']}",
        ]
        if args.limit:
            eval_command.extend(["--limit", str(args.limit)])
        run_command(eval_command)

    results = {}
    for item in plan:
        path = results_dir / f"{item['name']}.json"
        results[item["name"]] = json.loads(path.read_text(encoding="utf-8"))

    golden = load_golden(Path(args.golden))
    index_counts = load_index_counts(plan, args.user_id)
    structure_stats = (
        load_structure_stats(plan) if args.experiment == "markdown-strategy" else {}
    )
    markdown = build_report(
        run_id=run_id,
        configs=plan,
        results=results,
        golden=golden,
        k=args.k,
        bootstrap=args.bootstrap,
        experiment=args.experiment,
        corpus_format=corpus_format,
        index_counts=index_counts,
        structure_stats=structure_stats,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")
    print(f"\nreport: {report_path}")
    print(f"results: {results_dir}")


if __name__ == "__main__":
    main()
