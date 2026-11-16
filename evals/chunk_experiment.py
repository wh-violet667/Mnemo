"""Run a controlled chunking comparison and write a markdown report.

For every configuration the script:

1. deletes/creates a separate knowledge base (separate Milvus namespace);
2. uploads the same 12 DOCX files with a per-file chunk config;
3. runs ``run_eval.py`` at chunk (or parent) granularity;
4. compares each strategy to the first one with paired bootstrap intervals;
5. lists questions where the ranking flips, with the gold evidence quote.

The corpus and embedding/reranker models stay fixed.  Only chunking changes.

Usage:
    python evals/chunk_experiment.py --dry-run
    python evals/chunk_experiment.py --limit 10
    python evals/chunk_experiment.py --run-id full
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

DEFAULT_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "child-200-o0",
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 200,
        "child_chunk_size": 200,
        "child_chunk_overlap": 0,
        "granularity": "child",
    },
    {
        "name": "child-200-o50",
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 200,
        "child_chunk_size": 200,
        "child_chunk_overlap": 50,
        "granularity": "child",
    },
    {
        "name": "child-400-o0",
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 200,
        "child_chunk_size": 400,
        "child_chunk_overlap": 0,
        "granularity": "child",
    },
]

METRICS = [
    ("recall", "Recall"),
    ("rr", "MRR"),
    ("precision", "Precision"),
    ("ndcg", "nDCG"),
]


def _safe_name(value: str, limit: int = 40) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "-", value).strip("-")
    return (cleaned or "strategy")[:limit]


def load_configs(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return [dict(item) for item in DEFAULT_CONFIGS]
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
) -> str:
    baseline_name = configs[0]["name"]
    baseline = results[baseline_name]
    lines: List[str] = [
        "# Chunking 策略对比",
        "",
        f"> run: `{run_id}`；同一语料、同一 embedding/rerank 模型，只改 chunk 配置。",
        "",
        "## 实验配置",
        "",
        "| 策略 | 父块 | 子块 | 粒度 |",
        "|---|---:|---:|---|",
    ]
    for config in configs:
        lines.append(
            f"| `{config['name']}` | "
            f"{config.get('parent_chunk_size')}/{config.get('parent_chunk_overlap')} | "
            f"{config.get('child_chunk_size')}/{config.get('child_chunk_overlap')} | "
            f"{config.get('granularity', 'child')} |"
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
            "> 单次 50 题评测仍有抽样波动；本报告的 95% CI 是配对 bootstrap，解释时应同时看区间和效应量。",
            "",
        ]
    )
    return "\n".join(lines)


def run_command(command: List[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="对比不同 chunking 策略的检索质量")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--user-id", type=int, default=1)
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
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--skip-seed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = load_configs(args.configs or None)
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
            f"parent={item.get('parent_chunk_size')}/{item.get('parent_chunk_overlap')} "
            f"child={item.get('child_chunk_size')}/{item.get('child_chunk_overlap')} "
            f"granularity={item.get('granularity', 'child')}"
        )
    if args.dry_run:
        return

    for item in plan:
        kb_id = item["kb_id"]
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

        output_path = results_dir / f"{item['name']}.json"
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
    markdown = build_report(
        run_id=run_id,
        configs=plan,
        results=results,
        golden=golden,
        k=args.k,
        bootstrap=args.bootstrap,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")
    print(f"\nreport: {report_path}")
    print(f"results: {results_dir}")


if __name__ == "__main__":
    main()
