"""检索评测指标（纯函数，无第三方依赖）。"""

from __future__ import annotations

from typing import Dict, List, Sequence


def recall_at_k(relevances: Sequence[bool], k: int) -> float:
    """前 k 条里是否至少有一条相关。"""

    if k <= 0:
        raise ValueError("k must be positive")
    return 1.0 if any(relevances[:k]) else 0.0


def reciprocal_rank(relevances: Sequence[bool]) -> float:
    """第一条相关结果的倒数排名；没有相关结果时为 0。"""

    for rank, relevant in enumerate(relevances, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数，p 取 0-100。"""

    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * (p / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize(rows: Sequence[Dict], k: int = 5) -> Dict[str, float]:
    """汇总逐题结果，返回 Recall@k / MRR / P50 / P95。"""

    if not rows:
        return {
            f"recall@{k}": 0.0,
            "mrr": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
        }

    recalls = [float(row.get("recall", 0.0)) for row in rows]
    rrs = [float(row.get("rr", 0.0)) for row in rows]
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]

    return {
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
    }
