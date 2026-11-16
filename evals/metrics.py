"""检索评测指标：Recall / MRR / Precision / nDCG（纯函数，无第三方依赖）。"""

from __future__ import annotations

import math
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


def precision_at_k(relevances: Sequence[bool], k: int) -> float:
    """前 k 条中相关结果的比例；分母固定为 k，少召回也会被惩罚。"""

    if k <= 0:
        raise ValueError("k must be positive")
    return sum(1 for relevant in relevances[:k] if relevant) / float(k)


def ndcg_at_k(relevances: Sequence[bool], k: int) -> float:
    """二值相关性 nDCG@k（IDCG 用前 k 条中实际相关条数）。"""

    if k <= 0:
        raise ValueError("k must be positive")

    gains = [1.0 if relevant else 0.0 for relevant in relevances[:k]]
    if not any(gains):
        return 0.0

    dcg = sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
        if gain
    )
    ideal_count = min(sum(1 for gain in gains if gain), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    return dcg / idcg if idcg else 0.0


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
            f"precision@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
        }

    recalls = [float(row.get("recall", 0.0)) for row in rows]
    rrs = [float(row.get("rr", 0.0)) for row in rows]
    precisions = [float(row.get("precision", 0.0)) for row in rows]
    ndcgs = [float(row.get("ndcg", 0.0)) for row in rows]
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]

    return {
        f"recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
        f"precision@{k}": sum(precisions) / len(precisions),
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
    }
