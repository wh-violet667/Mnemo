from evals.metrics import (
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)


def test_recall_at_k_checks_only_top_k():
    assert recall_at_k([False, False, True], k=3) == 1.0
    assert recall_at_k([False, False, True], k=2) == 0.0
    assert recall_at_k([], k=5) == 0.0


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank([False, True, True]) == 0.5
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_precision_at_k_and_ndcg():
    relevances = [False, True, False, True]

    assert precision_at_k(relevances, 4) == 0.5
    assert precision_at_k(relevances, 2) == 0.5
    assert 0.0 < ndcg_at_k(relevances, 4) < 1.0
    assert ndcg_at_k([False, False], 2) == 0.0


def test_percentile_interpolates():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 25.0
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 40.0
    assert percentile([], 95) == 0.0


def test_summarize_aggregates_rows():
    rows = [
        {"recall": 1.0, "rr": 1.0, "latency_ms": 100.0},
        {"recall": 0.0, "rr": 0.0, "latency_ms": 300.0},
    ]

    summary = summarize(rows, k=5)

    assert summary["recall@5"] == 0.5
    assert summary["mrr"] == 0.5
    assert summary["p50_ms"] == 200.0
    assert summary["p95_ms"] == 290.0
