from evals.chunk_experiment import build_report, paired_bootstrap_ci


def test_paired_bootstrap_ci_is_deterministic():
    baseline = [1.0, 0.0, 1.0, 0.0]
    candidate = [1.0, 1.0, 1.0, 0.0]
    first = paired_bootstrap_ci(baseline, candidate, n_resamples=200, seed=7)
    second = paired_bootstrap_ci(baseline, candidate, n_resamples=200, seed=7)
    assert first == second
    assert first[0] == 0.25
    assert first[1] <= first[0] <= first[2]


def test_build_report_contains_metrics_and_flip_case():
    configs = [
        {"name": "a", "parent_chunk_size": 1000, "parent_chunk_overlap": 200,
         "child_chunk_size": 200, "child_chunk_overlap": 0, "granularity": "child"},
        {"name": "b", "parent_chunk_size": 1000, "parent_chunk_overlap": 200,
         "child_chunk_size": 200, "child_chunk_overlap": 50, "granularity": "child"},
    ]
    results = {
        "a": {
            "summary": {"recall@5": 1.0, "mrr": 1.0, "precision@5": 0.2,
                        "ndcg@5": 0.8, "p50_ms": 10, "p95_ms": 20},
            "results": [
                {"id": "q1", "query": "问题", "recall": 1.0, "rr": 1.0,
                 "precision": 0.2, "ndcg": 0.8, "latency_ms": 10,
                 "top_files": ["a.docx"], "best_evidence_overlap": 1.0},
            ],
        },
        "b": {
            "summary": {"recall@5": 0.0, "mrr": 0.0, "precision@5": 0.0,
                        "ndcg@5": 0.0, "p50_ms": 10, "p95_ms": 20},
            "results": [
                {"id": "q1", "query": "问题", "recall": 0.0, "rr": 0.0,
                 "precision": 0.0, "ndcg": 0.0, "latency_ms": 10,
                 "top_files": ["b.docx"], "best_evidence_overlap": 0.2},
            ],
        },
    }
    markdown = build_report(
        run_id="test",
        configs=configs,
        results=results,
        golden={"q1": {"gold_evidence": [{"quote": "证据"}]}},
        k=5,
        bootstrap=100,
    )

    assert "Chunking 策略对比" in markdown
    assert "Recall@k" in markdown
    assert "ΔRecall" in markdown
    assert "q1" in markdown
    assert "best_overlap" in markdown
