from evals.answer_metrics import (
    aggregate_records,
    citation_metrics,
    compute_retrieval_metrics,
    extract_citation_marks,
    match_required_facts,
)


def test_match_required_facts_normalises_spacing_and_case():
    result = match_required_facts(
        "制作公司是京都动画（Kyoto Animation）。",
        ["京都动画", "Kyoto Animation"],
    )
    assert result["coverage"] == 1.0
    assert result["missing"] == []


def test_extract_citation_marks_supports_multiple_forms():
    assert extract_citation_marks("答案 [1][3]，另见 [2, 4]。") == [1, 3, 2, 4]


def test_multi_hop_coverage_exposes_missing_document():
    row = {
        "expected_files": ["01-overview.docx", "07-production.docx"],
    }
    only_one = [
        {"parent_id": "p1", "file_name": "01-overview.docx", "text": "..."},
    ]
    both = only_one + [
        {"parent_id": "p2", "file_name": "07-production.docx", "text": "..."},
    ]

    partial = compute_retrieval_metrics(row, only_one)
    full = compute_retrieval_metrics(row, both)

    assert partial["hit_at_5"] == 1.0
    assert partial["expected_context_coverage"] == 0.5
    assert partial["all_evidence_hit"] == 0.0

    assert full["expected_context_coverage"] == 1.0
    assert full["all_evidence_hit"] == 1.0
    assert full["mrr"] == 1.0


def test_citation_metrics_separates_validity_precision_and_chunk_support():
    answer = "制作公司是京都动画 [1][2]。"
    sources = [
        {
            "index": 1,
            "parent_id": "p1",
            "file_name": "07-production.docx",
            "snippet": "京都动画制作",
        },
        {
            "index": 2,
            "parent_id": "p2",
            "file_name": "01-overview.docx",
            "snippet": "作品概况",
        },
    ]
    judge_b = {
        "claims": [
            {
                "claim": "制作公司是京都动画",
                "verdict": "supported",
                "needs_citation": True,
                "cited_marks": [1],
            }
        ],
        "citations": [
            {
                "mark": 1,
                "valid": True,
                "supports_claim": True,
                "quote": "京都动画",
            },
            {
                "mark": 2,
                "valid": True,
                "supports_claim": False,
                "quote": "",
            },
        ],
    }

    metrics = citation_metrics(
        answer,
        sources,
        judge_b,
        {"p1": "京都动画制作了本作。", "p2": "作品概况。"},
    )

    assert metrics["citation_validity"] == 1.0
    assert metrics["citation_precision"] == 0.5
    assert metrics["citation_coverage"] == 1.0
    assert metrics["chunk_support_rate"] == 1.0


def test_aggregate_separates_correctness_when_retrieval_hits_or_misses():
    base = {
        "query_type": "single_hop",
        "judge_b": None,
        "citation_metrics": {},
        "fact_check": {"coverage": 1.0},
        "latency_ms": 100.0,
        "errors": [],
    }
    hit_correct = {
        **base,
        "judge_a": {"score": 2, "refused": False},
        "retrieval_metrics": {"hit_at_5": 1.0, "mrr": 1.0},
    }
    miss_wrong = {
        **base,
        "judge_a": {"score": 0, "refused": False},
        "retrieval_metrics": {"hit_at_5": 0.0, "mrr": 0.0},
    }

    summary = aggregate_records([hit_correct, miss_wrong])["overall"]

    assert summary["n_hit"] == 1
    assert summary["n_miss"] == 1
    assert summary["correctness_given_hit"] == 1.0
    assert summary["correctness_given_miss"] == 0.0
