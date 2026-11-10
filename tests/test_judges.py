from evals.judges import (
    normalize_judge_a,
    normalize_judge_b,
    parse_json_object,
)


def test_parse_json_object_strips_code_fence():
    data = parse_json_object('```json\n{"score": 2, "reason": "ok"}\n```')
    assert data["score"] == 2


def test_normalize_judge_a_clamps_and_cleans_fields():
    result = normalize_judge_a(
        {
            "score": 9,
            "refused": True,
            "missing_facts": ["  A ", ""],
            "wrong_facts": ["B"],
        }
    )
    assert result["score"] == 2
    assert result["refused"] is True
    assert result["missing_facts"] == ["A"]
    assert result["wrong_facts"] == ["B"]


def test_normalize_judge_a_does_not_give_zero_without_missing_or_wrong_facts():
    result = normalize_judge_a(
        {"score": 0, "refused": False, "missing_facts": [], "wrong_facts": []}
    )
    assert result["score"] == 2


def test_normalize_judge_a_keeps_zero_for_refusal():
    result = normalize_judge_a(
        {"score": 0, "refused": True, "missing_facts": [], "wrong_facts": []}
    )
    assert result["score"] == 0


def test_normalize_judge_b_filters_bad_claims_and_marks():
    result = normalize_judge_b(
        {
            "claims": [
                {"claim": "A", "verdict": "SUPPORTED", "cited_marks": ["1", "x"]},
                {"claim": "", "verdict": "supported"},
            ],
            "citations": [
                {"mark": "1", "valid": True, "supports_claim": True},
                {"mark": "bad", "valid": True},
            ],
        }
    )
    assert len(result["claims"]) == 1
    assert result["claims"][0]["verdict"] == "supported"
    assert result["claims"][0]["cited_marks"] == [1]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["mark"] == 1
