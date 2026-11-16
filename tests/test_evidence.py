from evals.evidence import (
    best_paragraph_match,
    document_evidence_score,
    find_quote_span,
    is_document_relevant,
    quote_overlap,
)


def test_quote_overlap_exact_and_boundary():
    quote = "退款申请必须在购买后 30 天内提交。"
    assert quote_overlap("说明：退款申请必须在购买后 30 天内提交。", quote) == 1.0
    assert quote_overlap("退款申请必须在购买后 30 天内", quote) > 0.55
    assert quote_overlap("完全无关的内容", quote) == 0.0


def test_evidence_relevance_requires_both_file_and_quote():
    row = {
        "gold_evidence": [
            {"file": "refund.docx", "quote": "退款申请必须在购买后 30 天内提交。"}
        ]
    }
    hit = {
        "file_name": "refund.docx",
        "text": "退款申请必须在购买后 30 天内提交，逾期不受理。",
    }
    wrong_file = {**hit, "file_name": "other.docx"}
    wrong_text = {"file_name": "refund.docx", "text": "退款需要联系客服。"}

    assert is_document_relevant(hit, row)
    assert not is_document_relevant(wrong_file, row)
    assert not is_document_relevant(wrong_text, row)
    assert document_evidence_score(hit, row) >= 0.9


def test_find_quote_span_returns_offsets_and_context():
    text = "前言。退款申请必须在购买后 30 天内提交。后记。"
    result = find_quote_span(text, "30 天")
    assert result is not None
    quote, start, end = result
    assert "30 天" in quote
    assert text[start:end] == quote


def test_best_paragraph_match_finds_paraphrased_answer():
    text = "第一段。\n\n退款期限是购买后 30 天内，逾期不受理。\n\n第三段。"
    result = best_paragraph_match(text, "购买后 30 天内可以退款吗")
    assert result is not None
    quote, start, end, score = result
    assert "30 天" in quote
    assert text[start:end] == quote
    assert score > 0.4
