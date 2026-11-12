from backend.citations import (
    build_numbered_context,
    claims_by_marker,
    dedupe_documents,
    enrich_sources_with_quotes,
    locate_quote,
    order_by_rerank,
    select_supporting_quote,
)


def test_build_numbered_context_keeps_order_and_skips_empty():
    documents = [
        {"parent_id": "p1", "file_name": "a.pdf", "text": "第一段"},
        {"parent_id": "p2", "file_name": "b.pdf", "text": "   "},
        {"parent_id": "p3", "file_name": "c.pdf", "text": "第三段", "score": 0.8},
    ]

    text, sources = build_numbered_context(documents)

    assert "[1] 来源：a.pdf" in text
    assert "[2] 来源：c.pdf" in text
    assert "[3]" not in text
    assert [source["index"] for source in sources] == [1, 2]
    assert sources[1]["score"] == 0.8


def test_build_numbered_context_continues_from_start_index():
    documents = [{"parent_id": "p1", "file_name": "a.pdf", "text": "内容"}]

    text, sources = build_numbered_context(documents, start_index=4)

    assert text.startswith("[4] 来源：a.pdf")
    assert sources[0]["index"] == 4


def test_build_numbered_context_returns_empty_for_blank_documents():
    text, sources = build_numbered_context([{"text": " "}, {}])

    assert text == ""
    assert sources == []


def test_dedupe_documents_preserves_first_occurrence_order():
    documents = [
        {"parent_id": "p1", "text": "一"},
        {"parent_id": "p2", "text": "二"},
        {"parent_id": "p1", "text": "一（重复）"},
    ]

    result = dedupe_documents(documents)

    assert [doc["parent_id"] for doc in result] == ["p1", "p2"]
    assert result[0]["text"] == "一"


def test_order_by_rerank_maps_index_to_original_document():
    documents = [
        {"parent_id": "p1", "file_name": "a.pdf", "text": "甲"},
        {"parent_id": "p2", "file_name": "b.pdf", "text": "乙"},
    ]
    rerank_result = {
        "output": {
            "results": [
                {"index": 1, "document": {"text": "乙"}, "relevance_score": 0.9},
                {"index": 0, "document": {"text": "甲"}, "relevance_score": 0.5},
            ]
        }
    }

    ordered = order_by_rerank(documents, rerank_result)

    assert [doc["parent_id"] for doc in ordered] == ["p2", "p1"]
    assert ordered[0]["file_name"] == "b.pdf"
    assert ordered[0]["score"] == 0.9


def test_order_by_rerank_falls_back_to_text_and_original_order():
    documents = [
        {"parent_id": "p1", "text": "甲"},
        {"parent_id": "p2", "text": "乙"},
    ]

    by_text = order_by_rerank(
        documents,
        {"output": {"results": [{"document": {"text": "乙"}, "relevance_score": 0.8}]}},
    )
    assert [doc["parent_id"] for doc in by_text] == ["p2"]

    fallback = order_by_rerank(documents, None)
    assert [doc["parent_id"] for doc in fallback] == ["p1", "p2"]


def test_claims_by_marker_handles_consecutive_markers():
    claims = claims_by_marker("导演是石立太一 [1][2]。播出时间是 2018 年 [3]。")

    assert claims[1] == ["导演是石立太一"]
    assert claims[2] == ["导演是石立太一"]
    assert claims[3] == ["播出时间是 2018 年"]


def test_select_supporting_quote_picks_the_matching_sentence():
    source = (
        "制作信息\n\n制作公司：京都动画（Kyoto Animation）。导演：石立太一。\n\n"
        "播出信息：2018 年 1 月首播。"
    )

    quote = select_supporting_quote("导演是谁？石立太一。", source)

    assert "石立太一" in quote
    assert "京都动画" not in quote


def test_locate_quote_returns_paragraph_number():
    source = "第一段内容。\n\n第二段有目标句。\n\n第三段收尾。"

    assert locate_quote(source, "目标句") == "第 2 段"
    assert locate_quote(source, "不存在的句子") == "第 1 段"


def test_enrich_sources_adds_quote_and_locator_without_full_text():
    source_text = (
        "制作信息\n\n制作公司：京都动画。导演：石立太一。\n\n播出信息：2018 年 1 月。"
    )
    sources = [
        {
            "index": 1,
            "file_name": "07-production.docx",
            "parent_id": "p1",
            "snippet": source_text[:240],
        }
    ]

    enriched = enrich_sources_with_quotes(
        "导演是石立太一 [1]。",
        sources,
        {"p1": source_text},
    )

    assert len(enriched) == 1
    assert "石立太一" in enriched[0]["quote"]
    assert enriched[0]["locator"] == "第 2 段"
    assert "text" not in enriched[0]
    assert enriched[0]["file_name"] == "07-production.docx"


def test_enrich_sources_falls_back_to_snippet_without_source_text():
    sources = [
        {
            "index": 1,
            "file_name": "a.pdf",
            "parent_id": "p1",
            "snippet": "只有片段，没有全文。",
        }
    ]

    enriched = enrich_sources_with_quotes("问题 [1]。", sources, {})

    assert enriched[0]["quote"] == "只有片段，没有全文。"
    assert enriched[0]["locator"] == "第 1 段"
