from backend.citations import (
    build_numbered_context,
    dedupe_documents,
    order_by_rerank,
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
