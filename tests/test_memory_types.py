import time

from memory_types import (
    SUPERSEDE_TYPES,
    TTL_DAYS,
    dedupe_key,
    normalize_memory_item,
    score_memory,
)


NOW = 1_800_000_000
DAY = 86400


def test_semantic_does_not_decay_with_hourly_recency():
    old = {
        "memory_type": "semantic",
        "importance": 0.8,
        "confidence": 0.9,
        "created_at": NOW - 30 * DAY,
        "last_access_at": NOW - 30 * DAY,
    }
    new = {
        "memory_type": "semantic",
        "importance": 0.8,
        "confidence": 0.9,
        "created_at": NOW,
        "last_access_at": NOW,
    }

    assert score_memory(old, relevance=0.8, now=NOW) == score_memory(
        new, relevance=0.8, now=NOW
    )


def test_summary_and_episodic_do_decay():
    old_summary = {
        "memory_type": "summary",
        "created_at": NOW - 30 * DAY,
        "last_access_at": NOW - 30 * DAY,
    }
    new_summary = {
        "memory_type": "summary",
        "created_at": NOW,
        "last_access_at": NOW,
    }
    assert score_memory(old_summary, relevance=0.8, now=NOW) < score_memory(
        new_summary, relevance=0.8, now=NOW
    )

    old_event = {
        "memory_type": "episodic",
        "event_time": NOW - 180 * DAY,
        "created_at": NOW - 180 * DAY,
    }
    new_event = {
        "memory_type": "episodic",
        "event_time": NOW,
        "created_at": NOW,
    }
    assert score_memory(old_event, relevance=0.8, now=NOW) < score_memory(
        new_event, relevance=0.8, now=NOW
    )


def test_procedural_success_rate_affects_score():
    good = {
        "memory_type": "procedural",
        "importance": 0.7,
        "confidence": 0.7,
        "usage_count": 5,
        "success_count": 9,
        "failure_count": 1,
    }
    bad = {
        "memory_type": "procedural",
        "importance": 0.7,
        "confidence": 0.7,
        "usage_count": 5,
        "success_count": 1,
        "failure_count": 9,
    }
    assert score_memory(good, relevance=0.8, now=NOW) > score_memory(
        bad, relevance=0.8, now=NOW
    )


def test_semantic_dedupe_uses_subject_and_predicate():
    first = dedupe_key(
        "semantic",
        "用户喜欢表格",
        {"subject": "用户", "predicate": "回答风格"},
    )
    second = dedupe_key(
        "semantic",
        "用户偏好结构化回答",
        {"subject": "用户", "predicate": "回答风格"},
    )
    assert first == second


def test_episodic_dedupe_is_day_scoped():
    first = dedupe_key(
        "episodic",
        "用户比较 FNN 和 CNN",
        event_time=NOW,
    )
    second = dedupe_key(
        "episodic",
        "用户比较 FNN 和 CNN",
        event_time=NOW + DAY,
    )
    assert first != second
    assert "episodic" not in SUPERSEDE_TYPES


def test_normalize_episodic_requires_event_time_and_ttl():
    item = normalize_memory_item(
        "episodic",
        {"content": "用户展示论文摘要", "importance_score": 0.7},
        now=NOW,
        default_event_time=NOW - DAY,
    )
    assert item is not None
    assert item["event_time"] == NOW - DAY
    assert item["expires_at"] == NOW + TTL_DAYS["episodic"] * DAY
    assert item["attributes"]["event_time"] == NOW - DAY


def test_normalize_summary_has_ttl_semantic_does_not():
    summary = normalize_memory_item(
        "summary",
        {"content": "用户讨论了 RAG", "attributes": {"thread_id": "u1:c1"}},
        now=NOW,
        thread_id="u1:c1",
        summary_id="sum-1",
    )
    semantic = normalize_memory_item(
        "semantic",
        {"content": "用户偏好表格", "attributes": {"subject": "用户"}},
        now=NOW,
    )

    assert summary is not None and summary["expires_at"] == NOW + TTL_DAYS["summary"] * DAY
    assert semantic is not None and semantic["expires_at"] == 0


def test_normalize_procedural_keeps_steps_and_success():
    item = normalize_memory_item(
        "procedural",
        {
            "content": "技术方案对比时先列维度再给建议",
            "steps": ["列维度", "给建议"],
            "success": False,
        },
        now=int(time.time()),
    )
    assert item is not None
    assert item["attributes"]["steps"] == ["列维度", "给建议"]
    assert item["attributes"]["success"] is False

    default_success = normalize_memory_item(
        "procedural",
        {"content": "默认成功的流程"},
        now=int(time.time()),
    )
    assert default_success is not None
    assert default_success["attributes"]["success"] is True
