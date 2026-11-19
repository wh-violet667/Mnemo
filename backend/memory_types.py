"""Contracts and type-specific behavior for Mnemo's four memory types.

The four types are not interchangeable rows in one table.  Each one has a
different truth condition, retrieval signal and lifecycle:

* ``summary``   — what happened in one conversation window;
* ``semantic``  — stable facts and preferences;
* ``episodic``  — a specific event with a time and outcome;
* ``procedural``— a reusable method with success/failure feedback.

Keeping these rules here (instead of spread across prompts and scoring code)
makes extraction validation, deduplication and ranking testable without a
database or an LLM.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

MEMORY_TYPES = ("summary", "semantic", "episodic", "procedural")

# Only these types may replace an older active memory.  An episodic memory is
# an event: two similar events can both be true, so similarity-based supersede
# would silently delete history.
SUPERSEDE_TYPES = frozenset({"semantic", "procedural"})

DEFAULT_RETRIEVAL_K = {
    "summary": 1,
    "semantic": 3,
    "episodic": 2,
    "procedural": 2,
}

# Time-based archival.  Semantic/procedural memories are kept until explicitly
# superseded or deleted; summary/episodic memories have a natural age.
TTL_DAYS = {
    "summary": 30,
    "semantic": 0,
    "episodic": 180,
    "procedural": 0,
}

_WHITESPACE = re.compile(r"\s+")
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


EXTRACTION_CONTRACT = """
【记忆类型硬约束】
1. summary：只记录这一批对话的摘要、结论、未决问题；不要写长期偏好、通用知识或操作步骤。
2. semantic：只记录稳定、去时间化的事实/偏好/概念。尽量给出 subject、predicate、value；不要写“某天发生了什么”。
3. episodic：只记录具体事件，必须带 event_time（YYYY-MM-DD）、actor、action、result；不要写通用建议。
4. procedural：只记录可复用的任务做法，建议给出 task、trigger、steps、success、failure_notes；不要写用户个人信息。
每个记忆项除 content/importance_score 外，尽量输出：
- confidence: 0.0-1.0
- event_time: episodic 必填，其他类型可省略
- attributes: 类型专属字段的对象
禁止把同一条信息同时写进多个类型；无法判断时优先 semantic 或 episodic，不要重复。
"""


def clamp(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return round(max(0.0, min(1.0, number)), 4)


def normalize_text(value: Any) -> str:
    return _WHITESPACE.sub(" ", str(value or "")).strip().casefold()


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def parse_epoch(value: Any, default: int | None = None) -> int | None:
    """Parse seconds/milliseconds/ISO date strings into Unix seconds."""

    if value is None or value == "":
        return default

    if isinstance(value, (int, float)):
        number = int(value)
        if number > 10_000_000_000:  # milliseconds
            number //= 1000
        return number

    text = str(value).strip()
    if not text:
        return default
    try:
        number = int(float(text))
        if number > 10_000_000_000:
            number //= 1000
        return number
    except ValueError:
        pass

    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            return int(parsed.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    except ValueError:
        return default


def dedupe_key(
    memory_type: str,
    content: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    thread_id: str | None = None,
    event_time: int | None = None,
    summary_id: str | None = None,
) -> str:
    """Build a stable key for exact duplicate detection.

    The key is intentionally type-specific: semantic facts merge by
    subject+predicate, episodic events by day+content, procedural methods by
    task name, and summaries by their source window.
    """

    attributes = dict(attributes or {})
    normalized = normalize_text(content)

    if memory_type == "summary":
        window = f"{attributes.get('window_start', '')}:{attributes.get('window_end', '')}"
        return f"summary:{thread_id or ''}:{summary_id or _short_hash(window + normalized)}"

    if memory_type == "semantic":
        subject = normalize_text(attributes.get("subject"))
        predicate = normalize_text(attributes.get("predicate"))
        if subject or predicate:
            return f"semantic:{subject}:{predicate}"
        return f"semantic:{_short_hash(normalized)}"

    if memory_type == "episodic":
        timestamp = event_time or parse_epoch(attributes.get("event_time"))
        day = (
            datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%d")
            if timestamp
            else "unknown"
        )
        return f"episodic:{day}:{_short_hash(normalized)}"

    if memory_type == "procedural":
        task = normalize_text(attributes.get("task"))
        return f"procedural:{task or _short_hash(normalized)}"

    return f"{memory_type}:{_short_hash(normalized)}"


def normalize_memory_item(
    memory_type: str,
    item: Mapping[str, Any],
    *,
    now: int | None = None,
    default_event_time: int | None = None,
    thread_id: str | None = None,
    summary_id: str | None = None,
) -> dict[str, Any] | None:
    """Validate one extracted item and attach lifecycle metadata."""

    if memory_type not in MEMORY_TYPES:
        return None

    now = int(now or time.time())
    content = str(item.get("content") or "").strip()
    if not content:
        return None

    attributes = item.get("attributes")
    attributes = dict(attributes) if isinstance(attributes, Mapping) else {}
    if memory_type == "summary":
        attributes.setdefault("thread_id", thread_id)
    if memory_type == "procedural":
        success_value = item.get("success")
        attributes.setdefault(
            "success", True if success_value is None else bool(success_value)
        )
        steps = item.get("steps")
        if isinstance(steps, list):
            attributes.setdefault("steps", [str(step) for step in steps])

    event_time = parse_epoch(item.get("event_time"), None)
    if memory_type == "episodic":
        event_time = event_time or default_event_time or now
        attributes.setdefault("event_time", event_time)
    else:
        event_time = event_time or 0

    importance = clamp(item.get("importance_score", item.get("importance", 0.5)))
    confidence = clamp(item.get("confidence", 0.7))
    key = dedupe_key(
        memory_type,
        content,
        attributes,
        thread_id=thread_id,
        event_time=event_time or None,
        summary_id=summary_id,
    )
    ttl_days = TTL_DAYS.get(memory_type, 0)
    return {
        "memory_type": memory_type,
        "content": content,
        "importance": importance,
        "confidence": confidence,
        "event_time": int(event_time),
        "attributes": attributes,
        "dedupe_key": key,
        "expires_at": int(now + ttl_days * 86400) if ttl_days else 0,
    }


def score_memory(
    memory: Mapping[str, Any],
    *,
    relevance: float,
    now: int | None = None,
) -> float:
    """Rank one memory with the scoring policy for its type.

    Semantic facts deliberately do not use hour-level recency decay.  A stable
    preference from a month ago should not become irrelevant merely because it
    was not retrieved recently.
    """

    now = float(now or time.time())
    memory_type = str(memory.get("memory_type") or "semantic")
    relevance = clamp(relevance)
    importance = clamp(memory.get("importance"), 0.5)
    confidence = clamp(memory.get("confidence"), 0.7)

    if memory_type == "summary":
        created = float(memory.get("created_at") or now)
        age_hours = max(0.0, now - created) / 3600.0
        recency = 0.5 ** (age_hours / (24.0 * 7.0))
        return 0.4 * relevance + 0.6 * recency

    if memory_type == "semantic":
        return 0.6 * relevance + 0.2 * confidence + 0.2 * importance

    if memory_type == "episodic":
        event_time = float(memory.get("event_time") or memory.get("created_at") or now)
        age_days = max(0.0, now - event_time) / 86400.0
        recency = 0.5 ** (age_days / 30.0)
        return 0.4 * relevance + 0.4 * recency + 0.2 * importance

    if memory_type == "procedural":
        success = float(memory.get("success_count") or 0)
        failure = float(memory.get("failure_count") or 0)
        success_rate = success / (success + failure) if success + failure else confidence
        usage = float(memory.get("usage_count") or 0)
        usage_score = min(1.0, math.log1p(usage) / math.log1p(10)) if usage > 0 else 0.0
        return 0.5 * relevance + 0.3 * success_rate + 0.1 * usage_score + 0.1 * importance

    return relevance


def retrieval_k(memory_type: str, requested: int | None = None) -> int:
    if requested is not None and int(requested) > 0:
        return int(requested)
    return DEFAULT_RETRIEVAL_K.get(memory_type, 2)


def memory_ttl_days(memory_type: str) -> int:
    return TTL_DAYS.get(memory_type, 0)
