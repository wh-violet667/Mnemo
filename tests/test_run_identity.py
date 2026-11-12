"""Tests for the evaluation run identity (KB fingerprint + runtime identity)."""

from __future__ import annotations

import json

from evals.run_identity import (
    build_kb_block,
    build_kb_identity,
    build_run_identity,
    normalize_kb_files,
    stable_identity,
    write_json_atomic,
)
from runtime_identity import build_runtime_identity


def _files() -> list[dict]:
    return [
        {"file_hash": "b" * 64, "file_name": "02-story.docx", "status": "ready"},
        {"file_hash": "a" * 64, "file_name": "01-overview.docx", "status": "ready"},
    ]


def test_kb_fingerprint_is_order_independent() -> None:
    forward = build_kb_block("violet-kb", 1, _files())
    reverse = build_kb_block("violet-kb", 1, list(reversed(_files())))

    assert forward["fingerprint"] == reverse["fingerprint"]
    assert forward["file_count"] == 2


def test_kb_fingerprint_changes_with_content_or_status() -> None:
    base = build_kb_block("violet-kb", 1, _files())

    changed_hash = build_kb_block(
        "violet-kb",
        1,
        [
            {"file_hash": "c" * 64, "file_name": "02-story.docx", "status": "ready"},
            {"file_hash": "a" * 64, "file_name": "01-overview.docx", "status": "ready"},
        ],
    )
    changed_status = build_kb_block(
        "violet-kb",
        1,
        [
            {"file_hash": "b" * 64, "file_name": "02-story.docx", "status": "failed"},
            {"file_hash": "a" * 64, "file_name": "01-overview.docx", "status": "ready"},
        ],
    )

    assert base["fingerprint"] != changed_hash["fingerprint"]
    assert base["fingerprint"] != changed_status["fingerprint"]


def test_normalize_kb_files_projects_and_sorts() -> None:
    rows = normalize_kb_files(
        [
            {"file_hash": "b", "file_name": "z.docx", "status": "READY", "extra": 1},
            {"file_hash": "a", "file_name": "a.docx"},
        ]
    )

    assert rows == [
        {"file_hash": "a", "file_name": "a.docx", "status": "unknown"},
        {"file_hash": "b", "file_name": "z.docx", "status": "ready"},
    ]


def test_empty_kb_identity_is_deterministic() -> None:
    first = build_kb_identity([])
    second = build_kb_identity([])

    assert first == second
    assert first["file_count"] == 0
    assert first["fingerprint"].startswith("sha256:")


def test_run_identity_shape_and_stable_identity() -> None:
    kb = build_kb_block("violet-kb", 1, _files())
    meta = build_run_identity(
        runtime={"schema_version": 1, "git": {"sha": "abc", "dirty": False}},
        kb_blocks=[kb],
        eval_config={"script": "answer_eval", "top_k": 5},
    )

    assert meta["schema_version"] == 1
    assert "created_at" in meta
    assert meta["kb"]["file_count"] == 2
    assert meta["eval"]["top_k"] == 5

    stable = stable_identity(meta)
    assert "created_at" not in stable
    assert stable["runtime"]["git"] == {"sha": "abc"}
    # The written meta keeps the dirty flag for provenance; only the resume
    # comparison drops it.
    assert meta["runtime"]["git"]["dirty"] is False


def test_runtime_identity_is_secret_free_and_complete() -> None:
    identity = build_runtime_identity()

    assert identity["schema_version"] == 1
    assert {"git", "chunk_config", "models", "generation", "retrieval"} <= set(identity)
    assert identity["chunk_config"]["parent_size"] > 0
    assert identity["chunk_config"]["child_size"] > 0
    assert set(identity["models"]) == {
        "agent",
        "summarization",
        "embedding",
        "rerank",
    }
    assert "DASHSCOPE_API_KEY" not in json.dumps(identity)
    assert "api_key" not in json.dumps(identity).lower()


def test_write_json_atomic_replaces_the_file(tmp_path) -> None:
    target = tmp_path / "run_meta.json"
    write_json_atomic(target, {"first": 1})
    write_json_atomic(target, {"second": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"second": 2}
    assert not list(tmp_path.glob(".tmp-*"))
