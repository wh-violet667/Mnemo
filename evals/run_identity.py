"""Provenance block attached to every evaluation run.

An evaluation number is only trustworthy if the run records *what* was
evaluated and *what* produced it. This module builds that block:

* ``kb`` — a deterministic fingerprint over the knowledge-base files
  ``(file_hash, file_name, status)``. Re-seeding, deleting or re-processing a
  file changes the fingerprint, so an old run can never be mistaken for a new
  corpus;
* ``runtime`` — the backend's secret-free runtime identity (git sha, chunk
  configuration, model names, retrieval defaults), fetched from ``/runtime``;
* ``eval`` — the golden-set hash and the knobs that change the numbers
  (retrieval mode, top-k, judge model, limits).

The pure functions are intentionally dependency-free so tests can exercise
them without PostgreSQL, Milvus or a live LLM.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # keep the pure helpers importable without httpx installed
    import httpx

RUN_IDENTITY_SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    """Canonical JSON used for fingerprints: sorted keys, no whitespace."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """Return a prefixed SHA-256 digest for a string."""

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a text file with platform-independent newline handling."""

    return sha256_text(Path(path).read_text(encoding="utf-8"))


def normalize_kb_files(files: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Keep only the stable identity fields, sorted so file order cannot matter."""

    normalized = [
        {
            "file_hash": str(item.get("file_hash") or ""),
            "file_name": str(item.get("file_name") or ""),
            "status": str(item.get("status") or "unknown").lower(),
        }
        for item in files
    ]
    return sorted(normalized, key=lambda row: (row["file_name"], row["file_hash"]))


def build_kb_block(
    kb_id: str,
    user_id: int,
    files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fingerprint one knowledge base's current file set."""

    normalized = normalize_kb_files(files)
    return {
        "kb_id": str(kb_id),
        "user_id": int(user_id),
        "file_count": len(normalized),
        "fingerprint": sha256_text(canonical_json(normalized)),
        "files": normalized,
    }


def build_kb_identity(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine one or more KB blocks into the run-level KB identity."""

    ordered = sorted(blocks, key=lambda block: (str(block["kb_id"]), int(block["user_id"])))
    return {
        "fingerprint": sha256_text(canonical_json(ordered)),
        "file_count": sum(int(block["file_count"]) for block in ordered),
        "knowledge_bases": list(ordered),
    }


def build_run_identity(
    *,
    runtime: Mapping[str, Any],
    kb_blocks: Sequence[Mapping[str, Any]],
    eval_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the full ``*_meta.json`` payload for one evaluation run."""

    return {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": dict(runtime),
        "kb": build_kb_identity(kb_blocks),
        "eval": dict(eval_config),
    }


def stable_identity(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Drop volatile fields so ``--resume`` compares configuration, not noise.

    ``created_at`` changes on every process start, and ``runtime.git.dirty``
    flips when an unrelated file is edited. Neither changes the code revision,
    models or corpus, so neither should block a resume. The full values stay in
    the written meta for provenance.
    """

    stable = {key: value for key, value in meta.items() if key != "created_at"}
    runtime = stable.get("runtime")
    if isinstance(runtime, Mapping):
        runtime_copy = dict(runtime)
        git = runtime_copy.get("git")
        if isinstance(git, Mapping):
            runtime_copy["git"] = {key: value for key, value in git.items() if key != "dirty"}
        stable["runtime"] = runtime_copy
    return stable


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a temp file + ``os.replace``.

    A reader sees either the previous complete file or the new complete file,
    never a half-written meta file.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".tmp-{os.getpid()}-{target.name}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return target


async def fetch_kb_files(
    client: httpx.AsyncClient,
    base_url: str,
    kb_id: str,
    user_id: int,
) -> list[dict[str, Any]]:
    """Read the KB file listing from the running backend."""

    response = await client.get(
        f"{base_url.rstrip('/')}/knowledge-bases/{kb_id}/files",
        params={"user_id": user_id},
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("success") is False:
        raise RuntimeError(
            payload.get("message") or f"knowledge base {kb_id!r} file listing failed"
        )
    files = payload.get("files")
    if not isinstance(files, list):
        raise RuntimeError(
            f"knowledge base {kb_id!r} file listing did not return a files list"
        )
    return files


async def fetch_runtime(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    """Read the backend runtime identity from ``GET /runtime``."""

    response = await client.get(f"{base_url.rstrip('/')}/runtime")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("/runtime did not return a JSON object")
    return payload


__all__ = [
    "RUN_IDENTITY_SCHEMA_VERSION",
    "build_kb_block",
    "build_kb_identity",
    "build_run_identity",
    "canonical_json",
    "fetch_kb_files",
    "fetch_runtime",
    "normalize_kb_files",
    "sha256_file",
    "sha256_text",
    "stable_identity",
    "write_json_atomic",
]
