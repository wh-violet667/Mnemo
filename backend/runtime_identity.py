"""Runtime identity for evaluation provenance.

Every evaluation number should answer two questions: *what was evaluated* and
*what produced it*. The knowledge-base fingerprint covers the first; this
module covers the second without touching the database or the network:

* the code revision (git sha + dirty flag),
* the chunk configuration,
* the configured generation / embedding / rerank models,
* the retrieval defaults that can change a result.

The module is deliberately dependency-free and safe to import from the API and
from the eval scripts. Secrets are never included.
"""

from __future__ import annotations

import copy
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

IDENTITY_SCHEMA_VERSION = 1


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _chunk_config() -> dict[str, Any]:
    """Read the static chunk configuration from ``backend/config.py``."""

    try:
        import config  # backend/config.py when backend/ is on sys.path
    except ModuleNotFoundError:
        from backend import config  # package import from the repository root

    return {
        "parent_size": int(config.parent_chunk_size),
        "parent_overlap": int(config.parent_chunk_overlap),
        "child_size": int(config.child_chunk_size),
        "child_overlap": int(config.child_chunk_overlap),
        "separators": [str(item) for item in config.separators],
    }


def _git_state() -> dict[str, Any]:
    """Return the short SHA and dirty flag, or ``unknown`` outside a checkout."""

    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"sha": "unknown", "dirty": None}
    return {"sha": sha or "unknown", "dirty": bool(status.strip())}


@lru_cache(maxsize=1)
def _runtime_identity_cached() -> dict[str, Any]:
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "git": _git_state(),
        "chunk_config": _chunk_config(),
        "models": {
            "agent": _env("AGENT_BASE_MODEL", "unknown"),
            "summarization": _env("SUMMARIZATION_MODEL", "unknown"),
            "embedding": _env("EMBEDDING_MODEL", "unknown"),
            "rerank": _env("RERANK_MODEL", "unknown"),
        },
        "generation": {
            "temperature": _env_float("AGENT_TEMPERATURE", 0.7),
            "default_top_k": _env_int("AGENT_TOP_K", 5),
        },
        "retrieval": {
            "default_mode": _env("AGENTIC_DEFAULT_MODE", "fast"),
            "max_rounds": _env_int("AGENTIC_MAX_ROUNDS", 2),
            "budget_ms": _env_int("AGENTIC_TIMEOUT_MS", 12000),
            "max_queries": _env_int("AGENTIC_MAX_QUERIES", 3),
        },
    }


def build_runtime_identity() -> dict[str, Any]:
    """Return a copy of the process identity.

    Cached because it only changes when the process or its environment changes;
    the copy keeps callers from mutating the cached value.
    """

    return copy.deepcopy(_runtime_identity_cached())


__all__ = ["IDENTITY_SCHEMA_VERSION", "build_runtime_identity"]
