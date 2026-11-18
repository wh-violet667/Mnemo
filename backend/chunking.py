"""Validated chunk configuration for document ingestion.

The indexing path is shared by normal uploads and the chunking experiment.
Keeping the bounds in one place prevents an evaluation run from silently
creating an index with an invalid or incomparable configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import config


DEFAULT_CHUNK_PROFILE = "fine-overlap"

NAMED_CHUNK_PROFILES: dict[str, dict[str, int]] = {
    DEFAULT_CHUNK_PROFILE: {
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 200,
        "child_chunk_size": 200,
        "child_chunk_overlap": 50,
    },
    "coarse-no-overlap": {
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 200,
        "child_chunk_size": 400,
        "child_chunk_overlap": 0,
    },
}


@dataclass(frozen=True)
class ChunkConfig:
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int
    separators: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["separators"] = list(self.separators)
        return payload


def default_chunk_config() -> ChunkConfig:
    return ChunkConfig(
        parent_chunk_size=int(config.parent_chunk_size),
        parent_chunk_overlap=int(config.parent_chunk_overlap),
        child_chunk_size=int(config.child_chunk_size),
        child_chunk_overlap=int(config.child_chunk_overlap),
        separators=tuple(config.separators),
    )


def named_chunk_config(name: str | None = None) -> ChunkConfig:
    """Resolve a stable, human-readable chunking profile."""

    profile_name = (name or DEFAULT_CHUNK_PROFILE).strip()
    params = NAMED_CHUNK_PROFILES.get(profile_name)
    if params is None:
        choices = ", ".join(NAMED_CHUNK_PROFILES)
        raise ValueError(f"未知分块策略 {profile_name!r}，可选值: {choices}")
    return validate_chunk_config(**params)


def list_named_chunk_configs() -> list[dict[str, Any]]:
    """Return profile metadata suitable for an API or UI selector."""

    return [
        {"name": name, **named_chunk_config(name).as_dict()}
        for name in NAMED_CHUNK_PROFILES
    ]


def validate_chunk_config(
    *,
    parent_chunk_size: int | None = None,
    parent_chunk_overlap: int | None = None,
    child_chunk_size: int | None = None,
    child_chunk_overlap: int | None = None,
    separators: tuple[str, ...] | list[str] | None = None,
) -> ChunkConfig:
    """Return a validated config, filling omitted values from ``config.py``."""

    defaults = default_chunk_config()
    parent_size = int(
        defaults.parent_chunk_size if parent_chunk_size is None else parent_chunk_size
    )
    parent_overlap = int(
        defaults.parent_chunk_overlap
        if parent_chunk_overlap is None
        else parent_chunk_overlap
    )
    child_size = int(
        defaults.child_chunk_size if child_chunk_size is None else child_chunk_size
    )
    child_overlap = int(
        defaults.child_chunk_overlap
        if child_chunk_overlap is None
        else child_chunk_overlap
    )
    separator_values = tuple(
        str(value) for value in (defaults.separators if separators is None else separators)
    )

    if not 100 <= parent_size <= 8000:
        raise ValueError("parent_chunk_size 必须在 100-8000 之间")
    if not 50 <= child_size <= parent_size:
        raise ValueError("child_chunk_size 必须 >=50 且不能大于 parent_chunk_size")
    if not 0 <= parent_overlap < parent_size:
        raise ValueError("parent_chunk_overlap 必须 >=0 且小于 parent_chunk_size")
    if not 0 <= child_overlap < child_size:
        raise ValueError("child_chunk_overlap 必须 >=0 且小于 child_chunk_size")
    if not separator_values:
        raise ValueError("separators 不能为空")

    return ChunkConfig(
        parent_chunk_size=parent_size,
        parent_chunk_overlap=parent_overlap,
        child_chunk_size=child_size,
        child_chunk_overlap=child_overlap,
        separators=separator_values,
    )
