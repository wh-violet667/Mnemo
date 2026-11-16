"""Request identifier validation shared by the API and Milvus filters.

Knowledge-base names are user-visible and may contain Chinese characters, so
we do not restrict them to ASCII.  We do reject quotes, backslashes and control
characters: those are the characters that can break out of a Milvus string
literal.  ``milvus_literal`` is still used as a second layer of defence.
"""

from __future__ import annotations

import re

MAX_KNOWLEDGE_BASE_ID_LENGTH = 128

# Allow Unicode names, spaces and punctuation, but not the characters that can
# terminate or escape a double-quoted Milvus string.
_KNOWLEDGE_BASE_ID_RE = re.compile(r'^[^"\\\x00-\x1f\x7f]{1,128}$')
_FILE_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class InvalidIdentifier(ValueError):
    """Raised when a client-supplied identifier is not safe to use."""


def normalize_knowledge_base_id(
    value: str, *, allow_empty: bool = False
) -> str:
    """Validate a knowledge-base ID without changing its visible spelling."""

    candidate = str(value or "").strip()
    if not candidate:
        if allow_empty:
            return ""
        raise InvalidIdentifier("knowledge_base_id 不能为空")
    if len(candidate) > MAX_KNOWLEDGE_BASE_ID_LENGTH:
        raise InvalidIdentifier(
            f"knowledge_base_id 不能超过 {MAX_KNOWLEDGE_BASE_ID_LENGTH} 个字符"
        )
    if not _KNOWLEDGE_BASE_ID_RE.fullmatch(candidate):
        raise InvalidIdentifier(
            "knowledge_base_id 不能包含引号、反斜杠或控制字符"
        )
    return candidate


def normalize_file_hash(value: str) -> str:
    """Validate the SHA-256 file hash used in paths and Milvus filters."""

    candidate = str(value or "").strip().lower()
    if not _FILE_HASH_RE.fullmatch(candidate):
        raise InvalidIdentifier("file_hash 必须是 64 位十六进制字符串")
    return candidate


def milvus_literal(value: str) -> str:
    """Quote a value for a Milvus expression.

    Values are validated before they reach this helper; escaping here keeps
    internal callers safe even if they bypass the API-level validation.
    """

    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
