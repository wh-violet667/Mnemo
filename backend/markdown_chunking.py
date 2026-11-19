"""Structure-aware chunking for Markdown documents.

The recursive splitter treats a document as an undifferentiated character
stream, so it can cut through a heading, a table row, or a sentence.  This
module keeps Markdown structure instead:

* ATX headings become a stable ``heading_path`` (``作品概况 > 基本信息``);
* fenced code blocks and pipe tables stay intact where possible;
* a table that must be split repeats its header row in every piece;
* every parent/child chunk carries source offsets and its heading path.

Only blocks that exceed the configured budget fall back to sentence or
character splitting.  Overlap is intentionally unsupported: heading
boundaries replace overlap, which keeps neighboring sections from bleeding
into each other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from chunking import ChunkConfig

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(```+|~~~+)")
_TABLE_SEPARATOR_RE = re.compile(r"^[\s|:-]+$")
_LIST_RE = re.compile(r"^([-*+]\s+|\d+[.)]\s+)")
_SENTENCE_BOUNDARY_RE = re.compile(r"[。！？!?；;]+|\n+")

# Reserve room for a synthetic ``# heading > path`` prefix when a chunk starts
# in the middle of a section.  Chunks that start with a heading line do not
# need it, but a fixed reserve keeps packing simple and the size bound safe.
_HEADING_PREFIX_RESERVE = 80


@dataclass(frozen=True)
class MarkdownChunk:
    """One child chunk with enough metadata to trace it back to the source."""

    text: str
    heading_path: tuple[str, ...]
    start: int
    end: int
    block_type: str
    heading_paths: tuple[str, ...] = ()

    def metadata(self) -> dict[str, Any]:
        paths = self.heading_paths
        if not paths and self.heading_path:
            paths = (" > ".join(self.heading_path),)
        return {
            "heading_path": " > ".join(self.heading_path),
            "heading_paths": list(paths),
            "start": self.start,
            "end": self.end,
            "block_type": self.block_type,
        }


@dataclass(frozen=True)
class MarkdownParent:
    """One parent chunk plus the child chunks cut from it."""

    text: str
    heading_path: tuple[str, ...]
    start: int
    end: int
    children: tuple[MarkdownChunk, ...]


@dataclass(frozen=True)
class _Block:
    text: str
    heading_path: tuple[str, ...]
    start: int
    end: int
    block_type: str


@dataclass(frozen=True)
class _Piece:
    text: str
    heading_path: tuple[str, ...]
    start: int
    end: int
    block_type: str


@dataclass(frozen=True)
class _Section:
    heading: _Block | None
    blocks: tuple[_Block, ...]

    @property
    def heading_path(self) -> tuple[str, ...]:
        if self.heading is not None:
            return self.heading.heading_path
        if self.blocks:
            return self.blocks[0].heading_path
        return ()


def strip_front_matter(text: str) -> tuple[str, int]:
    """Return the Markdown body and the offset where it starts.

    The corpus uses YAML front matter for document-level metadata.  It is not
    part of the searchable prose, so both chunking strategies skip it.
    """

    if not text.startswith("---"):
        return text, 0

    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return text, 0

    for index in range(1, min(len(lines), 200)):
        if lines[index].strip() in {"---", "..."}:
            start = sum(len(line) for line in lines[: index + 1])
            return text[start:], start
    return text, 0


def parse_markdown_blocks(text: str, base_offset: int = 0) -> list[_Block]:
    """Split Markdown into heading, paragraph, list, table and code blocks.

    ``text`` is the body returned by :func:`strip_front_matter`; offsets are
    shifted by ``base_offset`` so they remain relative to the original file.
    """

    blocks: list[_Block] = []
    heading_stack: list[tuple[int, str]] = []
    current: list[tuple[int, str]] = []
    current_start = 0
    current_end = 0
    current_path: tuple[str, ...] = ()

    fence_lines: list[tuple[int, str]] = []
    fence_start = 0
    fence_path: tuple[str, ...] = ()
    fence_marker = ""
    in_fence = False

    def flush_current() -> None:
        nonlocal current_start, current_end
        if not current:
            return
        body = "\n".join(line for _, line in current)
        if body.strip():
            blocks.append(
                _Block(
                    text=body,
                    heading_path=current_path,
                    start=current_start,
                    end=current_end,
                    block_type=_classify_block(current[0][1]),
                )
            )
        current.clear()
        current_start = 0
        current_end = 0

    def flush_fence() -> None:
        nonlocal fence_start
        if not fence_lines:
            return
        body = "\n".join(line for _, line in fence_lines)
        if body.strip():
            blocks.append(
                _Block(
                    text=body,
                    heading_path=fence_path,
                    start=fence_start,
                    end=fence_lines[-1][0] + len(fence_lines[-1][1]),
                    block_type="code",
                )
            )
        fence_lines.clear()
        fence_start = 0

    offset = base_offset
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        line_end = offset + len(line)

        if in_fence:
            fence_lines.append((offset, line))
            if line.strip().startswith(fence_marker):
                flush_fence()
                in_fence = False
                fence_marker = ""
            offset = line_end + (len(raw_line) - len(line))
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            flush_current()
            in_fence = True
            fence_marker = fence.group(1)
            fence_start = offset
            fence_path = tuple(title for _, title in heading_stack)
            fence_lines = [(offset, line)]
            offset = line_end + (len(raw_line) - len(line))
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_current()
            level = len(heading.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading.group(2).strip()))
            current_path = tuple(title for _, title in heading_stack)
            blocks.append(
                _Block(
                    text=line.strip(),
                    heading_path=current_path,
                    start=offset,
                    end=line_end,
                    block_type="heading",
                )
            )
            offset = line_end + (len(raw_line) - len(line))
            continue

        if not line.strip():
            flush_current()
            offset = line_end + (len(raw_line) - len(line))
            continue

        if not current:
            current_start = offset
            current_path = tuple(title for _, title in heading_stack)
        current.append((offset, line))
        current_end = line_end
        offset = line_end + (len(raw_line) - len(line))

    flush_current()
    if in_fence:
        flush_fence()
    return blocks


def split_markdown_document(
    text: str,
    config: ChunkConfig,
) -> list[MarkdownParent]:
    """Build structure-aware parent/child chunks from one Markdown document.

    Adjacent sections that share a first-level heading are packed together up
    to the configured size.  This avoids the failure mode where a corpus with
    many short ``##`` sections produces hundreds of 100–200 character chunks:
    heading lines stay in the text, but short siblings share one parent/child.
    """

    body, base_offset = strip_front_matter(text)
    blocks = [
        block for block in parse_markdown_blocks(body, base_offset) if block.text.strip()
    ]
    sections = [section for section in _build_sections(blocks) if section.blocks]
    if not sections:
        return []

    parent_budget = max(
        50, config.parent_chunk_size - _HEADING_PREFIX_RESERVE
    )
    child_budget = max(50, config.child_chunk_size - _HEADING_PREFIX_RESERVE)

    parents: list[MarkdownParent] = []
    for section_group in _group_sections(sections):
        parent_pieces: list[_Piece] = []
        for section in section_group:
            if section.heading is not None:
                parent_pieces.append(_heading_piece(section.heading))
            for block in section.blocks:
                parent_pieces.extend(_split_block(block, parent_budget))

        for parent_group in _pack_pieces(parent_pieces, parent_budget):
            if not _has_body(parent_group):
                continue
            parent_text = _chunk_text(parent_group)
            if not parent_text.strip():
                continue

            child_pieces: list[_Piece] = []
            for piece in parent_group:
                if piece.block_type == "heading":
                    child_pieces.append(piece)
                else:
                    child_pieces.extend(
                        _split_block(_as_block(piece), child_budget)
                    )

            children: list[MarkdownChunk] = []
            for child_group in _pack_pieces(child_pieces, child_budget):
                if not _has_body(child_group):
                    continue
                child_text = _chunk_text(child_group)
                if not child_text.strip():
                    continue
                children.append(
                    MarkdownChunk(
                        text=child_text,
                        heading_path=child_group[0].heading_path,
                        start=child_group[0].start,
                        end=child_group[-1].end,
                        block_type=_combined_type(child_group),
                        heading_paths=_distinct_paths(child_group),
                    )
                )

            if not children:
                continue
            parents.append(
                MarkdownParent(
                    text=parent_text,
                    heading_path=parent_group[0].heading_path,
                    start=parent_group[0].start,
                    end=parent_group[-1].end,
                    children=tuple(children),
                )
            )

    return parents


def _build_sections(blocks: Sequence[_Block]) -> list[_Section]:
    sections: list[_Section] = []
    heading: _Block | None = None
    body: list[_Block] = []

    for block in blocks:
        if block.block_type == "heading":
            if heading is not None or body:
                sections.append(_Section(heading=heading, blocks=tuple(body)))
            heading = block
            body = []
            continue
        body.append(block)

    if heading is not None or body:
        sections.append(_Section(heading=heading, blocks=tuple(body)))
    return sections


def _group_sections(sections: Sequence[_Section]) -> list[list[_Section]]:
    """Group consecutive sections by their first-level heading."""

    groups: list[list[_Section]] = []
    current: list[_Section] = []
    current_key: tuple[str, ...] | None = None

    for section in sections:
        key = section.heading_path[:1]
        if current and key != current_key:
            groups.append(current)
            current = []
        if not current:
            current_key = key
        current.append(section)

    if current:
        groups.append(current)
    return groups


def _heading_piece(block: _Block) -> _Piece:
    level = len(block.text) - len(block.text.lstrip("#"))
    return _Piece(
        text=_heading_line(block.heading_path, level),
        heading_path=block.heading_path,
        start=block.start,
        end=block.end,
        block_type="heading",
    )


def _heading_line(heading_path: Sequence[str], level: int = 1) -> str:
    if not heading_path:
        return ""
    return "#" * max(1, min(level, 6)) + " " + " > ".join(heading_path)


def _chunk_text(pieces: Sequence[_Piece]) -> str:
    body = "\n\n".join(piece.text for piece in pieces)
    if pieces and pieces[0].block_type != "heading":
        prefix = _heading_prefix(pieces[0].heading_path)
        if prefix:
            return prefix + body
    return body


def _distinct_paths(pieces: Sequence[_Piece]) -> tuple[str, ...]:
    paths: list[str] = []
    for piece in pieces:
        joined = " > ".join(piece.heading_path)
        if joined and joined not in paths:
            paths.append(joined)
    return tuple(paths)


def _heading_prefix(heading_path: Sequence[str]) -> str:
    if not heading_path:
        return ""
    return "# " + " > ".join(heading_path) + "\n\n"


def _classify_block(first_line: str) -> str:
    stripped = first_line.lstrip()
    if stripped.startswith("|"):
        return "table"
    if stripped.startswith(">"):
        return "quote"
    if _LIST_RE.match(stripped):
        return "list"
    return "paragraph"


def _is_table_separator(line: str) -> bool:
    return "|" in line and "-" in line and bool(_TABLE_SEPARATOR_RE.match(line))


def _has_body(pieces: Sequence[_Piece]) -> bool:
    return any(piece.block_type != "heading" for piece in pieces)


def _combined_type(pieces: Sequence[_Piece]) -> str:
    types = {piece.block_type for piece in pieces if piece.block_type != "heading"}
    if not types:
        return "heading"
    return types.pop() if len(types) == 1 else "mixed"


def _as_block(piece: _Piece) -> _Block:
    return _Block(
        text=piece.text,
        heading_path=piece.heading_path,
        start=piece.start,
        end=piece.end,
        block_type=piece.block_type,
    )


def _make_piece(
    text: str,
    heading_path: tuple[str, ...],
    start: int,
    end: int,
    block_type: str,
) -> _Piece | None:
    raw = text[start:end]
    if not raw.strip():
        return None
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw.rstrip())
    return _Piece(
        text=raw.strip(),
        heading_path=heading_path,
        start=start + leading,
        end=start + trailing,
        block_type=block_type,
    )


def _split_block(block: _Block, max_chars: int) -> list[_Piece]:
    if len(block.text) <= max_chars:
        piece = _make_piece(
            block.text,
            block.heading_path,
            0,
            len(block.text),
            block.block_type,
        )
        if piece is None:
            return []
        return [
            _Piece(
                text=piece.text,
                heading_path=block.heading_path,
                start=block.start + piece.start,
                end=block.start + piece.end,
                block_type=block.block_type,
            )
        ]

    if block.block_type == "table":
        return _split_table(block, max_chars)
    if block.block_type == "code":
        return _split_hard(block, max_chars)
    return _split_sentences(block, max_chars)


def _split_table(block: _Block, max_chars: int) -> list[_Piece]:
    """Split a pipe table by rows while repeating the header in each piece."""

    lines = block.text.splitlines()
    if len(lines) < 2 or not _is_table_separator(lines[1]):
        return _split_sentences(block, max_chars)

    header = lines[:2]
    rows = lines[2:]
    if not rows:
        return _split_sentences(block, max_chars)

    pieces: list[_Piece] = []
    group: list[str] = []

    def emit() -> None:
        if not group:
            return
        table_text = "\n".join(header + group)
        if table_text.strip():
            pieces.append(
                _Piece(
                    text=table_text,
                    heading_path=block.heading_path,
                    # Offsets are block-level: repeated header rows make a
                    # precise per-piece offset misleading.
                    start=block.start,
                    end=block.end,
                    block_type="table",
                )
            )
        group.clear()

    for row in rows:
        if group and len("\n".join(header + group + [row])) > max_chars:
            emit()
        group.append(row)
    emit()
    return pieces


def _split_sentences(block: _Block, max_chars: int) -> list[_Piece]:
    spans = _sentence_spans(block.text)
    if not spans:
        return []
    pieces: list[_Piece] = []
    window_start: int | None = None
    window_end = 0

    def flush() -> None:
        nonlocal window_start, window_end
        if window_start is None:
            return
        piece = _make_piece(
            block.text,
            block.heading_path,
            window_start,
            window_end,
            block.block_type,
        )
        if piece is not None:
            pieces.append(
                _Piece(
                    text=piece.text,
                    heading_path=block.heading_path,
                    start=block.start + piece.start,
                    end=block.start + piece.end,
                    block_type=block.block_type,
                )
            )
        window_start = None
        window_end = 0

    for start, end in spans:
        if end - start > max_chars:
            flush()
            for hard_start, hard_end in _hard_spans(block.text, start, end, max_chars):
                piece = _make_piece(
                    block.text,
                    block.heading_path,
                    hard_start,
                    hard_end,
                    block.block_type,
                )
                if piece is not None:
                    pieces.append(
                        _Piece(
                            text=piece.text,
                            heading_path=block.heading_path,
                            start=block.start + piece.start,
                            end=block.start + piece.end,
                            block_type=block.block_type,
                        )
                    )
            continue

        if window_start is None:
            window_start, window_end = start, end
        elif end - window_start <= max_chars:
            window_end = end
        else:
            flush()
            window_start, window_end = start, end

    flush()
    return pieces


def _split_hard(block: _Block, max_chars: int) -> list[_Piece]:
    pieces: list[_Piece] = []
    for start, end in _hard_spans(block.text, 0, len(block.text), max_chars):
        piece = _make_piece(
            block.text,
            block.heading_path,
            start,
            end,
            block.block_type,
        )
        if piece is not None:
            pieces.append(
                _Piece(
                    text=piece.text,
                    heading_path=block.heading_path,
                    start=block.start + piece.start,
                    end=block.start + piece.end,
                    block_type=block.block_type,
                )
            )
    return pieces


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.end()
        if end > start:
            spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _hard_spans(
    text: str,
    start: int,
    end: int,
    max_chars: int,
) -> Iterable[tuple[int, int]]:
    position = start
    while position < end:
        stop = min(position + max_chars, end)
        if stop < end:
            window = text[position:stop]
            cut = max(
                window.rfind("\n"),
                window.rfind("。"),
                window.rfind("！"),
                window.rfind("？"),
                window.rfind("；"),
                window.rfind(" "),
            )
            if cut >= max_chars // 2:
                stop = position + cut + 1
        yield position, stop
        position = stop


def _pack_pieces(pieces: Sequence[_Piece], max_chars: int) -> list[list[_Piece]]:
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    current_len = 0

    for piece in pieces:
        extra = len(piece.text) + (2 if current else 0)
        if current and current_len + extra > max_chars:
            groups.append(current)
            current = []
            current_len = 0
            extra = len(piece.text)
        current.append(piece)
        current_len += extra

    if current:
        groups.append(current)
    return groups
