from chunking import named_chunk_config, validate_chunk_config
from markdown_chunking import (
    parse_markdown_blocks,
    split_markdown_document,
    strip_front_matter,
)


def _markdown_config(**overrides):
    values = {
        "strategy": "markdown-heading",
        "parent_chunk_size": 1000,
        "parent_chunk_overlap": 0,
        "child_chunk_size": 400,
        "child_chunk_overlap": 0,
    }
    values.update(overrides)
    return validate_chunk_config(**values)


def test_strip_front_matter_keeps_source_offset():
    text = "---\ntitle: demo\ncategory: test\n---\n\n# Doc\n\n正文。\n"
    body, offset = strip_front_matter(text)

    assert body.startswith("\n# Doc")
    assert "title: demo" not in body
    assert text[offset:].startswith("\n# Doc")


def test_heading_path_is_attached_to_every_chunk():
    text = (
        "---\ntitle: demo\n---\n\n"
        "# 作品\n\n"
        "## 基本信息\n\n"
        "这是基本信息。\n\n"
        "## 角色\n\n"
        "这是角色介绍。\n"
    )
    parents = split_markdown_document(text, _markdown_config())

    assert [parent.heading_path for parent in parents] == [
        ("作品", "基本信息"),
        ("作品", "角色"),
    ]
    assert "# 作品 > 基本信息" in parents[0].text
    assert "这是角色介绍" not in parents[0].text
    assert parents[0].start < parents[0].end
    assert parents[1].start < parents[1].end


def test_tables_keep_header_when_split_across_children():
    rows = "\n".join(f"| {index} | {'内容' * 20} |" for index in range(12))
    text = f"# 文档\n\n## 表格\n\n| 编号 | 内容 |\n|---|---|\n{rows}\n"
    parents = split_markdown_document(
        text,
        _markdown_config(parent_chunk_size=240, child_chunk_size=120),
    )

    children = [child for parent in parents for child in parent.children]
    assert len(children) > 1
    assert all("| 编号 | 内容 |" in child.text for child in children)
    assert all("|---|---|" in child.text for child in children)
    assert all(child.block_type == "table" for child in children)


def test_table_rows_are_not_cut_in_half():
    long_row = "| 1 | " + "很长的单元格" * 10 + " |"
    text = f"# 文档\n\n## 表格\n\n| 编号 | 内容 |\n|---|---|\n{long_row}\n"
    parents = split_markdown_document(
        text,
        _markdown_config(parent_chunk_size=120, child_chunk_size=80),
    )

    children = [child for parent in parents for child in parent.children]
    assert children
    assert all(long_row in child.text for child in children)


def test_prose_chunks_respect_child_budget():
    text = (
        "# 文档\n\n## 长文\n\n"
        + "这是第一句话。这是第二句话。这是第三句话。这是第四句话。"
        * 10
        + "\n"
    )
    config = _markdown_config(parent_chunk_size=300, child_chunk_size=80)
    parents = split_markdown_document(text, config)

    for parent in parents:
        for child in parent.children:
            assert len(child.text) <= config.child_chunk_size + 5
            assert child.start < child.end


def test_code_fence_is_not_parsed_as_headings():
    text = (
        "# 文档\n\n## 示例\n\n"
        "```python\n"
        "# 这不是标题\n"
        "print('hello')\n"
        "```\n"
    )
    blocks = parse_markdown_blocks(text)
    content = [block for block in blocks if block.block_type != "heading"]

    assert len(content) == 1
    assert content[0].block_type == "code"
    assert "# 这不是标题" in content[0].text


def test_markdown_split_is_deterministic():
    text = "# 文档\n\n## A\n\n甲。乙。\n\n## B\n\n丙。\n"
    config = named_chunk_config("markdown-heading")

    first = split_markdown_document(text, config)
    second = split_markdown_document(text, config)

    assert [(p.text, p.start, p.end) for p in first] == [
        (p.text, p.start, p.end) for p in second
    ]


def test_empty_document_returns_no_chunks():
    assert split_markdown_document("---\ntitle: empty\n---\n", _markdown_config()) == []
