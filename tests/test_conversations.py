import pytest

from backend.conversations import (
    make_conversation_title,
    make_thread_id,
    normalize_conversation_id,
)


def test_normalize_conversation_id_falls_back_to_default():
    assert normalize_conversation_id(None) == "default"
    assert normalize_conversation_id("   ") == "default"


def test_normalize_conversation_id_accepts_uuid_and_rejects_path_characters():
    assert normalize_conversation_id("abc-123_XYZ") == "abc-123_XYZ"

    with pytest.raises(ValueError):
        normalize_conversation_id("../../etc/passwd")
    with pytest.raises(ValueError):
        normalize_conversation_id("a" * 37)


def test_make_thread_id_is_scoped_by_user_and_conversation():
    assert make_thread_id(1, "abc") == "1:abc"
    assert make_thread_id(1, "abc") != make_thread_id(2, "abc")
    assert make_thread_id(1, "abc") != make_thread_id(1, "def")


def test_make_conversation_title_collapses_whitespace_and_truncates():
    assert make_conversation_title("  你好\n\n世界  ") == "你好 世界"
    assert make_conversation_title("") == "新对话"
    assert len(make_conversation_title("一" * 100, limit=10)) == 10
