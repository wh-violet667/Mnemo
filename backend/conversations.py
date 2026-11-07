"""会话 ID 与标题的纯函数工具。

放在这里是为了让 API / Agent / 前端使用同一套规则，并且可以单测。
"""

from __future__ import annotations

import re
import uuid

CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,36}$")


def new_conversation_id() -> str:
    """生成一个新的会话 ID（UUID4，36 字符，符合 Redis key 与 URL 要求）。"""

    return str(uuid.uuid4())


def normalize_conversation_id(value: str | None) -> str:
    """校验并规范化会话 ID。

    只允许字母、数字、下划线和短横线，避免拼进 Redis key 或 URL 时出问题。
    空值回落到 ``default``，兼容老客户端。
    """

    conversation_id = (value or "").strip()
    if not conversation_id:
        return "default"
    if not CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
        raise ValueError("conversation_id 只能包含字母、数字、下划线或短横线，且不超过 36 个字符")
    return conversation_id


def make_thread_id(user_id: int, conversation_id: str) -> str:
    """LangGraph checkpoint 使用的线程 ID：用户 + 会话，保证多会话隔离。"""

    return f"{int(user_id)}:{conversation_id}"


def make_conversation_title(message: str, limit: int = 30) -> str:
    """用首条用户消息生成标题，压缩空白并截断。"""

    title = " ".join((message or "").split())
    if not title:
        return "新对话"
    return title[:limit]
