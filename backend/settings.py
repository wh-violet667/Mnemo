"""启动前的环境变量校验。

缺配置时给出可读的错误，而不是让 pydantic 在 import 阶段丢一堆 traceback。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV = [
    "DASHSCOPE_API_KEY",
    "BASE_URL",
    "AGENT_BASE_MODEL",
    "EMBEDDING_MODEL",
    "DATABASE_URL",
    "REDIS_URL",
    "Milvus_url",
    "knowledge_base_collection",
    "memory_collection",
]

PLACEHOLDER_MARKERS = ("your_", "your-", "<", "changeme", "xxxx")


def validate_env() -> None:
    """检查必填变量；缺失或仍是示例占位值时直接给出中文提示。"""

    required = list(REQUIRED_ENV)
    if "opencode.ai" in (os.getenv("BASE_URL") or "").lower():
        required.append("OPENCODE_API_KEY")

    missing = []
    for key in required:
        value = (os.getenv(key) or "").strip().lower()
        if not value or any(marker in value for marker in PLACEHOLDER_MARKERS):
            missing.append(key)

    if missing:
        raise RuntimeError(
            "缺少环境变量："
            + ", ".join(missing)
            + "\n请复制 .env.example 为 .env，填写 OpenCode Go、DashScope、Milvus、PostgreSQL、Redis 配置后重试。"
        )
