"""把旧版 memory_collection（auto_id=True）迁移到 Postgres 事实源 + 新 Milvus 投影。

用法（先停止后端）：

    cd backend
    ..\\.venv\\Scripts\\python.exe migrate_memory_v2.py

脚本会：
1. 把旧 Milvus 行备份到临时目录；
2. 写入 Postgres 的 summaries / memories / memory_versions / memory_sources；
3. 删除旧 collection，按新 schema 重建；
4. 从 Postgres 重新生成 embedding 并回填 Milvus。
"""

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import DashScopeEmbeddings
from memory_manager import MemoryManager
from postgresql_client import get_postgresql_client
from pymilvus import AsyncMilvusClient


def split_conversation_id(thread_id: str | None) -> str | None:
    if not thread_id:
        return None
    if ":" in thread_id:
        return thread_id.split(":", 1)[1]
    return thread_id


async def migrate() -> None:
    collection = os.getenv("memory_collection")
    dim = int(os.getenv("dense_dimension", "1024"))
    uri = os.getenv("Milvus_url")
    token = os.getenv("Token")

    client = AsyncMilvusClient(uri=uri, token=token)
    pg_client = await get_postgresql_client()
    await pg_client.init_pool()

    if not await client.has_collection(collection):
        print(f"collection {collection} 不存在，无需迁移。")
        await client.close()
        return

    desc = await client.describe_collection(collection)
    fields = {field["name"]: field for field in desc["fields"]}
    id_field = fields.get("id", {})
    if not id_field.get("auto_id") and "status" in fields:
        print("集合已经是新 schema，无需迁移。")
        await client.close()
        return

    rows = await client.query(
        collection_name=collection,
        filter="user_id >= 0",
        output_fields=[
            "id",
            "user_id",
            "thread_id",
            "memory_type",
            "content",
            "summary_id",
            "importance",
            "created_at",
            "last_access_at",
        ],
        limit=16384,
    )
    print(f"旧集合读取到 {len(rows)} 条记忆。")

    backup_dir = Path(tempfile.gettempdir()) / "opencode"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"memory_collection_backup_{uuid.uuid4().hex}.json"
    backup_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"旧数据已备份到 {backup_path}")

    if not rows:
        print("旧集合为空，跳过数据迁移。")
    else:
        async with pg_client.pool.acquire() as conn:
            existing = await conn.fetchval("SELECT count(*) FROM memories")
            if existing:
                print(
                    f"Postgres 已有 {existing} 条 memories，拒绝重复迁移。"
                    "如需重来，请先清理 memories 相关表。"
                )
                await client.close()
                return

            async with conn.transaction():
                for row in rows:
                    memory_id = f"mem_{uuid.uuid4().hex}"
                    user_id = int(row["user_id"])
                    memory_type = row["memory_type"]
                    thread_id = row.get("thread_id")
                    conversation_id = split_conversation_id(thread_id)
                    summary_id = row.get("summary_id")
                    created_at = int(row.get("created_at") or 0)
                    last_access_at = int(row.get("last_access_at") or 0)
                    content = row.get("content") or ""
                    importance = float(row.get("importance") or 0.5)

                    if memory_type == "summary" and summary_id:
                        message_count = await conn.fetchval(
                            """
                            SELECT count(*) FROM raw_conversations
                            WHERE summary_id = $1 AND user_id = $2
                            """,
                            summary_id,
                            user_id,
                        )
                        await conn.execute(
                            """
                            INSERT INTO summaries (
                                summary_id, user_id, thread_id, conversation_id,
                                content, importance, token_count, message_count,
                                model, status, created_at
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, NULL, $7, NULL, 'active', $8)
                            ON CONFLICT (summary_id) DO NOTHING
                            """,
                            summary_id,
                            user_id,
                            thread_id or "",
                            conversation_id,
                            content,
                            importance,
                            message_count,
                            created_at,
                        )

                    await conn.execute(
                        """
                        INSERT INTO memories (
                            memory_id, user_id, memory_type, content, importance,
                            status, source_summary_id, thread_id, conversation_id,
                            embedding_model, embedding_dim, index_status,
                            created_at, last_access_at
                        )
                        VALUES (
                            $1, $2, $3, $4, $5,
                            'active', $6, $7, $8,
                            $9, $10, 'pending',
                            $11, $12
                        )
                        """,
                        memory_id,
                        user_id,
                        memory_type,
                        content,
                        importance,
                        summary_id if memory_type == "summary" else None,
                        thread_id,
                        conversation_id,
                        os.getenv("EMBEDDING_MODEL"),
                        dim,
                        created_at,
                        last_access_at,
                    )
                    await conn.execute(
                        """
                        INSERT INTO memory_versions (
                            version_id, memory_id, content, importance,
                            status, valid_from
                        )
                        VALUES ($1, $2, $3, $4, 'active', to_timestamp($5))
                        """,
                        f"ver_{uuid.uuid4().hex}",
                        memory_id,
                        content,
                        importance,
                        created_at or 0,
                    )

                    if memory_type == "summary" and summary_id:
                        await conn.execute(
                            """
                            INSERT INTO memory_sources (
                                memory_id, message_id, conversation_id, summary_id
                            )
                            SELECT $1, id, conversation_id, summary_id
                            FROM raw_conversations
                            WHERE summary_id = $2 AND user_id = $3
                            ON CONFLICT DO NOTHING
                            """,
                            memory_id,
                            summary_id,
                            user_id,
                        )

    print("Postgres 写入完成，开始重建 Milvus collection...")
    await client.drop_collection(collection)
    await client.close()

    new_client = AsyncMilvusClient(uri=uri, token=token)
    embeddings = DashScopeEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    manager = MemoryManager(
        client=new_client,
        embeddings=embeddings,
        collection_name=collection,
        dense_dim=dim,
    )
    await manager.init_collection()

    records = await pg_client.get_memories_for_reindex()
    print(f"从 Postgres 读取 {len(records)} 条记忆，开始回填向量...")
    for start in range(0, len(records), 32):
        chunk = records[start : start + 32]
        vectors = await embeddings.aembed_documents(
            [record["content"] for record in chunk]
        )
        ok = await manager.index_memory_records(chunk, vectors)
        if not ok:
            raise RuntimeError("回填 Milvus 失败")
        await pg_client.mark_memories_index_status(
            [record["memory_id"] for record in chunk], "indexed"
        )

    print(f"迁移完成：{len(records)} 条记忆已写入新集合。")
    print(f"备份文件：{backup_path}")
    await new_client.close()
    await pg_client.close()


if __name__ == "__main__":
    asyncio.run(migrate())
