"""从 Postgres 重建 Milvus 记忆投影。

用法：

    cd backend
    ..\\.venv\\Scripts\\python.exe reindex_memories.py

    # 删除旧 collection 后全量重建（会短暂不可检索）
    ..\\.venv\\Scripts\\python.exe reindex_memories.py --rebuild
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import DashScopeEmbeddings
from memory_manager import MemoryManager
from postgresql_client import get_postgresql_client
from pymilvus import AsyncMilvusClient


async def reindex(rebuild: bool) -> None:
    collection = os.getenv("memory_collection")
    dim = int(os.getenv("dense_dimension", "1024"))

    pg_client = await get_postgresql_client()
    await pg_client.init_pool()

    client = AsyncMilvusClient(
        uri=os.getenv("Milvus_url"),
        token=os.getenv("Token"),
    )

    if rebuild and await client.has_collection(collection):
        print(f"删除旧集合 {collection} ...")
        await client.drop_collection(collection)

    embeddings = DashScopeEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    manager = MemoryManager(
        client=client,
        embeddings=embeddings,
        collection_name=collection,
        dense_dim=dim,
    )
    await manager.init_collection()

    records = await pg_client.get_memories_for_reindex()
    print(f"Postgres 读取到 {len(records)} 条记忆。")

    for start in range(0, len(records), 32):
        chunk = records[start : start + 32]
        vectors = await embeddings.aembed_documents(
            [record["content"] for record in chunk]
        )
        ok = await manager.index_memory_records(chunk, vectors)
        if not ok:
            raise RuntimeError("写入 Milvus 失败")
        await pg_client.mark_memories_index_status(
            [record["memory_id"] for record in chunk], "indexed"
        )
        print(f"  已回填 {min(start + 32, len(records))}/{len(records)}")

    print("reindex 完成。")
    await client.close()
    await pg_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="先删除 collection，再按新 schema 全量重建",
    )
    args = parser.parse_args()
    asyncio.run(reindex(args.rebuild))
