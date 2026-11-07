import os
import asyncio
import time
from dotenv import load_dotenv
from pymilvus import AsyncMilvusClient
from langchain_community.embeddings import DashScopeEmbeddings
import logging
from memory_manager import MemoryManager
from knowledge_base_manager import KnowledgeBaseManager


logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

_global_milvus_client = None


async def get_milvus_client():
    """获取全局 Milvus 客户端实例"""
    global _global_milvus_client
    if _global_milvus_client is None:
        _global_milvus_client = AsyncMilvusClientWrapper()
        await _global_milvus_client.ensure_collection()
    else:
        await _global_milvus_client.ensure_connection()
    return _global_milvus_client


class AsyncMilvusClientWrapper:
    """Milvus 客户端包装器 - 只负责向量数据库连接和集合初始化"""

    _instance = None
    _singleton_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._singleton_initialized:
            return

        self.knowledge_base_collection = os.getenv("knowledge_base_collection")
        self.memory_collection = os.getenv("memory_collection")

        self.embeddings = DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        self.client = AsyncMilvusClient(
            uri=os.getenv("Milvus_url"),
            token=os.getenv("Token"),
        )
        self.dense_dim = int(os.getenv("dense_dimension", "1024"))
        self._collection_initialized = False
        self._collection_lock = asyncio.Lock()
        self._reconnect_lock = asyncio.Lock()
        self._last_health_check = 0.0

        self._singleton_initialized = True

        self.memory_manager = MemoryManager(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.memory_collection,
            dense_dim=self.dense_dim,
        )

        self.knowledge_base_manager = KnowledgeBaseManager(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.knowledge_base_collection,
            dense_dim=self.dense_dim,
        )

    async def ensure_connection(self):
        """定期探活；Zilliz 长连接空闲后会被关闭，检测到失效时自动重连。"""
        now = time.monotonic()
        if now - self._last_health_check < 60:
            return

        self._last_health_check = now
        try:
            await asyncio.wait_for(self.client.list_collections(), timeout=5)
        except Exception as exc:
            logger.warning("Milvus 连接不可用，准备重连: %s", exc)
            try:
                await self.reconnect()
            except Exception:
                self._last_health_check = 0.0
                raise

    async def reconnect(self):
        """重建底层 gRPC 连接，并让两个 manager 使用新 client。"""
        async with self._reconnect_lock:
            try:
                await self.client.close()
            except Exception:
                pass

            self.client = AsyncMilvusClient(
                uri=os.getenv("Milvus_url"),
                token=os.getenv("Token"),
            )
            self.memory_manager.client = self.client
            self.knowledge_base_manager.client = self.client
            self._collection_initialized = False
            await self.ensure_collection()
            self._last_health_check = time.monotonic()
            logger.info("Milvus 客户端重连完成")

    async def ensure_collection(self):
        """确保集合已初始化（只在启动时调用一次）"""
        if self._collection_initialized:
            return

        async with self._collection_lock:
            if self._collection_initialized:
                return
            await self._init_knowledge_base_collection()
            await self._init_memory_collection()
            self._collection_initialized = True

    async def _init_memory_collection(self):
        """初始化对话记忆集合"""
        await self.memory_manager.init_collection()

    async def _init_knowledge_base_collection(self):
        """初始化知识库集合"""
        await self.knowledge_base_manager.init_collection()

    async def close(self):
        """关闭客户端连接"""
        if self.client:
            await self.client.close()
            logger.info("Milvus 客户端已关闭")

    # ==================== 记忆管理方法代理 ====================
    async def hybrid_retrieval_memories(
        self,
        query: str,
        user_id: int,
        summary_k: int,
        semantic_k: int,
        episodic_k: int,
        procedural_k: int,
    ):
        """混合检索记忆 - 代理到 memory_manager"""
        return await self.memory_manager.hybrid_retrieval_memories(
            query=query,
            user_id=user_id,
            summary_k=summary_k,
            semantic_k=semantic_k,
            episodic_k=episodic_k,
            procedural_k=procedural_k,
        )

    async def find_superseded_candidates(
        self, items: list[dict], user_id: int
    ) -> tuple[dict, list[list[float]]]:
        """找出应被替换的旧记忆 - 代理到 memory_manager"""
        return await self.memory_manager.find_superseded_candidates(items, user_id)

    async def list_memories(
        self, user_id: int, memory_type: str = None, limit: int = 100
    ):
        """读取长期记忆 - 代理到 memory_manager"""
        return await self.memory_manager.list_memories(
            user_id=user_id, memory_type=memory_type, limit=limit
        )

    async def set_memory_status(self, memory_ids: list[str], status: str) -> bool:
        """更新记忆状态 - 代理到 memory_manager"""
        return await self.memory_manager.set_memory_status(memory_ids, status)

    async def index_memory_records(
        self, records: list[dict], vectors: list[list[float]] | None = None
    ) -> bool:
        """写入记忆投影 - 代理到 memory_manager"""
        return await self.memory_manager.index_memory_records(records, vectors)

    # ==================== 知识库管理方法代理 ====================

    async def hybrid_retrieval_knowledge_base(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        user_id: int,
        mode: str = "hybrid",
    ):
        """混合检索知识库 - 代理到 knowledge_base_manager"""
        return await self.knowledge_base_manager.hybrid_retrieval_knowledge_base(
            query=query,
            knowledge_base_id=knowledge_base_id,
            top_k=top_k,
            user_id=user_id,
            mode=mode,
        )

    async def add_chunks_batch(
        self, knowledge_base_id: str, chunks, user_id: int
    ) -> bool:
        """批量添加知识块 - 代理到 knowledge_base_manager"""
        return await self.knowledge_base_manager.add_chunks_batch(
            knowledge_base_id, chunks, user_id
        )

    async def delete_knowledge_file_chunks(
        self, knowledge_base_id: str, user_id: int
    ) -> bool:
        """删除知识库文件块 - 代理到 knowledge_base_manager"""
        return await self.knowledge_base_manager.delete_knowledge_file_chunks(
            knowledge_base_id, user_id
        )

    async def delete_file_chunks(
        self, knowledge_base_id: str, file_hash: str, user_id: int
    ):
        """删除知识库文件 - 代理到 knowledge_base_manager"""
        return await self.knowledge_base_manager.delete_file_chunks(
            knowledge_base_id=knowledge_base_id, file_hash=file_hash, user_id=user_id
        )
