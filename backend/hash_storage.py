# hash_storage.py
from typing import Any, Dict, Optional
import logging
from postgresql_client import get_postgresql_client

logger = logging.getLogger(__name__)


class HashStorage:
    """
    基于 PostgreSQL 的哈希存储
    文件哈希存储在 file_metadata 表中
    块哈希存储在 chunk_hashes 表中，关联文件哈希
    """
    
    def __init__(self):
        # 延迟初始化，避免循环依赖
        self._pg_client = None
    
    async def _get_pg_client(self):
        """延迟获取 PostgreSQL 客户端"""
        if self._pg_client is None:
            self._pg_client = await get_postgresql_client()
        return self._pg_client
    
    # ============ 文件哈希管理 ============

    async def get_file_status(
        self, file_hash: str, knowledge_base_id: str, user_id: int
    ) -> Optional[str]:
        """返回文件入库状态；不存在时返回 None。"""
        client = await self._get_pg_client()
        return await client.get_file_status(file_hash, knowledge_base_id, user_id)

    async def add_file_hash(
        self,
        file_hash: str,
        file_name: str,
        knowledge_base_id: str,
        user_id: int,
        chunk_profile: Optional[str] = None,
        chunk_config: Optional[Dict[str, Any]] = None,
    ):
        """添加文件哈希值（同时添加文件元数据和 chunk 配置）。"""
        client = await self._get_pg_client()
        await client.add_file_metadata(
            file_hash,
            file_name,
            knowledge_base_id,
            user_id,
            chunk_profile=chunk_profile,
            chunk_config=chunk_config,
        )