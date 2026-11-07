import asyncpg
import logging
import json
from typing import Any, List, Optional, Dict, Set
from dotenv import load_dotenv
import os
import asyncio
import time
import uuid
from urllib.parse import unquote, urlparse
from conversations import make_conversation_title, new_conversation_id
from redis_cache import get_redis_client 

load_dotenv()

logger = logging.getLogger(__name__)

async def ensure_database_exists(dsn: str) -> None:
    """确保数据库存在，不存在则自动创建。

    优先使用 DATABASE_URL 解析出的连接信息；.env 里的分项配置（user/password/...）
    仍然可以覆盖，方便本地开发。
    """

    parsed = urlparse(dsn or os.getenv("DATABASE_URL") or "")
    # DATABASE_URL 优先；分项环境变量只作为兜底。
    # 注意 Windows 环境变量不区分大小写，PORT（uvicorn）会和 port（PostgreSQL）冲突。
    user = unquote(parsed.username or "") or os.getenv("user")
    if parsed.password is not None:
        password = unquote(parsed.password)
    else:
        password = os.getenv("password")
    host = parsed.hostname or os.getenv("host") or "localhost"
    port = int(parsed.port or os.getenv("port") or 5432)
    db_name = (
        (parsed.path.lstrip("/") if parsed.path else "")
        or os.getenv("db_name")
        or "postgres"
    )

    if not user or not db_name:
        raise RuntimeError("DATABASE_URL 缺少用户名或数据库名")

    try:
        conn = await asyncpg.connect(
            user=user, password=password, host=host, port=port, database=db_name
        )
        await conn.close()
        logger.info(f"Database '{db_name}' already exists")
        return
    except asyncpg.InvalidCatalogNameError:
        logger.info(f"Database '{db_name}' does not exist. Creating...")
    except Exception as e:
        # 认证失败、网络不通等错误直接抛出，不要误判成“数据库不存在”
        logger.error(f"连接 PostgreSQL 失败: {e}")
        raise

    conn = await asyncpg.connect(
        user=user, password=password, host=host, port=port, database="postgres"
    )
    try:
        await conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await conn.close()
    logger.info(f"Created database: {db_name}")


_global_postgresql_client = None


async def get_postgresql_client():
    """获取全局 PostgreSQL 客户端实例"""
    global _global_postgresql_client
    if _global_postgresql_client is None:
        _global_postgresql_client = PostgreSQLParentClient()
        await _global_postgresql_client.init_pool()
    return _global_postgresql_client


class PostgreSQLParentClient:
    """PostgreSQL 客户端，用于存储父块（单例模式）"""

    _instance = None
    _singleton_initialized = False

    def __new__(
        cls,
        dsn: Optional[str] = None,
        min_size: int = 10,
        max_size: int = 50,
        auto_create_db: bool = True,
    ):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        dsn: Optional[str] = None,
        min_size: int = 10,
        max_size: int = 50,
        auto_create_db: bool = True,
    ):
        """
        初始化 PostgreSQL 客户端

        Args:
            dsn: 数据库连接字符串，如果不提供则从环境变量 DATABASE_URL 读取
            min_size: 连接池最小连接数
            max_size: 连接池最大连接数
            auto_create_db: 是否自动创建数据库
        """
        if self._singleton_initialized:
            return

        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise ValueError(
                "DSN must be provided or set in DATABASE_URL environment variable"
            )

        self.min_size = min_size
        self.max_size = max_size
        self.auto_create_db = auto_create_db
        self.pool: Optional[asyncpg.Pool] = None
        self._pool_initialized = False
        self._pool_lock = asyncio.Lock()

        self._singleton_initialized = True

    async def init_pool(self) -> None:
        """初始化连接池并创建表（只执行一次）"""
        if self._pool_initialized:
            return

        async with self._pool_lock:
            if self._pool_initialized:
                return

            try:
                if self.auto_create_db:
                    await ensure_database_exists(self.dsn)

                self.pool = await asyncpg.create_pool(
                    self.dsn,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    command_timeout=60,
                    server_settings={
                        "timezone": "Asia/Shanghai"
                    },  # 设置会话时区为北京时间
                )

                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        -- 1. 用户表
                        CREATE TABLE IF NOT EXISTS users (
                            user_id SERIAL PRIMARY KEY,
                            user_profile TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );

                        -- 2. 会话表（一个用户可有多个会话）
                        CREATE TABLE IF NOT EXISTS conversations (
                            conversation_id VARCHAR(64) NOT NULL,
                            user_id INTEGER NOT NULL,
                            title VARCHAR(200) NOT NULL DEFAULT '新对话',
                            knowledge_base_id VARCHAR(128),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (conversation_id, user_id),
                            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                        );

                        -- 3. 知识库表（关联用户，支持级联删除）
                        CREATE TABLE IF NOT EXISTS knowledge_bases (
                            knowledge_base_id VARCHAR(128) NOT NULL,
                            user_id INTEGER NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (knowledge_base_id, user_id),
                            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                        );

                        -- 4. 文件信息表（关联知识库和用户，支持级联删除）
                        CREATE TABLE IF NOT EXISTS file_metadata (
                            file_hash VARCHAR(64) NOT NULL,
                            file_name VARCHAR(255) NOT NULL,
                            knowledge_base_id VARCHAR(128) NOT NULL,
                            user_id INTEGER NOT NULL,
                            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (file_hash, knowledge_base_id, user_id),
                            FOREIGN KEY (knowledge_base_id, user_id) REFERENCES knowledge_bases(knowledge_base_id, user_id) ON DELETE CASCADE
                        );               
                        
                        -- 5. 父块表（关联文件，支持级联删除）
                        CREATE TABLE IF NOT EXISTS parent_chunks (
                            parent_id VARCHAR(128) NOT NULL,
                            knowledge_base_id VARCHAR(128) NOT NULL,
                            user_id INTEGER NOT NULL,
                            text TEXT NOT NULL,
                            file_name VARCHAR(255) NOT NULL,
                            file_hash VARCHAR(64) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (parent_id, knowledge_base_id, user_id),
                            FOREIGN KEY (file_hash, knowledge_base_id, user_id) REFERENCES file_metadata(file_hash, knowledge_base_id, user_id) ON DELETE CASCADE
                        );
                        
                        -- 6. 块哈希表（关联文件，支持级联删除）
                        CREATE TABLE IF NOT EXISTS chunk_hashes (
                            chunk_hash VARCHAR(64) NOT NULL,
                            file_hash VARCHAR(64) NOT NULL,
                            knowledge_base_id VARCHAR(128) NOT NULL,
                            user_id INTEGER NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (chunk_hash, file_hash, knowledge_base_id, user_id),
                            FOREIGN KEY (file_hash, knowledge_base_id, user_id) REFERENCES file_metadata(file_hash, knowledge_base_id, user_id) ON DELETE CASCADE
                        );

                        -- 7. 原始对话表（可以通过 summary_id / conversation_id 查询）
                        CREATE TABLE IF NOT EXISTS raw_conversations (
                            id VARCHAR(50) PRIMARY KEY,
                            user_id INT NOT NULL,
                            thread_id VARCHAR(50) NOT NULL,
                            conversation_id VARCHAR(64),
                            summary_id VARCHAR(50) NULL,
                            role VARCHAR(10) NOT NULL,
                            content TEXT NOT NULL,
                            sources TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );

                        -- 兼容旧库：补列 + 回填（CREATE TABLE IF NOT EXISTS 不会修改已存在的表）
                        ALTER TABLE raw_conversations ADD COLUMN IF NOT EXISTS conversation_id VARCHAR(64);
                        ALTER TABLE raw_conversations ADD COLUMN IF NOT EXISTS sources TEXT;
                        ALTER TABLE conversations ADD COLUMN IF NOT EXISTS knowledge_base_id VARCHAR(128);
                        UPDATE raw_conversations
                        SET conversation_id = thread_id
                        WHERE conversation_id IS NULL OR conversation_id = '';

                        INSERT INTO users (user_id)
                        SELECT DISTINCT user_id FROM raw_conversations
                        ON CONFLICT (user_id) DO NOTHING;

                        INSERT INTO conversations (conversation_id, user_id, title)
                        SELECT DISTINCT thread_id, user_id, '历史对话'
                        FROM raw_conversations
                        WHERE conversation_id = thread_id
                        ON CONFLICT (conversation_id, user_id) DO NOTHING;

                        -- 独立创建索引，加速查询
                        CREATE INDEX IF NOT EXISTS idx_user_created ON raw_conversations(user_id, created_at);
                        CREATE INDEX IF NOT EXISTS idx_thread_created ON raw_conversations(thread_id, created_at);
                        CREATE INDEX IF NOT EXISTS idx_summary_id ON raw_conversations(summary_id);
                        CREATE INDEX IF NOT EXISTS idx_raw_conversation ON raw_conversations(user_id, conversation_id, created_at);
                        CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);

                        -- 8. 摘要表（每次记忆压缩批次的事实记录）
                        CREATE TABLE IF NOT EXISTS summaries (
                            summary_id VARCHAR(64) PRIMARY KEY,
                            user_id INTEGER NOT NULL,
                            thread_id VARCHAR(64) NOT NULL,
                            conversation_id VARCHAR(64),
                            content TEXT NOT NULL DEFAULT '',
                            importance REAL DEFAULT 0.5,
                            token_count INTEGER,
                            message_count INTEGER,
                            model VARCHAR(100),
                            status VARCHAR(20) NOT NULL DEFAULT 'active',
                            created_at BIGINT NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_summaries_user_created
                            ON summaries(user_id, created_at DESC);

                        -- 9. 记忆事实表（Postgres 是唯一事实源）
                        CREATE TABLE IF NOT EXISTS memories (
                            memory_id VARCHAR(64) PRIMARY KEY,
                            user_id INTEGER NOT NULL,
                            memory_type VARCHAR(30) NOT NULL,
                            content TEXT NOT NULL,
                            importance REAL DEFAULT 0.5,
                            status VARCHAR(20) NOT NULL DEFAULT 'active',
                            superseded_by VARCHAR(64),
                            source_summary_id VARCHAR(64) REFERENCES summaries(summary_id) ON DELETE SET NULL,
                            thread_id VARCHAR(64),
                            conversation_id VARCHAR(64),
                            embedding_model VARCHAR(100),
                            embedding_dim INTEGER,
                            index_status VARCHAR(20) NOT NULL DEFAULT 'pending',
                            indexed_at TIMESTAMP,
                            created_at BIGINT NOT NULL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            deleted_at TIMESTAMP,
                            last_access_at BIGINT DEFAULT 0
                        );
                        CREATE INDEX IF NOT EXISTS idx_memories_user_status_type
                            ON memories(user_id, status, memory_type);
                        CREATE INDEX IF NOT EXISTS idx_memories_summary
                            ON memories(source_summary_id);
                        CREATE INDEX IF NOT EXISTS idx_memories_index_status
                            ON memories(index_status);

                        -- 10. 记忆版本表（只追加，不覆盖）
                        CREATE TABLE IF NOT EXISTS memory_versions (
                            version_id VARCHAR(64) PRIMARY KEY,
                            memory_id VARCHAR(64) NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                            content TEXT NOT NULL,
                            importance REAL,
                            status VARCHAR(20) NOT NULL,
                            valid_from TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            valid_to TIMESTAMP,
                            superseded_by VARCHAR(64),
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        );
                        CREATE INDEX IF NOT EXISTS idx_memory_versions_memory
                            ON memory_versions(memory_id, created_at DESC);

                        -- 11. 记忆来源表（记忆 -> 原始消息）
                        CREATE TABLE IF NOT EXISTS memory_sources (
                            memory_id VARCHAR(64) NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                            message_id VARCHAR(50) NOT NULL,
                            conversation_id VARCHAR(64),
                            summary_id VARCHAR(64),
                            relation VARCHAR(20) NOT NULL DEFAULT 'derived_from',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (memory_id, message_id)
                        );
                        CREATE INDEX IF NOT EXISTS idx_memory_sources_message
                            ON memory_sources(message_id);
                        """
                    )
                    # 为业务表创建索引，加速多表关联查询和过滤操作
                    await conn.execute(
                        """
                        -- 索引优化
                        CREATE INDEX IF NOT EXISTS idx_file_kb_user_time ON file_metadata(knowledge_base_id, user_id, uploaded_at DESC);-- 加速查询：查询某知识库下某用户的所有文件,同时支持 WHERE 条件和 ORDER BY
                        CREATE INDEX IF NOT EXISTS idx_parent_kb_user ON parent_chunks(knowledge_base_id, user_id);-- 加速查询：查询某用户在某知识库中的所有父块
                        CREATE INDEX IF NOT EXISTS idx_parent_file_hash ON parent_chunks(file_hash);-- 加速查询：根据文件哈希查找所有父块
                        CREATE INDEX IF NOT EXISTS idx_chunk_hash ON chunk_hashes(chunk_hash);-- 加速查询：快速查找特定哈希值的块（用于去重）
                        CREATE INDEX IF NOT EXISTS idx_chunk_file_hash ON chunk_hashes(file_hash, knowledge_base_id, user_id);-- 复合索引：加速多条件查询，场景：查找某文件在特定知识库中的所有块哈希
                        CREATE INDEX IF NOT EXISTS idx_kb_user ON knowledge_bases(user_id);-- 加速查询：查询某用户的所有知识库
                    """
                    )

                self._pool_initialized = True
                logger.info("PostgreSQL client initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL client: {e}")
                raise

    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

    # ============ 会话管理 ============
    async def create_conversation(
        self, user_id: int, title: str = "新对话", knowledge_base_id: Optional[str] = None
    ) -> Dict[str, any]:
        """创建会话；同一用户可以有多个会话。"""

        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        conversation_id = new_conversation_id()
        title = (title or "").strip()[:200] or "新对话"

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO users (user_id, created_at)
                    VALUES ($1, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id) DO NOTHING
                """,
                    user_id,
                )
                row = await conn.fetchrow(
                    """
                    INSERT INTO conversations (conversation_id, user_id, title, knowledge_base_id)
                    VALUES ($1, $2, $3, $4)
                    RETURNING conversation_id, title, knowledge_base_id, created_at, updated_at
                """,
                    conversation_id,
                    user_id,
                    title,
                    knowledge_base_id,
                )

        logger.info(f"为用户 {user_id} 创建会话: {conversation_id}")
        return dict(row)

    async def ensure_conversation(
        self, conversation_id: str, user_id: int
    ) -> None:
        """确保会话存在；直接调用 API 时也不会产生“没有会话的孤儿消息”。"""

        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO users (user_id, created_at)
                    VALUES ($1, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id) DO NOTHING
                """,
                    user_id,
                )
                await conn.execute(
                    """
                    INSERT INTO conversations (conversation_id, user_id)
                    VALUES ($1, $2)
                    ON CONFLICT (conversation_id, user_id) DO NOTHING
                """,
                    conversation_id,
                    user_id,
                )

    async def set_conversation_knowledge_base(
        self, conversation_id: str, user_id: int, knowledge_base_id: str
    ) -> None:
        """记录会话当前使用的知识库；传空字符串表示清空选择。"""

        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE conversations
                SET knowledge_base_id = $3, updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = $1 AND user_id = $2
            """,
                conversation_id,
                user_id,
                knowledge_base_id or None,
            )

    async def list_conversations(self, user_id: int) -> List[Dict[str, any]]:
        """按最近活跃时间倒序列出用户会话。"""

        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT conversation_id, title, knowledge_base_id, created_at, updated_at
                FROM conversations
                WHERE user_id = $1
                ORDER BY updated_at DESC, created_at DESC
            """,
                user_id,
            )

        return [dict(row) for row in rows]

    async def delete_last_turn(
        self, conversation_id: str, user_id: int
    ) -> int:
        """删除会话最后一轮问答，供前端“重新生成”使用。"""
        if not self.pool:
            return 0
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                DELETE FROM raw_conversations
                WHERE id IN (
                    SELECT id FROM raw_conversations
                    WHERE conversation_id = $1 AND user_id = $2
                    ORDER BY created_at DESC
                    LIMIT 2
                )
                RETURNING id
                """,
                conversation_id,
                user_id,
            )
        return len(rows)

    async def rename_conversation(
        self, conversation_id: str, user_id: int, title: str
    ) -> Optional[Dict[str, any]]:
        """重命名会话；不存在或不属于该用户时返回 None。"""

        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        title = (title or "").strip()[:200] or "新对话"

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE conversations
                SET title = $3, updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = $1 AND user_id = $2
                RETURNING conversation_id, title, knowledge_base_id, created_at, updated_at
            """,
                conversation_id,
                user_id,
                title,
            )

        return dict(row) if row else None

    async def delete_conversation(self, conversation_id: str, user_id: int) -> bool:
        """删除会话及其原始消息；不存在时返回 False。"""

        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    DELETE FROM raw_conversations
                    WHERE user_id = $1 AND conversation_id = $2
                """,
                    user_id,
                    conversation_id,
                )
                result = await conn.execute(
                    """
                    DELETE FROM conversations
                    WHERE conversation_id = $1 AND user_id = $2
                """,
                    conversation_id,
                    user_id,
                )

        deleted = (
            int(result.split()[-1]) if result.startswith("DELETE ") else 0
        )
        return deleted > 0

    async def ensure_conversation_title(
        self, conversation_id: str, user_id: int, message: str
    ) -> bool:
        """首条消息到达时，用消息内容替换默认标题。"""

        if not self.pool:
            return False

        title = make_conversation_title(message, limit=200)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE conversations
                SET title = $3, updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = $1
                  AND user_id = $2
                  AND (title IS NULL OR title = '' OR title = '新对话')
                RETURNING conversation_id
            """,
                conversation_id,
                user_id,
                title,
            )

        return row is not None

    async def get_conversation_messages(
        self, conversation_id: str, user_id: int, limit: int = 200
    ) -> List[Dict[str, any]]:
        """读取会话的原始消息，用于切换会话时恢复前端历史。"""

        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content, sources, created_at
                FROM raw_conversations
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY created_at ASC, id ASC
                LIMIT $3
            """,
                user_id,
                conversation_id,
                limit,
            )

        messages = []
        for row in rows:
            item = dict(row)
            raw_sources = item.get("sources")
            if raw_sources:
                try:
                    item["sources"] = json.loads(raw_sources)
                except Exception:
                    item["sources"] = []
            else:
                item["sources"] = []
            messages.append(item)
        return messages

    # ============ 知识库管理 ============
    async def create_knowledge_base(
        self, knowledge_base_id: str, user_id: int
    ) -> Dict[str, any]:
        """为用户创建知识库
        Returns:
            Dict[str, any]: {
                "success": bool,
                "message": str,
                "knowledge_base_id": str (可选)
            }
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 先检查知识库是否已存在（可选，但保持风格一致）
                    exists = await conn.fetchval(
                        """
                        SELECT knowledge_base_id FROM knowledge_bases 
                        WHERE knowledge_base_id = $1 AND user_id = $2
                    """,
                        knowledge_base_id,
                        user_id,
                    )

                    if exists:
                        logger.warning(
                            f"知识库 {knowledge_base_id} 已存在（用户 {user_id}）"
                        )
                        return {
                            "success": False,
                            "message": f"知识库 {knowledge_base_id} 已存在",
                            "knowledge_base_id": knowledge_base_id,
                        }
                    # 1. 先确保用户存在
                    await conn.execute(
                        """
                        INSERT INTO users (user_id,created_at)
                        VALUES ($1,CURRENT_TIMESTAMP)
                        ON CONFLICT (user_id) DO NOTHING
                    """,
                        user_id,
                    )
                    # 插入新知识库
                    await conn.execute(
                        """
                        INSERT INTO knowledge_bases (knowledge_base_id, user_id)
                        VALUES ($1, $2)
                    """,
                        knowledge_base_id,
                        user_id,
                    )

                    logger.info(f"为用户 {user_id} 创建知识库: {knowledge_base_id}")

                    return {
                        "success": True,
                        "message": f"知识库 {knowledge_base_id} 创建成功",
                        "knowledge_base_id": knowledge_base_id,
                    }

        except Exception as e:
            logger.error(f"创建知识库 {knowledge_base_id} 失败: {e}")
            raise

    async def delete_knowledge_base(
        self, knowledge_base_id: str, user_id: int
    ) -> Dict[str, any]:
        """
        删除整个知识库（级联删除会自动处理 file_metadata、parent_chunks、chunk_hashes）
        返回删除的统计信息
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 先检查知识库是否存在且属于该用户
                    exists = await conn.fetchval(
                        """
                        SELECT knowledge_base_id FROM knowledge_bases 
                        WHERE knowledge_base_id = $1 AND user_id = $2
                    """,
                        knowledge_base_id,
                        user_id,
                    )

                    if not exists:
                        logger.warning(
                            f"知识库 {knowledge_base_id} 不存在或不属于用户 {user_id}"
                        )
                        return {
                            "success": False,
                            "message": "知识库不存在或无权删除",
                            "files_deleted": 0,
                        }

                    # 获取要删除的文件数量（用于日志）
                    file_count = await conn.fetchval(
                        """
                        SELECT COUNT(*) FROM file_metadata 
                        WHERE knowledge_base_id = $1 AND user_id = $2
                    """,
                        knowledge_base_id,
                        user_id,
                    )

                    # 删除知识库（由于外键级联，会自动删除所有关联数据）
                    await conn.execute(
                        """
                        DELETE FROM knowledge_bases 
                        WHERE knowledge_base_id = $1 AND user_id = $2
                    """,
                        knowledge_base_id,
                        user_id,
                    )

                    logger.info(
                        f"成功删除知识库 {knowledge_base_id}（用户 {user_id}），以及其中包含的{file_count} 个文件"
                    )

                    return {
                        "success": True,
                        "message": f"知识库删除成功，共删除 {file_count} 个文件",
                        "files_deleted": file_count,
                    }

        except Exception as e:
            logger.error(f"删除知识库 {knowledge_base_id} 失败: {e}")
            raise

    async def get_user_knowledge_bases(self, user_id: int) -> Dict[str, any]:
        """获取用户的所有知识库ID
        Returns:
            Dict[str, any]: {
                "success": bool,
                "message": str,
                "knowledge_bases": List[str] (可选),
                "count": int (可选)
            }
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    rows = await conn.fetch(
                        """
                        SELECT knowledge_base_id 
                        FROM knowledge_bases 
                        WHERE user_id = $1
                        ORDER BY created_at DESC
                    """,
                        user_id,
                    )

                    knowledge_bases = [row["knowledge_base_id"] for row in rows]
                    count = len(knowledge_bases)

                    logger.info(f"获取用户 {user_id} 的知识库列表，共 {count} 个")

                    return {
                        "success": True,
                        "message": f"成功获取 {count} 个知识库",
                        "knowledge_bases": knowledge_bases,
                        "count": count,
                    }

        except Exception as e:
            logger.error(f"获取用户 {user_id} 的知识库列表失败: {e}")
            raise

    # ============ 文件哈希管理 ============
    async def is_file_duplicate(
        self, file_hash: str, knowledge_base_id: str, user_id: int
    ) -> bool:
        """检查文件哈希在指定用户的知识库中是否已存在"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(
                    """
                    SELECT 1 FROM file_metadata 
                    WHERE file_hash = $1 AND knowledge_base_id = $2 AND user_id = $3
                    LIMIT 1
                """,
                    file_hash,
                    knowledge_base_id,
                    user_id,
                )
                return result is not None
        except Exception as e:
            logger.error(f"Error checking file duplicate: {e}")
            raise

    async def add_file_metadata(
        self, file_hash: str, file_name: str, knowledge_base_id: str, user_id: int
    ):
        """添加文件元数据"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 1. 先确保用户存在
                    await conn.execute(
                        """
                        INSERT INTO users (user_id, created_at)
                        VALUES ($1, CURRENT_TIMESTAMP)
                        ON CONFLICT (user_id) DO NOTHING
                    """,
                        user_id,
                    )

                    # 2. 再确保知识库存在
                    await conn.execute(
                        """
                        INSERT INTO knowledge_bases (knowledge_base_id, user_id)
                        VALUES ($1, $2)
                        ON CONFLICT (knowledge_base_id, user_id) DO NOTHING
                    """,
                        knowledge_base_id,
                        user_id,
                    )

                    # 3. 最后添加文件元数据
                    await conn.execute(
                        """
                        INSERT INTO file_metadata (file_hash, file_name, knowledge_base_id, user_id)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (file_hash, knowledge_base_id, user_id) DO UPDATE 
                        SET file_name = EXCLUDED.file_name
                    """,
                        file_hash,
                        file_name,
                        knowledge_base_id,
                        user_id,
                    )

                    logger.info(
                        f"添加文件元数据: {file_hash[:16]}... 到知识库 {knowledge_base_id}（用户 {user_id}）"
                    )
        except Exception as e:
            logger.error(f"Error adding file metadata: {e}")
            raise

    async def delete_file(
        self, file_hash: str, knowledge_base_id: str, user_id: int
    ) -> Dict[str, any]:
        """
        删除单个文件（级联删除会自动处理 parent_chunks 和 chunk_hashes）
        返回删除的统计信息
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 先检查文件是否存在且属于该用户
                    exists = await conn.fetchval(
                        """
                        SELECT file_hash FROM file_metadata 
                        WHERE file_hash = $1 AND knowledge_base_id = $2 AND user_id = $3
                    """,
                        file_hash,
                        knowledge_base_id,
                        user_id,
                    )

                    if not exists:
                        logger.warning(
                            f"文件 {file_hash[:16]} 不存在或不属于用户 {user_id}"
                        )
                        return {
                            "success": False,
                            "message": "文件不存在或无权删除",
                            "files_deleted": 0,
                            "parent_chunks_deleted": 0,
                        }

                    # 获取要删除的父块数量（用于统计）
                    parent_count = await conn.fetchval(
                        """
                        SELECT COUNT(*) FROM parent_chunks 
                        WHERE file_hash = $1 AND knowledge_base_id = $2 AND user_id = $3
                    """,
                        file_hash,
                        knowledge_base_id,
                        user_id,
                    )

                    # 删除文件元数据（由于外键级联，会自动删除关联的父块和块哈希）
                    result = await conn.execute(
                        """
                        DELETE FROM file_metadata 
                        WHERE file_hash = $1 AND knowledge_base_id = $2 AND user_id = $3
                    """,
                        file_hash,
                        knowledge_base_id,
                        user_id,
                    )

                    deleted_file_count = (
                        int(result.split()[-1]) if result.startswith("DELETE ") else 0
                    )

                    logger.info(
                        f"删除文件 {file_hash[:16]}（知识库 {knowledge_base_id}，用户 {user_id}），关联的 {parent_count} 个父块将被级联删除"
                    )

                    return {
                        "success": True,
                        "message": "文件删除成功",
                        "files_deleted": deleted_file_count,
                        "parent_chunks_deleted": parent_count,
                    }

        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            raise

    async def get_knowledge_base_files(
        self, knowledge_base_id: str, user_id: int
    ) -> Dict[str, any]:
        """获取知识库中的所有文件
        Returns:
            Dict[str, any]: {
                "success": bool,
                "message": str,
                "files": List[Dict],  # 每个文件包含 file_hash, file_name, uploaded_at
                "count": int
            }
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT file_hash, file_name, uploaded_at
                    FROM file_metadata
                    WHERE knowledge_base_id = $1 AND user_id = $2
                    ORDER BY uploaded_at DESC
                """,
                    knowledge_base_id,
                    user_id,
                )

                files = [dict(row) for row in rows]
                count = len(files)

                logger.info(f"获取知识库 {knowledge_base_id} 的文件列表，共 {count} 个")

                return {
                    "success": True,
                    "message": f"成功获取 {count} 个文件",
                    "files": files,
                    "count": count,
                }

        except Exception as e:
            logger.error(f"获取知识库 {knowledge_base_id} 的文件列表失败: {e}")
            return {"success": False, "message": str(e), "files": [], "count": 0}

    # ============ 父块管理 ============
    async def add_parent_chunk_batch(
        self, parents_data: List[Dict[str, any]], user_id: int
    ) -> int:
        """
        批量添加父块
        parents_data 格式: [{"parent_id": "...", "knowledge_base_id": "...", "text": "...", "file_name": "...", "file_hash": "..."}]
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        if not parents_data:
            return 0

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for data in parents_data:
                        await conn.execute(
                            """
                            INSERT INTO parent_chunks (parent_id, knowledge_base_id, user_id, text, file_name, file_hash)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (parent_id, knowledge_base_id, user_id) DO UPDATE 
                            SET text = EXCLUDED.text, 
                                file_name = EXCLUDED.file_name, 
                                file_hash = EXCLUDED.file_hash,
                                updated_at = CURRENT_TIMESTAMP
                        """,
                            data["parent_id"],
                            data["knowledge_base_id"],
                            user_id,
                            data["text"],
                            data["file_name"],
                            data["file_hash"],
                        )

            logger.info(f"批量添加 {len(parents_data)} 个父块（用户 {user_id}）")
            return len(parents_data)
        except Exception as e:
            logger.error(f"Error batch adding parents: {e}")
            raise

    async def get_parents(
        self, parent_ids: List[str], knowledge_base_id: str, user_id: int
    ) -> List[Dict[str, any]]:
        """批量获取父块，保持检索顺序并带文件名，供引用溯源使用。"""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        if not parent_ids:
            return []

        # 去重但保留首次出现的顺序，避免 set 打乱 rerank 前的候选顺序
        ordered_ids = list(dict.fromkeys(parent_ids))

        try:
            async with self.pool.acquire() as conn:
                # 默认知识库的情况下查询所有知识库的父块文本，其他知识库名就限制在知识库里查找
                if knowledge_base_id == "默认知识库":
                    rows = await conn.fetch(
                        """
                    SELECT parent_id, text, file_name
                    FROM parent_chunks
                    WHERE parent_id = ANY($1::text[]) 
                    AND user_id = $2
                """,
                        ordered_ids,
                        user_id,
                    )
                else:
                    rows = await conn.fetch(
                        """
                    SELECT parent_id, text, file_name
                    FROM parent_chunks
                    WHERE parent_id = ANY($1::text[]) 
                    AND knowledge_base_id = $2 
                    AND user_id = $3
                """,
                        ordered_ids,
                        knowledge_base_id,
                        user_id,
                    )

            by_id = {row["parent_id"]: dict(row) for row in rows}
            return [by_id[parent_id] for parent_id in ordered_ids if parent_id in by_id]
        except Exception as e:
            logger.error(f"Error getting parents: {e}")
            raise

    # 文档块去重
    async def batch_check_chunk_duplicates(
        self,
        chunk_hashes: List[str],
        file_hash: str,
        knowledge_base_id: str,
        user_id: int,
    ) -> Set[str]:
        """
        批量检查块哈希是否已被当前用户的其他文件使用过
        注意：排除当前文件自身的块哈希

        Args:
            chunk_hashes: 哈希值列表
            file_hash: 当前文件的哈希值
            knowledge_base_id: 知识库ID
            user_id: 用户ID

        Returns:
            已被其他文件使用过的哈希值集合
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        if not chunk_hashes:
            return set()

        try:
            async with self.pool.acquire() as conn:
                # 查找被同一用户同一知识库中其他文件使用过的块哈希
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT chunk_hash FROM chunk_hashes 
                    WHERE chunk_hash = ANY($1::text[])
                    AND file_hash != $2
                    AND knowledge_base_id = $3
                    AND user_id = $4
                """,
                    chunk_hashes,
                    file_hash,
                    knowledge_base_id,
                    user_id,
                )

                return {row["chunk_hash"] for row in rows}
        except Exception as e:
            logger.error(f"Error batch checking chunk duplicates: {e}")
            raise

    async def batch_add_chunk_hashes(
        self,
        chunk_hashes: List[str],
        file_hash: str,
        knowledge_base_id: str,
        user_id: int,
    ):
        """
        批量添加块哈希，关联到指定文件

        Args:
            chunk_hashes: 哈希值列表
            file_hash: 文件哈希值
            knowledge_base_id: 知识库ID
            user_id: 用户ID
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        if not chunk_hashes:
            return

        try:
            async with self.pool.acquire() as conn:
                # 构建批量插入参数
                values = [
                    (chunk_hash, file_hash, knowledge_base_id, user_id)
                    for chunk_hash in chunk_hashes
                ]
                await conn.executemany(
                    """
                    INSERT INTO chunk_hashes (chunk_hash, file_hash, knowledge_base_id, user_id)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (chunk_hash, file_hash, knowledge_base_id, user_id) DO NOTHING
                """,
                    values,
                )
                logger.info(
                    f"批量添加 {len(chunk_hashes)} 个块哈希，关联文件 {file_hash[:16]}（用户 {user_id}）"
                )
        except Exception as e:
            logger.error(f"Error batch adding chunk hashes: {e}")
            raise

    async def add_conversation_message(
        self,
        user_id: int,
        thread_id: str,
        role: str,
        content: str,
        conversation_id: Optional[str] = None,
        sources: Optional[List[Dict]] = None,
    ) -> None:
        """
        添加对话消息

        Args:
            user_id: 用户ID
            thread_id: LangGraph checkpoint 线程 ID（user_id:conversation_id）
            role: 角色 ('human' 或 'ai')
            content: 消息内容
            conversation_id: 会话 ID；为空时回落到 thread_id，兼容旧数据
            sources: AI 回答引用的来源列表，序列化为 JSON 存库
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        conversation_id = conversation_id or thread_id
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        """
                        INSERT INTO raw_conversations
                            (id, user_id, thread_id, conversation_id, role, content, sources)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                        str(uuid.uuid4()),
                        user_id,
                        thread_id,
                        conversation_id,
                        role,
                        content,
                        sources_json,
                    )
                    await conn.execute(
                        """
                        UPDATE conversations
                        SET updated_at = CURRENT_TIMESTAMP
                        WHERE conversation_id = $1 AND user_id = $2
                    """,
                        conversation_id,
                        user_id,
                    )

        except Exception as e:
            logger.error(f"Error adding conversation message: {e}")
            raise

    async def update_messages_with_summary_id(
        self, message_ids: List[str], summary_id: str
    ) -> bool:
        """更新被压缩的消息的 summary_id（后台任务，不抛出异常）"""
        if not message_ids:
            return False

        try:
            if not self.pool:
                logger.error("Connection pool not initialized")
                return False

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        """
                        UPDATE raw_conversations
                        SET summary_id = $1
                        WHERE id = ANY($2::text[])
                    """,
                        summary_id,
                        message_ids,
                    )
                    logger.info(
                        f"成功更新 {len(message_ids)} 条消息的 summary_id 为 {summary_id}"
                    )
                    return True

        except Exception as e:
            logger.error(f"更新消息的 summary_id 失败: {e}", exc_info=True)
            return False

    async def get_user_profile(self, user_id: int) -> Optional[str]:
        """查询用户画像 - 使用 Redis 缓存"""
        
        try:
            # 1. 先从 Redis 缓存读取
            redis_client = await get_redis_client()
            cache_key = f"user_profile:{user_id}"
            
            # AsyncRedisSaver 底层是 redis.asyncio.Redis
            cached_profile = await redis_client.get(cache_key)
            if cached_profile:
                logger.info(f"用户画像查询（Redis 缓存命中）user_id={user_id}")
                return cached_profile
            
            # 2. 缓存未命中，查询数据库
            if not self.pool:
                return None
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT user_profile FROM users WHERE user_id = $1",
                    user_id,
                )
                
                if row and row["user_profile"]:
                    profile = row["user_profile"]
                    
                    # 3. 写入 Redis 缓存（TTL 1小时）
                    await redis_client.setex(cache_key, 3600, profile)
                    logger.info(f"用户画像查询（数据库）user_id={user_id}, 已写入缓存")
                    return profile
                
                return None
                
        except Exception as e:
            logger.error(f"查询用户画像失败: {e}")
            return None

    async def update_user_profile(self, user_id: int, user_profile: str) -> bool:
        """
        更新用户画像，同时写入 Redis 缓存

        Args:
            user_id: 用户ID
            user_profile: 用户画像内容

        Returns:
            bool: 更新成功返回 True，失败返回 False
        """
        if not self.pool:
            logger.error("Connection pool not initialized")
            return False

        if not user_profile or not user_profile.strip():
            logger.warning(f"用户画像内容为空，跳过更新")
            return False

        try:
            # 1. 先更新数据库
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO users (user_id, user_profile, created_at, updated_at)
                    VALUES ($1, $2, NOW(), NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        user_profile = EXCLUDED.user_profile,
                        updated_at = NOW()
                """,
                    user_id,
                    user_profile.strip(),
                )

            # 2.更新 Redis 缓存
            try:
                redis_client = await get_redis_client()
                cache_key = f"user_profile:{user_id}"
                await redis_client.setex(cache_key, 3600, user_profile.strip())# 缓存 1 小时
                logger.info(f"用户画像缓存已更新 user_id={user_id}")
            except Exception as e:
                logger.warning(f"更新 Redis 缓存失败: {e}")
            return True

        except Exception as e:
            logger.error(f"更新用户画像失败: {e}")
            return False

    async def cache_user_profile(self, user_id: int, user_profile: str) -> None:
        """把画像写入 Redis；缓存失败不影响 Postgres 事实源。"""

        try:
            redis_client = await get_redis_client()
            await redis_client.setex(
                f"user_profile:{user_id}", 3600, user_profile.strip()
            )
        except Exception as e:
            logger.warning(f"更新用户画像缓存失败: {e}")

    async def store_memory_batch(
        self,
        *,
        user_id: int,
        thread_id: str,
        conversation_id: Optional[str],
        summary_id: str,
        summary_content: str,
        summary_importance: float,
        token_count: int,
        message_count: int,
        model_name: Optional[str],
        memories: List[Dict[str, Any]],
        superseded: List[Dict[str, str]],
        message_ids: List[str],
        user_profile: Optional[str],
    ) -> bool:
        """在一个 Postgres 事务里写入 summary、memory、version、source。

        Milvus 只在事务提交后做索引投影，不再是事实源。
        """

        if not self.pool:
            logger.error("Connection pool not initialized")
            return False

        if not memories:
            return False

        created_at = int(memories[0].get("created_at") or time.time())
        summary_content = (summary_content or "").strip()

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 1. 摘要批次
                    await conn.execute(
                        """
                        INSERT INTO summaries (
                            summary_id, user_id, thread_id, conversation_id,
                            content, importance, token_count, message_count,
                            model, status, created_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'active', $10)
                        ON CONFLICT (summary_id) DO NOTHING
                        """,
                        summary_id,
                        user_id,
                        thread_id,
                        conversation_id,
                        summary_content,
                        float(summary_importance or 0.5),
                        int(token_count),
                        int(message_count),
                        model_name,
                        created_at,
                    )

                    # 2. 旧记忆标记为 superseded，并关闭它的开放版本
                    for item in superseded:
                        old_id = item.get("memory_id")
                        new_id = item.get("superseded_by")
                        if not old_id or not new_id:
                            continue

                        closed = await conn.fetch(
                            """
                            UPDATE memory_versions
                            SET valid_to = NOW(), superseded_by = $1
                            WHERE memory_id = $2 AND valid_to IS NULL
                            RETURNING version_id
                            """,
                            new_id,
                            old_id,
                        )
                        if not closed:
                            old_row = await conn.fetchrow(
                                """
                                SELECT content, importance
                                FROM memories
                                WHERE memory_id = $1 AND user_id = $2
                                """,
                                old_id,
                                user_id,
                            )
                            if old_row:
                                await conn.execute(
                                    """
                                    INSERT INTO memory_versions (
                                        version_id, memory_id, content, importance,
                                        status, valid_from, valid_to, superseded_by
                                    )
                                    VALUES ($1, $2, $3, $4, 'superseded', NOW(), NOW(), $5)
                                    """,
                                    f"ver_{uuid.uuid4().hex}",
                                    old_id,
                                    old_row["content"],
                                    old_row["importance"],
                                    new_id,
                                )

                        await conn.execute(
                            """
                            UPDATE memories
                            SET status = 'superseded',
                                superseded_by = $1,
                                updated_at = NOW(),
                                index_status = 'pending'
                            WHERE memory_id = $2
                              AND user_id = $3
                              AND status = 'active'
                            """,
                            new_id,
                            old_id,
                            user_id,
                        )

                    # 3. 新记忆 + 初始版本
                    for mem in memories:
                        memory_id = mem["memory_id"]
                        content = mem["content"]
                        importance = float(mem.get("importance", 0.5))
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
                            mem["memory_type"],
                            content,
                            importance,
                            mem.get("source_summary_id") or summary_id,
                            thread_id,
                            conversation_id,
                            mem.get("embedding_model"),
                            mem.get("embedding_dim"),
                            int(mem.get("created_at") or created_at),
                            int(mem.get("last_access_at") or created_at),
                        )
                        await conn.execute(
                            """
                            INSERT INTO memory_versions (
                                version_id, memory_id, content, importance,
                                status, valid_from
                            )
                            VALUES ($1, $2, $3, $4, 'active', NOW())
                            """,
                            f"ver_{uuid.uuid4().hex}",
                            memory_id,
                            content,
                            importance,
                        )

                    # 4. 记忆来源：每条记忆都能回到具体消息
                    if message_ids:
                        source_rows = [
                            (mem["memory_id"], message_id, conversation_id, summary_id)
                            for mem in memories
                            for message_id in message_ids
                        ]
                        await conn.executemany(
                            """
                            INSERT INTO memory_sources (
                                memory_id, message_id, conversation_id, summary_id
                            )
                            VALUES ($1, $2, $3, $4)
                            ON CONFLICT DO NOTHING
                            """,
                            source_rows,
                        )

                    # 5. 标记这批消息已压缩
                    if message_ids:
                        await conn.execute(
                            """
                            UPDATE raw_conversations
                            SET summary_id = $1
                            WHERE id = ANY($2::text[])
                            """,
                            summary_id,
                            message_ids,
                        )

                    # 6. 画像与记忆在同一个事务里
                    if user_profile and user_profile.strip():
                        await conn.execute(
                            """
                            INSERT INTO users (user_id, user_profile, created_at, updated_at)
                            VALUES ($1, $2, NOW(), NOW())
                            ON CONFLICT (user_id) DO UPDATE SET
                                user_profile = EXCLUDED.user_profile,
                                updated_at = NOW()
                            """,
                            user_id,
                            user_profile.strip(),
                        )

            if user_profile and user_profile.strip():
                await self.cache_user_profile(user_id, user_profile)

            logger.info(
                "Postgres 记忆事务完成: user=%s summary=%s memories=%s sources=%s",
                user_id,
                summary_id,
                len(memories),
                len(memories) * len(message_ids),
            )
            return True

        except Exception as e:
            logger.error(f"写入记忆事务失败: {e}", exc_info=True)
            return False

    async def list_memories(
        self,
        user_id: int,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """从 Postgres 读取当前 active 记忆（前端记忆抽屉的事实源）。"""

        if not self.pool:
            return []

        limit = max(1, min(int(limit), 200))
        params: List[Any] = [user_id]
        sql = """
            SELECT memory_id AS id, memory_type, content, importance,
                   source_summary_id AS summary_id, created_at, last_access_at
            FROM memories
            WHERE user_id = $1 AND status = 'active'
        """
        if memory_type:
            params.append(memory_type)
            sql += f" AND memory_type = ${len(params)}"
        params.append(limit)
        sql += f" ORDER BY created_at DESC LIMIT ${len(params)}"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            {
                "id": str(row["id"]),
                "memory_type": row["memory_type"],
                "content": row["content"] or "",
                "importance": row["importance"],
                "summary_id": row["summary_id"],
                "created_at": row["created_at"],
                "last_access_at": row["last_access_at"],
            }
            for row in rows
        ]

    async def soft_delete_memory(self, memory_id: str, user_id: int) -> int:
        """软删除：保留事实与版本，Milvus 状态由调用方同步。"""

        if not self.pool:
            return 0

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(
                        """
                        UPDATE memories
                        SET status = 'deleted',
                            deleted_at = NOW(),
                            updated_at = NOW(),
                            index_status = 'pending'
                        WHERE memory_id = $1
                          AND user_id = $2
                          AND status <> 'deleted'
                        RETURNING memory_id
                        """,
                        memory_id,
                        user_id,
                    )
                    if not row:
                        return 0

                    await conn.execute(
                        """
                        UPDATE memory_versions
                        SET valid_to = NOW()
                        WHERE memory_id = $1 AND valid_to IS NULL
                        """,
                        memory_id,
                    )
                    return 1
        except Exception as e:
            logger.error(f"软删除记忆失败: {e}", exc_info=True)
            return 0

    async def mark_memories_index_status(
        self, memory_ids: List[str], status: str
    ) -> None:
        """记录 Milvus 投影状态；失败不抛错，交给 reindex 修复。"""

        if not self.pool or not memory_ids:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE memories
                    SET index_status = $1::varchar,
                        indexed_at = CASE
                            WHEN $1::text = 'indexed' THEN NOW()
                            ELSE indexed_at
                        END
                    WHERE memory_id = ANY($2::varchar[])
                    """,
                    status,
                    memory_ids,
                )
        except Exception as e:
            logger.warning(f"更新索引状态失败: {e}")

    async def update_memory_last_access(
        self, memory_ids: List[str], timestamp: int
    ) -> None:
        """同步记忆访问时间；Postgres 是展示与审计的事实源。"""

        if not self.pool or not memory_ids:
            return
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE memories
                    SET last_access_at = $1
                    WHERE memory_id = ANY($2::varchar[])
                    """,
                    int(timestamp),
                    memory_ids,
                )
        except Exception as e:
            logger.warning(f"更新记忆访问时间失败: {e}")

    async def get_memories_for_reindex(
        self, user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """读取全部记忆（含 superseded/deleted），供重建 Milvus 使用。"""

        if not self.pool:
            return []

        sql = """
            SELECT memory_id, user_id, memory_type, content, importance, status,
                   source_summary_id, thread_id, conversation_id,
                   embedding_model, embedding_dim, created_at, last_access_at
            FROM memories
        """
        params: List[Any] = []
        if user_id is not None:
            sql += " WHERE user_id = $1"
            params.append(user_id)
        sql += " ORDER BY created_at ASC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [dict(row) for row in rows]

    async def get_raw_conversation_by_summary_id(
        self, summary_id: str, user_id: int
    ) -> str:
        """
        通过 summary_id 检索对话，并返回格式化的原始对话文本。
        summary_id 本身是每次压缩生成的 UUID，已能唯一定位一批消息，因此不再需要 thread_id。

        Args:
            summary_id: 摘要ID
            user_id: 用户ID

        Returns:
            str: 格式化的对话文本，格式如：
                [2026-04-13 18:32:17] human: 你好
                [2026-04-13 18:32:17] ai: 你好，小明！
                ...
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        if not summary_id:
            return ""

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT role, content, created_at
                    FROM raw_conversations
                    WHERE summary_id = $1 
                    AND user_id = $2 
                    ORDER BY created_at ASC
                    """,
                    summary_id,
                    user_id,
                )

                if not rows:
                    logger.warning(
                        f"未找到 summary_id={summary_id} 对应的对话 "
                        f"(user_id={user_id})"
                    )
                    return f"未找到 summary_id={summary_id} 对应的对话"

                formatted_lines = []
                for row in rows:
                    # 格式化时间戳
                    created_at = row["created_at"]
                    if hasattr(created_at, "strftime"):
                        time_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        time_str = str(created_at)

                    # 构建格式化的消息行
                    formatted_lines.append(
                        f"[{time_str}] {row['role']}: {row['content']}"
                    )

                result = f"以下是summary_id={summary_id}对应的原始对话记录:\n" + "\n".join(formatted_lines)
                logger.info(f"成功查询到 summary_id={summary_id} 对应的原始对话记录，共 {len(rows)} 条消息")

                return result

        except Exception as e:
            logger.error(f"通过 summary_id 检索并格式化对话失败: {e}")
            return f"通过 summary_id 检索对话失败: {e}"


# 测试函数
async def test_query_by_user(user_id: int):
    """测试按 user_id 查询对话"""

    print(f"\n📋 查询 user_id={user_id} 的对话记录")
    print("-" * 60)

    try:
        pg_client = await get_postgresql_client()

        async with pg_client.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, summary_id, role, content, created_at
                FROM raw_conversations
                WHERE user_id = $1
                ORDER BY created_at ASC
                LIMIT 100
            """,
                user_id,
            )

            if not rows:
                print(f"⚠️  未找到 user_id={user_id} 的对话记录")
                return

            print(f"✅ 找到 {len(rows)} 条记录:\n")
            for row in rows:
                print(
                    f"  [{row['created_at']}(summary_id):{row['summary_id']})] {row['role']}: {row['content'][:200]}"
                )
    except Exception as e:
        print(f"❌ 查询失败: {e}")


# 测试函数
async def reset_summary_id_to_null(user_id: Optional[int] = None):
    """
    将消息的 summary_id 重置为 NULL

    Args:
        user_id: 可选，如果指定则只重置该用户的记录，否则重置所有用户的记录

    Returns:
        int: 更新的记录数量
    """
    try:
        pg_client = await get_postgresql_client()

        async with pg_client.pool.acquire() as conn:
            # 构建 SQL 语句
            if user_id is not None:
                # 只重置指定用户的记录
                result = await conn.execute(
                    """
                    UPDATE raw_conversations
                    SET summary_id = NULL
                    WHERE user_id = $1
                """,
                    user_id,
                )

                # 获取影响的行数
                rows_affected = result.split(" ")[-1] if result else "0"

                print(f"✅ 已重置 user_id={user_id} 的所有记录的 summary_id 为 NULL")
                print(f"   影响记录数: {rows_affected}")

            else:
                # 重置所有用户的记录
                result = await conn.execute(
                    """
                    UPDATE raw_conversations
                    SET summary_id = NULL
                """
                )

                # 获取影响的行数
                rows_affected = result.split(" ")[-1] if result else "0"

                print(f"✅ 已重置所有记录的 summary_id 为 NULL")
                print(f"   影响记录数: {rows_affected}")

            return int(rows_affected) if rows_affected.isdigit() else 0

    except Exception as e:
        print(f"❌ 重置 summary_id 失败: {e}")
        return 0


if __name__ == "__main__":
    # 重置用户ID=1的所有记录的summary_id为NULL（调试用，用于auto_summrize_psql_message设置summary_id后的重置）
    #asyncio.run(reset_summary_id_to_null(1))

    # 查看对话历史记录（用户ID=1），100条
    asyncio.run(test_query_by_user(1))
