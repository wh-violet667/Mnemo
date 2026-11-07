import os
import asyncio
import time
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import AsyncMilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest, RRFRanker
import logging

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

SUPERSEDE_SIMILARITY = 0.9


def select_superseded_candidates(
    candidates: List[tuple[int, str, float]],
    threshold: float = SUPERSEDE_SIMILARITY,
) -> Dict[int, Dict[str, Any]]:
    """同一条旧记忆只保留相似度最高的一条新记忆来替换。"""

    ordered = sorted(
        [
            (item_idx, old_id, float(score))
            for item_idx, old_id, score in candidates
            if float(score) >= threshold
        ],
        key=lambda item: item[2],
        reverse=True,
    )

    mapping: Dict[int, Dict[str, Any]] = {}
    used_old_ids = set()
    for item_idx, old_id, score in ordered:
        if old_id in used_old_ids:
            continue
        used_old_ids.add(old_id)
        mapping[item_idx] = {"memory_id": old_id, "score": score}
    return mapping


class MemoryManager:
    """记忆管理器 - 负责记忆的初始化、存储、更新、检索"""

    def __init__(
        self,
        client: AsyncMilvusClient,
        embeddings: DashScopeEmbeddings,
        collection_name: str,
        dense_dim: int,
    ):
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.dense_dim = dense_dim

    async def init_collection(self):
        """初始化对话记忆集合。

        Postgres 是事实源，Milvus 只保存可重建的检索投影，因此主键必须由
        Postgres 生成（auto_id=False），否则无法做幂等 upsert 和 reindex。
        """
        if await self.client.has_collection(self.collection_name):
            desc = await self.client.describe_collection(self.collection_name)
            fields = {field["name"]: field for field in desc["fields"]}
            id_field = fields.get("id", {})
            if id_field.get("auto_id") or "status" not in fields:
                raise RuntimeError(
                    f"集合 {self.collection_name} 是旧版 schema（auto_id=True 或缺少 status）。"
                    "请先运行 backend/migrate_memory_v2.py 完成迁移，再启动后端。"
                )
            logger.info(f"对话记忆集合 {self.collection_name} 已存在")
            await self.client.load_collection(self.collection_name)
            return

        # 创建 Schema
        schema = self.client.create_schema(enable_dynamic_field=True)

        # 主键：由 Postgres 生成，Milvus 只做投影
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            auto_id=False,
        )

        # 用户标识
        schema.add_field(
            field_name="user_id", datatype=DataType.INT64, is_partition_key=True
        )

        # 会话标识
        schema.add_field(
            field_name="thread_id", datatype=DataType.VARCHAR, max_length=100
        )

        # 记忆类型
        schema.add_field(
            field_name="memory_type", datatype=DataType.VARCHAR, max_length=30
        )

        # 记忆状态：active / superseded / deleted
        schema.add_field(
            field_name="status",
            datatype=DataType.VARCHAR,
            max_length=20,
            default_value="active",
        )

        # 记忆内容
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )

        # 关联的 summary_id
        schema.add_field(
            field_name="summary_id",
            datatype=DataType.VARCHAR,
            max_length=50,
            nullable=True,
        )

        # 重要性评分
        schema.add_field(field_name="importance", datatype=DataType.FLOAT)

        # 时间戳
        schema.add_field(field_name="created_at", datatype=DataType.INT64)
        schema.add_field(
            field_name="last_access_at",
            datatype=DataType.INT64,
            default_value=0,
        )

        # 向量字段
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim
        )

        # BM25 稀疏向量
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 配置 BM25 内置函数
        bm25_function = Function(
            name="bm25_func",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
        )
        schema.add_function(bm25_function)

        # 配置索引
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        index_params.add_index(field_name="user_id", index_type="STL_SORT")
        index_params.add_index(field_name="thread_id", index_type="STL_SORT")
        index_params.add_index(field_name="memory_type", index_type="STL_SORT")
        index_params.add_index(field_name="status", index_type="STL_SORT")
        index_params.add_index(field_name="importance", index_type="STL_SORT")
        index_params.add_index(field_name="last_access_at", index_type="STL_SORT")
        index_params.add_index(field_name="created_at", index_type="STL_SORT")

        # 创建集合
        await self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            properties={"partitionkey.isolation": True},
        )
        await self.client.load_collection(self.collection_name)

    async def get_dense_vector(self, query: str) -> List[float]:
        """生成稠密向量，带重试机制"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                dense_vector = await self.embeddings.aembed_query(query)
                break
            except Exception as e:
                logger.error(
                    f"Embedding API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    logger.error(f"Embedding API 最终失败，返回空结果")
                    return []
                else:
                    logger.warning(f"等待 1 秒后重试...")
                    await asyncio.sleep(1)
                    continue
        return dense_vector

    async def hybrid_retrieval_memories(
        self,
        query: str,
        user_id: int,
        summary_k: int,
        semantic_k: int,
        episodic_k: int,
        procedural_k: int,
    ) -> Dict[str, List[Dict]]:
        """混合检索，每种记忆类型并行检索、去重，返回召回的相似记忆内容"""

        memory_configs = {
            key: {
                "k": k,
                "filter": (
                    f"user_id == {user_id} and memory_type == '{key}' "
                    "and status == 'active'"
                ),
            }
            for key, k in {
                "summary": summary_k,
                "semantic": semantic_k,
                "episodic": episodic_k,
                "procedural": procedural_k,
            }.items()
            if k
        }

        dense_vector = await self.get_dense_vector(query)
        if not dense_vector:
            return {}

        async def search_memory_type(memory_type: str, config: dict):
            k = config["k"]
            filter_expr = config["filter"]

            try:
                dense_req = AnnSearchRequest(
                    data=[dense_vector],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"ef": 64}},
                    limit=k * 6,
                    expr=filter_expr,
                )

                sparse_req = AnnSearchRequest(
                    data=[query],
                    anns_field="sparse_vector",
                    param={"metric_type": "BM25"},
                    limit=k * 4,
                    expr=filter_expr,
                )

                results = await self.client.hybrid_search(
                    collection_name=self.collection_name,
                    reqs=[dense_req, sparse_req],
                    ranker=RRFRanker(),
                    limit=k * 3,
                    output_fields=[
                        "id",
                        "memory_type",
                        "content",
                        "summary_id",
                        "importance",
                        "last_access_at",
                    ],
                )

                memories = []
                seen_ids = set()

                if results and len(results) > 0:
                    for hit in results[0]:
                        entity = hit["entity"]
                        memory_id = entity.get("id")

                        if memory_id and memory_id not in seen_ids:
                            seen_ids.add(memory_id)
                            memories.append(
                                {
                                    "id": memory_id,
                                    "memory_type": entity.get("memory_type"),
                                    "content": entity.get("content"),
                                    "summary_id": entity.get("summary_id"),
                                    "importance": entity.get("importance"),
                                    "last_access_at": entity.get("last_access_at"),
                                    "score": hit["distance"],
                                }
                            )

                return {memory_type: memories[: k * 2]}

            except Exception as e:
                logger.error(
                    f"记忆类型 {memory_type} 混合检索失败：{e}, 降级为稠密向量检索"
                )

                try:
                    results = await self.client.search(
                        collection_name=self.collection_name,
                        data=[dense_vector],
                        anns_field="vector",
                        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                        limit=k * 6,
                        filter=filter_expr,
                        output_fields=[
                            "id",
                            "memory_type",
                            "content",
                            "summary_id",
                            "importance",
                            "last_access_at",
                        ],
                    )

                    memories = []
                    if results and len(results) > 0:
                        for hit in results[0]:
                            entity = hit["entity"]
                            memories.append(
                                {
                                    "id": entity.get("id"),
                                    "memory_type": entity.get("memory_type"),
                                    "content": entity.get("content"),
                                    "summary_id": entity.get("summary_id"),
                                    "importance": entity.get("importance"),
                                    "last_access_at": entity.get("last_access_at"),
                                    "score": hit["distance"],
                                }
                            )

                    return {memory_type: memories[: k * 2]}

                except Exception as search_e:
                    logger.error(f"记忆类型 {memory_type} 稠密检索也失败：{search_e}")
                    return {memory_type: []}

        tasks = [
            search_memory_type(mem_type, config)
            for mem_type, config in memory_configs.items()
        ]

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = {
            "summary": [],
            "semantic": [],
            "episodic": [],
            "procedural": [],
        }

        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"并行检索任务失败：{result}")
                continue

            if result and isinstance(result, dict):
                for mem_type, memories in result.items():
                    if mem_type in final_results:
                        final_results[mem_type].extend(memories)

        top_k_memories = self.get_the_top_k_memories(
            memory_dict=final_results,
            memory_configs=memory_configs,
        )

        await self.update_memory_last_access_time(top_k_memories)

        return top_k_memories
        """
        返回的结构类似下面这样
        {
        "summary": 
            {
                "id": "mem_001",#这里的id是记忆的id，方便后续更新记忆的last_access_at时使用
                "memory_type": "summary",
                "content": "用户喜欢喝美式咖啡，不加糖",
                "summary_id": "sum_001",或者None
                "last_access_at": 1734019200.0,  # 时间戳
            },
        "semantic": [],
        "episodic": [],
        "procedural": [],
        }
        """

    async def update_memory_last_access_time(
        self, top_k_memories: Dict[str, List[Dict]]
    ):
        """更新记忆的最后访问时间"""
        current_timestamp = int(time.time())
        update_data = []

        for memories in top_k_memories.values():
            for mem in memories:
                if "id" in mem:
                    update_data.append(
                        {"id": mem["id"], "last_access_at": current_timestamp}
                    )

        if update_data:
            res = await self.client.upsert(
                collection_name=self.collection_name,
                data=update_data,
                partial_update=True,
            )
            print(f"更新记忆访问时间结果：{res}")

    def get_the_top_k_memories(
        self,
        memory_dict: Dict[str, List[Dict]],
        alpha: float = 0.45,
        beta: float = 0.25,
        gamma: float = 0.3,
        memory_configs: Dict[str, Dict] = None,
        type_weights: Dict[str, float] = None,
    ) -> Dict[str, List[Dict]]:
        """从记忆字典中提取每个记忆类型的前 k 条记录"""
        if type_weights is None:
            type_weights = {
                "summary": 0.7,
                "semantic": 1.3,
                "episodic": 1.0,
                "procedural": 1.2,
            }

        current_time = time.time()
        DECAY_RATE = 0.995

        result = {}

        for mem_type, memories in memory_dict.items():
            if not memories:
                result[mem_type] = []
                continue

            type_weight = type_weights.get(mem_type, 1.0)
            scored_memories = []

            for mem in memories:
                semantic_score = mem.get("score", 0.5)

                last_access = mem.get("last_access_at", current_time)
                hours_passed = (current_time - last_access) / 3600
                recency_score = DECAY_RATE**hours_passed

                importance_score = mem.get("importance", 0.5)

                final_score = (
                    alpha * semantic_score
                    + beta * recency_score
                    + gamma * importance_score
                ) * type_weight

                scored_memories.append((mem, final_score))

            scored_memories.sort(key=lambda x: x[1], reverse=True)
            top_k = memory_configs[mem_type]["k"]

            result[mem_type] = [
                {
                    "id": mem.get("id"),
                    "memory_type": mem.get("memory_type"),
                    "content": mem.get("content"),
                    "last_access_at": mem.get("last_access_at"),
                    "summary_id": mem.get("summary_id"),
                }
                for mem, _ in scored_memories[:top_k]
            ]

        return result

    SUPERSEDE_SIMILARITY = 0.9

    async def find_superseded_candidates(
        self,
        items: List[Dict[str, Any]],
        user_id: int,
    ) -> tuple[Dict[int, Dict[str, Any]], List[List[float]]]:
        """找出每条新记忆应替换的旧记忆，不删除任何数据。

        返回 ``(mapping, vectors)``：

        - ``mapping``: ``{新记忆下标: {"memory_id": 旧记忆ID, "score": 相似度}}``
        - ``vectors``: 与新记忆同序的 embedding，供后续写 Milvus 复用
        """

        if not items:
            return {}, []

        texts = [item["content"] for item in items]
        vectors = await self.embeddings.aembed_documents(texts)

        from collections import defaultdict

        type_to_pairs = defaultdict(list)
        for idx, (item, vector) in enumerate(zip(items, vectors)):
            type_to_pairs[item["memory_type"]].append((idx, vector))

        candidates = []

        async def search_type(mem_type, pairs):
            indices = [pair[0] for pair in pairs]
            type_vectors = [pair[1] for pair in pairs]
            results = await self.client.search(
                collection_name=self.collection_name,
                data=type_vectors,
                anns_field="vector",
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=1,
                filter=(
                    f"user_id == {user_id} and memory_type == '{mem_type}' "
                    "and status == 'active'"
                ),
                output_fields=["id"],
            )
            for local_idx, hits in enumerate(results):
                if (
                    hits
                    and float(hits[0].get("distance", 0))
                    >= self.SUPERSEDE_SIMILARITY
                ):
                    candidates.append(
                        (
                            indices[local_idx],
                            str(hits[0]["id"]),
                            float(hits[0]["distance"]),
                        )
                    )

        await asyncio.gather(
            *[
                search_type(mem_type, pairs)
                for mem_type, pairs in type_to_pairs.items()
            ]
        )

        return (
            select_superseded_candidates(
                candidates, threshold=self.SUPERSEDE_SIMILARITY
            ),
            vectors,
        )

    async def list_memories(
        self,
        user_id: int,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """按用户读取长期记忆，供前端记忆抽屉展示。"""

        filter_expr = f"user_id == {int(user_id)} and status == 'active'"
        if memory_type:
            filter_expr += f" and memory_type == '{memory_type}'"

        rows = await self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr,
            output_fields=[
                "id",
                "memory_type",
                "content",
                "summary_id",
                "importance",
                "created_at",
                "last_access_at",
            ],
            limit=max(1, min(int(limit), 200)),
        )

        memories = []
        for row in rows:
            memories.append(
                {
                    "id": str(row.get("id")),
                    "memory_type": row.get("memory_type"),
                    "content": row.get("content") or "",
                    "summary_id": row.get("summary_id"),
                    "importance": row.get("importance"),
                    "created_at": row.get("created_at"),
                    "last_access_at": row.get("last_access_at"),
                }
            )

        memories.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
        return memories

    async def set_memory_status(self, memory_ids: List[str], status: str) -> bool:
        """更新 Milvus 投影里的状态；只更新已存在的行。"""

        if not memory_ids:
            return True

        quoted = ", ".join(f'"{memory_id}"' for memory_id in memory_ids)
        try:
            rows = await self.client.query(
                collection_name=self.collection_name,
                filter=f"id in [{quoted}]",
                output_fields=["id"],
            )
            existing_ids = {str(row["id"]) for row in rows}
            data = [
                {"id": memory_id, "status": status}
                for memory_id in memory_ids
                if memory_id in existing_ids
            ]
            if data:
                await self.client.upsert(
                    collection_name=self.collection_name,
                    data=data,
                    partial_update=True,
                )
            return True
        except Exception as e:
            logger.error(f"更新 Milvus 记忆状态失败: {e}")
            return False

    async def index_memory_records(
        self,
        records: List[Dict[str, Any]],
        vectors: Optional[List[List[float]]] = None,
    ) -> bool:
        """把 Postgres 记忆投影到 Milvus（幂等 upsert）。"""

        if not records:
            return True

        if vectors is None:
            vectors = await self.embeddings.aembed_documents(
                [record["content"] for record in records]
            )

        if len(vectors) != len(records):
            raise ValueError("vectors 数量与 records 不一致")

        data = []
        for record, vector in zip(records, vectors):
            data.append(
                {
                    "id": str(record["memory_id"]),
                    "user_id": int(record["user_id"]),
                    "thread_id": record.get("thread_id") or "",
                    "memory_type": record["memory_type"],
                    "content": record["content"],
                    "summary_id": record.get("source_summary_id")
                    or record.get("summary_id"),
                    "importance": float(record.get("importance", 0.5)),
                    "status": record.get("status", "active"),
                    "created_at": int(record.get("created_at") or time.time()),
                    "last_access_at": int(record.get("last_access_at") or 0),
                    "vector": vector,
                }
            )

        try:
            for start in range(0, len(data), 100):
                await self.client.upsert(
                    collection_name=self.collection_name,
                    data=data[start : start + 100],
                )
            logger.info(f"Milvus 记忆投影完成: {len(data)} 条")
            return True
        except Exception as e:
            logger.error(f"写入 Milvus 记忆投影失败: {e}")
            return False


async def test_update_access_time():
    """测试：更新记忆访问时间"""
    from milvus_client import get_milvus_client  # 根据实际导入路径调整

    client = AsyncMilvusClient(
        uri=os.getenv("Milvus_url"),
        token=os.getenv("Token"),
    )
    embeddings = DashScopeEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    memory_manager = MemoryManager(
        client=client,
        embeddings=embeddings,
        collection_name=os.getenv("memory_collection"),  # 你的集合名
        dense_dim=int(os.getenv("dense_dimension", "1024")),  # 你的向量维度
    )

    test_memory = {
        "semantic": [
            {
                "id": "465201085538812576",  # 这里的id是记忆的id
                "memory_type": "semantic",
                "content": "用户喜欢喝美式咖啡",
                "summary_id": None,
                "last_access_at": time.time(),
            },
        ]
    }

    await memory_manager.update_memory_last_access_time(test_memory)


if __name__ == "__main__":
    print("开始更新记忆访问时间")
    asyncio.run(test_update_access_time())
