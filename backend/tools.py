import asyncio
from citations import build_numbered_context, dedupe_documents, order_by_rerank
from documents_process import rerank_documents
from postgresql_client import get_postgresql_client
from milvus_client import get_milvus_client
import logging
import sys
from langgraph.config import get_stream_writer
import time
from schemas import ContextSchema
from langchain.tools import tool, ToolRuntime
import config


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@tool("get_raw_conversation_by_summary_id")
async def get_raw_conversation_by_summary_id(
    summary_id: str,
    runtime: ToolRuntime[ContextSchema]
) -> str:
    """
    获取指定摘要ID对应的完整原始对话记录。
    
    该工具返回按时间顺序排列的原始对话文本，包含每条消息的时间戳、角色和完整内容。
    
    适用场景：
    - 摘要内容不够详细，需要查看完整的对话细节
    - 用户明确要求查看原始对话记录
    - 需要对对话内容进行精确分析、引用或二次处理
    - 摘要中遗漏了关键信息，需要补充上下文
    
    注意：
    - 只能在获取到对应摘要summary_id后调用，确保摘要存在且已压缩
    - 原始对话可能较长，调用前应评估是否需要完整内容
    - 仅在摘要无法满足需求时调用，避免不必要的开销
    - 返回格式：[时间戳] 角色: 消息内容
    
    Args:
        summary_id: 摘要ID，用于标识需要检索的对话摘要

    Returns:
        str: 格式化的原始对话文本，每条消息独立成行，包含时间戳、角色和内容
    """
    user_id = runtime.context.user_id
    postgresql_client = await get_postgresql_client()
    writer = get_stream_writer()
    writer(f"🔍 正在检索原始对话...")
    start_time = time.time()
    raw_conversation = await postgresql_client.get_raw_conversation_by_summary_id(
        summary_id=summary_id,
        user_id=user_id,
    )
    writer(f"🔍 原始对话检索完成，耗时{time.time() - start_time:.2f}秒")
    return raw_conversation


@tool("get_memory")
async def get_memory(
    query: str,
    semantic_k: int,
    episodic_k: int,
    procedural_k: int,
    runtime: ToolRuntime[ContextSchema],
) -> str:
    """
    检索用户长期记忆，用于个性化回答、事实验证和历史连贯性。

    【调用时机】满足以下任一条件即调用：
    - 用户提到之前、刚才、上次等回忆性词汇
    - 重复提问或用户表现困惑
    - 询问操作步骤、个人偏好、习惯设置
    - 任何需要记住历史对话的场景

    【参数决策】（k值 0-5，0=不需要，3-5=核心依赖）
    - semantic_k（事实/概念）：事实问答3-5，操作问题2-3，历史回忆1-2
    - episodic_k（历史/对话）：重复提问/回忆3-5，事实问答2-3，操作问题1-2
    - procedural_k（步骤/方法）：操作问题3-5，事实问答2-3，历史回忆1-2

    Args:
        query: 用户原始问题（如果过于模糊或口语化可做适当改写）
        semantic_k: 语义记忆召回数量
        episodic_k: 情节记忆召回数量
        procedural_k: 程序记忆召回数量
    """
    logger.info(f"{'---'*20}开始检索记忆hybrid_retrieval_memories{'---'*20}")
    logger.info(f"ai希望检索的各种记忆数量: 语义记忆{semantic_k}, 情节记忆{episodic_k}, 程序记忆{procedural_k}")
    user_id = runtime.context.user_id
    writer = get_stream_writer()
    writer(f"🔍 正在检索记忆库...")
    start_time = time.time()
    milvus_client = await get_milvus_client()
    memories_dict = await milvus_client.hybrid_retrieval_memories(
        query,
        user_id=user_id,
        summary_k=1,
        semantic_k=semantic_k,
        episodic_k=episodic_k,
        procedural_k=procedural_k,
    )
    count = 1
    for memory in memories_dict.values():
        if isinstance(memory, list):
            count += len(memory)
        
    writer(f"🔍 检索到{count}条记忆，耗时{time.time() - start_time:.2f}秒")
    now = int(time.time())
    memories_text = config.MEMORY_USAGE_PROMPT.format(
        memories_dict=memories_dict,
        current_timestamp=now,
    )
    return memories_text


@tool("search_knowledge_base")
async def search_knowledge_base(
    query: str, runtime: ToolRuntime[ContextSchema]  # 只需要 query，其他从 runtime 获取
) -> str:
    """
    **功能**：从私有知识库检索相关文档、专业知识、业务规则和内部资料
    - **适用场景**：
    - 问题涉及领域专业知识、技术细节或特定业务规则
    - 需要引用内部文档、官方资料或准确事实
    - 对不确定的信息进行验证，避免产生幻觉
    - 用户明确要求查询知识库或资料
    Args:
        query: 搜索查询词
    """
    # 从强类型 context 获取参数
    user_id = runtime.context.user_id
    knowledge_base_id = runtime.context.knowledge_base_id
    top_k = runtime.context.top_k

    logger.info(f"用户 {user_id} 查询知识库 {knowledge_base_id}: {query}")
    writer = get_stream_writer()
    writer(f"🔍 正在检索知识库【{knowledge_base_id}】...")
    logger.info(f"{'---'*20}开始混合检索hybrid_retrieval_knowledge_base{'---'*20}")
    start_time = time.time()

    # 使用全局 Milvus 客户端
    milvus_client = await get_milvus_client()
    parent_chunkId_list = await milvus_client.hybrid_retrieval_knowledge_base(
        query=query, knowledge_base_id=knowledge_base_id, top_k=top_k, user_id=user_id
    )

    logger.info(
        f"{'---'*20}开始从PostgreSQL获取父块get_parents，一共需要检索出{len(parent_chunkId_list)}个父块{'---'*20}"
    )

    if not parent_chunkId_list:
        writer("⚠️ 未检索到相关文档，请尝试更换关键词或稍后再试。")
        logger.info(f"{'---'*20}未检索到相关文档{'---'*20}")
        return "未检索到相关文档"

    # 使用全局 PostgreSQL 客户端
    postgresql_client = await get_postgresql_client()
    parent_documents = await postgresql_client.get_parents(
        knowledge_base_id=knowledge_base_id,
        parent_ids=parent_chunkId_list,
        user_id=user_id,
    )

    if not parent_documents:
        writer("⚠️ 检索到片段但未能取回原文，请稍后再试。")
        logger.info(f"{'---'*20}父块为空{'---'*20}")
        return "未检索到相关文档"

    # 重排序父块。输入纯文本，结果里的 index 用来映射回文件名。
    logger.info(f"{'---'*20}开始重排序父块rerank_documents{'---'*20}")
    rerank_result = await rerank_documents(
        query, [doc["text"] for doc in parent_documents], top_k
    )

    ranked_documents = order_by_rerank(parent_documents, rerank_result)
    ranked_documents = dedupe_documents(ranked_documents)[:top_k]

    snippets, sources = build_numbered_context(
        ranked_documents, start_index=len(runtime.context.sources) + 1
    )
    if not snippets:
        return "未检索到相关文档"

    runtime.context.sources.extend(sources)

    end_time = time.time()
    writer(f"✓ 检索完成，本次检索耗时{end_time - start_time:.2f}秒")
    logger.info(f"{'---'*20}检索完成{'---'*20}")

    return f"检索到的相关文档如下（回答时请用方括号编号标注来源）：\n{snippets}"

"""rerank_result的结构示例：
{'output': {'results': [{'document': {'text': '20 世纪80 年代末'}, 'index': 0, 
'relevance_score': 0.886919463597282}]}, 'usage': {'total_tokens': 1224}, 
'request_id': '2252323b-a3ba-4ef5-a203-e305b64249e1'}  
"""