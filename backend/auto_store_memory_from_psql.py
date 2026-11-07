"""
定时自动压缩psql中的历史对话，压缩函数提取的是langchain中的SummarizationMiddleware，
压缩时检查对话是否超过 400 个 token（TOKEN_THRESHOLD），超过则压缩，否则不压缩。
新增逻辑：大模型识别后半部分语义差异大且不完整的消息，仅压缩前半部分相关内容，返回过滤的消息ID列表
"""

import asyncio
import logging
import time
import uuid
import json
from typing import List, Dict, Any
from datetime import datetime
import config
from json_repair import repair_json
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately
from postgresql_client import get_postgresql_client
from langchain_openai import ChatOpenAI
from milvus_client import get_milvus_client
import os

logger = logging.getLogger(__name__)

TOKEN_THRESHOLD = 400  # 低于此阈值的对话不做自动记忆抽取，可按交流频率调整
MEMORY_EXTRACT_PROMPT = config.MEMORY_EXTRACT_PROMPT
USER_PROFILE_MERGE_PROMPT = config.USER_PROFILE_MERGE_PROMPT


async def extract_memories(
    messages: List[Dict[str, Any]],
    model: BaseChatModel,
) -> Dict[str, Any]:
    """
    从对话中提取多种类型的记忆

    Args:
        messages: 原始消息列表（包含id/role/content/created_at）
        model: 用于提取记忆的 LLM 模型

    Returns:
        dict: {
            "summary": {"content": str, "importance_score": float},
            "semantic_memory": [{"content": str, "importance_score": float}, ...],
            "episodic_memory": [{"content": str, "importance_score": float}, ...],
            "procedural_memory": [{"content": str, "importance_score": float}, ...],
            "user_profile": str,
            "filtered_message_ids": list
        }
    """
    if not messages:
        return {
            "summary": {"content": "", "importance_score": 0.0},
            "semantic_memory": [],
            "episodic_memory": [],
            "procedural_memory": [],
            "user_profile": "",
            "filtered_message_ids": [],
        }

    # 格式化对话文本，每个消息占一行，格式为：id | role | content | time
    lines = []
    for msg in messages:
        lines.append(
            f"{msg.get('id', '')} | {msg.get('role', '')} | {msg.get('content', '')} | {msg.get('created_at', 'N/A')}"
        )
    conversation_text = "\n\n".join(lines)

    # 调用模型
    response = await model.ainvoke(
        [
            HumanMessage(
                content=MEMORY_EXTRACT_PROMPT.format(
                    conversation_text=conversation_text
                )
            )
        ]
    )

    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    try:
        result = json.loads(content.strip())
    except json.JSONDecodeError:
        # 小模型偶尔会把 JSON 字符串里的引号写坏，先尝试自动修复再失败
        logger.warning(
            "记忆提取返回非法 JSON，尝试修复: %s", content[:500]
        )
        try:
            result = json.loads(repair_json(content))
        except Exception as repair_error:
            logger.error(
                "记忆提取 JSON 修复失败: %s; raw=%s",
                repair_error,
                content[:2000],
            )
            raise

    # 解析结果
    def safe_score(v):
        try:
            return round(
                max(0.0, min(1.0, float(v))), 1
            )  # 将importance_score安全地限制在 0.0 到 1.0 之间，并保留 1 位小数
        except:
            return 0.5

    # 解析记忆项，确保 content 字段存在且非空字符串
    def parse_items(items):
        if not isinstance(items, list):
            return []
        return [
            {
                "content": item["content"].strip(),
                "importance_score": safe_score(item.get("importance_score", 0.5)),
            }
            for item in items
            if isinstance(item, dict)
            and item.get("content", "").strip()  # 确保 content 字段存在且非空字符串
        ]

    summary_raw = result.get("summary", {})

    return {
        "summary": {
            "content": summary_raw.get("content", "").strip(),
            "importance_score": safe_score(summary_raw.get("importance_score", 0.5)),
        },
        "semantic_memory": parse_items(result.get("semantic_memory")),
        "episodic_memory": parse_items(result.get("episodic_memory")),
        "procedural_memory": parse_items(result.get("procedural_memory")),
        "user_profile": result.get("user_profile", "").strip(),
        "filtered_message_ids": result.get("filtered_message_ids", []),
    }


async def get_unsunmarized_conversations(
    user_id: int, thread_id: str
) -> List[Dict[str, Any]]:
    """获取指定用户和会话中未摘要的对话消息"""
    try:
        pg_client = await get_postgresql_client()

        if not pg_client.pool:
            logger.error("数据库连接池未初始化，无法获取未摘要消息")
            return []

        async with pg_client.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content, created_at, thread_id, conversation_id
                FROM raw_conversations
                WHERE user_id = $1 
                AND thread_id = $2
                AND (summary_id IS NULL OR summary_id = '')
                ORDER BY created_at ASC
            """,
                user_id,
                thread_id,
            )

            messages = [dict(row) for row in rows]
            logger.info(
                f"获取到用户 {user_id} 会话 {thread_id} 的 {len(messages)} 条未摘要消息"
            )
            return messages

    except Exception as e:
        logger.error(
            f"获取未摘要消息失败 user_id={user_id}, thread_id={thread_id}: {e}",
            exc_info=True,
        )
        return []


# 这个才是存入记忆的关键
async def extract_and_append_memory(
    messages: List[Dict[str, Any]], model: BaseChatModel, user_id: int, thread_id: str
) -> Dict[str, Any]:
    """压缩单个会话的消息并生成摘要"""
    if not messages:
        return {"success": False, "reason": "没有消息需要压缩"}

    # 1. 计算 token 数
    conversation_text = "\n\n".join(
        [f"{m.get('role')}: {m.get('content')}" for m in messages]
    )
    total_tokens = count_tokens_approximately([HumanMessage(content=conversation_text)])

    if total_tokens <= TOKEN_THRESHOLD:
        return {
            "success": False,
            "reason": f"token 数未超过阈值 ({total_tokens} <= {TOKEN_THRESHOLD})",
            "token_count": total_tokens,
            "message_count": len(messages),
        }

    # 2. 提取记忆
    try:
        extract_result = await extract_memories(messages, model)
    except Exception as e:
        logger.error(f"记忆提取失败: {e}", exc_info=True)
        return {
            "success": False,
            "reason": f"记忆提取失败: {str(e)}",
            "token_count": total_tokens,
            "message_count": len(messages),
        }

    # 3. 存储记忆
    filtered_message_ids = extract_result.get("filtered_message_ids", [])
    try:
        return await _store_memories(
            messages=messages,
            extract_result=extract_result,
            user_id=user_id,
            thread_id=thread_id,
            total_tokens=total_tokens,
            filtered_message_ids=filtered_message_ids,
            model=model,
        )
    except Exception as e:
        logger.error(f"存储记忆失败: {e}", exc_info=True)
        return {
            "success": False,
            "reason": f"存储记忆失败: {str(e)}",
            "filtered_message_ids": filtered_message_ids,
            "token_count": total_tokens,
            "message_count": len(messages),
        }


async def _store_memories(
    messages: List[Dict[str, Any]],
    extract_result: Dict[str, Any],
    user_id: int,
    thread_id: str,
    total_tokens: int,
    filtered_message_ids: List[str],
    model: BaseChatModel,
) -> Dict[str, Any]:
    """把一批抽取结果写入 Postgres（事实源），再投影到 Milvus。

    Postgres 事务提交后，Milvus 失败不会回滚事实数据；索引状态会标记为
    failed，交给 reindex 脚本修复。
    """
    milvus_client = await get_milvus_client()
    pg_client = await get_postgresql_client()

    new_user_profile = (extract_result.get("user_profile") or "").strip()
    summary_id = str(uuid.uuid4())
    created_at = int(time.time())

    # 确定需要更新的消息 ID
    all_message_ids = [str(msg["id"]) for msg in messages]
    update_message_ids = [
        mid for mid in all_message_ids if mid not in filtered_message_ids
    ]

    if not update_message_ids:
        logger.warning(f"会话 {thread_id} 所有消息都被过滤")
        return {
            "success": False,
            "reason": "所有消息都被过滤",
            "token_count": total_tokens,
            "message_count": len(messages),
        }

    conversation_id = messages[0].get("conversation_id") if messages else None
    if not conversation_id and ":" in thread_id:
        conversation_id = thread_id.split(":", 1)[1]

    embedding_model = os.getenv("EMBEDDING_MODEL")
    embedding_dim = int(os.getenv("dense_dimension", "1024"))

    # 构造稳定的 memory_id；summary 和非 summary 都带同一批 summary_id
    new_memories: List[Dict[str, Any]] = []

    summary = extract_result.get("summary") or {}
    summary_content = (summary.get("content") or "").strip()
    if summary_content:
        new_memories.append(
            {
                "memory_id": f"mem_{uuid.uuid4().hex}",
                "user_id": user_id,
                "memory_type": "summary",
                "content": summary_content,
                "importance": float(summary.get("importance_score", 0.5)),
                "source_summary_id": summary_id,
                "status": "active",
                "thread_id": thread_id,
                "conversation_id": conversation_id,
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
                "created_at": created_at,
                "last_access_at": created_at,
            }
        )

    type_mapping = {
        "semantic_memory": "semantic",
        "episodic_memory": "episodic",
        "procedural_memory": "procedural",
    }
    for key, memory_type in type_mapping.items():
        for item in extract_result.get(key, []) or []:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            new_memories.append(
                {
                    "memory_id": f"mem_{uuid.uuid4().hex}",
                    "user_id": user_id,
                    "memory_type": memory_type,
                    "content": content,
                    "importance": float(item.get("importance_score", 0.5)),
                    "source_summary_id": summary_id,
                    "status": "active",
                    "thread_id": thread_id,
                    "conversation_id": conversation_id,
                    "embedding_model": embedding_model,
                    "embedding_dim": embedding_dim,
                    "created_at": created_at,
                    "last_access_at": created_at,
                }
            )

    if not new_memories:
        return {
            "success": False,
            "reason": "没有可写入的记忆",
            "filtered_message_ids": filtered_message_ids,
            "token_count": total_tokens,
            "message_count": len(messages),
        }

    # 用旧记忆做 supersede 判定；不删除旧记忆，只记录新版本关系
    try:
        superseded_map, vectors = await milvus_client.find_superseded_candidates(
            new_memories, user_id
        )
    except Exception as e:
        logger.error(f"记忆冲突检测失败: {e}", exc_info=True)
        return {
            "success": False,
            "reason": f"记忆冲突检测失败: {str(e)}",
            "filtered_message_ids": filtered_message_ids,
            "token_count": total_tokens,
            "message_count": len(messages),
        }

    superseded = [
        {
            "memory_id": match["memory_id"],
            "superseded_by": new_memories[idx]["memory_id"],
        }
        for idx, match in superseded_map.items()
    ]

    # 使用小模型合并新旧用户画像
    if new_user_profile:
        old_user_profile = await pg_client.get_user_profile(user_id)
        response = await model.ainvoke(
            [
                HumanMessage(
                    content=USER_PROFILE_MERGE_PROMPT.format(
                        old_user_profile=old_user_profile,
                        new_user_profile=new_user_profile,
                    )
                )
            ]
        )
        new_user_profile = response.content.strip()

    # 1. Postgres 单事务：summary + memories + versions + sources + 消息标记 + 画像
    stored = await pg_client.store_memory_batch(
        user_id=user_id,
        thread_id=thread_id,
        conversation_id=conversation_id,
        summary_id=summary_id,
        summary_content=summary_content,
        summary_importance=float(summary.get("importance_score", 0.5)),
        token_count=total_tokens,
        message_count=len(update_message_ids),
        model_name=getattr(model, "model_name", None)
        or os.getenv("SUMMARIZATION_MODEL"),
        memories=new_memories,
        superseded=superseded,
        message_ids=update_message_ids,
        user_profile=new_user_profile or None,
    )

    if not stored:
        return {
            "success": False,
            "reason": "Postgres 记忆事务写入失败",
            "filtered_message_ids": filtered_message_ids,
            "token_count": total_tokens,
            "message_count": len(messages),
        }

    # 2. Milvus 投影：新记忆 upsert，旧记忆状态改为 superseded
    new_ids = [memory["memory_id"] for memory in new_memories]
    old_ids = [item["memory_id"] for item in superseded]
    try:
        index_ok = await milvus_client.index_memory_records(new_memories, vectors)
        status_ok = True
        if old_ids:
            status_ok = await milvus_client.set_memory_status(
                old_ids, "superseded"
            )
    except Exception as e:
        logger.error(f"Milvus 投影异常: {e}", exc_info=True)
        index_ok = False
        status_ok = False

    indexed = bool(index_ok and status_ok)
    if indexed:
        await pg_client.mark_memories_index_status(new_ids, "indexed")
        if old_ids:
            await pg_client.mark_memories_index_status(old_ids, "indexed")
    else:
        await pg_client.mark_memories_index_status(new_ids, "failed")
        if old_ids:
            await pg_client.mark_memories_index_status(old_ids, "failed")
        logger.error(
            "Postgres 已提交，但 Milvus 投影失败: summary=%s new=%s old=%s；"
            "请运行 backend/reindex_memories.py 修复索引。",
            summary_id,
            len(new_ids),
            len(old_ids),
        )

    logger.info(
        "成功压缩会话 %s，摘要 %s，新记忆 %s 条，superseded %s 条，indexed=%s",
        thread_id,
        summary_id,
        len(new_memories),
        len(superseded),
        indexed,
    )
    return {
        "success": True,
        "summary_id": summary_id,
        "user_profile": new_user_profile,
        "filtered_message_ids": filtered_message_ids,
        "token_count": total_tokens,
        "message_count": len(messages),
        "updated_message_count": len(update_message_ids),
        "memory_ids": new_ids,
        "superseded_ids": old_ids,
        "indexed": indexed,
    }


async def process_all_users_conversations(
    model: BaseChatModel,
) -> Dict[str, Any]:
    """处理所有用户的未摘要对话（后台任务，不抛出异常）"""
    try:
        pg_client = await get_postgresql_client()

        if not pg_client.pool:
            logger.error("数据库连接池未初始化，无法执行压缩任务")
            return {
                "total_conversations_processed": 0,
                "compressed_count": 0,
                "skipped_count": 0,
                "failed_count": 0,
                "error": "数据库连接池未初始化",
                "details": [],
            }

        results = {
            "total_conversations_processed": 0,
            "compressed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "details": [],
        }

        # 获取所有未摘要的会话
        async with pg_client.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT user_id, thread_id
                FROM raw_conversations
                WHERE summary_id IS NULL
                ORDER BY user_id, thread_id
            """
            )

        # 处理每个会话，捕获单个会话的异常
        for row in rows:
            user_id = row["user_id"]
            thread_id = row["thread_id"]

            try:
                # 获取未摘要的会话消息
                messages = await get_unsunmarized_conversations(user_id, thread_id)
                if messages:
                    # 压缩会话消息
                    result = await extract_and_append_memory(
                        messages, model, user_id, thread_id
                    )
                    results["total_conversations_processed"] += 1

                    if result.get("success"):
                        results["compressed_count"] += 1
                    elif "未超过阈值" in result.get("reason", ""):
                        results["skipped_count"] += 1
                    else:
                        results["failed_count"] += 1

                    results["details"].append(
                        {"user_id": user_id, "thread_id": thread_id, **result}
                    )
            except Exception as e:
                # 单个会话处理失败，记录错误但继续处理其他会话
                logger.error(
                    f"处理会话失败 user_id={user_id}, thread_id={thread_id}: {e}",
                    exc_info=True,
                )
                results["failed_count"] += 1
                results["details"].append(
                    {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "success": False,
                        "reason": f"处理异常: {str(e)}",
                        "filtered_message_ids": [],
                    }
                )

        # 输出汇总信息
        logger.info("=" * 60)
        logger.info("压缩任务完成汇总:")
        logger.info(f"  处理会话数: {results['total_conversations_processed']}")
        logger.info(f"  成功压缩数: {results['compressed_count']}")
        logger.info(f"  跳过（未超阈值）: {results['skipped_count']}")
        logger.info(f"  失败数: {results['failed_count']}")
        logger.info("=" * 60)

        return results

    except Exception as e:
        # 捕获整个任务的致命错误
        logger.error(f"压缩任务执行失败: {e}", exc_info=True)
        return {
            "total_conversations_processed": 0,
            "compressed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "error": str(e),
            "details": [],
        }


async def run_compression_task(model: BaseChatModel):
    """定时任务入口函数（兼容user_id参数，后台任务不抛出错误）"""
    logger.info("开始执行记忆存储任务...")
    start_time = datetime.now()

    try:
        pg_client = await get_postgresql_client()
        await pg_client.init_pool()

        # 兼容原参数（实际未使用user_id，保持接口一致）
        results = await process_all_users_conversations(model)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"记忆存储任务完成，耗时: {duration:.2f} 秒")

        return results
    except Exception as e:
        logger.error(f"记忆存储任务执行失败: {e}", exc_info=True)
        return {
            "total_conversations_processed": 0,
            "compressed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "error": str(e),
            "details": [],
        }


# ============ 使用示例 ============
if __name__ == "__main__":
    import sys

    sys.path.append(".")

    async def main():
        summarize_model = ChatOpenAI(
            model=os.getenv("SUMMARIZATION_MODEL", "qwen-turbo"),
            openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
            openai_api_base=os.getenv("BASE_URL"),
            temperature=0.5,  # 总结模型温度，控制总结对话的随机性，0-1之间，0越确定，1越随机
        )

        results = await run_compression_task(summarize_model)
        print(results)
        print("\n压缩结果:")
        for detail in results["details"]:
            if detail.get("success"):
                filter_count = len(detail.get("filtered_message_ids", []))
                print(
                    f"  ✅ 用户 {detail['user_id']}, 会话 {detail['thread_id'][:20]}...: "
                    f"压缩 {detail['message_count']} 条消息 (过滤 {filter_count} 条) | {detail['token_count']} tokens"
                )
            elif "未超过阈值" in detail.get("reason", ""):
                print(
                    f"  ⏭️  用户 {detail['user_id']}, 会话 {detail['thread_id'][:20]}...: "
                    f"跳过 ({detail['token_count']} tokens)"
                )

    asyncio.run(main())
