"""检索路由与查询改写纯函数。

只放不依赖 DB / LLM / Web 框架的规则逻辑，方便单元测试：
- needs_deep：auto 模式下判断是否需要深度检索；
- build_initial_queries：deep 第一轮查询生成（规则兜底，planner 失败时用）；
- build_refined_query：根据 critic 的 missing 维度改写第二轮查询；
- merge_parent_ids：多轮 parent_id 保序去重合并；
- should_stop / stop_reason：统一停止条件。
"""

from __future__ import annotations

import re
from typing import Literal, Optional

RetrievalMode = Literal["auto", "fast", "deep"]

# 命中任一提示词即判 deep（比较/版本/适用域/多对象类问题的表层特征）
DEEP_HINTS = (
    "比较", "区别", "差异", "分别", "各自", "同时", "以及", "还是",
    "版本", "适用", "生效", "过期", "之前", "之后", "冲突",
    "关系",
)

# 粗粒度实体识别：书名号《...》、三个及以上字符的大写开头拉丁词
# （EMEA/VPN/MFA/OVA 等；TV/OP/ED 等两字符媒介缩写不算内容实体）
_ENTITY_PATTERN = re.compile(r"《[^》]+》|[A-Z][A-Za-z0-9]{2,}")

# 子查询拆分分隔符，按优先级排列（先长后短，避免"以及"被"与"截断）
_SPLIT_PATTERN = re.compile(r"以及|还是|区别|差异|、|和|与")

_QUERY_MIN_LEN = 2
_MAX_QUERIES = 3
_MAX_MISSING_DIMS = 3


def needs_deep(query: str) -> bool:
    """规则判断是否需要深度检索。

    命中任一条件即判 deep：
    - 包含 DEEP_HINTS 提示词；
    - 查询长度 >= 40 字；
    - 包含两个及以上明显实体。
    """
    q = (query or "").strip()
    if not q:
        return False
    if any(hint in q for hint in DEEP_HINTS):
        return True
    if len(q) >= 40:
        return True
    if len(_ENTITY_PATTERN.findall(q)) >= 2:
        return True
    return False


def _clean_part(part: str) -> str:
    return part.strip(" ？?，,。.:：;；")


def build_initial_queries(query: str) -> list[str]:
    """生成第一轮查询，最多 3 条。

    - 总是包含原问题；
    - 含"和/与/、/以及/还是/区别/差异"时，按分隔符拆出至多两个子查询；
    - 去重、去空、截断到 3 条；不做同义词改写。
    """
    q = (query or "").strip()
    if not q:
        return []

    queries = [q]
    if _SPLIT_PATTERN.search(q):
        parts = [_clean_part(p) for p in _SPLIT_PATTERN.split(q)]
        for part in parts:
            if len(part) >= _QUERY_MIN_LEN and part != q and part not in queries:
                queries.append(part)
            if len(queries) >= _MAX_QUERIES:
                break
    return queries[:_MAX_QUERIES]


def build_refined_query(query: str, missing: list[str]) -> str:
    """根据 critic 的 missing 维度改写第二轮查询。

    missing 为空或原问题为空时返回空字符串，由调用方触发停止。
    """
    dims = [str(m).strip() for m in (missing or []) if str(m).strip()]
    base = (query or "").strip()
    if not dims or not base:
        return ""
    return f"{base}（{'、'.join(dims[:_MAX_MISSING_DIMS])}）"


def merge_parent_ids(rounds: list[list[str]]) -> list[str]:
    """按轮次顺序合并 parent_id，保序去重。"""
    seen = set()
    merged: list[str] = []
    for ids in rounds or []:
        for pid in ids or []:
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            merged.append(pid)
    return merged


def stop_reason(
    *,
    iteration: int,
    max_iterations: int,
    new_parent_count: int,
    critic: Optional[dict],
    elapsed_ms: int,
    budget_ms: int,
) -> Optional[str]:
    """返回停止原因；不应停止时返回 None。"""
    if iteration >= max_iterations:
        return "max_iterations"
    if new_parent_count <= 0:
        return "no_new_parents"
    if critic and critic.get("sufficient") is True:
        return "critic_sufficient"
    if elapsed_ms >= budget_ms:
        return "budget_exhausted"
    return None


def should_stop(
    *,
    iteration: int,
    max_iterations: int,
    new_parent_count: int,
    critic: Optional[dict],
    elapsed_ms: int,
    budget_ms: int,
) -> bool:
    """统一停止条件：轮次上限 / 无新增证据 / critic 充分 / 预算耗尽。"""
    return (
        stop_reason(
            iteration=iteration,
            max_iterations=max_iterations,
            new_parent_count=new_parent_count,
            critic=critic,
            elapsed_ms=elapsed_ms,
            budget_ms=budget_ms,
        )
        is not None
    )
