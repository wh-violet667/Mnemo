"""Agentic-lite planner / critic 真实模型冒烟脚本。

用 SUMMARIZATION_MODEL 对 3 条样例问题调用 planner / critic，
验证结构化 JSON 输出可解析、字段符合契约，以及坏 JSON 的修复路径可用。

运行（项目根目录）：
    .venv\\Scripts\\python.exe scripts\\smoke_agentic.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv

for candidate in (ROOT / ".env", ROOT / "backend" / ".env"):
    if candidate.exists():
        load_dotenv(candidate)

from retrieval import (  # noqa: E402
    _default_critic,
    _default_planner,
    _parse_json_object,
)


SAMPLE_EVIDENCE = [
    {
        "file_name": "travel_policy.pdf",
        "text": "国内航班一律经济舱。住宿上限 500 CNY/晚。",
    },
    {
        "file_name": "travel_policy_emea.pdf",
        "text": "EMEA 地区出差住宿上限 800 EUR/晚，适用于正式员工。",
    },
]


async def smoke_planner() -> bool:
    planner = _default_planner()
    ok = True

    cases = [
        "退款期限是多久？",
        "合同工出差 EMEA 的住宿上限和 MFA 要求是什么？",
        "正式员工与合同工的安全要求有哪些差异？",
    ]
    for question in cases:
        queries = await planner(query=question)
        valid = (
            isinstance(queries, list)
            and 1 <= len(queries) <= 3
            and all(isinstance(q, str) and q.strip() for q in queries)
        )
        ok = ok and valid
        print(f"[planner] {question}")
        print(f"  -> {queries}  {'OK' if valid else 'FAIL'}")
    return ok


async def smoke_critic() -> bool:
    critic = _default_critic()

    sufficient_result = await critic(
        query="EMEA 出差住宿上限是多少？",
        documents=[dict(item) for item in SAMPLE_EVIDENCE],
    )
    fields_ok = (
        isinstance(sufficient_result, dict)
        and isinstance(sufficient_result.get("sufficient"), bool)
        and isinstance(sufficient_result.get("missing"), list)
        and isinstance(sufficient_result.get("rewrite"), str)
        and isinstance(sufficient_result.get("reason"), str)
    )
    print(f"[critic] 证据充分场景 -> {sufficient_result}  {'OK' if fields_ok else 'FAIL'}")

    insufficient_result = await critic(
        query="合同工出差 EMEA 的住宿上限和 MFA 要求是什么？",
        documents=[dict(item) for item in SAMPLE_EVIDENCE[:1]],
    )
    contract_ok = isinstance(insufficient_result, dict) and isinstance(
        insufficient_result.get("sufficient"), bool
    )
    print(
        f"[critic] 证据不足场景 -> {insufficient_result}  {'OK' if contract_ok else 'FAIL'}"
    )
    if isinstance(insufficient_result, dict) and not insufficient_result.get(
        "sufficient"
    ):
        if not insufficient_result.get("missing"):
            print("[critic] FAIL: sufficient=false 但 missing 为空")
            contract_ok = False

    return fields_ok and contract_ok


def smoke_bad_json_repair() -> bool:
    broken = '{"sufficient": false, "missing": ["生效时间"], "rewrite": "查询", "reason": "缺时间'
    repaired = _parse_json_object(broken)
    ok = isinstance(repaired, dict) and repaired.get("sufficient") is False
    print(f"[json_repair] 坏 JSON 修复 -> {repaired}  {'OK' if ok else 'FAIL'}")

    garbage = _parse_json_object("这不是 JSON")
    ok = ok and garbage is None
    print(f"[json_repair] 非 JSON 返回 None -> {'OK' if garbage is None else 'FAIL'}")
    return ok


async def main() -> int:
    results = [
        await smoke_planner(),
        await smoke_critic(),
        smoke_bad_json_repair(),
    ]
    print("\n总体:", "全部通过" if all(results) else "存在失败项")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
