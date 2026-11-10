"""LLM judges for the violet answer evaluation.

Judge A checks factual correctness against the reference answer.
Judge B checks claim-level faithfulness and citation support against the
retrieved context.  The two prompts are intentionally separate so that
"matches the reference" and "supported by the retrieved context" cannot
bias each other.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from evals.answer_metrics import extract_citation_marks


JUDGE_A_SYSTEM = """\
你是严格的 RAG 答案评分器。你只能使用用户问题、题型、参考答案、required_facts 和待评答案，不得使用外部知识。

输出必须是单个 JSON 对象，不要 Markdown，不要代码块，不要解释。

普通可回答问题按 0/1/2 评分：
- 2：关键事实全部正确，无矛盾，无编造的关键事实。
- 1：主结论正确，但漏了次要关键事实，或数字/单位不够精确，且无矛盾。
- 0：关键事实错误、与参考答案矛盾、答非所问，或该回答时拒答。

如果题型是 unanswerable，按以下标准评分：
- 2：明确说明知识库未提供/无法回答，且没有编造具体事实。
- 1：有拒答倾向，但仍给出了不确定的具体信息。
- 0：编造具体事实，或给出确定答案。

评分原则：
1. 不要因为答案比参考答案长、表达不同、或包含额外正确信息而扣分；额外细节不构成 wrong_facts。
2. 只判断 required_facts / 参考答案中的关键事实是否正确，不要做文风评分，也不要判断额外细节的真实性。
3. 如果答案明确拒答，但题目其实可回答，score 必须为 0，refused=true。
4. 只有与参考答案或 required_facts 明确矛盾的事实，才写入 wrong_facts；参考答案没有提到的额外细节不要写入 wrong_facts。
5. 如果 required_facts 全部正确，即使答案更长、补充了额外细节，也应给 2 分。
6. 数值题不要求只输出数字；只要 required_facts 中的数值和对象出现在答案中且无矛盾，就给 2 分。
7. 参考答案中的时间、因果、先后、适用域关系也是关键事实；答案否认、反转或用“无直接先后/并行/未说明”等表述弱化参考答案明确给出的关系时，必须在 wrong_facts 中列出并给 0 分。
8. 如果 missing_facts 和 wrong_facts 都为空，且答案不是误拒，score 应为 2。

输出格式：
{
  "score": 0,
  "refused": false,
  "missing_facts": [],
  "wrong_facts": [],
  "reason": ""
}
"""


JUDGE_B_SYSTEM = """\
你是严格的 RAG 忠实度和引用审核器。你只能依据提供的检索上下文判断，不得使用外部知识。

答案中的 [n] 是引用编号；[C1]...[C5] 是本次检索到的全部证据，两者不要混淆。

任务：
1. 将答案拆成原子 claim。每个 claim 判断：
   - verdict: supported / partial / unsupported
   - needs_citation: 这个 claim 是否需要证据引用
   - cited_marks: 这个 claim 附近实际出现的引用编号列表
   - evidence_source: 支持它的上下文编号，例如 C1
   - quote: 支持它的原文片段；没有证据时为空字符串
2. 对答案中出现的每一个 [n] 输出一条 citation 判断：
   - valid: 这个编号是否存在于引用来源列表
   - supports_claim: 被引来源是否真的支持它所标注的 claim
   - claim_index: 对应的 claim 下标，从 0 开始
   - quote: 被引来源中支持该 claim 的原文片段
3. 每个支持判断都必须给出 quote，且 quote 必须能在对应来源中找到。
4. 多跳综合允许 partial，但每个前提应分别有引用。
5. 拒答、说明性文字、纯粹的结构性句子可以 needs_citation=false。
6. 不要因为答案简洁而扣分；只判断证据关系。

输出必须是单个 JSON 对象，不要 Markdown，不要代码块，不要解释。

输出格式：
{
  "claims": [
    {
      "claim": "",
      "verdict": "supported",
      "needs_citation": true,
      "cited_marks": [1],
      "evidence_source": "C1",
      "quote": ""
    }
  ],
  "citations": [
    {
      "mark": 1,
      "valid": true,
      "supports_claim": true,
      "claim_index": 0,
      "quote": ""
    }
  ]
}
"""


def build_judge(
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
) -> tuple[ChatOpenAI, ChatOpenAI]:
    """Return a JSON-mode judge and a plain fallback judge."""

    common = {
        "model": model,
        "openai_api_key": api_key,
        "openai_api_base": base_url,
        "temperature": temperature,
        "timeout": 60.0,
        "max_retries": 1,
    }
    primary = ChatOpenAI(
        **common,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    fallback = ChatOpenAI(**common)
    return primary, fallback


def parse_json_object(raw: str) -> Dict[str, Any]:
    """Parse a JSON object from a model response."""

    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("响应中没有 JSON 对象")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层不是对象")
    return data


async def _invoke_json(
    primary: ChatOpenAI,
    fallback: ChatOpenAI,
    messages: Sequence[Any],
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for model in (primary, fallback):
        try:
            response = await model.ainvoke(list(messages))
            return parse_json_object(str(response.content))
        except Exception as exc:  # pragma: no cover - network/model dependent
            last_error = exc
    raise RuntimeError(f"judge 输出无法解析为 JSON: {last_error}")


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_judge_a(data: Dict[str, Any]) -> Dict[str, Any]:
    score = max(0, min(2, _as_int(data.get("score"), 0)))
    refused = bool(data.get("refused", False))
    missing_facts = [
        str(item).strip()
        for item in (data.get("missing_facts") or [])
        if str(item).strip()
    ]
    wrong_facts = [
        str(item).strip()
        for item in (data.get("wrong_facts") or [])
        if str(item).strip()
    ]
    # 一致性保护：judge 自己都没有列出缺失/错误事实时，不应给 0 分。
    if score == 0 and not refused and not missing_facts and not wrong_facts:
        score = 2
    return {
        "score": score,
        "refused": refused,
        "missing_facts": missing_facts,
        "wrong_facts": wrong_facts,
        "reason": str(data.get("reason") or "").strip(),
    }


def normalize_judge_b(data: Dict[str, Any]) -> Dict[str, Any]:
    claims: List[Dict[str, Any]] = []
    for claim in data.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("claim") or "").strip()
        if not text:
            continue
        verdict = str(claim.get("verdict") or "unsupported").lower()
        if verdict not in {"supported", "partial", "unsupported"}:
            verdict = "unsupported"
        cited_marks = []
        for mark in claim.get("cited_marks") or []:
            parsed = _as_int(mark)
            if parsed >= 0:
                cited_marks.append(parsed)
        claims.append(
            {
                "claim": text,
                "verdict": verdict,
                "needs_citation": bool(claim.get("needs_citation", True)),
                "cited_marks": cited_marks,
                "evidence_source": str(claim.get("evidence_source") or "").strip(),
                "quote": str(claim.get("quote") or "").strip(),
            }
        )

    citations: List[Dict[str, Any]] = []
    for citation in data.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        mark = _as_int(citation.get("mark"))
        if mark < 0:
            continue
        supports = citation.get("supports_claim")
        citations.append(
            {
                "mark": mark,
                "valid": bool(citation.get("valid", False)),
                "supports_claim": bool(supports) if supports is not None else None,
                "claim_index": _as_int(citation.get("claim_index"), -1),
                "quote": str(citation.get("quote") or "").strip(),
            }
        )

    return {"claims": claims, "citations": citations}


async def judge_correctness(
    primary: ChatOpenAI,
    fallback: ChatOpenAI,
    row: Dict[str, Any],
    answer: str,
) -> Dict[str, Any]:
    required_facts = row.get("required_facts") or []
    user_prompt = f"""\
问题：
{row.get("query") or ""}

题型：
{row.get("query_type") or "unknown"}

参考答案：
{row.get("reference_answer") or ""}

required_facts：
{json.dumps(required_facts, ensure_ascii=False)}

待评答案：
{answer}
"""
    data = await _invoke_json(
        primary,
        fallback,
        [
            SystemMessage(content=JUDGE_A_SYSTEM),
            HumanMessage(content=user_prompt),
        ],
    )
    return normalize_judge_a(data)


def _format_source(source: Dict[str, Any], label: str) -> str:
    text = str(source.get("text") or source.get("snippet") or "").strip()
    if len(text) > 3000:
        text = text[:3000] + "\n...[截断]"
    return (
        f"[{label}] file={source.get('file_name') or '未知文件'} "
        f"parent_id={source.get('parent_id') or '-'}\n{text}"
    )


async def judge_grounding(
    primary: ChatOpenAI,
    fallback: ChatOpenAI,
    row: Dict[str, Any],
    answer: str,
    cited_sources: Sequence[Dict[str, Any]],
    retrieved_contexts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    marks = extract_citation_marks(answer)
    cited_text = (
        "\n\n".join(
            _format_source(source, str(source.get("index")))
            for source in cited_sources
        )
        or "（无引用来源）"
    )
    context_text = (
        "\n\n".join(
            _format_source(
                source, f"C{index}"
            )
            for index, source in enumerate(retrieved_contexts, start=1)
        )
        or "（没有检索到上下文）"
    )

    user_prompt = f"""\
问题：
{row.get("query") or ""}

待评答案：
{answer}

答案中出现的引用编号：
{json.dumps(marks, ensure_ascii=False)}

引用来源（[n]）：
{cited_text}

本次检索到的全部上下文（C1...）：
{context_text}

请严格按系统提示输出 JSON。每个引用编号都必须有一条 citation 判断。
"""
    data = await _invoke_json(
        primary,
        fallback,
        [
            SystemMessage(content=JUDGE_B_SYSTEM),
            HumanMessage(content=user_prompt),
        ],
    )
    return normalize_judge_b(data)
