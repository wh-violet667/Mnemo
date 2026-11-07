import pytest

from retrieval import run_retrieval


class FakeRetriever:
    """query -> parent_id 列表的映射；记录调用次数与查询内容。"""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    async def __call__(self, *, query):
        self.calls.append(query)
        return list(self.mapping.get(query, []))


def fake_load_documents(documents_by_id):
    async def _load(*, parent_ids):
        return [documents_by_id[pid] for pid in parent_ids if pid in documents_by_id]

    return _load


def make_documents(ids):
    return {
        pid: {"parent_id": pid, "text": f"正文 {pid}", "file_name": f"{pid}.pdf"}
        for pid in ids
    }


def identity_rerank():
    calls = []

    async def _rerank(*, query, documents):
        calls.append(query)
        return [dict(doc) for doc in documents]

    return _rerank, calls


def critic_queue(results):
    calls = []
    queue = list(results)

    async def _critic(*, query, documents):
        calls.append((query, documents))
        return queue.pop(0) if queue else {"sufficient": True}

    return _critic, calls


def planner_returning(queries):
    async def _plan(*, query):
        return list(queries)

    return _plan


@pytest.mark.asyncio
async def test_fast_calls_retriever_exactly_once():
    docs = make_documents(["p1", "p2"])
    retriever = FakeRetriever({"退款期限是多久？": ["p1", "p2"]})
    rerank, rerank_calls = identity_rerank()

    result = await run_retrieval(
        "退款期限是多久？",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="fast",
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        rerank=rerank,
    )

    assert len(retriever.calls) == 1
    assert result.trace.mode == "fast"
    assert result.trace.rounds == 1
    assert result.trace.critic == "skipped"
    assert result.sufficient is None
    assert len(rerank_calls) == 1
    assert result.snippets
    assert [s["index"] for s in result.sources] == [1, 2]


@pytest.mark.asyncio
async def test_fast_empty_result_returns_empty_snippets():
    retriever = FakeRetriever({})
    rerank, rerank_calls = identity_rerank()

    result = await run_retrieval(
        "不存在的条款",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="fast",
        retriever=retriever,
        load_documents=fake_load_documents({}),
        rerank=rerank,
    )

    assert result.snippets == ""
    assert result.sources == []
    assert result.documents == []
    assert rerank_calls == []  # 无证据不做 rerank


@pytest.mark.asyncio
async def test_deep_two_rounds_when_critic_not_sufficient():
    docs = make_documents(["p1", "p2", "p3"])
    retriever = FakeRetriever(
        {
            "住宿和交通标准": ["p1"],
            "住宿和交通标准（地区适用域）": ["p2", "p3"],
        }
    )
    critic, critic_calls = critic_queue(
        [
            {"sufficient": False, "missing": ["地区适用域"]},
            {"sufficient": False, "missing": ["地区适用域"]},
        ]
    )
    rerank, _ = identity_rerank()

    result = await run_retrieval(
        "住宿和交通标准",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="deep",
        max_rounds=2,
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        critic=critic,
        planner=planner_returning(["住宿和交通标准"]),
        rerank=rerank,
    )

    assert result.trace.rounds == 2
    assert len(critic_calls) == 2
    assert result.sufficient is False
    assert set(result.trace.queries) == {"住宿和交通标准", "住宿和交通标准（地区适用域）"}
    assert {doc["parent_id"] for doc in result.documents} == {"p1", "p2", "p3"}


@pytest.mark.asyncio
async def test_deep_stops_at_max_rounds():
    docs = make_documents(["p1", "p2", "p3"])
    retriever = FakeRetriever(
        {
            "q": ["p1"],
            "q（m）": ["p2"],
        }
    )
    critic, _ = critic_queue(
        [
            {"sufficient": False, "missing": ["m"]},
            {"sufficient": False, "missing": ["m"]},
        ]
    )
    rerank, _ = identity_rerank()

    result = await run_retrieval(
        "q",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="deep",
        max_rounds=2,
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        critic=critic,
        planner=planner_returning(["q"]),
        rerank=rerank,
    )

    assert result.trace.rounds == 2
    assert result.sufficient is False


@pytest.mark.asyncio
async def test_deep_stops_when_critic_sufficient():
    docs = make_documents(["p1", "p2"])
    retriever = FakeRetriever({"q": ["p1"], "q（m）": ["p2"]})
    critic, critic_calls = critic_queue([{"sufficient": True}])
    rerank, _ = identity_rerank()

    result = await run_retrieval(
        "q",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="deep",
        max_rounds=3,
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        critic=critic,
        planner=planner_returning(["q"]),
        rerank=rerank,
    )

    assert result.trace.rounds == 1
    assert len(critic_calls) == 1
    assert result.sufficient is True
    assert result.trace.critic == "sufficient"


@pytest.mark.asyncio
async def test_deep_stops_when_no_new_parents():
    docs = make_documents(["p1"])
    retriever = FakeRetriever({"q": ["p1"], "q（m）": ["p1"]})
    critic, _ = critic_queue([{"sufficient": False, "missing": ["m"]}])
    rerank, _ = identity_rerank()

    result = await run_retrieval(
        "q",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="deep",
        max_rounds=3,
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        critic=critic,
        planner=planner_returning(["q"]),
        rerank=rerank,
    )

    assert result.trace.rounds == 2
    assert {doc["parent_id"] for doc in result.documents} == {"p1"}


@pytest.mark.asyncio
async def test_auto_escalates_when_fast_returns_nothing():
    docs = make_documents(["p1"])
    # fast 用原查询检索为空；升级 deep 后 planner 换角度查询命中证据
    retriever = FakeRetriever({"冷门问题": [], "冷门问题换角度": ["p1"]})
    critic, _ = critic_queue([{"sufficient": True}])
    rerank, _ = identity_rerank()

    result = await run_retrieval(
        "冷门问题",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="auto",
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        critic=critic,
        planner=planner_returning(["冷门问题换角度"]),
        rerank=rerank,
    )

    assert result.trace.escalated is True
    assert result.trace.mode == "deep"
    assert {doc["parent_id"] for doc in result.documents} == {"p1"}


@pytest.mark.asyncio
async def test_auto_does_not_escalate_when_fast_has_results():
    docs = make_documents(["p1"])
    retriever = FakeRetriever({"退款期限是多久？": ["p1"]})
    rerank, _ = identity_rerank()

    result = await run_retrieval(
        "退款期限是多久？",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="auto",
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        rerank=rerank,
    )

    assert result.trace.escalated is False
    assert result.trace.mode == "fast"
    assert result.trace.rounds == 1


@pytest.mark.asyncio
async def test_auto_routes_comparison_query_to_deep():
    docs = make_documents(["p1"])
    retriever = FakeRetriever({"A 和 B 的区别": ["p1"]})
    critic, _ = critic_queue([{"sufficient": True}])
    rerank, _ = identity_rerank()

    result = await run_retrieval(
        "A 和 B 的区别",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="auto",
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        critic=critic,
        planner=planner_returning(["A 和 B 的区别"]),
        rerank=rerank,
    )

    assert result.trace.mode == "deep"
    assert result.trace.escalated is False


@pytest.mark.asyncio
async def test_source_numbering_starts_from_start_index_and_stays_contiguous():
    docs = make_documents(["p1", "p2", "p3"])
    retriever = FakeRetriever({"q": ["p1", "p2", "p3"]})
    rerank, _ = identity_rerank()

    result = await run_retrieval(
        "q",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="fast",
        start_index=4,
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        rerank=rerank,
    )

    assert [s["index"] for s in result.sources] == [4, 5, 6]
    assert "[4]" in result.snippets
    assert "[6]" in result.snippets


@pytest.mark.asyncio
async def test_disabled_agentic_degrades_deep_to_fast():
    docs = make_documents(["p1"])
    retriever = FakeRetriever({"q": ["p1"]})
    rerank, _ = identity_rerank()

    result = await run_retrieval(
        "q",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="deep",
        enabled=False,
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        rerank=rerank,
    )

    assert result.trace.mode == "fast"
    assert result.trace.degraded is True
    assert result.trace.rounds == 1


@pytest.mark.asyncio
async def test_critic_failure_stops_but_returns_evidence():
    docs = make_documents(["p1"])
    retriever = FakeRetriever({"q": ["p1"]})

    async def failing_critic(*, query, documents):
        return None

    rerank, _ = identity_rerank()

    result = await run_retrieval(
        "q",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="deep",
        max_rounds=2,
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        critic=failing_critic,
        planner=planner_returning(["q"]),
        rerank=rerank,
    )

    assert result.trace.critic == "failed"
    assert result.sufficient is None
    assert result.snippets  # 仍返回已有证据


@pytest.mark.asyncio
async def test_deep_always_runs_first_round_even_when_planner_eats_budget():
    """planner 延迟耗尽预算时，第 1 轮检索仍必须执行（m001 回归用例）。"""
    docs = make_documents(["p1"])
    retriever = FakeRetriever({"q": ["p1"]})

    async def slow_planner(*, query):
        import asyncio as _asyncio

        await _asyncio.sleep(0.2)
        return ["q"]

    critic, _ = critic_queue([{"sufficient": True}])
    rerank, _ = identity_rerank()

    result = await run_retrieval(
        "q",
        user_id=1,
        knowledge_base_id="kb",
        top_k=5,
        mode="deep",
        max_rounds=2,
        budget_ms=100,  # planner 0.2s > 预算 0.1s
        retriever=retriever,
        load_documents=fake_load_documents(docs),
        critic=critic,
        planner=slow_planner,
        rerank=rerank,
    )

    assert result.trace.rounds >= 1
    assert {doc["parent_id"] for doc in result.documents} == {"p1"}

