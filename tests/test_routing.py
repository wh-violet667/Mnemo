from routing import (
    build_initial_queries,
    build_refined_query,
    merge_parent_ids,
    needs_deep,
    should_stop,
    stop_reason,
)


class TestNeedsDeep:
    def test_single_hop_question_is_fast(self):
        assert needs_deep("退款期限是多久？") is False
        assert needs_deep("申请退款需要提供什么信息？") is False

    def test_comparison_hint_triggers_deep(self):
        assert needs_deep("现行退款政策是 30 天还是 14 天？") is True
        assert needs_deep("正式员工与合同工的安全要求有哪些差异？") is True

    def test_version_hint_triggers_deep(self):
        assert needs_deep("2024 年之前下的订单适用哪个版本的退款政策") is True

    def test_multiple_entities_trigger_deep(self):
        assert needs_deep("VPN 和 MFA 的配置要求是什么？") is True

    def test_long_query_triggers_deep(self):
        query = "请详细说明差旅政策中关于住宿标准、交通工具、餐饮补贴等各项费用的报销额度和审批流程"
        assert len(query) >= 40
        assert needs_deep(query) is True

    def test_two_letter_media_abbreviations_do_not_count_as_entities(self):
        # TV/OP/ED 等两字符媒介缩写不是内容实体（violet 语料误路由回归用例）
        assert needs_deep("TV 动画《紫罗兰永恒花园》由哪家动画公司制作？") is False
        assert needs_deep("TV 动画的片头曲（OP）叫什么？由谁演唱？") is False

    def test_two_question_marks_do_not_trigger_deep(self):
        # 双问号是同文档复合问句的弱信号，已移除该规则
        assert needs_deep("退款期限是多久？退款处理又要多久？") is False

    def test_enumeration_marker_does_not_trigger_deep(self):
        assert needs_deep("生还的基尔伯特失去了身体的哪些部分？") is False

    def test_empty_query_is_fast(self):
        assert needs_deep("") is False
        assert needs_deep("   ") is False


class TestBuildInitialQueries:
    def test_always_includes_original_query(self):
        queries = build_initial_queries("退款期限是多久？")
        assert queries[0] == "退款期限是多久？"

    def test_no_separator_returns_only_original(self):
        assert build_initial_queries("退款期限是多久？") == ["退款期限是多久？"]

    def test_splits_on_separator(self):
        queries = build_initial_queries("合同工和正式员工的住宿上限")
        assert queries[0] == "合同工和正式员工的住宿上限"
        assert "合同工" in queries
        assert len(queries) <= 3

    def test_splits_on_yiji(self):
        queries = build_initial_queries("住宿上限以及 MFA 要求")
        assert "住宿上限" in queries
        assert "MFA 要求" in queries

    def test_dedupes_and_caps_at_three(self):
        queries = build_initial_queries("A 和 B 与 C 以及 D")
        assert len(queries) <= 3
        assert len(queries) == len(set(queries))

    def test_empty_query_returns_empty(self):
        assert build_initial_queries("") == []

    def test_drops_tiny_fragments(self):
        queries = build_initial_queries("退款政策和、流程")
        assert all(len(q) >= 2 for q in queries)


class TestBuildRefinedQuery:
    def test_appends_missing_dimensions(self):
        refined = build_refined_query("住宿上限是多少", ["角色适用域", "生效时间"])
        assert refined.startswith("住宿上限是多少")
        assert "角色适用域" in refined
        assert "生效时间" in refined

    def test_empty_missing_returns_empty_string(self):
        assert build_refined_query("住宿上限是多少", []) == ""
        assert build_refined_query("住宿上限是多少", None) == ""

    def test_empty_query_returns_empty_string(self):
        assert build_refined_query("", ["角色适用域"]) == ""

    def test_caps_missing_dimensions(self):
        refined = build_refined_query("q", ["a", "b", "c", "d"])
        assert "d" not in refined


class TestMergeParentIds:
    def test_merges_in_order_and_dedupes(self):
        merged = merge_parent_ids([["a", "b", "c"], ["b", "d"], ["e", "a"]])
        assert merged == ["a", "b", "c", "d", "e"]

    def test_handles_empty_input(self):
        assert merge_parent_ids([]) == []
        assert merge_parent_ids([[], None]) == []

    def test_skips_none_ids(self):
        assert merge_parent_ids([["a", None, "b"]]) == ["a", "b"]


class TestShouldStop:
    base = dict(
        iteration=1,
        max_iterations=2,
        new_parent_count=3,
        critic=None,
        elapsed_ms=1000,
        budget_ms=12000,
    )

    def test_continues_when_evidence_flowing(self):
        assert should_stop(**self.base) is False
        assert stop_reason(**self.base) is None

    def test_stops_at_max_iterations(self):
        kwargs = {**self.base, "iteration": 2}
        assert should_stop(**kwargs) is True
        assert stop_reason(**kwargs) == "max_iterations"

    def test_stops_when_no_new_parents(self):
        kwargs = {**self.base, "new_parent_count": 0}
        assert should_stop(**kwargs) is True
        assert stop_reason(**kwargs) == "no_new_parents"

    def test_stops_when_critic_sufficient(self):
        kwargs = {**self.base, "critic": {"sufficient": True}}
        assert should_stop(**kwargs) is True
        assert stop_reason(**kwargs) == "critic_sufficient"

    def test_stops_when_budget_exhausted(self):
        kwargs = {**self.base, "elapsed_ms": 12000}
        assert should_stop(**kwargs) is True
        assert stop_reason(**kwargs) == "budget_exhausted"

    def test_critic_false_does_not_stop(self):
        kwargs = {**self.base, "critic": {"sufficient": False, "missing": ["时间"]}}
        assert should_stop(**kwargs) is False
