"""测试：_derive_search_plan —— 搜索计划生成。"""

import json

from mult_agents.nodes.planning import _derive_search_plan


class TestDeriveSearchPlan:
    def test_generates_queries_from_outline(self):
        outline = [
            {"id": "sec_1", "search_queries": ["AI Agent 框架", "LangGraph 使用"]}
        ]
        result = _derive_search_plan(outline, [], [], "AI Agent")
        assert len(result) >= 2
        # Should contain queries from outline
        queries = [item["query"] for item in result]
        assert any("AI Agent" in q for q in queries)

    def test_fallback_when_no_sections(self):
        result = _derive_search_plan([], [], [], "测试查询")
        assert len(result) >= 1
        assert result[0]["query"] == "测试查询"
        assert result[0]["source_preference"] == "hybrid"

    def test_deduplicates_duplicate_queries(self):
        outline = [
            {"id": "sec_1", "search_queries": ["AI Agent", "AI Agent"]},
        ]
        result = _derive_search_plan(outline, [], [], "AI Agent")
        queries = [item["query"] for item in result]
        # "AI Agent" should appear at most once
        assert queries.count("AI Agent") <= 1

    def test_respects_max_six_queries(self):
        # Generate many sections to push past the cap
        outline = [
            {"id": f"sec_{i}", "search_queries": [f"query_{i}"]}
            for i in range(10)
        ]
        result = _derive_search_plan(outline, [], [], "base")
        assert len(result) <= 6
