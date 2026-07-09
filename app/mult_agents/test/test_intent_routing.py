"""测试：意图路由 + 图条件路由。

覆盖以下纯函数（零 mock，零外部依赖）：
- detect_intent        — nodes.py
- route_after_intent   — graph.py
- should_continue_research — graph.py
"""

from mult_agents.nodes import detect_intent
from mult_agents.graph import route_after_intent, should_continue_research


# ============================================================
# detect_intent
# ============================================================

class TestDetectIntent:
    """测试意图关键词路由规则。"""

    # --- "年份 + 趋势" 组合 ---
    def test_year_trend_multiagent(self):
        assert detect_intent("2025年人工智能趋势") == "multiagent"

    def test_year_news_multiagent(self):
        assert detect_intent("2024年AI新闻盘点") == "multiagent"

    def test_year_generic_without_trend_keyword(self):
        """2025年 但不含趋势/新闻/调研等 → 不命中第一道规则"""
        assert detect_intent("2025年计划安排") == "direct"

    # --- force_multiagent 关键词 ---
    def test_force_keyword_diaoyan(self):
        assert detect_intent("帮我调研一下AI Agent") == "multiagent"

    def test_force_keyword_laoyuan(self):
        assert detect_intent("最新来源") == "multiagent"

    def test_force_keyword_zhengju(self):
        assert detect_intent("提供证据") == "multiagent"

    def test_force_keyword_trend(self):
        assert detect_intent("市场趋势") == "multiagent"

    # --- 普通关键词 ---
    def test_keyword_analyze(self):
        assert detect_intent("分析一下") == "multiagent"

    def test_keyword_compare(self):
        assert detect_intent("方案对比") == "multiagent"

    def test_keyword_knowledge_base(self):
        assert detect_intent("检索知识库") == "multiagent"

    def test_keyword_report(self):
        assert detect_intent("写一份报告") == "multiagent"

    def test_keyword_code(self):
        assert detect_intent("写代码实现") == "multiagent"

    # --- 无关键词 → direct ---
    def test_greeting_direct(self):
        assert detect_intent("你好") == "direct"

    def test_identity_direct(self):
        assert detect_intent("你是谁") == "direct"

    def test_simple_question_direct(self):
        assert detect_intent("今天星期几") == "direct"

    def test_empty_string_direct(self):
        assert detect_intent("") == "direct"

    def test_whitespace_only_direct(self):
        assert detect_intent("   ") == "direct"

    def test_single_word_unknown_direct(self):
        """不命中任何关键词 → direct"""
        assert detect_intent("吃饭") == "direct"


# ============================================================
# route_after_intent
# ============================================================

class TestRouteAfterIntent:
    """测试意图路由分发。"""

    def test_direct_answer_route(self):
        assert route_after_intent({"intent": "direct"}) == "direct_answer"

    def test_multiagent_route(self):
        assert route_after_intent({"intent": "multiagent"}) == "plan"

    def test_unknown_intent_falls_to_plan(self):
        """未知 intent 值 → 走 plan（默认）"""
        assert route_after_intent({"intent": "unknown"}) == "plan"

    def test_missing_intent_falls_to_plan(self):
        assert route_after_intent({}) == "plan"


# ============================================================
# should_continue_research
# ============================================================

class TestShouldContinueResearch:
    """测试研究补搜条件路由。"""

    def test_reached_max_iterations_go_write(self):
        state = {"iteration": 3, "max_iterations": 3, "needs_more_research": False}
        assert should_continue_research(state) == "write"

    def test_under_max_and_needs_more_go_reflect(self):
        state = {"iteration": 1, "max_iterations": 3, "needs_more_research": True}
        assert should_continue_research(state) == "reflect"

    def test_under_max_no_gaps_go_write(self):
        state = {"iteration": 1, "max_iterations": 3, "needs_more_research": False}
        assert should_continue_research(state) == "write"

    def test_exactly_at_max_with_gaps_go_write(self):
        """即便 needs_more 为 True，已达 max 也应终止"""
        state = {"iteration": 3, "max_iterations": 3, "needs_more_research": True}
        assert should_continue_research(state) == "write"

    def test_zero_iterations_defaults(self):
        state = {"iteration": 0, "max_iterations": 2}
        assert should_continue_research(state) == "write"

    def test_needs_more_research_false_by_default(self):
        state = {"iteration": 0, "max_iterations": 2}
        assert should_continue_research(state) == "write"
