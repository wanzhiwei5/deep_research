"""测试：Agent 节点（mock LLM）。

使用 unittest.mock 拦截 agent.invoke()，不调用真实 LLM。

覆盖全部 9 个 Agent 节点：
- intent_node / direct_answer_node / plan_node / web_search_node
- local_rag_node / deep_dive_node / analyze_node / reflect_node / write_node
"""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from mult_agents.nodes import (
    intent_node,
    direct_answer_node,
    plan_node,
    web_search_node,
    local_rag_node,
    deep_dive_node,
    analyze_node,
    reflect_node,
    write_node,
)
from mult_agents.state import create_initial_state


# ============================================================
# 工具函数
# ============================================================

def _make_llm_result(content: str):
    """构造 mock agent.invoke() 返回值。"""
    mock_result = MagicMock()
    mock_result.__getitem__.side_effect = lambda key: {
        "messages": [AIMessage(content=content)]
    }[key]
    return mock_result


# ============================================================
# intent_node
# ============================================================

class TestIntentNode:
    def test_multiagent_routing(self):
        state = create_initial_state(
            query="帮我调研AI Agent", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result(
            json.dumps({"route": "multiagent", "reason": "需要检索"})
        )
        result = intent_node(state, mock_agent, "intent_router")
        assert result["intent"] == "multiagent"

    def test_direct_routing(self):
        state = create_initial_state(
            query="你好", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result(
            json.dumps({"route": "direct", "reason": "简单问候"})
        )
        result = intent_node(state, mock_agent, "intent_router")
        assert result["intent"] == "direct"

    def test_llm_route_falls_back_to_rule_route(self):
        """LLM 返回非预期值 → 回退到规则引擎的判断"""
        state = create_initial_state(
            query="帮我调研一下", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result("invalid json")
        result = intent_node(state, mock_agent, "intent_router")
        # "调研" 命中 force_multiagent → 回退为 multiagent
        assert result["intent"] == "multiagent"

    def test_llm_invoke_receives_query(self):
        state = create_initial_state(
            query="测试问题", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result(
            json.dumps({"route": "direct", "reason": "t"})
        )
        intent_node(state, mock_agent, "intent_router")
        called = mock_agent.invoke.call_args[0][0]
        assert "测试问题" in called["messages"][0].content

    def test_intent_output_has_draft(self):
        state = create_initial_state(
            query="hello", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result(
            json.dumps({"route": "direct", "reason": "greeting"})
        )
        result = intent_node(state, mock_agent, "intent_router")
        assert "draft" in result


# ============================================================
# direct_answer_node
# ============================================================

class TestDirectAnswerNode:
    def test_returns_content(self):
        state = create_initial_state(
            query="你好", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result("你好！有什么可以帮您的？")
        result = direct_answer_node(state, mock_agent, "direct_responder")
        assert "messages" in result

    def test_receives_query(self):
        state = create_initial_state(
            query="Python是什么", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result("Python是一门编程语言")
        direct_answer_node(state, mock_agent, "direct_responder")
        called = mock_agent.invoke.call_args[0][0]
        assert "Python" in called["messages"][0].content


# ============================================================
# plan_node
# ============================================================

class TestPlanNode:
    def test_returns_plan_with_outline(self):
        state = create_initial_state(
            query="调研Rust语言", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result(json.dumps({
            "objective": "调研Rust语言",
            "outline": [{"title": "概述", "depth": 1}],
            "sub_questions": ["Rust的特点是什么"],
            "research_questions": ["Rust vs Go"],
            "budget": {"max_depth": 2, "max_queries": 5},
        }))
        result = plan_node(state, mock_agent, "planner")
        assert result["phase"] == "planning completed"
        assert isinstance(result["outline"], list)
        assert isinstance(result["sub_questions"], list)
        assert "draft" in result

    def test_uses_fallback_on_invalid_json(self):
        state = create_initial_state(
            query="调研", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result("not json")
        result = plan_node(state, mock_agent, "planner")
        # Fallback _default_plan returns query as sub_questions
        assert isinstance(result["outline"], list)
        assert result["sub_questions"] == ["调研"]

    def test_receives_query(self):
        state = create_initial_state(
            query="量子计算", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result(json.dumps({
            "objective": "量子计算", "outline": [], "sub_questions": [], "research_questions": [], "budget": {},
        }))
        plan_node(state, mock_agent, "planner")
        called = mock_agent.invoke.call_args[0][0]
        assert "量子计算" in called["messages"][0].content


# ============================================================
# web_search_node
# ============================================================

class TestWebSearchNode:
    @patch("mult_agents.nodes.bocha_web_search_records")
    def test_returns_no_results_when_search_empty(self, mock_search):
        mock_search.return_value = []
        state = create_initial_state(
            query="测试搜索", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        result = web_search_node(state, mock_agent, "web_scout")
        assert "未检索到可用网页证据" in result["web_search"]
        assert "web_evidence" in result

    @patch("mult_agents.nodes.bocha_web_search_records")
    def test_processes_search_results(self, mock_search):
        mock_search.return_value = [
            {"title": "测试标题", "snippet": "测试摘要", "url": "https://example.com"},
        ]
        state = create_initial_state(
            query="测试", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result(json.dumps({
            "summary": "搜索结果总结",
            "evidence": [{"source_id": "WEB1_1-1", "title": "测试标题"}],
        }))
        result = web_search_node(state, mock_agent, "web_scout")
        assert "web_evidence" in result
        assert isinstance(result["web_evidence"], list)


# ============================================================
# local_rag_node
# ============================================================

class TestLocalRagNode:
    @patch("mult_agents.nodes.search_knowledge_base_records")
    def test_returns_no_results_when_empty(self, mock_search):
        mock_search.return_value = []
        state = create_initial_state(
            query="内部知识库查询", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        result = local_rag_node(state, mock_agent, "local_scout")
        assert "未检索到可用本地知识库" in result["local_rag"]
        assert "local_evidence" in result

    @patch("mult_agents.nodes.search_knowledge_base_records")
    def test_processes_search_results(self, mock_search):
        mock_search.return_value = [
            {"snippet": "本地文档内容", "title": "文档标题", "doc_id": "doc_1"},
        ]
        state = create_initial_state(
            query="本地查询", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result(json.dumps({
            "summary": "本地检索结果",
            "evidence": [{"source_id": "LOC1_1-1", "title": "文档标题"}],
        }))
        result = local_rag_node(state, mock_agent, "local_scout")
        assert "local_evidence" in result
        assert isinstance(result["local_evidence"], list)


# ============================================================
# deep_dive_node
# ============================================================

class TestDeepDiveNode:
    def test_returns_empty_when_no_evidence(self):
        state = create_initial_state(
            query="测试", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        result = deep_dive_node(state, mock_agent, "evidence_judge")
        assert result == {}

    def test_processes_evidence(self):
        state = create_initial_state(
            query="测试", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        state["web_evidence"] = [
            {"source_id": "w1", "source_type": "web", "title": "网络来源", "url": "https://example.com", "snippet": "内容"},
        ]
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result(json.dumps({
            "summary": "深度分析完成",
            "evidence_pool": [{"source_id": "w1", "source_type": "web", "title": "网络来源"}],
            "audit_flags": [],
        }))
        result = deep_dive_node(state, mock_agent, "evidence_judge")
        assert "evidence_pool" in result
        assert "audit_flags" in result
        assert len(result["evidence_pool"]) >= 1

    def test_uses_fallback_on_invalid_json(self):
        state = create_initial_state(
            query="测试", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        state["web_evidence"] = [
            {"source_id": "w1", "source_type": "web", "title": "来源", "url": "https://x.com", "snippet": "x"},
        ]
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result("bad json")
        # Fallback _fallback_audit builds evidence_pool from raw records
        result = deep_dive_node(state, mock_agent, "evidence_judge")
        assert isinstance(result.get("evidence_pool"), list)


# ============================================================
# analyze_node
# ============================================================

class TestAnalyzeNode:
    def test_returns_findings(self):
        state = create_initial_state(
            query="分析测试", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        state["evidence_pool"] = [{"source_id": "e1", "title": "证据1"}]
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result(json.dumps({
            "analysis_summary": "完成分析",
            "findings": [{"claim_id": "c1", "claim": "结论1", "confidence": "high"}],
            "claim_map": [{"claim_id": "c1", "source_ids": ["e1"]}],
            "needs_more_research": False,
            "missing_gaps": [],
        }))
        result = analyze_node(state, mock_agent, "analyst")
        assert isinstance(result["findings"], list)
        assert isinstance(result["claim_map"], list)
        assert "analysis" in result

    def test_uses_fallback_on_invalid_json(self):
        state = create_initial_state(
            query="分析", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result("{}")
        result = analyze_node(state, mock_agent, "analyst")
        assert isinstance(result.get("findings"), list)
        assert "messages" in result


# ============================================================
# reflect_node
# ============================================================

class TestReflectNode:
    def test_returns_supplementary_queries(self):
        state = create_initial_state(
            query="补搜测试", max_iterations=1,
            user_id="test", tenant_id="test"
        )
        state["missing_gaps"] = ["缺少对比数据"]
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result(json.dumps({
            "reflection_summary": "需要补充检索",
            "supplementary_queries": [{"section_id": "gap_1", "query": "补搜内容", "source_preference": "hybrid", "reason": "补充"}],
        }))
        result = reflect_node(state, mock_agent, "reflector")
        assert result["iteration"] == 1  # 初始 iteration=0 → +1
        assert isinstance(result["supplementary_queries"], list)

    def test_uses_fallback_when_no_missing_gaps(self):
        state = create_initial_state(
            query="无缺口", max_iterations=0,
            user_id="test", tenant_id="test"
        )
        state["missing_gaps"] = []
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result("bad json")
        result = reflect_node(state, mock_agent, "reflector")
        assert result["iteration"] == 1  # 初始 iteration=0 → +1
        # Fallback returns at least one supplementary query
        assert len(result["supplementary_queries"]) >= 1


# ============================================================
# write_node
# ============================================================

class TestWriteNode:
    def test_returns_markdown_report(self):
        state = create_initial_state(
            query="写报告", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result("# 研报标题\n\n正文内容")
        result = write_node(state, mock_agent, "writer")
        assert "draft" in result
        assert "final" in result
        assert "# 研报标题" in result["draft"]

    def test_receives_query(self):
        state = create_initial_state(
            query="Rust语言调研", max_iterations=3,
            user_id="test", tenant_id="test"
        )
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_llm_result("# 报告")
        write_node(state, mock_agent, "writer")
        called = mock_agent.invoke.call_args[0][0]
        assert "Rust语言调研" in called["messages"][0].content
