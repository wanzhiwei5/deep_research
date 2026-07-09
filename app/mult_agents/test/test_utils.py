"""测试：_invoke_json_agent —— 核心 LLM 调用函数。"""

import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from mult_agents.nodes.utils import _invoke_json_agent
from mult_agents.state import create_initial_state


def _make_result(content: str):
    mock_result = MagicMock()
    mock_result.__getitem__.side_effect = lambda key: {
        "messages": [AIMessage(content=content)]
    }[key]
    return mock_result


class TestInvokeJsonAgent:
    def test_returns_parsed_json(self):
        state = create_initial_state(query="测试", max_iterations=3, user_id="u", tenant_id="t")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_result(
            json.dumps({"key": "value", "num": 42})
        )
        payload, content, messages = _invoke_json_agent(
            state, "测试提示", mock_agent, "test_agent", "test_node", {}
        )
        assert payload == {"key": "value", "num": 42}
        assert "key" in content
        assert len(messages) == 2

    def test_uses_fallback_on_invalid_json(self):
        state = create_initial_state(query="测试", max_iterations=3, user_id="u", tenant_id="t")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_result("not json")
        fallback = {"default": True}
        payload, content, messages = _invoke_json_agent(
            state, "提示", mock_agent, "test_agent", "test_node", fallback
        )
        assert payload == fallback

    def test_parses_json_from_code_block(self):
        state = create_initial_state(query="测试", max_iterations=3, user_id="u", tenant_id="t")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_result(
            "```json\n{\"from_block\": true}\n```"
        )
        payload, content, messages = _invoke_json_agent(
            state, "提示", mock_agent, "test_agent", "test_node", {}
        )
        assert payload == {"from_block": True}

    def test_receives_prompt_with_memory_context(self):
        state = create_initial_state(query="Q", max_iterations=3, user_id="u", tenant_id="t")
        state["memory_context"] = "上次聊过AI"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_result(json.dumps({"ok": True}))
        _invoke_json_agent(state, "关于AI", mock_agent, "a", "n", {})
        called = mock_agent.invoke.call_args[0][0]
        prompt_text = called["messages"][0].content
        assert "关于AI" in prompt_text
        assert "上次聊过AI" in prompt_text
