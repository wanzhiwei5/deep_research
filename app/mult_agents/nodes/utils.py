"""共享辅助函数：颜色输出、LLM 调用、JSON 解析等。"""

import json
import logging
import os
import re
from functools import partial

from langchain_core.messages import HumanMessage

from ..state import ResearchState


logger = logging.getLogger("mult_agents")

ANSI = {
    "reset": "[0m",
    "cyan": "[36m",
    "magenta": "[35m",
    "yellow": "[33m",
    "green": "[32m",
    "red": "[31m",
}


def colorize(text: str, color: str) -> str:
    if os.getenv("NO_COLOR"):
        return text
    code = ANSI.get(color, "")
    if not code:
        return text
    return f"{code}{text}{ANSI['reset']}"
def emit(node: str, content: str):
    preview = content.replace("\n", " ")
    if len(preview) > 400:
        preview = preview[:400] + "..."
    logger.info("%s 输出: %s", colorize(f"[{node}]", "yellow"), preview)
def collect_tool_calls(messages) -> tuple[list, list]:
    tools = []
    tool_outputs = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else None
                if name:
                    tools.append(name)
        name = getattr(msg, "name", None)
        msg_type = getattr(msg, "type", None)
        if msg_type == "tool" and name:
            tools.append(name)
            output = getattr(msg, "content", "")
            if output:
                tool_outputs.append(f"{name}: {output}")
    return tools, tool_outputs
def with_memory_context(state: ResearchState, user_prompt: str) -> str:
    memory_context = state.get("memory_context", "").strip()
    if not memory_context:
        return user_prompt
    return f"{user_prompt}\n\n[跨会话记忆]\n{memory_context}"
def log_inputs(node: str, agent_name: str, payload: dict):
    preview = {
        key: (value[:200] + "..." if isinstance(value, str) and len(value) > 200 else value)
        for key, value in payload.items()
    }
    logger.info("%s 输入 | agent=%s | data=%s", colorize(f"[{node}]", "cyan"), colorize(agent_name, "magenta"), preview)
def bind_agent(node_func, agent, agent_name: str):
    return partial(node_func, agent=agent, agent_name=agent_name)
def _last_content(result) -> str:
    content = result["messages"][-1].content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
    return str(content)
def _extract_json_block(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned
def _load_json(text: str, fallback: dict) -> dict:
    try:
        value = json.loads(_extract_json_block(text))
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return fallback
def _invoke_json_agent(state: ResearchState, prompt: str, agent, agent_name: str, node: str, fallback: dict) -> tuple[dict, str, list]:
    human = HumanMessage(content=with_memory_context(state, prompt))
    # Optimization: Do NOT pass state["messages"] to avoid token accumulation
    # Each node only needs its specific instruction and the current state data
    result = agent.invoke({"messages": [human]})
    tools, tool_outputs = collect_tool_calls(result["messages"])
    logger.info("%s 工具: %s", colorize(f"[{node}]", "green"), ", ".join(tools) if tools else "无")
    for item in tool_outputs[:5]:
        logger.info("%s 工具输出: %s", colorize(f"[{node}]", "green"), item[:400])
    logger.info("%s LLM调用: 是 | 思考: 不可见", colorize(f"[{node}]", "yellow"))
    content = _last_content(result)
    emit(node, content)
    return _load_json(content, fallback), content, [human, result["messages"][-1]]
