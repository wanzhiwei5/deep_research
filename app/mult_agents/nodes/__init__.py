"""节点执行模块：Agent 节点函数 + 子模块 re-export。

拆分自原 nodes.py（1,354行），职责划分：
- utils.py       — 共享辅助函数（颜色、LLM 调用、JSON 解析）
- planning.py    — 意图检测与规划函数
- search.py      — 搜索构建、过滤、评分与证据处理
- fallbacks.py   — 各阶段回退函数
- citations.py   — 引用系统与报告渲染
"""

import json
import logging
import re

from langchain_core.messages import HumanMessage

from ..state import ResearchState
from ..tools import bocha_web_search_records, search_knowledge_base_records

# Re-export for test_tools.py
__all__ = [
    "bocha_web_search_records",
    "search_knowledge_base_records",
]

from .utils import (
    colorize,
    emit,
    collect_tool_calls,
    with_memory_context,
    log_inputs,
    bind_agent,
    _last_content,
    _extract_json_block,
    _load_json,
    _invoke_json_agent,
    logger,
    ANSI,
)
from .planning import (
    detect_intent,
    _default_plan,
    _guess_primary_entity,
    _derive_direct_search_queries,
    _is_query_grounded,
    _derive_search_plan,
)
from .search import (
    _build_queries,
    _extract_query_terms,
    _estimate_relevance,
    _is_bad_web_domain,
    _filter_web_records,
    _filter_local_records,
    _format_raw_records,
    _minimal_record_filter,
    _assign_source_ids,
    _enrich_evidence_from_raw,
    _prune_evidence_to_allowed_sources,
    _summarize_records,
    _normalize_source_ids,
    _finalize_query_traces,
    _is_official_domain,
    _score_evidence,
    _dedupe_sources,
)
from .fallbacks import (
    _fallback_web_evidence,
    _fallback_local_evidence,
    _fallback_audit,
    _fallback_analysis,
)
from .citations import (
    _render_fallback_report,
    _build_source_lookup,
    _extract_citation_ids,
    _validate_and_fix_citations,
    _render_reference_list,
    _render_execution_appendix,
    _ensure_reference_section,
)


# ============================================================
# Agent 节点函数（9 个）
# ============================================================

def intent_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[intent]", "cyan"), colorize(agent_name, "magenta"))
    rule_route = detect_intent(state["query"])
    prompt = (
        f"用户问题：{state['query']}\n"
        f"规则引擎初判：{rule_route}\n"
        "请输出 JSON：{\"route\":\"direct|multiagent\",\"reason\":\"...\"}"
    )
    payload, content, messages = _invoke_json_agent(
        state,
        prompt,
        agent,
        agent_name,
        "intent",
        {"route": rule_route, "reason": "rule"},
    )
    route = str(payload.get("route", rule_route)).strip().lower()
    if route not in {"direct", "multiagent"}:
        route = rule_route
    logger.info("%s 路由: %s", colorize("[intent]", "green"), route)
    return {"intent": route, "draft": content, "messages": messages}


def direct_answer_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[direct_answer]", "cyan"), colorize(agent_name, "magenta"))
    prompt = f"用户问题：{state['query']}"
    human = HumanMessage(content=with_memory_context(state, prompt))
    result = agent.invoke({"messages": [human]})
    content = _last_content(result).strip()
    emit("direct_answer", content)
    return {
        "intent": "direct",
        "final": content,
        "draft": content,
        "analysis_summary": content,
        "needs_more_research": False,
        "messages": [human, result["messages"][-1]],
    }


def plan_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[plan]", "cyan"), colorize(agent_name, "magenta"))
    log_inputs("plan", agent_name, {"query": state["query"]})
    fallback = _default_plan(state)
    payload, content, messages = _invoke_json_agent(
        state,
        f"用户需求：{state['query']}\n请先做大纲与问题拆解，再输出规划 JSON。",
        agent,
        agent_name,
        "plan",
        fallback,
    )
    outline = payload.get("outline") if isinstance(payload.get("outline"), list) else fallback["outline"]
    sub_questions = payload.get("sub_questions") if isinstance(payload.get("sub_questions"), list) else fallback["sub_questions"]
    research_questions = payload.get("research_questions") if isinstance(payload.get("research_questions"), list) else fallback["research_questions"]
    budget = payload.get("budget") if isinstance(payload.get("budget"), dict) else fallback["budget"]
    search_plan = _derive_search_plan(outline, sub_questions, research_questions, state["query"])
    plan_summary = payload.get("objective") or state["query"]
    return {
        "phase": "planning completed",
        "plan": plan_summary,
        "outline": outline,
        "sub_questions": sub_questions,
        "research_questions": research_questions,
        "search_plan": search_plan,
        "budget": budget,
        "messages": messages,
        "draft": content,
        "iteration": 0,
    }


def web_search_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[web_search]", "cyan"), colorize(agent_name, "magenta"))
    queries = _build_queries(state, "web")
    logger.info("[web_search_node] 构建查询 | 查询数量=%s | queries=%s", len(queries), [q.get("query", "") for q in queries])

    raw_records = []
    query_traces = state.get("web_search_trace", [])

    iteration = state.get("iteration", 0)
    prefix = f"WEB{iteration+1}"
    logger.info("[web_search_node] 迭代信息 | iteration=%s | prefix=%s", iteration, prefix)

    for query_index, item in enumerate(queries, 1):
        query_text = str(item.get("query", ""))
        logger.info("[web_search_node] 执行第 %s/%s 个查询 | query=%s | section_id=%s", query_index, len(queries), query_text, item.get("section_id"))
        records = bocha_web_search_records(query_text, count=4)
        logger.info("[web_search_node] 查询 %s 返回 | 记录数=%s", query_index, len(records))
        records = _assign_source_ids(records, f"{prefix}_{query_index}")
        for record in records:
            record["section_id"] = item.get("section_id")
            record["search_query"] = item.get("query")
        raw_records.extend(records)
        query_traces.append(
            {
                "iteration": iteration,
                "plan_step": query_index,
                "query": str(item.get("query", "")),
                "section_id": item.get("section_id"),
                "reason": item.get("reason", ""),
                "source_preference": item.get("source_preference", "web"),
                "raw_count": len(records),
                "raw_records": _summarize_records(records),
            }
        )
    raw_records = _dedupe_sources(raw_records, ["url", "title"])
    raw_records = _minimal_record_filter(raw_records, ["title", "snippet", "url"])
    logger.info("[web_search_node] 数据清洗后 | 去重过滤后记录数=%s", len(raw_records))

    web_retrieval_stats = state.get("web_retrieval_stats", {})
    web_retrieval_stats["query_count"] = web_retrieval_stats.get("query_count", 0) + len(queries)
    web_retrieval_stats["raw_count"] = web_retrieval_stats.get("raw_count", 0) + len(raw_records)

    log_inputs("web_search", agent_name, {"query_count": str(len(queries)), "raw_count": str(len(raw_records))})
    if not raw_records:
        logger.warning("[web_search_node] 无可用网页证据，跳过网页上下文注入 | 查询数=%s", len(queries))
        logger.info("%s 无可用网页证据，跳过网页上下文注入", colorize("[web_search]", "yellow"))
        return {
            "web_search": "未检索到可用网页证据，已跳过网页上下文注入。",
            "web_evidence": state.get("web_evidence", []),
            "web_retrieval_stats": web_retrieval_stats,
            "web_search_trace": query_traces,
        }
    logger.info("[web_search_node] 调用 LLM 整理证据 | raw_records=%s", len(raw_records))
    fallback = _fallback_web_evidence(raw_records)
    payload, content, messages = _invoke_json_agent(
        state,
        "请基于以下网页证据整理结构化 JSON。\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"原始网页证据：\n{_format_raw_records(raw_records, 'web')}",
        agent,
        agent_name,
        "web_search",
        fallback,
    )
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else fallback["evidence"]
    logger.info("[web_search_node] LLM 返回证据 | evidence数量=%s", len(evidence))
    allowed_source_ids = {str(item.get("source_id")) for item in raw_records if item.get("source_id")}
    evidence = _prune_evidence_to_allowed_sources(evidence, allowed_source_ids)
    evidence = _enrich_evidence_from_raw(evidence, raw_records)

    web_retrieval_stats["kept_count"] = web_retrieval_stats.get("kept_count", 0) + len(evidence)
    web_retrieval_stats["dropped_count"] = web_retrieval_stats.get("dropped_count", 0) + max(len(raw_records) - len(evidence), 0)

    kept_ids = {str(item.get("source_id")) for item in evidence if item.get("source_id")}
    query_traces = _finalize_query_traces(
        query_traces,
        kept_ids,
        payload.get("rejected_source_ids", []),
        str(payload.get("reject_reason", "")).strip(),
    )

    existing_evidence = state.get("web_evidence", [])
    logger.info("[web_search_node] 节点完成 | 新增证据=%s | 累计证据=%s", len(evidence), len(existing_evidence) + len(evidence))
    return {
        "web_search": payload.get("summary", content),
        "web_evidence": existing_evidence + evidence,
        "web_retrieval_stats": web_retrieval_stats,
        "web_search_trace": query_traces,
        "messages": messages,
    }


def local_rag_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[local_rag]", "cyan"), colorize(agent_name, "magenta"))
    queries = _build_queries(state, "local")
    raw_records = []
    query_traces = state.get("local_rag_trace", [])

    iteration = state.get("iteration", 0)
    prefix = f"LOC{iteration+1}"

    for query_index, item in enumerate(queries, 1):
        records = search_knowledge_base_records(str(item.get("query", "")), limit=4)
        records = _assign_source_ids(records, f"{prefix}_{query_index}")
        for record in records:
            record["section_id"] = item.get("section_id")
            record["search_query"] = item.get("query")
        raw_records.extend(records)
        query_traces.append(
            {
                "iteration": iteration,
                "plan_step": query_index,
                "query": str(item.get("query", "")),
                "section_id": item.get("section_id"),
                "reason": item.get("reason", ""),
                "source_preference": item.get("source_preference", "local"),
                "raw_count": len(records),
                "raw_records": _summarize_records(records),
            }
        )
    raw_records = _dedupe_sources(raw_records, ["doc_id", "snippet"])
    raw_records = _minimal_record_filter(raw_records, ["snippet", "title", "doc_id"])

    local_retrieval_stats = state.get("local_retrieval_stats", {})
    local_retrieval_stats["query_count"] = local_retrieval_stats.get("query_count", 0) + len(queries)
    local_retrieval_stats["raw_count"] = local_retrieval_stats.get("raw_count", 0) + len(raw_records)

    log_inputs("local_rag", agent_name, {"query_count": str(len(queries)), "raw_count": str(len(raw_records))})
    if not raw_records:
        logger.info("%s 无可用本地证据，跳过本地上下文注入", colorize("[local_rag]", "yellow"))
        return {
            "local_rag": "未检索到可用本地知识库证据，已跳过本地上下文注入。",
            "local_evidence": state.get("local_evidence", []),
            "local_retrieval_stats": local_retrieval_stats,
            "local_rag_trace": query_traces,
        }
    fallback = _fallback_local_evidence(raw_records)
    payload, content, messages = _invoke_json_agent(
        state,
        "请基于以下知识库证据整理结构化 JSON。\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"原始知识库证据：\n{_format_raw_records(raw_records, 'local')}",
        agent,
        agent_name,
        "local_rag",
        fallback,
    )
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else fallback["evidence"]
    allowed_source_ids = {str(item.get("source_id")) for item in raw_records if item.get("source_id")}
    evidence = _prune_evidence_to_allowed_sources(evidence, allowed_source_ids)

    local_retrieval_stats["kept_count"] = local_retrieval_stats.get("kept_count", 0) + len(evidence)
    local_retrieval_stats["dropped_count"] = local_retrieval_stats.get("dropped_count", 0) + max(len(raw_records) - len(evidence), 0)

    kept_ids = {str(item.get("source_id")) for item in evidence if item.get("source_id")}
    query_traces = _finalize_query_traces(
        query_traces,
        kept_ids,
        payload.get("rejected_source_ids", []),
        str(payload.get("reject_reason", "")).strip(),
    )

    existing_evidence = state.get("local_evidence", [])
    return {
        "local_rag": payload.get("summary", content),
        "local_evidence": existing_evidence + evidence,
        "local_retrieval_stats": local_retrieval_stats,
        "local_rag_trace": query_traces,
        "messages": messages,
    }


def deep_dive_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[deep_dive]", "cyan"), colorize(agent_name, "magenta"))
    if not state.get("web_evidence") and not state.get("local_evidence"):
        logger.info("%s 等待检索结果", colorize("[deep_dive]", "yellow"))
        return {}
    fallback = _fallback_audit(state)
    payload, content, messages = _invoke_json_agent(
        state,
        "请对 web 与 local 证据进行评分、去重、冲突审计，并只输出 JSON。\n"
        f"问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"web_evidence：{json.dumps(state.get('web_evidence', []), ensure_ascii=False)}\n"
        f"local_evidence：{json.dumps(state.get('local_evidence', []), ensure_ascii=False)}",
        agent,
        agent_name,
        "deep_dive",
        fallback,
    )
    payload_pool = payload.get("evidence_pool") if isinstance(payload.get("evidence_pool"), list) else []
    raw_evidence = state.get("web_evidence", []) + state.get("local_evidence", [])
    allowed_source_ids = {str(item.get("source_id", "")).strip() for item in raw_evidence if item.get("source_id")}
    evidence_pool = []
    for item in payload_pool:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("source_id", "")).strip()
        if sid and sid in allowed_source_ids:
            evidence_pool.append(item)
    if not evidence_pool:
        evidence_pool = fallback["evidence_pool"]
    existing_ids = {str(item.get("source_id", "")).strip() for item in evidence_pool if isinstance(item, dict)}
    for record in raw_evidence:
        sid = str(record.get("source_id", "")).strip()
        if not sid or sid in existing_ids:
            continue
        score, reason = _score_evidence(record)
        evidence_pool.append(
            {
                "source_id": sid,
                "source_type": record.get("source_type", "source"),
                "title": record.get("title") or sid,
                "url": record.get("url", ""),
                "doc_id": record.get("doc_id", ""),
                "snippet": record.get("snippet", ""),
                "supports_questions": record.get("supports_questions", []),
                "reliability_score": score,
                "reliability_reason": reason,
                "source_label": record.get("title") or record.get("doc_id") or record.get("url") or sid,
            }
        )
        existing_ids.add(sid)
    audit_flags = payload.get("audit_flags") if isinstance(payload.get("audit_flags"), list) else fallback["audit_flags"]
    source_index = []
    for item in evidence_pool:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("source_id", "")).strip()
        if not sid:
            continue
        source_index.append(
            {
                "source_id": sid,
                "label": item.get("title") or item.get("source_label") or sid,
                "locator": item.get("url") or item.get("doc_id") or "",
                "source_type": item.get("source_type", "source"),
            }
        )
    source_index = _dedupe_sources(source_index, ["source_id"])
    return {
        "deep_dive": payload.get("summary", content),
        "audit": payload.get("summary", content),
        "evidence_pool": evidence_pool,
        "audit_flags": audit_flags,
        "source_index": source_index,
        "messages": messages,
    }


def analyze_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[analyze]", "cyan"), colorize(agent_name, "magenta"))
    fallback = _fallback_analysis(state)
    payload, content, messages = _invoke_json_agent(
        state,
        "请基于证据池输出结论映射 JSON，并评估证据完备性：\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"证据池：{json.dumps(state.get('evidence_pool', []), ensure_ascii=False)}\n"
        f"审计标记：{json.dumps(state.get('audit_flags', []), ensure_ascii=False)}",
        agent,
        agent_name,
        "analyze",
        fallback,
    )
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else fallback["findings"]
    claim_map = payload.get("claim_map") if isinstance(payload.get("claim_map"), list) else fallback["claim_map"]
    needs_more_research = payload.get("needs_more_research", False)
    missing_gaps = payload.get("missing_gaps", [])
    analysis_summary = payload.get("analysis_summary", content)
    return {
        "analysis": analysis_summary,
        "findings": findings,
        "claim_map": claim_map,
        "needs_more_research": needs_more_research,
        "missing_gaps": missing_gaps,
        "messages": messages,
    }


def reflect_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[reflect]", "cyan"), colorize(agent_name, "magenta"))

    missing_gaps = state.get("missing_gaps", [])
    log_inputs("reflect", agent_name, {"missing_gaps": str(missing_gaps)})

    fallback = {
        "reflection_summary": "默认补搜",
        "supplementary_queries": [{"section_id": "gap_1", "query": state["query"], "source_preference": "hybrid", "reason": "fallback"}]
    }

    prompt = (
        f"分析师指出当前证据不足以完全回答问题，存在以下信息缺口：\n{json.dumps(missing_gaps, ensure_ascii=False)}\n\n"
        f"原问题：{state['query']}\n"
        f"子问题：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n"
        f"已执行过的搜索计划：\n{json.dumps(state.get('search_plan', []), ensure_ascii=False)}\n"
        f"已执行过的补搜计划：\n{json.dumps(state.get('supplementary_queries', []), ensure_ascii=False)}\n\n"
        "请生成新的补搜计划以填补缺口。"
    )

    payload, content, messages = _invoke_json_agent(
        state,
        prompt,
        agent,
        agent_name,
        "reflect",
        fallback,
    )

    return {
        "iteration": state.get("iteration", 0) + 1,
        "supplementary_queries": payload.get("supplementary_queries", fallback["supplementary_queries"]),
        "messages": messages,
    }


def write_node(state: ResearchState, agent, agent_name: str) -> ResearchState:
    logger.info("%s 开始 | agent=%s", colorize("[write]", "cyan"), colorize(agent_name, "magenta"))
    valid_source_ids = [str(item.get("source_id", "")).strip() for item in state.get("source_index", []) if item.get("source_id")]
    valid_source_ids = [item for item in valid_source_ids if item][:80]
    valid_source_ids_set = set(valid_source_ids)

    prompt = (
        "请严格根据以下信息撰写最终的 Markdown 研报。请直接输出正文，绝对不要输出任何 JSON 结构，也不要复述你的指令。\n\n"
        f"核心问题：{state['query']}\n"
        f"子问题拆解：{json.dumps(state.get('sub_questions', []), ensure_ascii=False)}\n\n"
        "【分析结论 (Findings)】：\n"
        f"{json.dumps(state.get('findings', []), ensure_ascii=False)}\n\n"
        "【可用来源索引 (source_index)】：\n"
        f"{json.dumps(state.get('source_index', []), ensure_ascii=False)}\n\n"
        "【合法引用ID列表】：\n"
        f"{json.dumps(valid_source_ids, ensure_ascii=False)}\n\n"
        "【可能存在的风险/冲突 (Audit Flags)】：\n"
        f"{json.dumps(state.get('audit_flags', []), ensure_ascii=False)}\n\n"
        "要求：正文必须使用合法引用ID（例如 [WEB1_1-1]、[LOC1_1-3]）；禁止使用不存在的编号。"
        "结尾不需要你来列举引用列表，系统会自动拼接。"
    )
    human = HumanMessage(content=with_memory_context(state, prompt))

    result = agent.invoke({"messages": [human]})
    content = _last_content(result)

    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```markdown\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"```$", "", content.strip())

    content, used_citation_ids = _validate_and_fix_citations(content, valid_source_ids_set)

    final_content = _ensure_reference_section(content, state)
    emit("write", final_content)
    return {"draft": final_content, "final": final_content, "messages": [human, result["messages"][-1]]}
