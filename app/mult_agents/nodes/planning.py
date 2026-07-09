"""意图检测与规划函数。"""

import json
import logging
import re

from ..state import ResearchState
from .search import _extract_query_terms
from .search import _dedupe_sources


logger = logging.getLogger("mult_agents")


def detect_intent(query: str) -> str:
    normalized_query = query.strip()
    force_multiagent_keywords = [
        "调查",
        "调研",
        "来源",
        "证据",
        "检索统计",
        "来源清单",
        "重大新闻",
        "热门项目",
        "趋势",
        "新闻",
        "最新",
        "盘点",
    ]
    if re.search(r"20\d{2}年", normalized_query) and any(word in normalized_query for word in ["趋势", "新闻", "调研", "调查", "盘点"]):
        return "multiagent"
    if any(word in query for word in force_multiagent_keywords):
        return "multiagent"
    keywords = [
        "调研",
        "研究",
        "调查",
        "盘点",
        "热门",
        "趋势",
        "榜单",
        "分析",
        "方案",
        "架构",
        "设计",
        "对比",
        "报告",
        "代码",
        "实现",
        "落地",
        "检索",
        "知识库",
        "证据",
        "来源",
        "溯源",
        "资料",
        "手册",
        "验证",
        "数据",
        "模型",
    ]
    return "multiagent" if any(word in query for word in keywords) else "direct"

def _default_plan(state: ResearchState) -> dict:
    return {
        "objective": state["query"],
        "sub_questions": [state["query"]],
        "outline": [
            {
                "id": "sec_1",
                "title": "默认大纲",
                "description": "默认生成的大纲",
                "section_type": "mixed",
                "requires_data": False,
                "requires_chart": False,
                "priority": 1,
                "search_queries": [state["query"]],
                "status": "pending",
            }
        ],
        "research_questions": [state["query"]],
        "budget": {"max_rounds": 2, "max_sources": 12, "max_tokens": 12000, "max_seconds": 45},
    }

def _guess_primary_entity(query: str) -> str:
    lowered = query.lower()
    ascii_terms = re.findall(r"[a-z][a-z0-9_-]{2,}", lowered)
    for term in ascii_terms:
        if term not in {"latest", "trend", "news", "agent", "open", "using"}:
            return term
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    for term in chinese_terms:
        if term not in {"帮我", "调查", "最新", "使用趋势", "是什么", "多少", "情况"}:
            return term
    return ""

def _derive_direct_search_queries(query: str) -> list[str]:
    base_query = query.strip()
    if not base_query:
        return []
    entity = _guess_primary_entity(base_query)
    candidates = [base_query]
    if entity:
        candidates.extend(
            [
                f"{entity}是什么",
                f"{entity} GitHub",
                f"{entity} 官方文档",
                f"{entity} 使用趋势",
                f"{entity} AI Agent",
            ]
        )
    else:
        candidates.extend(
            [
                f"{base_query} 是什么",
                f"{base_query} GitHub",
                f"{base_query} 官方文档",
            ]
        )
    deduped: list[str] = []
    for item in candidates:
        text = item.strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped[:6]

def _is_query_grounded(candidate: str, user_query: str) -> bool:
    candidate_terms = set(_extract_query_terms(candidate))
    user_terms = set(_extract_query_terms(user_query))
    if not candidate_terms or not user_terms:
        return False
    if _guess_primary_entity(user_query) and _guess_primary_entity(user_query) in candidate.lower():
        return True
    overlap = candidate_terms & user_terms
    return len(overlap) >= 1

def _derive_search_plan(outline: list[dict], sub_questions: list[str], _research_questions: list[str], query: str) -> list[dict]:
    plan: list[dict] = []
    for direct_query in _derive_direct_search_queries(query):
        plan.append(
            {
                "section_id": "user_query",
                "query": direct_query,
                "source_preference": "hybrid",
                "reason": "围绕用户原始问题生成的直接检索词",
            }
        )
    for section in outline:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "sec")
        for item in section.get("search_queries", []) or []:
            text = str(item).strip()
            if text and _is_query_grounded(text, query):
                plan.append(
                    {
                        "section_id": section_id,
                        "query": text,
                        "source_preference": "hybrid",
                        "reason": f"来自大纲章节 {section_id}",
                    }
                )
    if not plan:
        plan.append({"section_id": "sec_1", "query": query, "source_preference": "hybrid", "reason": "fallback"})
    deduped = _dedupe_sources(plan, ["query"])
    return deduped[:6]
