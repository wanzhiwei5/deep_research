"""状态定义模块：声明多智能体工作流共享的 ResearchState 结构。"""

import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    query: str  # 用户问题
    user_id: str  # 用户id
    tenant_id: str  # 租户id
    memory_context: str  # 记忆上下文
    messages: Annotated[List[BaseMessage], operator.add]  # 消息列表
    intent: str  # 意图
    phase: str  # 工作流阶段
    plan: str  # 计划
    outline: list[dict]  # 大纲
    sub_questions: list[str]  # 子问题
    research_questions: list[str]  # 研究问题
    search_plan: list[dict]  # 搜索计划
    budget: dict  # 预算
    web_search: str  # 网络搜索结果文本
    local_rag: str  # 本地RAG检索结果文本
    web_evidence: list[dict]  # 网络证据
    local_evidence: list[dict]  # 本地证据
    evidence_pool: list[dict]  # 证据池
    deep_dive: str  # 深度挖掘
    audit: str  # 审计
    audit_flags: list[dict]  # 审计标志
    analysis: str  # 分析
    needs_more_research: bool  # 是否需要更多研究
    missing_gaps: list[str]  # 缺失的空白
    supplementary_queries: list[dict]  # 补充查询
    findings: list[dict]  # 研究结果
    claim_map: list[dict]  # 声明映射
    source_index: list[dict]  # 源索引
    web_retrieval_stats: dict  # 网络检索统计
    local_retrieval_stats: dict  # 本地检索统计
    web_search_trace: list[dict]  # 网络搜索追踪
    local_rag_trace: list[dict]  # 本地RAG追踪
    code: str  # 代码
    draft: str  # 草稿
    final: str  # 最终结果
    iteration: int  # 迭代次数
    max_iterations: int  # 最大迭代次数


def create_initial_state(
    query: str,
    max_iterations: int,
    user_id: str,
    tenant_id: str,
    memory_context: str = "",
) -> ResearchState:
    return {
        "query": query,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "memory_context": memory_context,
        "messages": [],
        "intent": "",
        "phase": "initialized",
        "plan": "",
        "outline": [],
        "sub_questions": [],
        "research_questions": [],
        "search_plan": [],
        "budget": {},
        "web_search": "",
        "local_rag": "",
        "web_evidence": [],
        "local_evidence": [],
        "evidence_pool": [],
        "deep_dive": "",
        "audit": "",
        "audit_flags": [],
        "analysis": "",
        "needs_more_research": False,
        "missing_gaps": [],
        "supplementary_queries": [],
        "findings": [],
        "claim_map": [],
        "source_index": [],
        "web_retrieval_stats": {},
        "local_retrieval_stats": {},
        "web_search_trace": [],
        "local_rag_trace": [],
        "code": "",
        "draft": "",
        "final": "",
        "iteration": 0,
        "max_iterations": max_iterations,
    }
