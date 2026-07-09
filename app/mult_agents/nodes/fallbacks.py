"""各阶段回退函数（LLM 调用失败时的默认值）。"""

from ..state import ResearchState
from .search import _is_official_domain, _score_evidence, _dedupe_sources


def _fallback_web_evidence(records: list[dict]) -> dict:
    evidence = []
    for record in records:
        evidence.append(
            {
                "source_id": record.get("source_id"),
                "title": record.get("title"),
                "url": record.get("url", ""),
                "snippet": record.get("snippet", ""),
                "domain": record.get("domain", ""),
                "source_type": "web",
                "reliability_hint": "official" if _is_official_domain(record.get("domain", "")) else "unknown",
                "supports": [],
                "notes": "",
            }
        )
    return {"summary": "完成网页证据采集。", "evidence": evidence, "gaps": []}

def _fallback_local_evidence(records: list[dict]) -> dict:
    evidence = []
    for record in records:
        evidence.append(
            {
                "source_id": record.get("source_id"),
                "doc_id": record.get("doc_id", ""),
                "title": record.get("title", "") or record.get("source_id", ""),
                "snippet": record.get("snippet", ""),
                "source_type": "local",
                "reliability_hint": "internal",
                "supports": [],
                "notes": "",
            }
        )
    return {"summary": "完成本地知识库证据采集。", "evidence": evidence, "gaps": []}

def _fallback_audit(state: ResearchState) -> dict:
    evidence_pool = []
    source_index = []
    audit_flags = []
    for record in state.get("web_evidence", []) + state.get("local_evidence", []):
        score, reason = _score_evidence(record)
        normalized = dict(record)
        normalized["reliability_score"] = score
        normalized["reliability_reason"] = reason
        normalized["source_label"] = record.get("title") or record.get("doc_id") or record.get("url") or record.get("source_id")
        normalized.setdefault("supports", [])
        normalized.setdefault("refutes", [])
        evidence_pool.append(normalized)
        locator = record.get("url") or record.get("doc_id") or ""
        if score < 0.6:
            audit_flags.append({"type": "low_confidence", "target": record.get("source_id"), "reason": reason})
        else:
            source_index.append(
                {
                    "source_id": record.get("source_id"),
                    "label": normalized["source_label"],
                    "locator": locator or "未提供定位信息",
                    "source_type": record.get("source_type", "source"),
                }
            )
    for hypo in state.get("hypotheses", []):
        hypo_id = hypo.get("id")
        related = [item for item in evidence_pool if hypo_id in item.get("supports", []) or hypo_id in item.get("refutes", [])]
        if not related:
            audit_flags.append({"type": "missing_evidence", "target": hypo_id, "reason": "缺少直接关联证据"})
    return {
        "summary": "完成证据评分与审计。",
        "evidence_pool": evidence_pool,
        "audit_flags": audit_flags,
        "source_index": _dedupe_sources(source_index, ["source_id"]),
    }

def _fallback_analysis(state: ResearchState) -> dict:
    return {
        "analysis_summary": "默认分析结论",
        "needs_more_research": False,
        "missing_gaps": [],
        "findings": [],
        "claim_map": [],
        "next_actions": [],
    }
