"""测试：证据过滤、评分与去重。

覆盖以下纯函数（零 mock，零外部依赖）：
- _estimate_relevance     — nodes.py
- _is_bad_web_domain      — nodes.py
- _is_official_domain     — nodes.py
- _score_evidence         — nodes.py
- _dedupe_sources         — nodes.py
- _filter_web_records     — nodes.py
- _filter_local_records   — nodes.py
- _assign_source_ids      — nodes.py
- _prune_evidence_to_allowed_sources — nodes.py
- _enrich_evidence_from_raw — nodes.py
- _minimal_record_filter  — nodes.py
- _summarize_records      — nodes.py
- _normalize_source_ids   — nodes.py
"""

from mult_agents.nodes import (
    _estimate_relevance,
    _is_bad_web_domain,
    _is_official_domain,
    _score_evidence,
    _dedupe_sources,
    _filter_web_records,
    _filter_local_records,
    _assign_source_ids,
    _prune_evidence_to_allowed_sources,
    _enrich_evidence_from_raw,
    _minimal_record_filter,
    _summarize_records,
    _normalize_source_ids,
)


# ============================================================
# _estimate_relevance
# ============================================================

class TestEstimateRelevance:
    def test_full_match_high_score(self):
        """查询词全部出现在文本中 → 高评分"""
        score = _estimate_relevance("AI Agent 框架对比", "AI Agent 框架对比分析")
        assert score > 0.5

    def test_partial_match_medium_score(self):
        score = _estimate_relevance("AI Agent 框架对比", "AI Agent 入门教程")
        assert 0 < score < 1.0

    def test_no_overlap_zero(self):
        score = _estimate_relevance("汽车保养", "今日股票市场走势分析")
        assert score == 0.0

    def test_empty_query_zero(self):
        score = _estimate_relevance("", "随便什么文本")
        assert score == 0.0

    def test_stopwords_excluded(self):
        """停用词不应参与评分计算"""
        score = _estimate_relevance("什么是AI Agent", "什么是AI Agent 趋势")
        assert score > 0

    def test_english_terms(self):
        score = _estimate_relevance("LangGraph", "LangGraph framework for building agent")
        assert score > 0

    def test_chinese_bigrams(self):
        score = _estimate_relevance("机器学习", "机器学习在自然语言处理中的应用")
        assert score > 0


# ============================================================
# _is_bad_web_domain
# ============================================================

class TestIsBadWebDomain:
    def test_bad_domain_detected(self):
        assert _is_bad_web_domain("datasheet.com") is True
        assert _is_bad_web_domain("bdtic.com") is True
        assert _is_bad_web_domain("doc88.com") is True
        assert _is_bad_web_domain("elecfans.com") is True

    def test_normal_domain_ok(self):
        assert _is_bad_web_domain("infoq.cn") is False
        assert _is_bad_web_domain("github.com") is False
        assert _is_bad_web_domain("example.com") is False

    def test_empty_domain_ok(self):
        assert _is_bad_web_domain("") is False


# ============================================================
# _is_official_domain
# ============================================================

class TestIsOfficialDomain:
    def test_gov_cn(self):
        assert _is_official_domain("www.gov.cn") is True

    def test_gov(self):
        assert _is_official_domain("whitehouse.gov") is True

    def test_edu(self):
        assert _is_official_domain("mit.edu") is True

    def test_edu_cn(self):
        assert _is_official_domain("pku.edu.cn") is True

    def test_regular_domain_not_official(self):
        assert _is_official_domain("example.com") is False

    def test_empty_not_official(self):
        assert _is_official_domain("") is False


# ============================================================
# _score_evidence
# ============================================================

class TestScoreEvidence:
    def test_local_source_high_score(self):
        score, reason = _score_evidence({"source_type": "local"})
        assert score == 0.92

    def test_official_domain_high_score(self):
        score, reason = _score_evidence({"source_type": "web", "domain": "whitehouse.gov"})
        assert score == 0.88

    def test_media_domain_medium_score(self):
        score, reason = _score_evidence({"source_type": "web", "domain": "news.yahoo.com"})
        assert score == 0.72

    def test_known_media_domain(self):
        score, reason = _score_evidence({"source_type": "web", "domain": "people.com.cn"})
        assert score == 0.72

    def test_regular_domain_low_score(self):
        score, reason = _score_evidence({"source_type": "web", "domain": "example.com"})
        assert score == 0.58

    def test_no_domain_lowest_score(self):
        score, reason = _score_evidence({"source_type": "web", "domain": ""})
        assert score == 0.45


# ============================================================
# _dedupe_sources
# ============================================================

class TestDedupeSources:
    def test_dedupe_by_source_id(self):
        items = [
            {"source_id": "WEB-1", "title": "A"},
            {"source_id": "WEB-2", "title": "B"},
            {"source_id": "WEB-1", "title": "A_dup"},
        ]
        result = _dedupe_sources(items, ["source_id"])
        assert len(result) == 2

    def test_no_duplicates(self):
        items = [{"id": "a"}, {"id": "b"}]
        result = _dedupe_sources(items, ["id"])
        assert len(result) == 2

    def test_empty_list(self):
        assert _dedupe_sources([], ["id"]) == []


# ============================================================
# _filter_web_records
# ============================================================

class TestFilterWebRecords:
    def test_keeps_relevant_drops_irrelevant(self):
        records = [
            {"title": "AI Agent 框架对比", "snippet": "LangGraph vs CrewAI 框架分析", "domain": "infoq.cn"},
            {"title": "今日菜价行情", "snippet": "白菜价格走势", "domain": "example.com"},
        ]
        kept, stats = _filter_web_records("AI Agent 框架", records)
        assert len(kept) == 1
        assert kept[0]["title"] == "AI Agent 框架对比"
        assert stats["dropped_irrelevant"] == 1

    def test_bad_domain_dropped(self):
        records = [
            {"title": "AI 框架", "snippet": "LangGraph 使用体验", "domain": "datasheet.com"},
        ]
        kept, stats = _filter_web_records("AI 框架", records)
        assert len(kept) == 0
        assert stats["dropped_domain"] == 1

    def test_empty_title_and_snippet_dropped(self):
        records = [
            {"title": "", "snippet": "", "domain": "example.com"},
        ]
        kept, stats = _filter_web_records("test", records)
        assert stats["dropped_empty"] == 1

    def test_official_domain_kept_even_low_relevance(self):
        """官方域名的结果即使相关度低也保留"""
        records = [
            {"title": "无关页面", "snippet": "一些内容", "domain": "gov.cn"},
        ]
        kept, stats = _filter_web_records("AI Agent", records)
        assert len(kept) == 1

    def test_empty_input_list(self):
        kept, stats = _filter_web_records("test", [])
        assert len(kept) == 0
        assert stats["raw_count"] == 0

    def test_relevance_score_attached(self):
        records = [{"title": "AI Agent", "snippet": "趋势分析", "domain": "infoq.cn"}]
        kept, _ = _filter_web_records("AI Agent", records)
        assert "relevance_score" in kept[0]
        assert kept[0]["relevance_score"] > 0


# ============================================================
# _filter_local_records
# ============================================================

class TestFilterLocalRecords:
    def test_keeps_relevant_drops_irrelevant(self):
        records = [
            {"title": "产品文档", "snippet": "AI Agent 配置说明", "doc_id": "doc-1"},
            {"title": "无关文档", "snippet": "食堂菜单", "doc_id": "doc-2"},
        ]
        kept, stats = _filter_local_records("AI Agent", records)
        assert len(kept) == 1
        assert stats["dropped_irrelevant"] == 1

    def test_empty_snippet_dropped(self):
        records = [{"title": "a", "snippet": "", "doc_id": "doc-1"}]
        kept, stats = _filter_local_records("test", records)
        assert stats["dropped_empty"] == 1

    def test_no_doc_id_low_relevance_dropped(self):
        records = [{"title": "一些内容", "snippet": "一些文本", "doc_id": ""}]
        kept, stats = _filter_local_records("AI Agent 框架", records)
        # 无 doc_id + 相关度低 → dropped_missing_doc
        assert stats["dropped_missing_doc"] >= 0

    def test_no_doc_id_but_high_relevance_kept(self):
        records = [{"title": "AI Agent 框架", "snippet": "AI Agent 框架 LangGraph 使用指南", "doc_id": ""}]
        kept, stats = _filter_local_records("AI Agent 框架", records)
        assert len(kept) == 1

    def test_empty_input(self):
        kept, stats = _filter_local_records("test", [])
        assert len(kept) == 0


# ============================================================
# _assign_source_ids
# ============================================================

class TestAssignSourceIds:
    def test_assigns_sequential_ids(self):
        records = [{"title": "a"}, {"title": "b"}]
        result = _assign_source_ids(records, "WEB")
        assert result[0]["source_id"] == "WEB-1"
        assert result[1]["source_id"] == "WEB-2"

    def test_empty_input(self):
        assert _assign_source_ids([], "WEB") == []

    def test_does_not_mutate_originals(self):
        records = [{"title": "a"}]
        _assign_source_ids(records, "WEB")
        assert "source_id" not in records[0]


# ============================================================
# _prune_evidence_to_allowed_sources
# ============================================================

class TestPruneEvidence:
    def test_keeps_allowed_source_ids(self):
        evidence = [
            {"source_id": "WEB-1", "content": "a"},
            {"source_id": "WEB-2", "content": "b"},
            {"source_id": "WEB-3", "content": "c"},
        ]
        result = _prune_evidence_to_allowed_sources(evidence, {"WEB-1", "WEB-3"})
        assert len(result) == 2
        assert result[0]["source_id"] == "WEB-1"

    def test_non_dict_items_skipped(self):
        evidence = [{"source_id": "WEB-1"}, "not a dict"]
        result = _prune_evidence_to_allowed_sources(evidence, {"WEB-1"})
        assert len(result) == 1

    def test_empty_allowed_set(self):
        evidence = [{"source_id": "WEB-1"}]
        result = _prune_evidence_to_allowed_sources(evidence, set())
        assert result == []


# ============================================================
# _enrich_evidence_from_raw
# ============================================================

class TestEnrichEvidence:
    def test_fills_missing_url_and_domain(self):
        evidence = [{"source_id": "WEB-1", "title": "t"}]
        raw = [{"source_id": "WEB-1", "url": "https://example.com", "domain": "example.com", "title": "原始标题"}]
        enriched = _enrich_evidence_from_raw(evidence, raw)
        assert enriched[0]["url"] == "https://example.com"
        assert enriched[0]["domain"] == "example.com"
        assert enriched[0]["title"] == "t"  # 不覆盖已有值

    def test_no_match_no_enrich(self):
        evidence = [{"source_id": "WEB-1"}]
        raw = [{"source_id": "WEB-99"}]
        enriched = _enrich_evidence_from_raw(evidence, raw)
        # source_id 不匹配，不填 url/domain
        assert enriched[0].get("url") is None or enriched[0].get("url") == ""


# ============================================================
# _minimal_record_filter
# ============================================================

class TestMinimalRecordFilter:
    def test_requires_at_least_one_field(self):
        records = [
            {"title": "a", "snippet": "b"},
            {"title": "", "snippet": ""},
        ]
        result = _minimal_record_filter(records, ["title", "snippet"])
        assert len(result) == 1


# ============================================================
# _summarize_records
# ============================================================

class TestSummarizeRecords:
    def test_summarizes_first_5(self):
        records = [{"source_id": f"W-{i}", "title": f"t{i}"} for i in range(10)]
        result = _summarize_records(records)
        assert len(result) == 5


# ============================================================
# _normalize_source_ids
# ============================================================

class TestNormalizeSourceIds:
    def test_deduplicates_and_strips(self):
        result = _normalize_source_ids([" WEB-1 ", "  WEB-1  "])
        assert len(result) == 1
        assert result[0] == "WEB-1"

    def test_empty_values_removed(self):
        result = _normalize_source_ids(["WEB-1", "", "WEB-2", "  "])
        assert len(result) == 2

    def test_none_input(self):
        assert _normalize_source_ids(None) == []
