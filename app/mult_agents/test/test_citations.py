"""测试：引用 ID 提取、校验与参考资料渲染。

覆盖以下纯函数（零 mock）：
- _extract_citation_ids        — nodes.py
- _validate_and_fix_citations  — nodes.py
"""

from mult_agents.nodes import _extract_citation_ids, _validate_and_fix_citations

# 合法的 source_id 必须匹配模式 [A-Z]+\\d+_\\d+-\\d+
# 例如：WEB1_1-1, LOC2_3-4
_VALID = "WEB1_1-1"
_VALID2 = "LOC2_3-4"
_INVALID = "FAKE1_0-0"  # 匹配模式但不在白名单中

# ============================================================
# _extract_citation_ids
# ============================================================

class TestExtractCitationIds:
    def test_extract_single_citation(self):
        result = _extract_citation_ids("详见[WEB1_1-1]")
        assert result == ["WEB1_1-1"]

    def test_extract_multiple_citations(self):
        content = "根据[WEB1_1-1]和[LOC2_3-4]可知"
        result = _extract_citation_ids(content)
        assert result == ["WEB1_1-1", "LOC2_3-4"]

    def test_deduplicates_preserves_order(self):
        content = "见[WEB1_1-1]，另见[WEB1_1-1]，还有[LOC2_3-4]"
        result = _extract_citation_ids(content)
        assert result == ["WEB1_1-1", "LOC2_3-4"]

    def test_no_citations_returns_empty(self):
        assert _extract_citation_ids("这是一段没有引用的文字") == []

    def test_empty_string(self):
        assert _extract_citation_ids("") == []

    def test_ignores_brackets_without_pattern(self):
        """普通方括号内容不匹配模式"""
        result = _extract_citation_ids("这是[一个]普通的[说明]")
        assert result == []

    def test_handles_lowercase_source_id(self):
        """模式要求大写字母开头，小写不匹配"""
        result = _extract_citation_ids("详见[web1_1-1]")
        assert result == []


# ============================================================
# _validate_and_fix_citations
# ============================================================

class TestValidateAndFixCitations:
    def test_keeps_valid_citations(self):
        content = f"根据[{_VALID}]可知"
        fixed, used = _validate_and_fix_citations(content, {_VALID})
        assert f"[{_VALID}]" in fixed
        assert used == [_VALID]

    def test_removes_invalid_citations(self):
        content = f"根据[{_VALID}]和[{_INVALID}]可知"
        fixed, used = _validate_and_fix_citations(content, {_VALID})
        assert f"[{_VALID}]" in fixed
        assert f"[{_INVALID}]" not in fixed
        assert used == [_VALID]

    def test_removes_all_invalid(self):
        content = f"见[{_INVALID}]"
        fixed, used = _validate_and_fix_citations(content, set())
        assert f"[{_INVALID}]" not in fixed
        assert used == []

    def test_no_citations_unchanged(self):
        content = "纯文本内容"
        fixed, used = _validate_and_fix_citations(content, {_VALID})
        assert fixed == content
        assert used == []

    def test_mixed_valid_and_invalid(self):
        content = f"研究[{_VALID}]发现，对比[{_VALID2}]结果，还有[{_INVALID}]不可信"
        fixed, used = _validate_and_fix_citations(content, {_VALID, _VALID2})
        assert f"[{_VALID}]" in fixed
        assert f"[{_VALID2}]" in fixed
        assert f"[{_INVALID}]" not in fixed
        assert len(used) == 2

    def test_duplicate_valid_kept_once(self):
        content = f"见[{_VALID}]和[{_VALID}]"
        fixed, used = _validate_and_fix_citations(content, {_VALID})
        assert used == [_VALID]
