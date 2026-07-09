"""测试：bocha_web_search_records 工具（mock HTTP 请求）。

使用 unittest.mock 拦截 urllib.request.urlopen，不发起真实网络请求。
"""

import json
from unittest.mock import patch

from mult_agents.nodes import bocha_web_search_records


def _make_urlopen(webpages: list[dict]):
    """返回一个可被 with 语句使用的 urlopen mock 函数。

    bocha_web_search_records 调用：
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")

    返回的函数是给 patch('urllib.request.urlopen') 用的替换值。
    """
    body = json.dumps({"data": {"webPages": webpages}}).encode("utf-8")

    class Response:
        status = 200

        def read(self):
            return body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def urlopen(*args, **kwargs):
        return Response()

    return urlopen


class TestBochaWebSearch:
    def test_parses_webpages_correctly(self):
        urlopen = _make_urlopen([
            {"url": "https://infoq.cn/ai-agent", "name": "AI Agent 趋势分析",
             "summary": "2025年AI Agent发展报告"},
        ])
        with patch("os.getenv", return_value="sk-test-key"), \
             patch("urllib.request.urlopen", new=urlopen):
            records = bocha_web_search_records("AI Agent 趋势", count=5)

        assert len(records) == 1
        assert records[0]["source_id"] == "WEB-1"
        assert records[0]["title"] == "AI Agent 趋势分析"
        assert records[0]["url"] == "https://infoq.cn/ai-agent"
        assert records[0]["domain"] == "infoq.cn"
        assert records[0]["source_type"] == "web"

    def test_multiple_results_assigned_sequential_ids(self):
        urlopen = _make_urlopen([
            {"url": "https://a.com/1", "name": "结果1", "summary": "摘要1"},
            {"url": "https://b.com/2", "name": "结果2", "summary": "摘要2"},
        ])
        with patch("os.getenv", return_value="sk-test"), \
             patch("urllib.request.urlopen", new=urlopen):
            records = bocha_web_search_records("test", count=5)

        assert len(records) == 2
        assert records[0]["source_id"] == "WEB-1"
        assert records[1]["source_id"] == "WEB-2"

    def test_empty_api_key_returns_empty(self):
        with patch("os.getenv", return_value=""):
            records = bocha_web_search_records("test")
        assert records == []

    def test_http_403_returns_empty(self):
        from urllib.error import HTTPError
        with patch("os.getenv", return_value="sk-test"), \
             patch("urllib.request.urlopen", side_effect=HTTPError(403, "Forbidden", None, None, None)):
            records = bocha_web_search_records("test")
        assert records == []

    def test_http_500_returns_empty(self):
        from urllib.error import HTTPError
        with patch("os.getenv", return_value="sk-test"), \
             patch("urllib.request.urlopen", side_effect=HTTPError(500, "Internal Server Error", None, None, None)):
            records = bocha_web_search_records("test")
        assert records == []

    def test_url_error_returns_empty(self):
        from urllib.error import URLError
        with patch("os.getenv", return_value="sk-test"), \
             patch("urllib.request.urlopen", side_effect=URLError("Name or service not known")):
            records = bocha_web_search_records("test")
        assert records == []

    def test_invalid_json_response_returns_empty(self):
        class BadResponse:
            status = 200
            def read(self):
                return b"not json"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        with patch("os.getenv", return_value="sk-test"), \
             patch("urllib.request.urlopen", return_value=BadResponse()):
            records = bocha_web_search_records("test")
        assert records == []

    def test_non_dict_webpages_handled(self):
        """webPages 为列表但元素可能非 dict"""
        urlopen = _make_urlopen(["just a string"])
        with patch("os.getenv", return_value="sk-test"), \
             patch("urllib.request.urlopen", new=urlopen):
            records = bocha_web_search_records("test")
        # 应跳过非 dict 元素
        assert records == []

    def test_webpages_as_dict_value(self):
        """webPages 可能是 dict 带 value 字段"""
        urlopen = _make_urlopen({"value": [
            {"url": "https://x.com/a", "name": "X", "summary": "关于X"},
        ]})
        with patch("os.getenv", return_value="sk-test"), \
             patch("urllib.request.urlopen", new=urlopen):
            records = bocha_web_search_records("test")
        assert len(records) == 1

    def test_count_limit_respected(self):
        urlopen = _make_urlopen([
            {"url": f"https://x.com/{i}", "name": str(i), "summary": str(i)}
            for i in range(20)
        ])
        with patch("os.getenv", return_value="sk-test"), \
             patch("urllib.request.urlopen", new=urlopen):
            records = bocha_web_search_records("test", count=3)
        assert len(records) == 3

    def test_missing_name_uses_fallback(self):
        urlopen = _make_urlopen([
            {"url": "https://x.com/a", "summary": "摘要"},
        ])
        with patch("os.getenv", return_value="sk-test"), \
             patch("urllib.request.urlopen", new=urlopen):
            records = bocha_web_search_records("test")
        assert records[0]["title"] == "web_result_1"

    def test_published_at_field(self):
        urlopen = _make_urlopen([
            {"url": "https://x.com/a", "name": "A", "summary": "s",
             "datePublished": "2025-01-01"},
        ])
        with patch("os.getenv", return_value="sk-test"), \
             patch("urllib.request.urlopen", new=urlopen):
            records = bocha_web_search_records("test")
        assert records[0]["published_at"] == "2025-01-01"
