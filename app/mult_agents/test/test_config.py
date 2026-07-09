"""测试：AppConfig 配置加载。"""

import json
import os
import pytest
import tempfile
from pathlib import Path

from mult_agents.config import AppConfig


class TestResolveStr:
    def test_env_takes_priority(self):
        os.environ["TEST_KEY"] = "from_env"
        result = AppConfig._resolve_str({"key": "from_file"}, "key", "TEST_KEY", "default")
        assert result == "from_env"
        del os.environ["TEST_KEY"]

    def test_file_when_env_missing(self):
        result = AppConfig._resolve_str({"key": "from_file"}, "key", "NONEXISTENT_VAR", "default")
        assert result == "from_file"

    def test_default_when_both_missing(self):
        result = AppConfig._resolve_str({}, "key", "NONEXISTENT_VAR", "default_val")
        assert result == "default_val"

    def test_empty_env_ignored(self):
        os.environ["EMPTY_KEY"] = ""
        result = AppConfig._resolve_str({"key": "from_file"}, "key", "EMPTY_KEY", "default")
        assert result == "from_file"
        del os.environ["EMPTY_KEY"]


class TestResolveBool:
    def test_true_from_env(self):
        os.environ["BOOL_KEY"] = "true"
        assert AppConfig._resolve_bool({}, "key", "BOOL_KEY", False) is True
        del os.environ["BOOL_KEY"]

    def test_false_from_env(self):
        os.environ["BOOL_KEY"] = "false"
        assert AppConfig._resolve_bool({}, "key", "BOOL_KEY", True) is False
        del os.environ["BOOL_KEY"]

    def test_default_from_kwargs(self):
        assert AppConfig._resolve_bool({}, "key", "NONEXISTENT", True) is True
        assert AppConfig._resolve_bool({}, "key", "NONEXISTENT", False) is False


class TestResolveInt:
    def test_from_env(self):
        os.environ["INT_KEY"] = "42"
        assert AppConfig._resolve_int({}, "key", "INT_KEY", 0) == 42
        del os.environ["INT_KEY"]

    def test_from_file(self):
        result = AppConfig._resolve_int({"key": "99"}, "key", "NONEXISTENT", 0)
        assert result == 99

    def test_default(self):
        assert AppConfig._resolve_int({}, "key", "NONEXISTENT", 7) == 7


class TestFromFile:
    def test_missing_api_key_raises(self):
        # Temporarily hide DASHSCOPE_API_KEY
        saved = os.environ.pop("DASHSCOPE_API_KEY", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w") as f:
                json.dump({}, f)
            with pytest.raises(ValueError, match="缺少 DASHSCOPE_API_KEY"):
                AppConfig.from_file(path)
        if saved is not None:
            os.environ["DASHSCOPE_API_KEY"] = saved

    def test_invalid_json_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            with open(path, "w") as f:
                f.write("not json")
            with pytest.raises(json.JSONDecodeError):
                AppConfig.from_file(path)

    def test_minimal_config_with_env_api_key(self):
        os.environ["DASHSCOPE_API_KEY"] = "test-key-123"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w") as f:
                json.dump({"model": "qwen-max"}, f)
            config = AppConfig.from_file(path)
            assert config.api_key == "test-key-123"
            assert config.model == "qwen-max"
        del os.environ["DASHSCOPE_API_KEY"]


class TestWithOverrides:
    def test_applies_non_none_overrides(self):
        config = AppConfig(api_key="k", model="m", thread_id="t", user_id="u",
                           tenant_id="t", max_iterations=3, enable_memory=True,
                           short_term_ttl_seconds=100, short_term_max_messages=10,
                           short_term_summary_threshold=5, short_term_backend="memory",
                           long_term_backend="disabled", long_term_scope="user",
                           save_conversation_task=False, checkpointer_backend="memory",
                           enable_milvus=False, memory_top_k=3,
                           redis_url="r", postgres_dsn="p",
                           milvus_host="h", milvus_port=1, milvus_collection="c")
        updated = config.with_overrides(user_id="new_user", max_iterations=5, enable_memory=None)
        assert updated.user_id == "new_user"
        assert updated.max_iterations == 5
        assert updated.enable_memory is True  # None 不覆盖
        # Unchanged fields remain
        assert updated.model == "m"

    def test_none_overrides_ignored(self):
        config = AppConfig(api_key="k", model="m", thread_id="t", user_id="u",
                           tenant_id="t", max_iterations=3, enable_memory=True,
                           short_term_ttl_seconds=100, short_term_max_messages=10,
                           short_term_summary_threshold=5, short_term_backend="memory",
                           long_term_backend="disabled", long_term_scope="user",
                           save_conversation_task=False, checkpointer_backend="memory",
                           enable_milvus=False, memory_top_k=3,
                           redis_url="r", postgres_dsn="p",
                           milvus_host="h", milvus_port=1, milvus_collection="c")
        updated = config.with_overrides(user_id=None, max_iterations=None)
        assert updated.user_id == config.user_id
        assert updated.max_iterations == config.max_iterations
