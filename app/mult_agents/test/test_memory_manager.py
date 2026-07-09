"""测试：MemoryManager（内存后端，不连外部服务）。

覆盖 MemoryManager 的核心路径，使用 short_term_backend="memory" 和
long_term_backend="disabled" 跳过所有 Redis/Milvus/PostgreSQL 连接。
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from mult_agents.memory import MemoryManager
from mult_agents.memory.base import MemoryType


@pytest.fixture
def mem() -> MemoryManager:
    """最小化 MemoryManager，不连任何外部服务。"""
    return MemoryManager(
        short_term_backend="memory",
        long_term_backend="disabled",
        enable_milvus=False,
        short_term_ttl=3600,
    )


class TestMemoryManagerInit:
    def test_init_without_external_services(self, mem):
        """不连 Redis/PG/Milvus，走内存降级"""
        assert mem.short_term is not None
        assert mem._redis_client is None
        assert mem._milvus_store is None


class TestShortTermMemory:
    def test_add_and_get_single_message(self, mem):
        mem.add_short_term_message("t1", HumanMessage(content="你好"), user_id="u1")
        msgs = mem.get_short_term_messages("t1", user_id="u1", include_summary=False)
        assert len(msgs) == 1
        assert msgs[0].content == "你好"

    def test_add_and_get_multiple_messages(self, mem):
        mem.add_short_term_message("t1", HumanMessage(content="你好"), user_id="u1")
        mem.add_short_term_message("t1", AIMessage(content="有什么可以帮您"), user_id="u1")
        msgs = mem.get_short_term_messages("t1", user_id="u1", include_summary=False)
        assert len(msgs) == 2
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)

    def test_get_empty_thread(self, mem):
        msgs = mem.get_short_term_messages("nonexistent", user_id="u1", include_summary=False)
        assert msgs == []

    def test_last_n_filter(self, mem):
        mem.add_short_term_message("t1", HumanMessage(content="m1"), user_id="u1")
        mem.add_short_term_message("t1", HumanMessage(content="m2"), user_id="u1")
        mem.add_short_term_message("t1", HumanMessage(content="m3"), user_id="u1")
        msgs = mem.get_short_term_messages("t1", user_id="u1", include_summary=False, last_n=2)
        assert len(msgs) == 2
        assert msgs[-1].content == "m3"

    def test_clear_thread(self, mem):
        mem.add_short_term_message("t1", HumanMessage(content="a"), user_id="u1")
        assert mem.clear_short_term("t1") is True
        msgs = mem.get_short_term_messages("t1", user_id="u1", include_summary=False)
        assert msgs == []

    def test_list_active_threads(self, mem):
        mem.add_short_term_message("t1", HumanMessage(content="a"), user_id="u1")
        mem.add_short_term_message("t2", HumanMessage(content="b"), user_id="u1")
        threads = mem.list_active_threads()
        assert "t1" in threads
        assert "t2" in threads

    def test_thread_isolation(self, mem):
        """不同线程互不干扰"""
        mem.add_short_term_message("t1", HumanMessage(content="线程1"), user_id="u1")
        mem.add_short_term_message("t2", HumanMessage(content="线程2"), user_id="u1")
        msgs1 = mem.get_short_term_messages("t1", user_id="u1", include_summary=False)
        assert msgs1[0].content == "线程1"
        msgs2 = mem.get_short_term_messages("t2", user_id="u1", include_summary=False)
        assert msgs2[0].content == "线程2"

    def test_clear_nonexistent_thread_returns_false(self, mem):
        assert mem.clear_short_term("nonexistent") is False

    def test_should_inject_long_term_on_new_thread(self, mem):
        """新线程应有长时记忆注入机会"""
        assert mem.should_inject_long_term("u1", "new-thread") is True

    def test_should_inject_long_term_false_after_message(self, mem):
        mem.add_short_term_message("t1", HumanMessage(content="a"), user_id="u1")
        assert mem.should_inject_long_term("u1", "t1") is False

    def test_add_short_term_messages_batch(self, mem):
        mem.add_short_term_messages(
            "t1",
            [HumanMessage(content="h1"), AIMessage(content="a1")],
            user_id="u1",
        )
        msgs = mem.get_short_term_messages("t1", user_id="u1", include_summary=False)
        assert len(msgs) == 2


class TestShortTermSummary:
    def test_summary_empty_on_new_thread(self, mem):
        summary = mem.get_short_term_summary("new-thread", user_id="u1")
        assert summary == ""

    def test_summary_after_messages(self, mem):
        mem.add_short_term_message("t1", HumanMessage(content="我喜欢Python"), user_id="u1")
        mem.add_short_term_message("t1", HumanMessage(content="也喜欢Go"), user_id="u1")
        # 短期记忆阈值较高（默认20），不会触发自动摘要
        summary = mem.get_short_term_summary("t1", user_id="u1")
        # 内存模式下，摘要为空（未触发压缩）
        assert summary == ""


class TestLongTermDisabled:
    def test_get_user_profile_returns_none(self, mem):
        assert mem.get_user_profile("u1") is None

    def test_save_user_profile_falls_through(self, mem):
        mem_id = mem.save_user_profile("u1", {"language": "zh"}, tenant_id="test")
        assert mem_id is not None

    def test_get_memory_stats(self, mem):
        stats = mem.get_memory_stats("u1")
        assert "short_term" in stats
        assert "modes" in stats
        assert stats["modes"]["long_term"] == "disabled"


class TestSerializeRoundTrip:
    def test_human_message_round_trip(self, mem):
        mem.add_short_term_message("t1", HumanMessage(content="测试"), user_id="u1")
        msgs = mem.get_short_term_messages("t1", user_id="u1", include_summary=False)
        assert isinstance(msgs[0], HumanMessage)

    def test_ai_message_round_trip(self, mem):
        mem.add_short_term_message("t1", AIMessage(content="回复"), user_id="u1")
        msgs = mem.get_short_term_messages("t1", user_id="u1", include_summary=False)
        assert isinstance(msgs[0], AIMessage)

    def test_system_message_round_trip(self, mem):
        mem.add_short_term_message("t1", SystemMessage(content="指令"), user_id="u1")
        msgs = mem.get_short_term_messages("t1", user_id="u1", include_summary=False)
        assert isinstance(msgs[0], SystemMessage)


class TestMarkInjectionSkipped:
    def test_mark_skipped_sets_trace(self, mem):
        mem.mark_injection_skipped("tenant1", "u1", "t1", "query", "reason")
        trace = mem.get_last_trace()
        assert trace["skipped"] is True
        assert trace["skip_reason"] == "reason"
        assert trace["query"] == "query"
