"""Tests for the core agent loop and context manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from probe_agent.agent import ProbeAgent
from probe_agent.config import Settings
from probe_agent.context import ContextManager
from probe_agent.registry import ToolRegistry
from probe_agent.types import AgentResult, LLMResponse, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides) -> Settings:
    """Create a Settings with test defaults."""
    defaults = {
        "llm_provider": "gemini",
        "llm_api_key": "test-key",
        "project_path": "/tmp/test-project",
        "max_steps": 10,
        "log_level": "WARNING",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_registry() -> ToolRegistry:
    """Create a registry with one dummy tool for testing."""
    registry = ToolRegistry()

    async def dummy_tool(name: str = "world") -> dict:
        return {"greeting": f"hello {name}"}

    registry.register(
        namespace="test",
        name="greet",
        fn=dummy_tool,
        description="A dummy greeting tool used for testing the agent loop.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to greet"},
            },
        },
    )
    return registry


def _text_response(content: str, tokens: int = 100) -> LLMResponse:
    """Create an LLMResponse with text only (no tool calls)."""
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage={"total_tokens": tokens},
    )


def _tool_call_response(
    tool_name: str,
    args: dict,
    call_id: str = "call_001",
    tokens: int = 50,
) -> LLMResponse:
    """Create an LLMResponse that requests a single tool call."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(name=tool_name, id=call_id, arguments=args)],
        usage={"total_tokens": tokens},
    )


# ===========================================================================
# ProbeAgent tests
# ===========================================================================


class TestProbeAgentTextAnswer:
    """Agent completes in 1 step when the LLM gives a text answer."""

    @pytest.mark.asyncio
    async def test_single_step_text_response(self) -> None:
        """LLM returns text immediately → 1 step, success=True."""
        llm = AsyncMock()
        llm.chat.return_value = _text_response("Everything looks healthy!")

        agent = ProbeAgent(
            config=_make_settings(),
            registry=_make_registry(),
            llm=llm,
        )

        result = await agent.run("check system health")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.steps == 1
        assert result.final_response == "Everything looks healthy!"
        assert result.total_tokens == 100
        assert llm.chat.call_count == 1


class TestProbeAgentToolThenText:
    """Agent completes in 2 steps: tool call → text answer."""

    @pytest.mark.asyncio
    async def test_tool_call_then_text(self) -> None:
        """LLM calls a tool, then answers → 2 steps."""
        llm = AsyncMock()
        llm.chat.side_effect = [
            _tool_call_response("test_greet", {"name": "agent"}),
            _text_response("Greeting complete: hello agent"),
        ]

        agent = ProbeAgent(
            config=_make_settings(),
            registry=_make_registry(),
            llm=llm,
        )

        result = await agent.run("greet the agent")

        assert result.success is True
        assert result.steps == 2
        assert result.final_response == "Greeting complete: hello agent"
        assert "test_greet" in result.tools_used
        assert result.total_tokens == 150  # 50 (tool call) + 100 (text)
        assert llm.chat.call_count == 2


class TestProbeAgentMaxSteps:
    """Agent stops when max_steps is reached."""

    @pytest.mark.asyncio
    async def test_stops_at_max_steps(self) -> None:
        """LLM keeps calling tools → agent stops after max_steps."""
        llm = AsyncMock()
        # Always return tool calls — never give a text answer.
        llm.chat.return_value = _tool_call_response(
            "test_greet", {"name": "loop"}, tokens=10,
        )

        agent = ProbeAgent(
            config=_make_settings(max_steps=3),
            registry=_make_registry(),
            llm=llm,
        )

        result = await agent.run("infinite loop test")

        assert result.steps == 3
        assert result.success is False
        assert result.final_response == ""

    @pytest.mark.asyncio
    async def test_single_max_step(self) -> None:
        """With max_steps=1 and a tool call, agent exits after 1 step."""
        llm = AsyncMock()
        llm.chat.return_value = _tool_call_response(
            "test_greet", {"name": "once"},
        )

        agent = ProbeAgent(
            config=_make_settings(max_steps=1),
            registry=_make_registry(),
            llm=llm,
        )

        result = await agent.run("one step only")
        assert result.steps == 1
        assert result.success is False


class TestProbeAgentSystemPrompt:
    """System prompt includes project path and tool namespaces."""

    def test_system_prompt_content(self) -> None:
        """System prompt references the project and available namespaces."""
        agent = ProbeAgent(
            config=_make_settings(project_path="/srv/myapp"),
            registry=_make_registry(),
            llm=AsyncMock(),
        )

        prompt = agent._build_system_prompt()

        assert "/srv/myapp" in prompt
        assert "test_*" in prompt
        assert "ProbeAgent" in prompt
        assert "1 tools" in prompt


class TestProbeAgentMultipleToolCalls:
    """LLM can request multiple tool calls in one turn."""

    @pytest.mark.asyncio
    async def test_multiple_parallel_tool_calls(self) -> None:
        """LLM asks for two tools in one turn."""
        llm = AsyncMock()
        llm.chat.side_effect = [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(name="test_greet", id="c1", arguments={"name": "alice"}),
                    ToolCall(name="test_greet", id="c2", arguments={"name": "bob"}),
                ],
                usage={"total_tokens": 80},
            ),
            _text_response("Greeted both alice and bob."),
        ]

        agent = ProbeAgent(
            config=_make_settings(),
            registry=_make_registry(),
            llm=llm,
        )

        result = await agent.run("greet alice and bob")
        assert result.success is True
        assert result.steps == 2
        assert "test_greet" in result.tools_used


# ===========================================================================
# ContextManager tests
# ===========================================================================


class TestContextManagerMessages:
    """Tests for adding and retrieving messages."""

    def test_user_message(self) -> None:
        """User messages are stored correctly."""
        ctx = ContextManager(max_messages=10)
        ctx.add_user_message("hello")
        msgs = ctx.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    def test_model_message(self) -> None:
        """Model messages are stored correctly."""
        ctx = ContextManager(max_messages=10)
        ctx.add_model_message("I'll check that.")
        msgs = ctx.get_messages()
        assert msgs[0]["role"] == "assistant"

    def test_tool_call_and_result(self) -> None:
        """Tool calls and results are stored in the right format."""
        ctx = ContextManager(max_messages=10)

        response = LLMResponse(
            content="",
            tool_calls=[ToolCall(name="fs_read", id="tc1", arguments={"path": "/x"})],
            usage={},
        )
        ctx.add_model_tool_calls(response)

        result = ToolResult(success=True, data={"content": "file data"}, duration_ms=5.0)
        ctx.add_tool_result("fs_read", result, tool_call_id="tc1")

        msgs = ctx.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["tool_calls"][0]["name"] == "fs_read"
        assert msgs[1]["role"] == "tool"
        assert msgs[1]["name"] == "fs_read"
        assert msgs[1]["tool_call_id"] == "tc1"


class TestContextManagerTruncation:
    """Tests for tool result truncation."""

    def test_truncates_large_results(self) -> None:
        """Tool results larger than 3000 chars are truncated."""
        ctx = ContextManager(max_messages=10)
        big_data = {"content": "x" * 5000}
        result = ToolResult(success=True, data=big_data, duration_ms=1.0)
        ctx.add_tool_result("big_tool", result)

        msgs = ctx.get_messages()
        content = msgs[0]["content"]
        assert "[truncated]" in content
        assert len(content) < 3200  # 3000 + truncation notice

    def test_small_results_not_truncated(self) -> None:
        """Small tool results are kept in full."""
        ctx = ContextManager(max_messages=10)
        result = ToolResult(success=True, data={"ok": True}, duration_ms=1.0)
        ctx.add_tool_result("small_tool", result)

        msgs = ctx.get_messages()
        assert "[truncated]" not in msgs[0]["content"]


class TestContextManagerRollingSummary:
    """Tests for context windowing and rolling summary."""

    def test_short_history_returns_all(self) -> None:
        """History ≤ max_messages returns everything."""
        ctx = ContextManager(max_messages=5)
        for i in range(5):
            ctx.add_user_message(f"msg {i}")

        msgs = ctx.get_messages()
        assert len(msgs) == 5

    def test_long_history_returns_window(self) -> None:
        """History > max_messages returns only the last N."""
        ctx = ContextManager(max_messages=3)
        for i in range(10):
            ctx.add_user_message(f"msg {i}")

        msgs = ctx.get_messages()
        # 3 recent messages (no summary yet, so no prefix).
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg 7"
        assert msgs[-1]["content"] == "msg 9"

    def test_rolling_summary_prepended(self) -> None:
        """When rolling_summary is set, it's prepended to the window."""
        ctx = ContextManager(max_messages=3)
        ctx.rolling_summary = "Previously: checked logs, found OOM errors."

        for i in range(10):
            ctx.add_user_message(f"msg {i}")

        msgs = ctx.get_messages()
        # 2 prefix (summary + ack) + 3 recent = 5
        assert len(msgs) == 5
        assert "[CONTEXT FROM PREVIOUS STEPS]" in msgs[0]["content"]
        assert "OOM errors" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["content"] == "msg 7"

    @pytest.mark.asyncio
    async def test_maybe_summarize_calls_llm(self) -> None:
        """maybe_summarize sends old messages to the LLM for compression."""
        ctx = ContextManager(max_messages=3)

        # Add enough messages to trigger summarisation.
        for i in range(10):
            ctx.add_user_message(f"step {i}")
        # Simulate 5 tool calls (triggers on multiples of 5).
        ctx._tools_used = ["a", "b", "c", "d", "e"]

        mock_llm = AsyncMock()
        mock_llm.chat.return_value = LLMResponse(
            content="Summary: inspected 10 things.",
            tool_calls=[],
            usage={"total_tokens": 20},
        )

        await ctx.maybe_summarize(mock_llm)

        assert ctx.rolling_summary == "Summary: inspected 10 things."
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_maybe_summarize_skips_when_short(self) -> None:
        """maybe_summarize does nothing when history is short."""
        ctx = ContextManager(max_messages=20)
        ctx.add_user_message("only one message")
        ctx._tools_used = ["a", "b", "c", "d", "e"]

        mock_llm = AsyncMock()
        await ctx.maybe_summarize(mock_llm)

        mock_llm.chat.assert_not_called()
        assert ctx.rolling_summary == ""


class TestContextManagerTokenTracking:
    """Tests for token tracking."""

    def test_accumulates_tokens(self) -> None:
        """track_tokens accumulates total_tokens from usage dicts."""
        ctx = ContextManager()
        ctx.track_tokens({"total_tokens": 100})
        ctx.track_tokens({"total_tokens": 50})
        ctx.track_tokens({"prompt_tokens": 30})  # no total_tokens key
        assert ctx.total_tokens == 150

    def test_tools_used_deduplicated(self) -> None:
        """get_tools_used returns unique names."""
        ctx = ContextManager()
        ctx._tools_used = ["fs_read", "fs_read", "docker_ps", "fs_read"]
        assert ctx.get_tools_used() == ["docker_ps", "fs_read"]
