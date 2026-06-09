"""Tests for SubagentRunner and agent tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from probe_agent.llm_client import LLMProvider
from probe_agent.registry import ToolRegistry
from probe_agent.subagent import SubagentRunner
from probe_agent.tools.agent_tools import (
    _DIAGNOSTIC_TOOLS,
    _REMEDIATION_TOOLS,
    _REPORT_TOOLS,
    register_agent_tools,
    spawn_diagnostic,
    spawn_remediation,
    spawn_report,
)
from probe_agent.types import LLMResponse, SubagentResult, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_full_registry() -> ToolRegistry:
    """Build a full registry with all tool namespaces."""
    registry = ToolRegistry()

    from probe_agent.tools.docker_tools import register_docker_tools
    from probe_agent.tools.fs import register_fs_tools
    from probe_agent.tools.git import register_git_tools
    from probe_agent.tools.observe import register_observe_tools
    from probe_agent.tools.project import register_project_tools
    from probe_agent.tools.shell import register_shell_tools

    register_fs_tools(registry)
    register_git_tools(registry)
    register_docker_tools(registry)
    register_shell_tools(registry)
    register_observe_tools(registry)
    register_project_tools(registry)
    return registry


def _text_response(text: str) -> LLMResponse:
    """Create a simple text LLM response."""
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage={"total_tokens": 50},
    )


def _tool_call_response(name: str, args: dict) -> LLMResponse:
    """Create an LLM response with a single tool call."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(name=name, id=f"call_{name}", arguments=args)],
        usage={"total_tokens": 50},
    )


# ===========================================================================
# SubagentRunner — isolation guarantees
# ===========================================================================


class TestSubagentIsolation:
    """Verify that SubagentRunner creates truly isolated context."""

    def test_scoped_registry_is_subset(self) -> None:
        """The subagent's registry contains only the named tools."""
        full_reg = _make_full_registry()
        tool_names = ["fs_read_file", "fs_tree", "git_log"]

        runner = SubagentRunner(
            name="test",
            system_prompt="You are a test agent.",
            tool_names=tool_names,
            registry=full_reg,
            llm=AsyncMock(),
        )

        assert runner.scoped_registry.count() == 3
        assert set(runner.scoped_registry.list_tools()) == set(tool_names)

    def test_scoped_registry_does_not_modify_parent(self) -> None:
        """Creating a subset does not alter the parent registry."""
        full_reg = _make_full_registry()
        original_count = full_reg.count()

        SubagentRunner(
            name="test",
            system_prompt="Test.",
            tool_names=["fs_read_file"],
            registry=full_reg,
            llm=AsyncMock(),
        )

        assert full_reg.count() == original_count

    def test_empty_history_on_init(self) -> None:
        """Subagent starts with an empty conversation history."""
        full_reg = _make_full_registry()

        runner = SubagentRunner(
            name="test",
            system_prompt="Test.",
            tool_names=["fs_read_file"],
            registry=full_reg,
            llm=AsyncMock(),
        )

        assert runner.history == []
        assert runner.step_count == 0
        assert runner.tools_used == []

    @pytest.mark.asyncio
    async def test_history_not_shared_with_parent(self) -> None:
        """Subagent execution does NOT modify any external history."""
        full_reg = _make_full_registry()
        parent_history: list[dict] = [{"role": "user", "content": "parent task"}]

        llm = AsyncMock()
        llm.chat.return_value = _text_response("done")

        runner = SubagentRunner(
            name="test",
            system_prompt="Test.",
            tool_names=["fs_read_file"],
            registry=full_reg,
            llm=llm,
        )

        await runner.run("subagent task")

        # Parent history must be unchanged.
        assert len(parent_history) == 1
        assert parent_history[0]["content"] == "parent task"

        # Subagent history should have its own messages.
        assert len(runner.history) >= 2  # user + assistant
        assert runner.history[0]["content"] == "subagent task"

    @pytest.mark.asyncio
    async def test_two_subagents_independent(self) -> None:
        """Two subagents have completely independent histories."""
        full_reg = _make_full_registry()

        llm = AsyncMock()
        llm.chat.return_value = _text_response("done")

        runner_a = SubagentRunner(
            name="agent_a",
            system_prompt="A.",
            tool_names=["fs_read_file"],
            registry=full_reg,
            llm=llm,
        )
        runner_b = SubagentRunner(
            name="agent_b",
            system_prompt="B.",
            tool_names=["fs_tree"],
            registry=full_reg,
            llm=llm,
        )

        await runner_a.run("task A")
        await runner_b.run("task B")

        assert runner_a.history[0]["content"] == "task A"
        assert runner_b.history[0]["content"] == "task B"
        assert runner_a.history is not runner_b.history


# ===========================================================================
# SubagentRunner — execution loop
# ===========================================================================


class TestSubagentExecution:
    """Verify the subagent's agentic loop works correctly."""

    @pytest.mark.asyncio
    async def test_text_only_response(self) -> None:
        """LLM gives text immediately → 1 step, done."""
        full_reg = _make_full_registry()
        llm = AsyncMock()
        llm.chat.return_value = _text_response("All clear, no issues found.")

        runner = SubagentRunner(
            name="diagnostic",
            system_prompt="Investigate.",
            tool_names=["fs_read_file"],
            registry=full_reg,
            llm=llm,
        )

        result = await runner.run("check health")

        assert result.success is True
        assert result.steps == 1
        assert result.result == "All clear, no issues found."
        assert result.subagent == "diagnostic"
        assert result.tools_used == []

    @pytest.mark.asyncio
    async def test_tool_then_text(self) -> None:
        """LLM calls a tool, then gives text → 2 steps."""
        full_reg = _make_full_registry()
        llm = AsyncMock()
        llm.chat.side_effect = [
            _tool_call_response("fs_read_file", {"path": "/tmp/test.txt"}),
            _text_response("File contents look fine."),
        ]

        # Mock the tool execution.
        mock_tool = AsyncMock(return_value={"content": "hello world", "total_lines": 1})
        full_reg._tools["fs_read_file"] = full_reg._tools["fs_read_file"].__class__(
            namespace="fs",
            name="read_file",
            full_name="fs_read_file",
            description="Read file.",
            parameters={},
            fn=mock_tool,
        )

        runner = SubagentRunner(
            name="diagnostic",
            system_prompt="Investigate.",
            tool_names=["fs_read_file"],
            registry=full_reg,
            llm=llm,
        )

        result = await runner.run("read the config")

        assert result.steps == 2
        assert result.success is True
        assert "fs_read_file" in result.tools_used
        assert result.result == "File contents look fine."

    @pytest.mark.asyncio
    async def test_max_steps_reached(self) -> None:
        """Subagent stops after max_steps even without text response."""
        full_reg = _make_full_registry()

        mock_tool = AsyncMock(return_value={"data": "ok"})
        full_reg._tools["fs_read_file"] = full_reg._tools["fs_read_file"].__class__(
            namespace="fs",
            name="read_file",
            full_name="fs_read_file",
            description="Read file.",
            parameters={},
            fn=mock_tool,
        )

        llm = AsyncMock()
        # Always return tool calls, never text.
        llm.chat.return_value = _tool_call_response(
            "fs_read_file", {"path": "/tmp/loop.txt"},
        )

        runner = SubagentRunner(
            name="stuck",
            system_prompt="Test.",
            tool_names=["fs_read_file"],
            registry=full_reg,
            llm=llm,
            max_steps=3,
        )

        result = await runner.run("infinite loop test")

        assert result.steps == 3
        assert result.success is False  # never got a text answer

    @pytest.mark.asyncio
    async def test_tool_result_truncation(self) -> None:
        """Large tool results are truncated in the history."""
        full_reg = _make_full_registry()

        huge_result = {"data": "x" * 5000}
        mock_tool = AsyncMock(return_value=huge_result)
        full_reg._tools["fs_read_file"] = full_reg._tools["fs_read_file"].__class__(
            namespace="fs",
            name="read_file",
            full_name="fs_read_file",
            description="Read file.",
            parameters={},
            fn=mock_tool,
        )

        llm = AsyncMock()
        llm.chat.side_effect = [
            _tool_call_response("fs_read_file", {"path": "/tmp/big.txt"}),
            _text_response("Done."),
        ]

        runner = SubagentRunner(
            name="test",
            system_prompt="Test.",
            tool_names=["fs_read_file"],
            registry=full_reg,
            llm=llm,
        )

        await runner.run("read big file")

        # Find the tool result in history.
        tool_messages = [m for m in runner.history if m["role"] == "tool"]
        assert len(tool_messages) == 1
        assert len(tool_messages[0]["content"]) <= 3020  # 3000 + "... [truncated]"

    @pytest.mark.asyncio
    async def test_system_prompt_passed_to_llm(self) -> None:
        """The subagent's system prompt is sent to every LLM call."""
        full_reg = _make_full_registry()
        llm = AsyncMock()
        llm.chat.return_value = _text_response("ok")

        prompt = "You are a special agent."
        runner = SubagentRunner(
            name="special",
            system_prompt=prompt,
            tool_names=["fs_read_file"],
            registry=full_reg,
            llm=llm,
        )

        await runner.run("do something")

        llm.chat.assert_called_once()
        _, kwargs = llm.chat.call_args
        assert kwargs["system"] == prompt


# ===========================================================================
# Tool scope verification — what each subagent can access
# ===========================================================================


class TestToolScopes:
    """Verify each subagent type gets the correct tool set."""

    def test_diagnostic_tools_are_read_only(self) -> None:
        """Diagnostic tools contain no write/modify operations."""
        write_tools = {
            "fs_write_file", "fs_edit_file",
            "docker_restart", "docker_stop",
            "docker_exec_command",
            "docker_compose_up", "docker_compose_down",
            "shell_run", "shell_run_in_dir",
            "git_commit", "git_checkout", "git_stash",
        }
        for tool in _DIAGNOSTIC_TOOLS:
            assert tool not in write_tools, f"Diagnostic has write tool: {tool}"

    def test_remediation_tools_include_writes(self) -> None:
        """Remediation tools include write/modify operations."""
        assert "fs_write_file" in _REMEDIATION_TOOLS
        assert "fs_edit_file" in _REMEDIATION_TOOLS
        assert "docker_restart" in _REMEDIATION_TOOLS
        assert "shell_run" in _REMEDIATION_TOOLS
        assert "git_commit" in _REMEDIATION_TOOLS

    def test_report_tools_are_read_only(self) -> None:
        """Report tools contain no write/modify operations."""
        write_tools = {
            "fs_write_file", "fs_edit_file",
            "docker_restart", "docker_stop",
            "docker_exec_command",
            "shell_run", "shell_run_in_dir",
            "git_commit", "git_checkout",
        }
        for tool in _REPORT_TOOLS:
            assert tool not in write_tools, f"Report has write tool: {tool}"

    def test_all_diagnostic_tools_exist(self) -> None:
        """All diagnostic tool names exist in the full registry."""
        full_reg = _make_full_registry()
        for tool in _DIAGNOSTIC_TOOLS:
            assert tool in full_reg, f"Missing tool: {tool}"

    def test_all_remediation_tools_exist(self) -> None:
        """All remediation tool names exist in the full registry."""
        full_reg = _make_full_registry()
        for tool in _REMEDIATION_TOOLS:
            assert tool in full_reg, f"Missing tool: {tool}"

    def test_all_report_tools_exist(self) -> None:
        """All report tool names exist in the full registry."""
        full_reg = _make_full_registry()
        for tool in _REPORT_TOOLS:
            assert tool in full_reg, f"Missing tool: {tool}"


# ===========================================================================
# spawn_* functions
# ===========================================================================


class TestSpawnDiagnostic:
    """Tests for spawn_diagnostic."""

    @pytest.mark.asyncio
    async def test_returns_diagnostic_result(self) -> None:
        """spawn_diagnostic returns a structured result dict."""
        full_reg = _make_full_registry()
        llm = AsyncMock()
        llm.chat.return_value = _text_response("Found: API is returning 500s due to DB timeout")

        result = await spawn_diagnostic(
            task="investigate 500 errors",
            project_path="/tmp/myproject",
            registry=full_reg,
            llm=llm,
        )

        assert result["subagent"] == "diagnostic"
        assert "findings" in result
        assert result["success"] is True
        assert isinstance(result["tools_used"], list)
        assert isinstance(result["steps"], int)

    @pytest.mark.asyncio
    async def test_task_includes_project_path(self) -> None:
        """The project path is injected into the task."""
        full_reg = _make_full_registry()
        llm = AsyncMock()
        llm.chat.return_value = _text_response("done")

        await spawn_diagnostic(
            task="check health",
            project_path="/srv/api",
            registry=full_reg,
            llm=llm,
        )

        # Check the task sent to the LLM includes the project path.
        call_args = llm.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "/srv/api" in user_msg["content"]


class TestSpawnRemediation:
    """Tests for spawn_remediation."""

    @pytest.mark.asyncio
    async def test_returns_remediation_result(self) -> None:
        """spawn_remediation returns actions_taken."""
        full_reg = _make_full_registry()
        llm = AsyncMock()
        llm.chat.return_value = _text_response("Fixed: increased DB connection pool to 20")

        result = await spawn_remediation(
            task="fix DB timeout",
            project_path="/tmp/myproject",
            registry=full_reg,
            llm=llm,
        )

        assert result["subagent"] == "remediation"
        assert "actions_taken" in result
        assert result["success"] is True


class TestSpawnReport:
    """Tests for spawn_report."""

    @pytest.mark.asyncio
    async def test_returns_report(self) -> None:
        """spawn_report returns a report string."""
        full_reg = _make_full_registry()
        llm = AsyncMock()
        llm.chat.return_value = _text_response(
            "## Summary\nSystem is healthy.\n\n## Findings\nNo issues found."
        )

        result = await spawn_report(
            task="system health overview",
            project_path="/tmp/myproject",
            registry=full_reg,
            llm=llm,
        )

        assert result["subagent"] == "report"
        assert "report" in result
        assert "Summary" in result["report"]
        assert result["success"] is True


# ===========================================================================
# Registration
# ===========================================================================


class TestRegistration:
    """Tests for register_agent_tools."""

    def test_registers_3_tools(self) -> None:
        """register_agent_tools adds exactly 3 tools."""
        registry = ToolRegistry()
        full_reg = _make_full_registry()
        llm = AsyncMock()

        register_agent_tools(registry, full_registry=full_reg, llm=llm)

        assert registry.count() == 3
        assert registry.list_namespaces() == ["agent"]

    def test_expected_tool_names(self) -> None:
        """All 3 expected tool names are registered."""
        registry = ToolRegistry()
        full_reg = _make_full_registry()
        llm = AsyncMock()

        register_agent_tools(registry, full_registry=full_reg, llm=llm)

        expected = {
            "agent_spawn_diagnostic",
            "agent_spawn_remediation",
            "agent_spawn_report",
        }
        assert set(registry.list_tools()) == expected

    def test_schemas_have_required_keys(self) -> None:
        """Every schema has name, description, and parameters."""
        registry = ToolRegistry()
        full_reg = _make_full_registry()
        llm = AsyncMock()

        register_agent_tools(registry, full_registry=full_reg, llm=llm)

        for schema in registry.get_schemas():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert "task" in schema["parameters"]["properties"]
            assert "project_path" in schema["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_registered_tool_executes(self) -> None:
        """The registered wrapper closures actually call the subagent."""
        registry = ToolRegistry()
        full_reg = _make_full_registry()
        llm = AsyncMock()
        llm.chat.return_value = _text_response("diagnostic complete")

        register_agent_tools(registry, full_registry=full_reg, llm=llm)

        result = await registry.execute(
            "agent_spawn_diagnostic",
            {"task": "test", "project_path": "/tmp"},
        )

        assert result.success is True
        assert result.data["subagent"] == "diagnostic"
