"""Tests for SessionRecorder and logging enhancements."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from probe_agent.logging_setup import bind_session, new_session_id, unbind_session
from probe_agent.session import SessionRecorder
from probe_agent.types import SubagentResult, ToolResult


# ===========================================================================
# SessionRecorder — recording steps
# ===========================================================================


class TestSessionRecorderSteps:
    """Test step recording."""

    def test_record_tool_step(self, tmp_path: Path) -> None:
        """record_step captures tool name, args, result, and timing."""
        recorder = SessionRecorder(session_id="test-001", output_dir=tmp_path)

        result = ToolResult(success=True, data={"status": "ok"}, duration_ms=42.5)
        recorder.record_step(
            step_index=1,
            tool_name="fs_read_file",
            args={"path": "/tmp/test.txt"},
            result=result,
        )

        assert recorder.step_count == 1
        step = recorder.steps[0]
        assert step["type"] == "tool_call"
        assert step["step"] == 1
        assert step["tool"] == "fs_read_file"
        assert step["args"]["path"] == "/tmp/test.txt"
        assert step["success"] is True
        assert step["duration_ms"] == 42.5
        assert "timestamp" in step

    def test_record_failed_step(self, tmp_path: Path) -> None:
        """Failed tool results record the error message."""
        recorder = SessionRecorder(session_id="test-002", output_dir=tmp_path)

        result = ToolResult(success=False, error="File not found", duration_ms=1.0)
        recorder.record_step(
            step_index=3,
            tool_name="fs_read_file",
            args={"path": "/nonexistent"},
            result=result,
        )

        step = recorder.steps[0]
        assert step["success"] is False
        assert step["result_preview"] == "File not found"

    def test_args_truncated(self, tmp_path: Path) -> None:
        """Long argument values are truncated to 200 chars."""
        recorder = SessionRecorder(session_id="test-003", output_dir=tmp_path)

        long_content = "x" * 500
        result = ToolResult(success=True, data="ok", duration_ms=1.0)
        recorder.record_step(
            step_index=1,
            tool_name="fs_write_file",
            args={"path": "/tmp/f.txt", "content": long_content},
            result=result,
        )

        assert len(recorder.steps[0]["args"]["content"]) == 200

    def test_result_preview_truncated(self, tmp_path: Path) -> None:
        """Large result data is truncated to 500 chars in preview."""
        recorder = SessionRecorder(session_id="test-004", output_dir=tmp_path)

        big_data = "y" * 1000
        result = ToolResult(success=True, data=big_data, duration_ms=1.0)
        recorder.record_step(
            step_index=1,
            tool_name="fs_read_file",
            args={"path": "/tmp/big.txt"},
            result=result,
        )

        assert len(recorder.steps[0]["result_preview"]) == 500

    def test_record_multiple_steps(self, tmp_path: Path) -> None:
        """Multiple steps accumulate in order."""
        recorder = SessionRecorder(session_id="test-005", output_dir=tmp_path)

        for i in range(5):
            result = ToolResult(success=True, data=f"result_{i}", duration_ms=float(i))
            recorder.record_step(
                step_index=i + 1,
                tool_name=f"tool_{i}",
                args={"x": str(i)},
                result=result,
            )

        assert recorder.step_count == 5
        assert recorder.steps[0]["step"] == 1
        assert recorder.steps[4]["step"] == 5


# ===========================================================================
# SessionRecorder — subagent recording
# ===========================================================================


class TestSessionRecorderSubagent:
    """Test subagent recording."""

    def test_record_subagent(self, tmp_path: Path) -> None:
        """record_subagent captures subagent metadata."""
        recorder = SessionRecorder(session_id="test-sub-001", output_dir=tmp_path)

        sub_result = SubagentResult(
            subagent="diagnostic",
            result="Found: DB timeout",
            steps=4,
            tools_used=["fs_read_file", "docker_logs"],
            success=True,
        )
        recorder.record_subagent(
            name="diagnostic",
            task="investigate 500 errors",
            result=sub_result,
        )

        assert recorder.step_count == 1
        step = recorder.steps[0]
        assert step["type"] == "subagent"
        assert step["subagent"] == "diagnostic"
        assert step["task"] == "investigate 500 errors"
        assert step["steps"] == 4
        assert step["success"] is True
        assert "timestamp" in step

    def test_subagent_task_truncated(self, tmp_path: Path) -> None:
        """Long subagent task descriptions are truncated to 200 chars."""
        recorder = SessionRecorder(session_id="test-sub-002", output_dir=tmp_path)

        sub_result = SubagentResult(
            subagent="report",
            result="done",
            steps=1,
            tools_used=[],
            success=True,
        )
        recorder.record_subagent(
            name="report",
            task="a" * 500,
            result=sub_result,
        )

        assert len(recorder.steps[0]["task"]) == 200


# ===========================================================================
# SessionRecorder — LLM call recording
# ===========================================================================


class TestSessionRecorderLLMCall:
    """Test LLM call recording."""

    def test_record_llm_call(self, tmp_path: Path) -> None:
        """record_llm_call captures message count, tool count, and usage."""
        recorder = SessionRecorder(session_id="test-llm-001", output_dir=tmp_path)

        recorder.record_llm_call(
            step_index=1,
            message_count=5,
            tool_count=10,
            usage={"total_tokens": 200, "prompt_tokens": 150},
        )

        assert recorder.step_count == 1
        step = recorder.steps[0]
        assert step["type"] == "llm_call"
        assert step["step"] == 1
        assert step["message_count"] == 5
        assert step["tool_count"] == 10
        assert step["usage"]["total_tokens"] == 200


# ===========================================================================
# SessionRecorder — save to disk
# ===========================================================================


class TestSessionRecorderSave:
    """Test JSON persistence."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """save() writes a valid JSON file."""
        recorder = SessionRecorder(session_id="test-save-001", output_dir=tmp_path)

        result = ToolResult(success=True, data="ok", duration_ms=10.0)
        recorder.record_step(1, "fs_tree", {"path": "."}, result)

        path = recorder.save()

        assert path.exists()
        assert path.name == "test-save-001.json"

    def test_save_creates_directories(self, tmp_path: Path) -> None:
        """save() creates parent directories if they don't exist."""
        deep_dir = tmp_path / "a" / "b" / "c"
        recorder = SessionRecorder(session_id="test-deep", output_dir=deep_dir)

        recorder.save()

        assert (deep_dir / "test-deep.json").exists()

    def test_save_json_structure(self, tmp_path: Path) -> None:
        """Saved JSON has all expected top-level fields."""
        recorder = SessionRecorder(session_id="test-json", output_dir=tmp_path)

        result = ToolResult(success=True, data="ok", duration_ms=5.0)
        recorder.record_step(1, "docker_ps", {"all": "True"}, result)
        recorder.record_step(2, "docker_logs", {"container": "api"}, result)

        path = recorder.save()
        data = json.loads(path.read_text())

        assert data["session_id"] == "test-json"
        assert "start_time" in data
        assert "end_time" in data
        assert "total_duration_s" in data
        assert data["total_steps"] == 2
        assert "docker_ps" in data["tools_used"]
        assert "docker_logs" in data["tools_used"]
        assert len(data["steps"]) == 2

    def test_save_duration_is_positive(self, tmp_path: Path) -> None:
        """total_duration_s is non-negative."""
        recorder = SessionRecorder(session_id="test-dur", output_dir=tmp_path)

        path = recorder.save()
        data = json.loads(path.read_text())

        assert data["total_duration_s"] >= 0

    def test_save_empty_session(self, tmp_path: Path) -> None:
        """Saving a session with no steps produces valid JSON."""
        recorder = SessionRecorder(session_id="test-empty", output_dir=tmp_path)

        path = recorder.save()
        data = json.loads(path.read_text())

        assert data["total_steps"] == 0
        assert data["tools_used"] == []
        assert data["steps"] == []

    def test_tools_used_deduplicated(self, tmp_path: Path) -> None:
        """tools_used in the saved JSON is deduplicated."""
        recorder = SessionRecorder(session_id="test-dedup", output_dir=tmp_path)

        result = ToolResult(success=True, data="ok", duration_ms=1.0)
        recorder.record_step(1, "fs_read_file", {"path": "a"}, result)
        recorder.record_step(2, "fs_read_file", {"path": "b"}, result)
        recorder.record_step(3, "docker_ps", {}, result)

        path = recorder.save()
        data = json.loads(path.read_text())

        assert len(data["tools_used"]) == 2  # fs_read_file + docker_ps


# ===========================================================================
# SessionRecorder — introspection properties
# ===========================================================================


class TestSessionRecorderIntrospection:
    """Test helper properties."""

    def test_tool_calls_property(self, tmp_path: Path) -> None:
        """tool_calls filters to only tool_call steps."""
        recorder = SessionRecorder(session_id="test-prop", output_dir=tmp_path)

        result = ToolResult(success=True, data="ok", duration_ms=1.0)
        recorder.record_step(1, "fs_read_file", {"path": "a"}, result)
        recorder.record_llm_call(1, 3, 10, {"total_tokens": 50})

        assert len(recorder.tool_calls) == 1
        assert recorder.tool_calls[0]["tool"] == "fs_read_file"

    def test_subagent_calls_property(self, tmp_path: Path) -> None:
        """subagent_calls filters to only subagent steps."""
        recorder = SessionRecorder(session_id="test-prop2", output_dir=tmp_path)

        sub_result = SubagentResult(
            subagent="diagnostic", result="ok", steps=1,
            tools_used=[], success=True,
        )
        recorder.record_subagent("diagnostic", "check health", sub_result)

        result = ToolResult(success=True, data="ok", duration_ms=1.0)
        recorder.record_step(1, "fs_tree", {}, result)

        assert len(recorder.subagent_calls) == 1
        assert recorder.subagent_calls[0]["subagent"] == "diagnostic"


# ===========================================================================
# Logging — session_id
# ===========================================================================


class TestSessionId:
    """Test session ID generation and binding."""

    def test_new_session_id_is_uuid(self) -> None:
        """new_session_id returns a valid UUID string."""
        import uuid

        sid = new_session_id()
        parsed = uuid.UUID(sid)  # Raises ValueError if invalid.
        assert str(parsed) == sid

    def test_new_session_ids_are_unique(self) -> None:
        """Each call produces a different ID."""
        ids = {new_session_id() for _ in range(100)}
        assert len(ids) == 100

    def test_bind_unbind_session(self) -> None:
        """bind_session and unbind_session don't crash."""
        sid = new_session_id()
        bind_session(sid)
        unbind_session()


# ===========================================================================
# Agent integration — session recording wired in
# ===========================================================================


class TestAgentSessionRecording:
    """Verify the agent creates and saves session recordings."""

    @pytest.mark.asyncio
    async def test_agent_saves_session(self, tmp_path: Path) -> None:
        """ProbeAgent.run() saves a session JSON file."""
        from unittest.mock import AsyncMock

        from probe_agent.agent import ProbeAgent
        from probe_agent.config import Settings
        from probe_agent.registry import ToolRegistry
        from probe_agent.types import LLMResponse

        settings = Settings(
            llm_provider="gemini",
            llm_api_key="fake-key",
            project_path=str(tmp_path),
            max_steps=5,
        )

        registry = ToolRegistry()
        llm = AsyncMock()
        llm.chat.return_value = LLMResponse(
            content="All good.",
            tool_calls=[],
            usage={"total_tokens": 50},
        )

        agent = ProbeAgent(config=settings, registry=registry, llm=llm)
        await agent.run("check health")

        # Session file should exist.
        sessions_dir = tmp_path / ".probe" / "sessions"
        assert sessions_dir.exists()

        files = list(sessions_dir.glob("*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        assert "session_id" in data
        assert data["total_steps"] >= 1  # At least the LLM call

    @pytest.mark.asyncio
    async def test_agent_records_tool_calls(self, tmp_path: Path) -> None:
        """Tool calls appear in the session recording."""
        from unittest.mock import AsyncMock

        from probe_agent.agent import ProbeAgent
        from probe_agent.config import Settings
        from probe_agent.registry import ToolRegistry
        from probe_agent.types import LLMResponse, ToolCall

        settings = Settings(
            llm_provider="gemini",
            llm_api_key="fake-key",
            project_path=str(tmp_path),
            max_steps=5,
        )

        registry = ToolRegistry()

        async def greet(name: str) -> dict:
            return {"greeting": f"hello {name}"}

        registry.register(
            namespace="test",
            name="greet",
            fn=greet,
            description="Greet someone.",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        )

        llm = AsyncMock()
        llm.chat.side_effect = [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(name="test_greet", id="c1", arguments={"name": "world"})],
                usage={"total_tokens": 30},
            ),
            LLMResponse(
                content="Done greeting.",
                tool_calls=[],
                usage={"total_tokens": 20},
            ),
        ]

        agent = ProbeAgent(config=settings, registry=registry, llm=llm)
        await agent.run("greet world")

        sessions_dir = tmp_path / ".probe" / "sessions"
        files = list(sessions_dir.glob("*.json"))
        data = json.loads(files[0].read_text())

        # Should have LLM calls and a tool call step.
        tool_steps = [s for s in data["steps"] if s.get("type") == "tool_call"]
        assert len(tool_steps) == 1
        assert tool_steps[0]["tool"] == "test_greet"
        assert tool_steps[0]["success"] is True
