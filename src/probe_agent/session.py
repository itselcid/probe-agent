"""Session recorder for ProbeAgent.

Records every step of an agent session — tool calls, subagent spawns, and
timing data — as a JSON file on disk.  This is essential for:

- **Debugging**: replay exactly what the agent did and why.
- **Evaluation**: measure tool usage, step counts, and success rates.
- **Auditability**: log every mutation for post-mortem analysis.

Session files are written to ``{project_path}/.probe/sessions/{session_id}.json``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from probe_agent.types import SubagentResult, ToolResult

log = structlog.get_logger(__name__)


class SessionRecorder:
    """Records every step of an agent session for debugging and evaluation.

    Args:
        session_id: Unique identifier for this session (typically a UUID).
        output_dir: Directory where the session JSON file will be saved.
    """

    def __init__(self, session_id: str, output_dir: Path) -> None:
        self.session_id = session_id
        self.steps: list[dict[str, Any]] = []
        self.output_dir = output_dir
        self.start_time: float = time.time()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        step_index: int,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
    ) -> None:
        """Record a tool execution step.

        Args:
            step_index: The agent loop step number (1-indexed).
            tool_name: The full name of the tool executed.
            args: Arguments passed to the tool.
            result: The outcome of the tool execution.
        """
        self.steps.append({
            "type": "tool_call",
            "step": step_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "args": {k: str(v)[:200] for k, v in args.items()},
            "success": result.success,
            "duration_ms": round(result.duration_ms, 1),
            "result_preview": (
                str(result.data)[:500] if result.success else result.error
            ),
        })

    def record_subagent(
        self,
        name: str,
        task: str,
        result: SubagentResult,
    ) -> None:
        """Record a subagent invocation.

        Args:
            name: The subagent name (e.g. ``"diagnostic"``).
            task: The task assigned to the subagent.
            result: The subagent's outcome.
        """
        self.steps.append({
            "type": "subagent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "subagent": name,
            "task": task[:200],
            "steps": result.steps,
            "tools_used": result.tools_used,
            "success": result.success,
        })

    def record_llm_call(
        self,
        step_index: int,
        message_count: int,
        tool_count: int,
        usage: dict[str, Any],
    ) -> None:
        """Record an LLM API call.

        Args:
            step_index: The agent loop step number.
            message_count: Number of messages sent to the LLM.
            tool_count: Number of tool schemas sent.
            usage: Token usage statistics from the LLM response.
        """
        self.steps.append({
            "type": "llm_call",
            "step": step_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_count": message_count,
            "tool_count": tool_count,
            "usage": usage,
        })

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        """Serialize the session to a JSON file on disk.

        Returns:
            The :class:`Path` to the written file.
        """
        end_time = time.time()

        # Collect tool usage stats.
        tool_names = [
            s["tool"] for s in self.steps if s.get("type") == "tool_call"
        ]

        output: dict[str, Any] = {
            "session_id": self.session_id,
            "start_time": datetime.fromtimestamp(
                self.start_time, tz=timezone.utc,
            ).isoformat(),
            "end_time": datetime.fromtimestamp(
                end_time, tz=timezone.utc,
            ).isoformat(),
            "total_duration_s": round(end_time - self.start_time, 2),
            "total_steps": len(self.steps),
            "tools_used": sorted(set(tool_names)),
            "steps": self.steps,
        }

        path = self.output_dir / f"{self.session_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, default=str))

        log.info(
            "session_saved",
            session_id=self.session_id,
            path=str(path),
            total_steps=len(self.steps),
        )

        return path

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Total number of recorded steps."""
        return len(self.steps)

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Only the tool-call steps."""
        return [s for s in self.steps if s.get("type") == "tool_call"]

    @property
    def subagent_calls(self) -> list[dict[str, Any]]:
        """Only the subagent steps."""
        return [s for s in self.steps if s.get("type") == "subagent"]
