"""Subagent runner for ProbeAgent.

A :class:`SubagentRunner` is a lightweight, isolated mini-agent.  It runs
its own independent conversation with the LLM, using a restricted set of
tools provided by the parent agent.

Key isolation guarantees:

- **Separate history**: the subagent starts with an empty message list;
  the parent's conversation is never visible.
- **Scoped tools**: only the tools explicitly named in *tool_names* are
  available.  The parent registry is never modified.
- **Bounded steps**: the subagent has its own ``max_steps`` limit.

Register convenience tools that spawn subagents via
:mod:`probe_agent.tools.agent_tools`.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from probe_agent.llm_client import LLMProvider
from probe_agent.registry import ToolRegistry
from probe_agent.types import SubagentResult

log = structlog.get_logger(__name__)


class SubagentRunner:
    """A self-contained mini-agent with isolated context.

    Args:
        name: Human-readable identifier (e.g. ``"diagnostic"``).
        system_prompt: The system prompt that defines this subagent's
            specialisation and behavioural constraints.
        tool_names: List of ``full_name`` strings (e.g. ``"fs_read_file"``)
            that this subagent is allowed to use.  Other tools in the parent
            registry are invisible.
        registry: The **parent** registry.  A scoped subset is created
            automatically — the parent is never modified.
        llm: The LLM provider shared with the parent (stateless, so sharing
            is safe).
        max_steps: Maximum reasoning steps before the subagent stops.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tool_names: list[str],
        registry: ToolRegistry,
        llm: LLMProvider,
        max_steps: int = 15,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.max_steps = max_steps

        # CRITICAL: Create a new registry with only the allowed tools.
        self.scoped_registry = registry.subset(tool_names)

        # CRITICAL: Fresh, empty history — NOT shared with parent.
        self.history: list[dict[str, Any]] = []

        self.tools_used: list[str] = []
        self.step_count: int = 0

        self.logger = structlog.get_logger().bind(subagent=name)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, task: str) -> SubagentResult:
        """Execute the subagent with its own isolated context.

        The loop mirrors :meth:`ProbeAgent.run` but is simpler — no
        rolling summary or token tracking.

        Args:
            task: Natural-language description of the sub-task.

        Returns:
            A :class:`SubagentResult` with findings and metadata.
        """
        self.logger.info("subagent_start", task=task[:200], max_steps=self.max_steps)

        # Seed the isolated history with the task.
        self.history.append({"role": "user", "content": task})

        final_response = ""

        while self.step_count < self.max_steps:
            self.step_count += 1

            tool_schemas = self.scoped_registry.get_schemas()

            self.logger.debug(
                "subagent_llm_call",
                step=self.step_count,
                messages=len(self.history),
                tools=len(tool_schemas),
            )

            # Ask the LLM.
            response = await self.llm.chat(
                messages=list(self.history),
                tools=tool_schemas,
                system=self.system_prompt,
            )

            # --- Case 1: tool calls -----------------------------------
            if response.tool_calls:
                # Record the assistant turn with tool calls.
                self.history.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        for tc in response.tool_calls
                    ],
                })

                for tool_call in response.tool_calls:
                    self.logger.info(
                        "subagent_tool_call",
                        step=self.step_count,
                        tool=tool_call.name,
                        args=tool_call.arguments,
                    )

                    result = await self.scoped_registry.execute(
                        tool_call.name, tool_call.arguments,
                    )

                    # Record tool name.
                    if tool_call.name not in self.tools_used:
                        self.tools_used.append(tool_call.name)

                    # Serialize the result for the LLM.
                    if result.success:
                        result_str = json.dumps(result.data, default=str)
                    else:
                        result_str = f"Error: {result.error}"

                    # Truncate large results.
                    if len(result_str) > 3000:
                        result_str = result_str[:3000] + "\n... [truncated]"

                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })

                    self.logger.info(
                        "subagent_tool_done",
                        step=self.step_count,
                        tool=tool_call.name,
                        success=result.success,
                        duration_ms=round(result.duration_ms, 1),
                    )

            # --- Case 2: text answer — done ---------------------------
            else:
                final_response = response.content or ""
                self.history.append({
                    "role": "assistant",
                    "content": final_response,
                })
                break

        success = self.step_count < self.max_steps or bool(final_response)

        self.logger.info(
            "subagent_end",
            steps=self.step_count,
            tools_used=self.tools_used,
            success=success,
        )

        return SubagentResult(
            subagent=self.name,
            result=final_response,
            steps=self.step_count,
            tools_used=self.tools_used,
            success=success,
        )
