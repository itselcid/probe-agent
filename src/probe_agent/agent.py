"""Core agent loop for ProbeAgent.

:class:`ProbeAgent` implements the main reasoning loop:

1. Send the user's task + tool schemas to the LLM.
2. If the LLM wants to call tools → execute them and send results back.
3. If the LLM gives a text answer → done.
4. Repeat until done or *max_steps* is reached.

The agent never imports a specific LLM SDK; it talks to any provider
through :class:`~probe_agent.llm_client.LLMProvider`.
"""

from __future__ import annotations

from typing import Any

import structlog

from probe_agent.config import Settings
from probe_agent.context import ContextManager
from probe_agent.llm_client import LLMProvider
from probe_agent.registry import ToolRegistry
from probe_agent.types import AgentResult

log = structlog.get_logger(__name__)


class ProbeAgent:
    """Autonomous DevOps / SRE agent.

    Args:
        config: Application settings (project path, max steps, etc.).
        registry: Tool registry containing all available tools.
        llm: The LLM provider to use for reasoning.
    """

    def __init__(
        self,
        config: Settings,
        registry: ToolRegistry,
        llm: LLMProvider,
    ) -> None:
        self.config = config
        self.registry = registry
        self.llm = llm
        self.context_mgr = ContextManager(max_messages=20)
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, task: str) -> AgentResult:
        """Run the agent on a task.

        This is the main agentic loop.  It alternates between LLM
        reasoning turns and tool execution until the LLM produces a
        final text answer or *max_steps* is exhausted.

        Args:
            task: Natural-language description of what the agent should do.

        Returns:
            An :class:`AgentResult` summarising the run.
        """
        system_prompt = self._build_system_prompt()
        self.context_mgr.add_user_message(task)

        final_response = ""

        log.info("agent_loop_start", task=task[:200], max_steps=self.config.max_steps)

        while self.step_count < self.config.max_steps:
            self.step_count += 1

            # Retrieve the (possibly truncated) conversation window.
            messages = self.context_mgr.get_messages()
            tool_schemas = self.registry.get_schemas()

            log.debug(
                "llm_call",
                step=self.step_count,
                messages=len(messages),
                tools=len(tool_schemas),
            )

            # Ask the LLM.
            response = await self.llm.chat(
                messages=messages,
                tools=tool_schemas,
                system=system_prompt,
            )

            # Track token usage.
            self.context_mgr.track_tokens(response.usage)

            # ----- Case 1: tool calls ---------------------------------
            if response.tool_calls:
                self.context_mgr.add_model_tool_calls(response)

                for tool_call in response.tool_calls:
                    log.info(
                        "executing_tool",
                        step=self.step_count,
                        tool=tool_call.name,
                        args=tool_call.arguments,
                    )

                    result = await self.registry.execute(
                        tool_call.name, tool_call.arguments,
                    )

                    self.context_mgr.add_tool_result(
                        tool_call.name, result, tool_call_id=tool_call.id,
                    )

                    log.info(
                        "tool_completed",
                        step=self.step_count,
                        tool=tool_call.name,
                        success=result.success,
                        duration_ms=round(result.duration_ms, 1),
                    )

                # Compress old context when the conversation grows long.
                await self.context_mgr.maybe_summarize(self.llm)

            # ----- Case 2: text answer — done -------------------------
            else:
                final_response = response.content or ""
                self.context_mgr.add_model_message(final_response)
                break

        success = self.step_count < self.config.max_steps or bool(final_response)

        log.info(
            "agent_loop_end",
            steps=self.step_count,
            total_tokens=self.context_mgr.total_tokens,
            tools_used=self.context_mgr.get_tools_used(),
            success=success,
        )

        return AgentResult(
            task=task,
            final_response=final_response,
            steps=self.step_count,
            total_tokens=self.context_mgr.total_tokens,
            tools_used=self.context_mgr.get_tools_used(),
            success=success,
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt that describes the agent's role."""
        namespaces = self.registry.list_namespaces()
        tool_count = self.registry.count()

        ns_list = "\n".join(f"- {ns}_*" for ns in namespaces)

        return (
            "You are ProbeAgent, an autonomous DevOps and SRE assistant.\n"
            "You operate on software projects by inspecting files, containers, "
            "logs, metrics, and code.\n\n"
            f"PROJECT: {self.config.project_path}\n\n"
            f"You have {tool_count} tools available across these namespaces:\n"
            f"{ns_list}\n\n"
            "GUIDELINES:\n"
            "1. Start by understanding the project. Use fs_tree or fs_list_dir first.\n"
            "2. Gather evidence before acting. Read logs and inspect state "
            "before making changes.\n"
            "3. After making changes, ALWAYS verify them (re-run tests, check status).\n"
            "4. Be concise. Focus on findings and actions, not verbose explanations.\n"
        )
