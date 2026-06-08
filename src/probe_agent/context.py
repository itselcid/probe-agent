"""Conversation context manager for ProbeAgent.

Manages the rolling message history sent to the LLM.  After many tool calls
the conversation grows very large; this module keeps the most recent messages
in full detail and compresses older ones into a compact rolling summary.

Why this matters
~~~~~~~~~~~~~~~~
- LLMs have finite context windows.  A 30-step investigation can easily
  generate 50 000 tokens of tool results.
- LLMs lose focus on extremely long conversations — the most recent
  messages carry the most signal.
- Summarising old context into a paragraph lets the agent "remember"
  without paying the full token cost.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from probe_agent.llm_client import LLMProvider
    from probe_agent.types import LLMResponse, ToolResult

log = structlog.get_logger(__name__)

_TOOL_RESULT_MAX_CHARS = 3_000


class ContextManager:
    """Manages conversation history with the LLM.

    Args:
        max_messages: Number of recent messages kept in full detail.
            Older messages are eligible for summarisation.
    """

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages

        # Full message history — provider-agnostic dicts.
        self.full_history: list[dict[str, Any]] = []

        # Compressed summary of messages that have been evicted.
        self.rolling_summary: str = ""

        # Cumulative token counts.
        self.total_tokens: int = 0

        # Every tool name invoked (with duplicates, for counting).
        self._tools_used: list[str] = []

    # ------------------------------------------------------------------
    # Adding messages
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        """Append a user message."""
        self.full_history.append({"role": "user", "content": content})

    def add_model_message(self, content: str) -> None:
        """Append a plain-text assistant message (no tool calls)."""
        self.full_history.append({"role": "assistant", "content": content})

    def add_model_tool_calls(self, response: LLMResponse) -> None:
        """Append the assistant's tool-call request to history.

        Stores tool calls in the provider-agnostic format that
        :meth:`LLMProvider.chat` expects on the next turn.
        """
        tool_calls_data = []
        for tc in response.tool_calls:
            tool_calls_data.append({
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            })

        self.full_history.append({
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": tool_calls_data,
        })

    def add_tool_result(
        self,
        tool_name: str,
        result: ToolResult,
        tool_call_id: str = "",
    ) -> None:
        """Append a tool execution result to history.

        Large results are truncated to prevent context overflow.

        Args:
            tool_name: Name of the tool that was called.
            result: The :class:`ToolResult` returned by the registry.
            tool_call_id: The LLM-assigned call ID for correlation.
        """
        self._tools_used.append(tool_name)

        # Serialise data to a string, truncating if necessary.
        if result.data is not None:
            try:
                data_str = json.dumps(result.data, indent=2, default=str)
            except (TypeError, ValueError):
                data_str = str(result.data)
        else:
            data_str = result.error or "No data"

        if len(data_str) > _TOOL_RESULT_MAX_CHARS:
            data_str = data_str[:_TOOL_RESULT_MAX_CHARS] + "\n... [truncated]"

        self.full_history.append({
            "role": "tool",
            "name": tool_name,
            "tool_call_id": tool_call_id,
            "content": data_str,
        })

    # ------------------------------------------------------------------
    # Retrieving messages for the LLM
    # ------------------------------------------------------------------

    def get_messages(self) -> list[dict[str, Any]]:
        """Get the message list to send to the LLM.

        If history is short (≤ *max_messages*), return everything.
        If history is long, prepend the rolling summary then the last
        *max_messages* messages.
        """
        if len(self.full_history) <= self.max_messages:
            return list(self.full_history)

        messages: list[dict[str, Any]] = []

        if self.rolling_summary:
            messages.append({
                "role": "user",
                "content": (
                    f"[CONTEXT FROM PREVIOUS STEPS]\n{self.rolling_summary}"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": (
                    "I have the context from previous steps. Continuing."
                ),
            })

        messages.extend(self.full_history[-self.max_messages:])
        return messages

    # ------------------------------------------------------------------
    # Context summarisation
    # ------------------------------------------------------------------

    async def maybe_summarize(self, llm: LLMProvider) -> None:
        """Periodically compress old messages into *rolling_summary*.

        Called after tool execution.  Every 5 accumulated tool calls,
        if the history exceeds *max_messages*, the overflow is summarised
        by the LLM into a compact paragraph.
        """
        if (
            len(self.full_history) > self.max_messages
            and len(self._tools_used) % 5 == 0
        ):
            old_messages = self.full_history[: -self.max_messages]
            formatted = self._format_for_summary(old_messages)

            try:
                summary_response = await llm.chat(
                    messages=[{
                        "role": "user",
                        "content": (
                            "Summarize these agent steps concisely. Focus on "
                            "key findings, decisions made, and important "
                            f"results:\n\n{formatted}"
                        ),
                    }],
                    tools=[],
                    system="You are a summarizer. Be concise. Output a paragraph.",
                )
                if summary_response.content:
                    self.rolling_summary = summary_response.content
                    log.info(
                        "context_summarized",
                        old_messages=len(old_messages),
                        summary_len=len(self.rolling_summary),
                    )
            except Exception:
                log.warning("context_summarization_failed", exc_info=True)

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def track_tokens(self, usage: dict[str, Any]) -> None:
        """Accumulate token counts from an LLM response."""
        self.total_tokens += usage.get("total_tokens", 0)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_tools_used(self) -> list[str]:
        """Return a deduplicated list of tool names invoked."""
        return sorted(set(self._tools_used))

    def message_count(self) -> int:
        """Return the current number of messages in history."""
        return len(self.full_history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_for_summary(messages: list[dict[str, Any]]) -> str:
        """Format message dicts into a text block for the summariser."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "tool":
                name = msg.get("name", "unknown")
                # Keep tool results very short in the summary input.
                content = content[:500] if content else ""
                parts.append(f"[TOOL {name}] {content}")
            elif role == "assistant" and msg.get("tool_calls"):
                calls = msg["tool_calls"]
                call_names = ", ".join(c.get("name", "?") for c in calls)
                parts.append(f"[ASSISTANT calls: {call_names}]")
            else:
                parts.append(f"[{role.upper()}] {content[:500]}")

        return "\n".join(parts)
