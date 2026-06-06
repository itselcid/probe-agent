"""Pydantic data models shared across ProbeAgent.

These models define the canonical shapes for tool results, LLM interactions,
sub-agent outcomes, and top-level agent runs.  They are used both for
runtime validation and as documentation of the data contracts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Outcome of a single tool invocation.

    Attributes:
        success: ``True`` if the tool completed without error.
        data: Arbitrary payload returned by the tool.  For LLM consumption
            this is typically a string (per the agent-tool-builder skill
            guideline: *return strings, not objects*).
        error: Human-readable error description if ``success`` is ``False``.
        duration_ms: Wall-clock execution time in milliseconds.
    """

    success: bool
    data: Any = None
    error: str | None = None
    duration_ms: float = Field(
        ...,
        description="Tool execution time in milliseconds.",
        ge=0.0,
    )


class ToolCall(BaseModel):
    """A single tool call requested by the LLM.

    Attributes:
        name: Registered tool name (must exist in the tool registry).
        id: Unique call identifier assigned by the LLM, used to correlate
            results with requests.
        arguments: Key/value arguments to pass to the tool function.
    """

    name: str = Field(
        ...,
        description="Tool name as registered in the tool registry.",
    )
    id: str = Field(
        ...,
        description="Unique call identifier assigned by the LLM.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to forward to the tool function.",
    )


class LLMResponse(BaseModel):
    """Parsed response from an LLM provider.

    Attributes:
        content: Text content of the response, if any.  Will be ``None``
            when the model only emits tool calls.
        tool_calls: Zero or more tool calls the model wants executed.
        usage: Token usage statistics (prompt, completion, total).
    """

    content: str | None = None
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool calls requested by the LLM.",
    )
    usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Token usage statistics from the API response.",
    )


class SubagentResult(BaseModel):
    """Outcome of a delegated sub-agent run.

    Attributes:
        subagent: Identifier or name of the sub-agent.
        result: Textual summary of what the sub-agent accomplished.
        steps: Number of agentic steps (tool calls) the sub-agent took.
        tools_used: Deduplicated list of tool names invoked during the run.
        success: ``True`` if the sub-agent completed its task successfully.
    """

    subagent: str
    result: str
    steps: int = Field(..., ge=0)
    tools_used: list[str] = Field(default_factory=list)
    success: bool


class AgentResult(BaseModel):
    """Top-level result of a full ProbeAgent run.

    Attributes:
        task: The original task description supplied by the user.
        final_response: The agent's final answer or summary.
        steps: Total number of tool calls made during the run.
        total_tokens: Cumulative token usage across all LLM calls.
        tools_used: Deduplicated list of tool names invoked during the run.
        success: ``True`` if the agent completed the task without fatal errors.
    """

    task: str
    final_response: str
    steps: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    tools_used: list[str] = Field(default_factory=list)
    success: bool
