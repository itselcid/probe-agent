"""Typed error hierarchy for ProbeAgent.

Every error carries a human-readable ``message`` and an optional ``context``
dict with structured metadata (tool name, HTTP status, etc.).  The hierarchy
mirrors the agent's subsystems so callers can catch at the right level of
granularity.

Hierarchy
---------
::

    AgentError
    ├── ToolError
    │   ├── ToolNotFoundError
    │   ├── ToolExecutionError
    │   ├── ToolTimeoutError
    │   └── ToolValidationError
    ├── LLMError
    │   ├── LLMRateLimitError
    │   ├── LLMContextOverflowError
    │   └── LLMResponseError
    ├── SubagentError
    └── DiscoveryError
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Base exception for all ProbeAgent errors.

    Attributes:
        message: A human-readable description of the error.
        context: Arbitrary structured metadata (tool name, request id, etc.).
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        self.message = message
        self.context: dict[str, Any] = context or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        ctx = f", context={self.context!r}" if self.context else ""
        return f"{type(self).__name__}({self.message!r}{ctx})"


# ---------------------------------------------------------------------------
# Tool errors
# ---------------------------------------------------------------------------

class ToolError(AgentError):
    """Base class for errors raised during tool execution."""


class ToolNotFoundError(ToolError):
    """The requested tool name does not exist in the registry.

    Attributes:
        tool_name: The name that was looked up.
    """

    def __init__(self, tool_name: str, context: dict[str, Any] | None = None) -> None:
        self.tool_name = tool_name
        super().__init__(
            message=f"Tool not found: {tool_name!r}",
            context={"tool_name": tool_name, **(context or {})},
        )


class ToolExecutionError(ToolError):
    """The tool function raised an unhandled exception during execution.

    Attributes:
        tool_name: Name of the tool that failed.
        cause: The original exception, if available.
    """

    def __init__(
        self,
        tool_name: str,
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(
            message=f"Tool {tool_name!r} raised {type(cause).__name__}: {cause}" if cause
            else f"Tool {tool_name!r} failed",
            context={"tool_name": tool_name, **(context or {})},
        )


class ToolTimeoutError(ToolError):
    """The tool did not complete within its allowed time budget.

    Attributes:
        tool_name: Name of the tool that timed out.
        timeout_seconds: The configured timeout.
    """

    def __init__(
        self,
        tool_name: str,
        timeout_seconds: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            message=f"Tool {tool_name!r} timed out after {timeout_seconds}s",
            context={"tool_name": tool_name, "timeout_seconds": timeout_seconds, **(context or {})},
        )


class ToolValidationError(ToolError):
    """Invalid arguments were passed to a tool.

    Attributes:
        tool_name: Name of the tool.
        validation_errors: List of validation error descriptions.
    """

    def __init__(
        self,
        tool_name: str,
        validation_errors: list[str],
        context: dict[str, Any] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.validation_errors = validation_errors
        super().__init__(
            message=f"Validation failed for tool {tool_name!r}: {'; '.join(validation_errors)}",
            context={
                "tool_name": tool_name,
                "validation_errors": validation_errors,
                **(context or {}),
            },
        )


# ---------------------------------------------------------------------------
# LLM errors
# ---------------------------------------------------------------------------

class LLMError(AgentError):
    """Base class for errors from the Gemini LLM API."""


class LLMRateLimitError(LLMError):
    """Gemini API returned HTTP 429 — rate limit exceeded.

    Attributes:
        retry_after_seconds: Suggested wait time before retrying, if provided
            by the API.
    """

    def __init__(
        self,
        retry_after_seconds: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        msg = "Gemini API rate limit exceeded"
        if retry_after_seconds is not None:
            msg += f" (retry after {retry_after_seconds}s)"
        super().__init__(
            message=msg,
            context={"retry_after_seconds": retry_after_seconds, **(context or {})},
        )


class LLMContextOverflowError(LLMError):
    """The input exceeded the model's maximum context length.

    Attributes:
        token_count: Number of tokens in the request, if known.
        max_tokens: Model's maximum context window, if known.
    """

    def __init__(
        self,
        token_count: int | None = None,
        max_tokens: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.token_count = token_count
        self.max_tokens = max_tokens
        parts = ["Input exceeds model context window"]
        if token_count is not None:
            parts.append(f"tokens={token_count}")
        if max_tokens is not None:
            parts.append(f"max={max_tokens}")
        super().__init__(
            message=", ".join(parts),
            context={
                "token_count": token_count,
                "max_tokens": max_tokens,
                **(context or {}),
            },
        )


class LLMResponseError(LLMError):
    """The Gemini API returned an unexpected or malformed response.

    Attributes:
        status_code: HTTP status code, if applicable.
        raw_response: The raw response body, truncated for safety.
    """

    def __init__(
        self,
        reason: str,
        status_code: int | None = None,
        raw_response: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.raw_response = raw_response
        super().__init__(
            message=f"Unexpected LLM response: {reason}",
            context={
                "status_code": status_code,
                "raw_response": (raw_response[:500] if raw_response else None),
                **(context or {}),
            },
        )


# ---------------------------------------------------------------------------
# Subagent & Discovery errors
# ---------------------------------------------------------------------------

class SubagentError(AgentError):
    """A delegated sub-agent failed to complete its task.

    Attributes:
        subagent_name: Identifier of the sub-agent that failed.
    """

    def __init__(
        self,
        subagent_name: str,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.subagent_name = subagent_name
        super().__init__(
            message=f"Subagent {subagent_name!r} failed: {reason}",
            context={"subagent_name": subagent_name, **(context or {})},
        )


class DiscoveryError(AgentError):
    """Project discovery or introspection failed.

    Raised when the agent cannot inspect the target project — e.g. the path
    does not exist, is not a valid project, or required files are missing.

    Attributes:
        project_path: The path that was being inspected.
    """

    def __init__(
        self,
        project_path: str,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.project_path = project_path
        super().__init__(
            message=f"Discovery failed for {project_path!r}: {reason}",
            context={"project_path": project_path, **(context or {})},
        )
