"""Anthropic Claude provider using the Messages API.

Supports Claude models via the Anthropic API using httpx (no SDK dependency).

Usage::

    LLM_PROVIDER=anthropic LLM_API_KEY=sk-ant-... probe-agent run ...
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx
import structlog

from probe_agent.errors import (
    LLMContextOverflowError,
    LLMRateLimitError,
    LLMResponseError,
)
from probe_agent.llm_client import LLMProvider
from probe_agent.rate_limiter import TokenBucketRateLimiter
from probe_agent.retry import retry
from probe_agent.types import LLMResponse, ToolCall

log = structlog.get_logger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------


def _messages_to_anthropic(
    messages: list[dict],
) -> list[dict[str, Any]]:
    """Convert provider-agnostic messages to Anthropic format.

    Anthropic uses a different structure:
    - No "system" role in messages (system is a top-level param)
    - Tool results are sent as user messages with tool_result content blocks
    - Assistant tool calls are content blocks with type "tool_use"

    Args:
        messages: Provider-agnostic message list.

    Returns:
        List of dicts compatible with Anthropic's Messages API.
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            result.append({"role": "user", "content": content})

        elif role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if content:
                content_blocks.append({"type": "text", "text": content})

            # If the assistant message carried tool_calls, add tool_use blocks.
            for tc in msg.get("tool_calls", []):
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", str(uuid.uuid4())),
                    "name": tc["name"],
                    "input": tc.get("arguments", {}),
                })

            result.append({
                "role": "assistant",
                "content": content_blocks if content_blocks else content,
            })

        elif role == "tool":
            # Anthropic expects tool results as user messages with tool_result blocks.
            result.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", str(uuid.uuid4())),
                        "content": content,
                    }
                ],
            })

    return result


def _tools_to_anthropic(tools: list[dict]) -> list[dict[str, Any]] | None:
    """Convert JSON-Schema tool definitions to Anthropic format.

    Args:
        tools: Provider-agnostic tool schemas.

    Returns:
        A list of Anthropic-compatible tool definitions, or ``None`` if empty.
    """
    if not tools:
        return None

    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
        }
        for tool in tools
    ]


def _parse_anthropic_response(data: dict[str, Any]) -> LLMResponse:
    """Extract an :class:`LLMResponse` from Anthropic's JSON response.

    Args:
        data: The parsed JSON from the Messages API.

    Returns:
        A normalised :class:`LLMResponse`.
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in data.get("content", []):
        if block["type"] == "text":
            text_parts.append(block["text"])
        elif block["type"] == "tool_use":
            tool_calls.append(
                ToolCall(
                    name=block["name"],
                    id=block.get("id", str(uuid.uuid4())),
                    arguments=block.get("input", {}),
                )
            )

    usage: dict[str, Any] = {}
    if "usage" in data:
        u = data["usage"]
        usage = {
            "prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": u.get("output_tokens", 0),
            "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
        }

    return LLMResponse(
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using the Messages API.

    Uses httpx directly — no ``anthropic`` SDK dependency required.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-sonnet-4-20250514",
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        """Initialise the Anthropic provider.

        Args:
            api_key: Anthropic API key.
            model_name: Claude model name.
            rate_limiter: Optional token-bucket limiter.
        """
        self._api_key = api_key
        self.model_name = model_name
        self._rate_limiter = rate_limiter

    def provider_name(self) -> str:
        """Return ``'anthropic'``."""
        return "anthropic"

    @retry(
        max_attempts=3,
        base_delay=2.0,
        max_delay=60.0,
        retryable_exceptions=(LLMRateLimitError,),
    )
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
    ) -> LLMResponse:
        """Send a chat request to Anthropic's Messages API.

        See :meth:`LLMProvider.chat` for the full contract.

        Raises:
            LLMRateLimitError: API returned 429.
            LLMContextOverflowError: Input exceeds the model's context.
            LLMResponseError: Any other unexpected failure.
        """
        # 1. Rate limiting
        if self._rate_limiter:
            await self._rate_limiter.acquire(estimated_tokens=1)

        # 2. Build request body
        anthropic_messages = _messages_to_anthropic(messages)
        anthropic_tools = _tools_to_anthropic(tools)

        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": 4096,
        }
        if system:
            body["system"] = system
        if anthropic_tools:
            body["tools"] = anthropic_tools

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

        # 3. Call the API
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    _ANTHROPIC_API_URL,
                    json=body,
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            raise LLMResponseError(
                reason=f"Request timed out: {exc}",
                context={"provider": "anthropic", "model": self.model_name},
            ) from exc
        except Exception as exc:
            raise LLMResponseError(
                reason=str(exc),
                context={"provider": "anthropic", "model": self.model_name},
            ) from exc

        duration_ms = (time.monotonic() - start) * 1000

        # 4. Handle HTTP errors
        if resp.status_code == 429:
            raise LLMRateLimitError(
                context={"provider": "anthropic", "model": self.model_name},
            )
        if resp.status_code == 400:
            error_text = resp.text
            if "context" in error_text.lower() or "token" in error_text.lower():
                raise LLMContextOverflowError(
                    context={"provider": "anthropic", "model": self.model_name},
                )
            raise LLMResponseError(
                reason=error_text,
                context={"provider": "anthropic", "model": self.model_name},
            )
        if resp.status_code >= 400:
            raise LLMResponseError(
                reason=f"HTTP {resp.status_code}: {resp.text[:500]}",
                context={"provider": "anthropic", "model": self.model_name},
            )

        # 5. Parse
        data = resp.json()
        llm_response = _parse_anthropic_response(data)

        # 6. Log
        log.info(
            "llm_chat_complete",
            provider="anthropic",
            model=self.model_name,
            duration_ms=round(duration_ms, 1),
            has_tool_calls=len(llm_response.tool_calls) > 0,
            usage=llm_response.usage,
        )

        return llm_response
