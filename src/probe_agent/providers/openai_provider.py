"""OpenAI-compatible provider (works with OpenAI, OpenRouter, Azure OpenAI).

Supports any API that speaks the OpenAI chat-completions format with
function calling.  Set ``OPENAI_BASE_URL`` to use OpenRouter or other
compatible endpoints.

Usage::

    # OpenAI direct
    LLM_PROVIDER=openai LLM_API_KEY=sk-... probe-agent run ...

    # OpenRouter (any model)
    LLM_PROVIDER=openai LLM_API_KEY=sk-or-... OPENAI_BASE_URL=https://openrouter.ai/api/v1 probe-agent run ...
"""

from __future__ import annotations

import json
import os
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

# Default OpenAI endpoint — overridable via OPENAI_BASE_URL env var.
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------


def _messages_to_openai(
    messages: list[dict],
    system: str = "",
) -> list[dict[str, Any]]:
    """Convert provider-agnostic messages to OpenAI format.

    Args:
        messages: Provider-agnostic message list.
        system: System prompt (prepended as a system message).

    Returns:
        List of dicts compatible with OpenAI's chat completions API.
    """
    result: list[dict[str, Any]] = []

    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            result.append({"role": "user", "content": content})

        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            if content:
                entry["content"] = content
            else:
                entry["content"] = None

            # If the assistant message carried tool_calls, include them.
            if "tool_calls" in msg and msg["tool_calls"]:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id", str(uuid.uuid4())),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(
                                tc.get("arguments", {}), default=str,
                            ),
                        },
                    }
                    for tc in msg["tool_calls"]
                ]
            result.append(entry)

        elif role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", str(uuid.uuid4())),
                "content": content,
            })

    return result


def _tools_to_openai(tools: list[dict]) -> list[dict[str, Any]] | None:
    """Convert JSON-Schema tool definitions to OpenAI function tools.

    Args:
        tools: Provider-agnostic tool schemas.

    Returns:
        A list of OpenAI-compatible tool definitions, or ``None`` if empty.
    """
    if not tools:
        return None

    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def _parse_openai_response(data: dict[str, Any]) -> LLMResponse:
    """Extract an :class:`LLMResponse` from OpenAI's JSON response.

    Args:
        data: The parsed JSON from the chat completions endpoint.

    Returns:
        A normalised :class:`LLMResponse`.
    """
    choice = data["choices"][0]
    message = choice["message"]

    content = message.get("content")
    tool_calls: list[ToolCall] = []

    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            fn = tc["function"]
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                arguments = {}

            tool_calls.append(
                ToolCall(
                    name=fn["name"],
                    id=tc.get("id", str(uuid.uuid4())),
                    arguments=arguments,
                )
            )

    usage: dict[str, Any] = {}
    if "usage" in data and data["usage"]:
        u = data["usage"]
        usage = {
            "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
        }

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider.

    Works with:
    - OpenAI API (``gpt-4o``, ``o3-mini``, etc.)
    - OpenRouter (set ``OPENAI_BASE_URL=https://openrouter.ai/api/v1``)
    - Azure OpenAI (set ``OPENAI_BASE_URL`` to your Azure endpoint)
    - Any OpenAI-compatible API
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        """Initialise the OpenAI provider.

        Args:
            api_key: API key (OpenAI, OpenRouter, etc.).
            model_name: Model name (e.g. ``"gpt-4o"``).
            rate_limiter: Optional token-bucket limiter.
        """
        self._api_key = api_key
        self.model_name = model_name
        self._rate_limiter = rate_limiter
        self._base_url = os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL)

    def provider_name(self) -> str:
        """Return ``'openai'``."""
        return "openai"

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
        """Send a chat request to an OpenAI-compatible endpoint.

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
        openai_messages = _messages_to_openai(messages, system)
        openai_tools = _tools_to_openai(tools)

        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": openai_messages,
        }
        if openai_tools:
            body["tools"] = openai_tools

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # OpenRouter requires HTTP-Referer header
        if "openrouter" in self._base_url:
            headers["HTTP-Referer"] = "https://github.com/probe-agent"
            headers["X-Title"] = "ProbeAgent"

        # 3. Call the API
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            raise LLMResponseError(
                reason=f"Request timed out: {exc}",
                context={"provider": "openai", "model": self.model_name},
            ) from exc
        except Exception as exc:
            raise LLMResponseError(
                reason=str(exc),
                context={"provider": "openai", "model": self.model_name},
            ) from exc

        duration_ms = (time.monotonic() - start) * 1000

        # 4. Handle HTTP errors
        if resp.status_code == 429:
            raise LLMRateLimitError(
                context={"provider": "openai", "model": self.model_name},
            )
        if resp.status_code in (400, 413):
            error_text = resp.text
            if "context" in error_text.lower() or "token" in error_text.lower():
                raise LLMContextOverflowError(
                    context={"provider": "openai", "model": self.model_name},
                )
            raise LLMResponseError(
                reason=error_text,
                context={"provider": "openai", "model": self.model_name},
            )
        if resp.status_code >= 400:
            raise LLMResponseError(
                reason=f"HTTP {resp.status_code}: {resp.text[:500]}",
                context={"provider": "openai", "model": self.model_name},
            )

        # 5. Parse
        data = resp.json()
        llm_response = _parse_openai_response(data)

        # 6. Log
        log.info(
            "llm_chat_complete",
            provider="openai",
            model=self.model_name,
            duration_ms=round(duration_ms, 1),
            has_tool_calls=len(llm_response.tool_calls) > 0,
            usage=llm_response.usage,
        )

        return llm_response
