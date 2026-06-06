"""Google Gemini provider using the ``google-generativeai`` SDK.

This is the default (and currently only fully-implemented) provider.  It
translates between ProbeAgent's provider-agnostic message format and Gemini's
native ``Content`` / ``Part`` structures, handles function-calling round-trips,
and maps Gemini-specific exceptions to ProbeAgent's error hierarchy.

Requires:
    ``GOOGLE_API_KEY`` (or pass ``api_key`` to the constructor).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import google.generativeai as genai
import structlog
from google.api_core import exceptions as google_exceptions

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


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------


def _messages_to_gemini_contents(
    messages: list[dict],
) -> list[dict[str, Any]]:
    """Convert provider-agnostic messages to Gemini ``Content`` dicts.

    Mapping rules:
    - ``role="user"``      → Gemini ``role="user"``
    - ``role="assistant"`` → Gemini ``role="model"``
    - ``role="tool"``      → Gemini ``role="user"`` with a
      ``function_response`` Part.

    Args:
        messages: Provider-agnostic message list.

    Returns:
        List of dicts compatible with Gemini's ``contents`` parameter.
    """
    contents: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            contents.append({
                "role": "user",
                "parts": [{"text": content}],
            })

        elif role == "assistant":
            parts: list[dict[str, Any]] = []
            if content:
                parts.append({"text": content})
            # If the assistant message carried tool_calls (echoed back),
            # include them as function_call Parts.
            for tc in msg.get("tool_calls", []):
                parts.append({
                    "function_call": {
                        "name": tc["name"],
                        "args": tc.get("arguments", {}),
                    },
                })
            contents.append({"role": "model", "parts": parts})

        elif role == "tool":
            contents.append({
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": msg.get("name", "unknown"),
                            "response": {"result": content},
                        },
                    },
                ],
            })

    return contents


def _tools_to_gemini_declarations(
    tools: list[dict],
) -> list[dict[str, Any]] | None:
    """Convert JSON-Schema tool definitions to Gemini FunctionDeclarations.

    Args:
        tools: Provider-agnostic tool schemas.

    Returns:
        A list of Gemini-compatible function declarations, or ``None`` if
        *tools* is empty (so no ``tools`` kwarg is sent to the API).
    """
    if not tools:
        return None

    declarations: list[dict[str, Any]] = []
    for tool in tools:
        decl: dict[str, Any] = {
            "name": tool["name"],
            "description": tool.get("description", ""),
        }
        if "parameters" in tool:
            decl["parameters"] = tool["parameters"]
        declarations.append(decl)

    return declarations


def _parse_gemini_response(response: Any) -> LLMResponse:
    """Extract an :class:`LLMResponse` from Gemini's native response object.

    Args:
        response: The object returned by ``model.generate_content()``.

    Returns:
        A normalised :class:`LLMResponse`.
    """
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append(
                    ToolCall(
                        name=fc.name,
                        id=str(uuid.uuid4()),
                        arguments=dict(fc.args) if fc.args else {},
                    )
                )

    usage: dict[str, Any] = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", 0),
            "completion_tokens": getattr(um, "candidates_token_count", 0),
            "total_tokens": getattr(um, "total_token_count", 0),
        }

    return LLMResponse(
        content="\n".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the ``google-generativeai`` SDK.

    Attributes:
        model_name: The Gemini model identifier (e.g. ``"gemini-2.5-flash"``).
        _rate_limiter: Optional rate limiter for RPM/TPM enforcement.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        """Initialise the Gemini provider.

        Args:
            api_key: Google API key.
            model_name: Gemini model name.
            rate_limiter: Optional token-bucket limiter.
        """
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self._rate_limiter = rate_limiter

    def provider_name(self) -> str:
        """Return ``'gemini'``."""
        return "gemini"

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
        """Send a chat request to Gemini and return a normalised response.

        See :meth:`LLMProvider.chat` for the full contract.

        Raises:
            LLMRateLimitError: Gemini returned 429 (resource exhausted).
            LLMContextOverflowError: Input too long for the model.
            LLMResponseError: Any other unexpected API failure.
        """
        # 1. Rate limiting
        if self._rate_limiter:
            await self._rate_limiter.acquire(estimated_tokens=1)

        # 2. Build Gemini-native structures
        contents = _messages_to_gemini_contents(messages)
        declarations = _tools_to_gemini_declarations(tools)

        # 3. Create the model
        model_kwargs: dict[str, Any] = {}
        if system:
            model_kwargs["system_instruction"] = system
        if declarations:
            model_kwargs["tools"] = [{"function_declarations": declarations}]

        model = genai.GenerativeModel(
            model_name=self.model_name,
            **model_kwargs,
        )

        # 4. Call the API
        start = time.monotonic()
        try:
            response = model.generate_content(contents)
        except google_exceptions.ResourceExhausted as exc:
            raise LLMRateLimitError(
                context={"provider": "gemini", "model": self.model_name},
            ) from exc
        except google_exceptions.InvalidArgument as exc:
            if "token" in str(exc).lower() or "context" in str(exc).lower():
                raise LLMContextOverflowError(
                    context={"provider": "gemini", "model": self.model_name},
                ) from exc
            raise LLMResponseError(
                reason=str(exc),
                context={"provider": "gemini", "model": self.model_name},
            ) from exc
        except Exception as exc:
            raise LLMResponseError(
                reason=str(exc),
                context={"provider": "gemini", "model": self.model_name},
            ) from exc

        duration_ms = (time.monotonic() - start) * 1000

        # 5. Parse into our generic format
        llm_response = _parse_gemini_response(response)

        # 6. Log
        log.info(
            "llm_chat_complete",
            provider="gemini",
            model=self.model_name,
            duration_ms=round(duration_ms, 1),
            has_tool_calls=len(llm_response.tool_calls) > 0,
            usage=llm_response.usage,
        )

        return llm_response
