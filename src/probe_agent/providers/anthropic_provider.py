"""Anthropic Claude provider (stub).

This module provides the architectural slot for Anthropic's Claude API.
The implementation is deferred — the class raises :class:`NotImplementedError`
with guidance on using the default Gemini provider in the meantime.

Once implemented, this provider will support:
- Anthropic API (``claude-sonnet-4-20250514``, ``claude-opus-4-20250514``, etc.)
- Tool use / function calling via Anthropic's native format
"""

from __future__ import annotations

from probe_agent.llm_client import LLMProvider
from probe_agent.rate_limiter import TokenBucketRateLimiter
from probe_agent.types import LLMResponse


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider.

    .. warning::
        Not yet implemented.  Use ``provider="gemini"`` (free tier) for now.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-sonnet-4-20250514",
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        """Store configuration for future use.

        Args:
            api_key: Anthropic API key.
            model_name: Model name (e.g. ``"claude-sonnet-4-20250514"``).
            rate_limiter: Optional token-bucket limiter.
        """
        self._api_key = api_key
        self.model_name = model_name
        self._rate_limiter = rate_limiter

    def provider_name(self) -> str:
        """Return ``'anthropic'``."""
        return "anthropic"

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
    ) -> LLMResponse:
        """Not yet implemented.

        Raises:
            NotImplementedError: Always — this provider is a stub.
        """
        raise NotImplementedError(
            "Anthropic provider not yet implemented. "
            "Use 'gemini' (free tier) for now.  "
            "Set LLM_PROVIDER=gemini in your environment."
        )
