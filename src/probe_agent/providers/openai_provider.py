"""OpenAI / OpenRouter provider (stub).

This module provides the architectural slot for OpenAI-compatible APIs.
The implementation is deferred — the class raises :class:`NotImplementedError`
with guidance on using the default Gemini provider in the meantime.

Once implemented, this provider will support:
- OpenAI API (``gpt-4o``, ``o3-mini``, etc.)
- OpenRouter (any model via ``OPENAI_BASE_URL`` override)
- Azure OpenAI (via ``OPENAI_API_BASE``)
"""

from __future__ import annotations

from probe_agent.llm_client import LLMProvider
from probe_agent.rate_limiter import TokenBucketRateLimiter
from probe_agent.types import LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI provider.  Works with OpenAI API and OpenRouter.

    .. warning::
        Not yet implemented.  Use ``provider="gemini"`` (free tier) for now.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        """Store configuration for future use.

        Args:
            api_key: OpenAI API key.
            model_name: Model name (e.g. ``"gpt-4o"``).
            rate_limiter: Optional token-bucket limiter.
        """
        self._api_key = api_key
        self.model_name = model_name
        self._rate_limiter = rate_limiter

    def provider_name(self) -> str:
        """Return ``'openai'``."""
        return "openai"

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
            "OpenAI provider not yet implemented. "
            "Use 'gemini' (free tier) for now.  "
            "Set LLM_PROVIDER=gemini in your environment."
        )
