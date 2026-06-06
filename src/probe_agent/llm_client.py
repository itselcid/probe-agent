"""Abstract LLM provider interface and factory function.

The agent loop calls :meth:`LLMProvider.chat` without knowing which vendor
is behind it.  Adding a new provider means implementing two methods in a new
file under ``providers/``, then registering it in :func:`create_llm_provider`.

Example::

    provider = create_llm_provider(
        provider="gemini",
        api_key="AIza...",
        rate_limiter=my_limiter,
    )
    response = await provider.chat(messages, tools, system="You are an SRE.")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probe_agent.rate_limiter import TokenBucketRateLimiter
    from probe_agent.types import LLMResponse


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Any LLM that supports function calling can be plugged in by
    implementing this interface.  The agent loop never knows which
    provider it's talking to.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
    ) -> LLMResponse:
        """Send messages with tool definitions, get back text or tool calls.

        Args:
            messages: Provider-**agnostic** message format::

                [
                    {"role": "user",      "content": "..."},
                    {"role": "assistant", "content": "..."},
                    {"role": "tool",      "name": "...", "tool_call_id": "...", "content": "..."},
                ]

            tools: Tool definitions in standard JSON Schema format::

                [
                    {
                        "name": "docker_ps",
                        "description": "List running containers.",
                        "parameters": {
                            "type": "object",
                            "properties": { ... },
                        },
                    },
                ]

            system: System instruction for the model.

        Returns:
            An :class:`~probe_agent.types.LLMResponse` with either ``content``
            (text) or ``tool_calls``, plus token usage stats.
        """

    @abstractmethod
    def provider_name(self) -> str:
        """Return the canonical provider identifier.

        Examples: ``"gemini"``, ``"openai"``, ``"anthropic"``.
        """


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Default model per provider.
_DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
}


def create_llm_provider(
    provider: str = "gemini",
    api_key: str = "",
    model_name: str | None = None,
    rate_limiter: TokenBucketRateLimiter | None = None,
) -> LLMProvider:
    """Instantiate the right :class:`LLMProvider` based on *provider* name.

    Provider modules are imported lazily so that optional SDK dependencies
    (``openai``, ``anthropic``) are only required when actually selected.

    Args:
        provider: One of ``"gemini"``, ``"openai"``, ``"anthropic"``.
        api_key: API key for the chosen provider.
        model_name: Model name override.  ``None`` selects the provider's
            default (see :data:`_DEFAULT_MODELS`).
        rate_limiter: Optional :class:`TokenBucketRateLimiter` shared across
            calls to enforce RPM / TPM limits.

    Returns:
        A concrete :class:`LLMProvider` instance.

    Raises:
        ValueError: If *provider* is not recognised.
    """
    resolved_model = model_name or _DEFAULT_MODELS.get(provider, "")

    if provider == "gemini":
        from probe_agent.providers.gemini import GeminiProvider

        return GeminiProvider(api_key, resolved_model, rate_limiter)

    if provider == "openai":
        from probe_agent.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key, resolved_model, rate_limiter)

    if provider == "anthropic":
        from probe_agent.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key, resolved_model, rate_limiter)

    raise ValueError(
        f"Unknown provider: {provider!r}. "
        "Use 'gemini', 'openai', or 'anthropic'."
    )
