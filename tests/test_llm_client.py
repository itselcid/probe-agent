"""Tests for the LLM client abstraction and factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from probe_agent.errors import LLMRateLimitError, LLMResponseError
from probe_agent.llm_client import LLMProvider, create_llm_provider
from probe_agent.providers.anthropic_provider import AnthropicProvider
from probe_agent.providers.gemini import GeminiProvider
from probe_agent.providers.openai_provider import OpenAIProvider
from probe_agent.types import LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestCreateLLMProvider:
    """Tests for :func:`create_llm_provider`."""

    def test_creates_gemini_provider(self) -> None:
        """Factory returns GeminiProvider for provider='gemini'."""
        provider = create_llm_provider(provider="gemini", api_key="fake-key")
        assert isinstance(provider, GeminiProvider)
        assert isinstance(provider, LLMProvider)
        assert provider.provider_name() == "gemini"
        assert provider.model_name == "gemini-2.5-flash"

    def test_creates_openai_provider(self) -> None:
        """Factory returns OpenAIProvider for provider='openai'."""
        provider = create_llm_provider(provider="openai", api_key="sk-fake")
        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name() == "openai"
        assert provider.model_name == "gpt-4o"

    def test_creates_anthropic_provider(self) -> None:
        """Factory returns AnthropicProvider for provider='anthropic'."""
        provider = create_llm_provider(provider="anthropic", api_key="sk-ant-fake")
        assert isinstance(provider, AnthropicProvider)
        assert provider.provider_name() == "anthropic"
        assert provider.model_name == "claude-sonnet-4-20250514"

    def test_custom_model_name(self) -> None:
        """Factory passes custom model_name through to the provider."""
        provider = create_llm_provider(
            provider="gemini",
            api_key="fake",
            model_name="gemini-2.5-pro",
        )
        assert isinstance(provider, GeminiProvider)
        assert provider.model_name == "gemini-2.5-pro"

    def test_unknown_provider_raises_value_error(self) -> None:
        """Factory raises ValueError for unrecognised provider names."""
        with pytest.raises(ValueError, match="Unknown provider: 'deepseek'"):
            create_llm_provider(provider="deepseek", api_key="x")

    def test_rate_limiter_forwarded(self) -> None:
        """Factory passes rate_limiter through to the provider."""
        from probe_agent.rate_limiter import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(rpm=10, tpm=50_000)
        provider = create_llm_provider(
            provider="gemini", api_key="fake", rate_limiter=limiter
        )
        assert provider._rate_limiter is limiter


# ---------------------------------------------------------------------------
# Gemini provider tests (mocked — no real API calls)
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    """Tests for :class:`GeminiProvider` with mocked google-generativeai SDK."""

    @pytest.mark.asyncio
    async def test_chat_returns_text_response(self) -> None:
        """Mocked generate_content returns text → LLMResponse.content is set."""
        provider = create_llm_provider(provider="gemini", api_key="fake-key")

        # Build a mock response object that mimics Gemini's structure.
        mock_part = MagicMock()
        mock_part.text = "Container nginx is healthy."
        mock_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 50
        mock_usage.candidates_token_count = 20
        mock_usage.total_token_count = 70

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        with patch("probe_agent.providers.gemini.genai") as mock_genai:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model

            result = await provider.chat(
                messages=[{"role": "user", "content": "Check health"}],
                tools=[],
                system="You are an SRE.",
            )

        assert isinstance(result, LLMResponse)
        assert result.content == "Container nginx is healthy."
        assert result.tool_calls == []
        assert result.usage["prompt_tokens"] == 50
        assert result.usage["completion_tokens"] == 20
        assert result.usage["total_tokens"] == 70

    @pytest.mark.asyncio
    async def test_chat_returns_function_call_response(self) -> None:
        """Mocked generate_content returns function_call → tool_calls populated."""
        provider = create_llm_provider(provider="gemini", api_key="fake-key")

        mock_fc = MagicMock()
        mock_fc.name = "docker_ps"
        mock_fc.args = {"all": True}

        mock_part = MagicMock()
        mock_part.text = ""
        mock_part.function_call = mock_fc

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 10
        mock_usage.total_token_count = 110

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        with patch("probe_agent.providers.gemini.genai") as mock_genai:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model

            result = await provider.chat(
                messages=[{"role": "user", "content": "List containers"}],
                tools=[{
                    "name": "docker_ps",
                    "description": "List running Docker containers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "all": {"type": "boolean", "description": "Show all containers"},
                        },
                    },
                }],
            )

        assert isinstance(result, LLMResponse)
        assert result.content is None  # No text, just tool calls.
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.name == "docker_ps"
        assert tc.arguments == {"all": True}
        assert tc.id  # Should have a generated UUID.

    @pytest.mark.asyncio
    async def test_rate_limit_error_mapping(self) -> None:
        """ResourceExhausted from google SDK → LLMRateLimitError."""
        from google.api_core import exceptions as google_exceptions

        provider = create_llm_provider(provider="gemini", api_key="fake-key")

        with patch("probe_agent.providers.gemini.genai") as mock_genai:
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = (
                google_exceptions.ResourceExhausted("429 quota exceeded")
            )
            mock_genai.GenerativeModel.return_value = mock_model

            with pytest.raises(LLMRateLimitError) as exc_info:
                # Disable retry for this test by calling the inner method.
                # We use max_attempts=1 on a fresh provider to avoid retries.
                provider_no_retry = GeminiProvider("fake-key", "gemini-2.5-flash")
                # Call the underlying function directly to bypass @retry.
                await provider_no_retry.chat.__wrapped__(
                    provider_no_retry,
                    messages=[{"role": "user", "content": "test"}],
                    tools=[],
                )

            assert exc_info.value.context["provider"] == "gemini"

    @pytest.mark.asyncio
    async def test_token_usage_tracking(self) -> None:
        """Usage metadata from the API response is captured in LLMResponse.usage."""
        provider = create_llm_provider(provider="gemini", api_key="fake-key")

        mock_part = MagicMock()
        mock_part.text = "Done."
        mock_part.function_call = None

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 200
        mock_usage.candidates_token_count = 50
        mock_usage.total_token_count = 250

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        with patch("probe_agent.providers.gemini.genai") as mock_genai:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model

            result = await provider.chat(
                messages=[{"role": "user", "content": "summarise"}],
                tools=[],
            )

        assert result.usage["prompt_tokens"] == 200
        assert result.usage["completion_tokens"] == 50
        assert result.usage["total_tokens"] == 250


# ---------------------------------------------------------------------------
# Stub provider tests
# ---------------------------------------------------------------------------


class TestStubProviders:
    """Verify that stub providers raise NotImplementedError with guidance."""

    @pytest.mark.asyncio
    async def test_openai_stub_raises(self) -> None:
        """OpenAIProvider.chat raises NotImplementedError."""
        provider = create_llm_provider(provider="openai", api_key="sk-fake")
        with pytest.raises(NotImplementedError, match="OpenAI provider not yet implemented"):
            await provider.chat(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
            )

    @pytest.mark.asyncio
    async def test_anthropic_stub_raises(self) -> None:
        """AnthropicProvider.chat raises NotImplementedError."""
        provider = create_llm_provider(provider="anthropic", api_key="sk-ant-fake")
        with pytest.raises(NotImplementedError, match="Anthropic provider not yet implemented"):
            await provider.chat(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
            )
