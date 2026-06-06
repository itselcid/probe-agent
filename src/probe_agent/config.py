"""Configuration module for ProbeAgent.

Uses ``pydantic-settings`` to load configuration from environment variables.
Every setting can be overridden via an env var of the same name (case-insensitive).

Switch LLM provider with environment variables::

    LLM_PROVIDER=gemini   LLM_API_KEY=AIza...  probe-agent --project . "task"
    LLM_PROVIDER=openai   LLM_API_KEY=sk-...   probe-agent --project . "task"
    LLM_PROVIDER=anthropic LLM_API_KEY=sk-ant-  probe-agent --project . "task"
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from the environment.

    Attributes:
        llm_provider: Which LLM backend to use (``"gemini"``, ``"openai"``,
            ``"anthropic"``).
        llm_api_key: API key for the selected provider.  **Required**.
        llm_model: Model name override.  ``None`` uses the provider's default
            (gemini → ``gemini-2.5-flash``, openai → ``gpt-4o``,
            anthropic → ``claude-sonnet-4-20250514``).
        project_path: Filesystem path to the project the agent operates on.
            Defaults to the current working directory.
        log_level: Logging verbosity.  Accepts any Python logging level name
            (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``).
        max_steps: Maximum number of tool calls the agent will make in a single
            run before it stops.  Acts as a safety guard against infinite loops.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    llm_provider: str = "gemini"
    """LLM provider identifier (``"gemini"``, ``"openai"``, ``"anthropic"``)."""

    llm_api_key: str
    """API key for the selected LLM provider.  Must be set via ``LLM_API_KEY`` env var."""

    llm_model: str | None = None
    """Model name override.  ``None`` uses the provider's default model."""

    project_path: str = "."
    """Path to the project under observation."""

    log_level: str = "INFO"
    """Logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``)."""

    max_steps: int = 30
    """Maximum tool calls per agent run."""


def load_settings() -> Settings:
    """Create and return a validated :class:`Settings` instance.

    Raises:
        pydantic.ValidationError: If required environment variables (e.g.
            ``LLM_API_KEY``) are missing.

    Returns:
        A fully-validated ``Settings`` object.
    """
    return Settings()  # type: ignore[call-arg]
