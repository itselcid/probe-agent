"""Configuration module for ProbeAgent.

Uses ``pydantic-settings`` to load configuration from environment variables.
Every setting can be overridden via an env var of the same name (case-insensitive).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from the environment.

    Attributes:
        google_api_key: Google Gemini API key.  **Required** — the agent cannot
            start without it.
        project_path: Filesystem path to the project the agent operates on.
            Defaults to the current working directory.
        log_level: Logging verbosity.  Accepts any Python logging level name
            (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``).
        max_steps: Maximum number of tool calls the agent will make in a single
            run before it stops.  Acts as a safety guard against infinite loops.
        model_name: Name of the Gemini model to use for reasoning.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    google_api_key: str
    """Gemini API key.  Must be set via ``GOOGLE_API_KEY`` env var."""

    project_path: str = "."
    """Path to the project under observation."""

    log_level: str = "INFO"
    """Logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``)."""

    max_steps: int = 30
    """Maximum tool calls per agent run."""

    model_name: str = "gemini-2.5-flash"
    """Gemini model name for LLM calls."""


def load_settings() -> Settings:
    """Create and return a validated :class:`Settings` instance.

    Raises:
        pydantic.ValidationError: If required environment variables (e.g.
            ``GOOGLE_API_KEY``) are missing.

    Returns:
        A fully-validated ``Settings`` object.
    """
    return Settings()  # type: ignore[call-arg]
