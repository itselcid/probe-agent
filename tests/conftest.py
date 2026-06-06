"""Shared test fixtures for ProbeAgent."""

from __future__ import annotations

import pytest

from probe_agent.config import Settings


@pytest.fixture()
def settings() -> Settings:
    """Return a ``Settings`` instance with a dummy API key for testing.

    This avoids requiring a real ``GOOGLE_API_KEY`` env var in CI.
    """
    return Settings(google_api_key="test-key-not-real")  # type: ignore[call-arg]
