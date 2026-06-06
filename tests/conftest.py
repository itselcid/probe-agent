"""Shared test fixtures for ProbeAgent."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure probe_agent is importable even without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from probe_agent.config import Settings


@pytest.fixture()
def settings() -> Settings:
    """Return a ``Settings`` instance with a dummy API key for testing.

    This avoids requiring a real ``LLM_API_KEY`` env var in CI.
    """
    return Settings(llm_api_key="test-key-not-real")  # type: ignore[call-arg]
