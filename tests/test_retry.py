"""Tests for the @retry decorator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from probe_agent.retry import retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TransientError(Exception):
    """Retryable exception used in tests."""


class FatalError(Exception):
    """Non-retryable exception used in tests."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_on_first_try() -> None:
    """If the function succeeds immediately, no retries happen."""
    mock = AsyncMock(return_value="ok")

    @retry(max_attempts=3, retryable_exceptions=(TransientError,))
    async def fn() -> str:
        return await mock()

    result = await fn()

    assert result == "ok"
    assert mock.await_count == 1


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failure() -> None:
    """The decorator retries and eventually succeeds."""
    call_count = 0

    @retry(
        max_attempts=3,
        base_delay=0.01,
        retryable_exceptions=(TransientError,),
    )
    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TransientError("boom")
        return "recovered"

    result = await fn()

    assert result == "recovered"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_gives_up_after_max_attempts() -> None:
    """After exhausting all attempts the last exception propagates."""
    call_count = 0

    @retry(
        max_attempts=3,
        base_delay=0.01,
        retryable_exceptions=(TransientError,),
    )
    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        raise TransientError(f"attempt {call_count}")

    with pytest.raises(TransientError, match="attempt 3"):
        await fn()

    assert call_count == 3


@pytest.mark.asyncio
async def test_non_retryable_exception_raises_immediately() -> None:
    """Exceptions not in retryable_exceptions propagate without retry."""
    call_count = 0

    @retry(
        max_attempts=5,
        base_delay=0.01,
        retryable_exceptions=(TransientError,),
    )
    async def fn() -> str:
        nonlocal call_count
        call_count += 1
        raise FatalError("fatal")

    with pytest.raises(FatalError, match="fatal"):
        await fn()

    # Should have been called exactly once — no retries.
    assert call_count == 1


@pytest.mark.asyncio
async def test_backoff_delay_increases() -> None:
    """Each retry should wait longer than the previous one.

    We capture wall-clock timestamps to verify the delays grow.
    Using very small base_delay so the test stays fast.
    """
    timestamps: list[float] = []

    @retry(
        max_attempts=4,
        base_delay=0.05,
        max_delay=10.0,
        backoff_factor=2.0,
        retryable_exceptions=(TransientError,),
    )
    async def fn() -> str:
        timestamps.append(asyncio.get_event_loop().time())
        raise TransientError("always fails")

    with pytest.raises(TransientError):
        await fn()

    assert len(timestamps) == 4

    # Compute inter-attempt delays.
    delays = [
        timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
    ]

    # Each delay should be roughly larger than the previous.
    # We allow generous tolerance because jitter adds randomness.
    for i in range(1, len(delays)):
        assert delays[i] > delays[i - 1] * 0.5, (
            f"Delay {i} ({delays[i]:.4f}s) should be larger than "
            f"delay {i-1} ({delays[i-1]:.4f}s)"
        )
