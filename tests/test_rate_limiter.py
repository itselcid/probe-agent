"""Tests for TokenBucketRateLimiter."""

from __future__ import annotations

import asyncio
import time

import pytest

from probe_agent.rate_limiter import TokenBucketRateLimiter


@pytest.mark.asyncio
async def test_acquire_when_bucket_has_tokens() -> None:
    """Acquiring when capacity is available should return instantly."""
    limiter = TokenBucketRateLimiter(rpm=60, tpm=100_000)

    start = time.monotonic()
    await limiter.acquire(estimated_tokens=100)
    elapsed = time.monotonic() - start

    # Should be effectively instant (well under 100 ms).
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_acquire_blocks_when_empty() -> None:
    """Once the bucket is drained, acquire should block until refill."""
    # Allow 1 request per minute → bucket starts with 1 token.
    limiter = TokenBucketRateLimiter(rpm=60, tpm=1_000_000)

    # Drain the request bucket.
    for _ in range(60):
        await limiter.acquire(estimated_tokens=1)

    # The next acquire must wait for a refill.
    start = time.monotonic()
    await limiter.acquire(estimated_tokens=1)
    elapsed = time.monotonic() - start

    # At 60 RPM the refill rate is 1 token/sec, so we should wait ~1 s.
    # Allow generous tolerance for CI jitter.
    assert elapsed > 0.5, f"Expected a wait, but only waited {elapsed:.3f}s"
    assert elapsed < 3.0, f"Waited too long: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_refill_over_time() -> None:
    """Tokens should refill continuously based on elapsed wall-clock time."""
    # 120 RPM → 2 tokens/sec refill.
    limiter = TokenBucketRateLimiter(rpm=120, tpm=1_000_000)

    # Drain the request bucket completely.
    for _ in range(120):
        await limiter.acquire(estimated_tokens=1)

    # Wait long enough for ~4 tokens to refill (2 tokens/sec × 2 s).
    await asyncio.sleep(2.0)

    # Should be able to acquire a few times without blocking.
    start = time.monotonic()
    for _ in range(3):
        await limiter.acquire(estimated_tokens=1)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, (
        f"Expected quick acquires after refill, but took {elapsed:.3f}s"
    )
