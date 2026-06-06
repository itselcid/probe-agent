"""Async token-bucket rate limiter for LLM API calls.

Implements a dual-bucket limiter that enforces both *requests per minute*
(RPM) and *tokens per minute* (TPM) limits.  Tokens refill continuously
rather than in bursts, so callers experience smooth throughput.

Example::

    limiter = TokenBucketRateLimiter(rpm=15, tpm=1_000_000)

    # Before each API call:
    await limiter.acquire(estimated_tokens=2048)
"""

from __future__ import annotations

import asyncio
import time

import structlog

log = structlog.get_logger(__name__)


class TokenBucketRateLimiter:
    """Dual token-bucket rate limiter (RPM + TPM).

    Each bucket refills continuously at a constant rate.  When a caller
    invokes :meth:`acquire`, it will block (``await asyncio.sleep``) until
    both buckets have enough capacity, then atomically consume from both.

    Attributes:
        rpm: Maximum requests per minute.
        tpm: Maximum tokens per minute.
    """

    def __init__(self, rpm: int = 15, tpm: int = 1_000_000) -> None:
        """Initialise the rate limiter.

        Args:
            rpm: Allowed requests per minute.  Each :meth:`acquire` call
                consumes exactly 1 request token.
            tpm: Allowed tokens per minute.  The ``estimated_tokens``
                argument to :meth:`acquire` is deducted from this bucket.
        """
        if rpm <= 0:
            raise ValueError(f"rpm must be positive, got {rpm}")
        if tpm <= 0:
            raise ValueError(f"tpm must be positive, got {tpm}")

        self.rpm: int = rpm
        self.tpm: int = tpm

        # --- Request bucket ---
        self._request_tokens: float = float(rpm)
        self._request_refill_rate: float = rpm / 60.0  # tokens per second

        # --- Token bucket ---
        self._token_tokens: float = float(tpm)
        self._token_refill_rate: float = tpm / 60.0  # tokens per second

        self._last_refill: float = time.monotonic()
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Refill both buckets based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        self._request_tokens = min(
            float(self.rpm),
            self._request_tokens + elapsed * self._request_refill_rate,
        )
        self._token_tokens = min(
            float(self.tpm),
            self._token_tokens + elapsed * self._token_refill_rate,
        )

    def _time_until_available(self, estimated_tokens: int) -> float:
        """Return seconds until both buckets can satisfy the request.

        Args:
            estimated_tokens: Number of TPM tokens the request will consume.

        Returns:
            ``0.0`` if capacity is available now, otherwise the wait in
            seconds.
        """
        wait_request = 0.0
        if self._request_tokens < 1.0:
            wait_request = (1.0 - self._request_tokens) / self._request_refill_rate

        wait_tokens = 0.0
        if self._token_tokens < estimated_tokens:
            deficit = estimated_tokens - self._token_tokens
            wait_tokens = deficit / self._token_refill_rate

        return max(wait_request, wait_tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self, estimated_tokens: int = 1) -> None:
        """Wait until rate-limit capacity is available, then consume.

        This method is safe to call concurrently from multiple coroutines;
        an internal ``asyncio.Lock`` serialises access to the buckets.

        Args:
            estimated_tokens: Approximate number of tokens the upcoming API
                call will use (prompt + expected completion).  Defaults to 1
                for simple request-only limiting.
        """
        async with self._lock:
            self._refill()

            wait = self._time_until_available(estimated_tokens)

            if wait > 0:
                log.info(
                    "rate_limited",
                    wait_seconds=round(wait, 3),
                    estimated_tokens=estimated_tokens,
                    request_tokens_available=round(self._request_tokens, 2),
                    token_tokens_available=round(self._token_tokens, 2),
                )
                await asyncio.sleep(wait)
                self._refill()

            # Consume from both buckets.
            self._request_tokens -= 1.0
            self._token_tokens -= float(estimated_tokens)
