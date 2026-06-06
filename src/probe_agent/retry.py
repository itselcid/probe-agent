"""Async retry decorator with exponential backoff and jitter.

Provides a generic ``@retry`` decorator for wrapping async functions that may
fail transiently (rate limits, network blips, etc.).  The decorator retries
on a configurable set of exception types, using exponential backoff with
random jitter to avoid thundering-herd problems.

Example::

    from probe_agent.errors import LLMRateLimitError
    from probe_agent.retry import retry

    @retry(max_attempts=3, retryable_exceptions=(LLMRateLimitError,))
    async def call_gemini(prompt: str) -> str:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import random
from typing import Any, Callable, TypeVar

import structlog

log = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Upper bound for jitter as a fraction of the computed delay.
_JITTER_FRACTION = 0.25


def retry(
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator factory that retries an async function on transient failures.

    Args:
        max_attempts: Total number of attempts (including the first call).
            Must be at least 1.
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Maximum delay cap in seconds; backoff will never exceed
            this value (before jitter).
        backoff_factor: Multiplier applied to the delay after each failed
            attempt.  Delay for attempt *n* (0-indexed) is
            ``min(base_delay * backoff_factor ** n, max_delay)``.
        retryable_exceptions: Tuple of exception classes that should trigger a
            retry.  Any exception *not* in this tuple propagates immediately.

    Returns:
        A decorator that wraps an ``async def`` function with retry logic.

    Raises:
        The last exception encountered if all attempts are exhausted.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: BaseException | None = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exception = exc

                    # If this was the final attempt, don't sleep — just raise.
                    if attempt + 1 >= max_attempts:
                        break

                    # Exponential backoff capped at max_delay.
                    delay = min(
                        base_delay * (backoff_factor ** attempt),
                        max_delay,
                    )

                    # Add 0–25 % random jitter.
                    jitter = delay * random.uniform(0, _JITTER_FRACTION)
                    delay += jitter

                    log.warning(
                        "retry_attempt",
                        function=func.__qualname__,
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        delay_seconds=round(delay, 3),
                        exception_type=type(exc).__name__,
                        exception_message=str(exc),
                    )

                    await asyncio.sleep(delay)

            # All attempts exhausted — re-raise the last exception.
            assert last_exception is not None  # noqa: S101
            raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator
