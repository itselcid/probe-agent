"""Evaluation metrics for ProbeAgent.

Defines :class:`EvalResult` — the outcome of running a single scenario —
and helper functions for computing aggregate quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Outcome of a single evaluation scenario.

    Attributes:
        scenario_name: Which scenario was run.
        passed: ``True`` if the agent completed without error and used
            at least one expected tool.
        tools_expected: Tools the scenario expected the agent to use.
        tools_used: Tools the agent actually used.
        tool_hit_rate: Fraction of expected tools that were actually used
            (0.0 to 1.0).  ``1.0`` means all expected tools were invoked.
        total_steps: Number of agent loop steps taken.
        total_tokens: Cumulative token usage across all LLM calls.
        duration_seconds: Wall-clock time for the scenario.
        final_response_preview: First 500 chars of the agent's final answer.
        error: Error message if the scenario failed, else ``None``.
    """

    scenario_name: str
    passed: bool
    tools_expected: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tool_hit_rate: float = 0.0
    total_steps: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    final_response_preview: str = ""
    error: str | None = None


def compute_tool_hit_rate(
    expected: list[str],
    used: list[str],
) -> float:
    """Compute the fraction of expected tools that were actually used.

    Args:
        expected: Tools the scenario says should be invoked.
        used: Tools the agent actually invoked.

    Returns:
        A float between 0.0 and 1.0.  Returns 1.0 if *expected* is empty
        (vacuously true — no expectations means "pass").
    """
    if not expected:
        return 1.0
    hits = set(expected) & set(used)
    return len(hits) / len(expected)


def summarize_results(results: list[EvalResult]) -> dict:
    """Compute aggregate metrics across multiple scenario results.

    Args:
        results: List of individual scenario outcomes.

    Returns:
        A dict with summary statistics::

            {
                "total_scenarios": int,
                "passed": int,
                "failed": int,
                "pass_rate": float,
                "avg_tool_hit_rate": float,
                "avg_steps": float,
                "avg_tokens": float,
                "avg_duration_s": float,
                "total_tokens": int,
            }
    """
    if not results:
        return {
            "total_scenarios": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "avg_tool_hit_rate": 0.0,
            "avg_steps": 0.0,
            "avg_tokens": 0.0,
            "avg_duration_s": 0.0,
            "total_tokens": 0,
        }

    n = len(results)
    passed = sum(1 for r in results if r.passed)

    return {
        "total_scenarios": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": passed / n,
        "avg_tool_hit_rate": sum(r.tool_hit_rate for r in results) / n,
        "avg_steps": sum(r.total_steps for r in results) / n,
        "avg_tokens": sum(r.total_tokens for r in results) / n,
        "avg_duration_s": sum(r.duration_seconds for r in results) / n,
        "total_tokens": sum(r.total_tokens for r in results),
    }
