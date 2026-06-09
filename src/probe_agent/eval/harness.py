"""Evaluation harness for ProbeAgent.

:class:`EvalHarness` wires up a fresh agent for each scenario, runs it
with a timeout, and scores the result using
:func:`~probe_agent.eval.metrics.compute_tool_hit_rate`.

Usage::

    from probe_agent.eval.harness import EvalHarness

    harness = EvalHarness(project_path="/srv/my-app", config=settings)
    results = asyncio.run(harness.run_all())
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.table import Table

from probe_agent.agent import ProbeAgent
from probe_agent.config import Settings
from probe_agent.eval.metrics import EvalResult, compute_tool_hit_rate, summarize_results
from probe_agent.eval.scenarios import SCENARIOS, Scenario
from probe_agent.llm_client import LLMProvider, create_llm_provider
from probe_agent.registry import ToolRegistry
from probe_agent.tools.agent_tools import register_agent_tools
from probe_agent.tools.docker_tools import register_docker_tools
from probe_agent.tools.fs import register_fs_tools
from probe_agent.tools.git import register_git_tools
from probe_agent.tools.observe import register_observe_tools
from probe_agent.tools.project import register_project_tools
from probe_agent.tools.shell import register_shell_tools

log = structlog.get_logger(__name__)


class EvalHarness:
    """Run evaluation scenarios against a real project.

    Each scenario gets a **fresh** agent instance with its own registry,
    context, and session recorder.  Results are scored on tool hit-rate
    and printed as a rich table.

    Args:
        project_path: Root directory of the project under evaluation.
        config: Application settings (LLM provider, API key, etc.).
    """

    def __init__(self, project_path: str, config: Settings) -> None:
        self.project_path = project_path
        self.config = config
        self.console = Console()

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_registry_and_llm(self) -> tuple[ToolRegistry, LLMProvider]:
        """Create a fresh registry with all tools and an LLM provider."""
        llm = create_llm_provider(
            provider=self.config.llm_provider,
            api_key=self.config.llm_api_key,
            model_name=self.config.llm_model,
        )

        registry = ToolRegistry()
        register_fs_tools(registry)
        register_git_tools(registry)
        register_docker_tools(registry)
        register_shell_tools(registry)
        register_observe_tools(registry)
        register_project_tools(registry)
        register_agent_tools(registry, full_registry=registry, llm=llm)

        return registry, llm

    # ------------------------------------------------------------------
    # Run a single scenario
    # ------------------------------------------------------------------

    async def run_scenario(self, scenario: Scenario) -> EvalResult:
        """Run a single scenario and evaluate the result.

        Builds a fresh agent, runs the task with a timeout, and scores
        tool usage against ``scenario.expected_tools``.

        Args:
            scenario: The scenario to execute.

        Returns:
            An :class:`EvalResult` with pass/fail, metrics, and timing.
        """
        log.info(
            "eval_scenario_start",
            scenario=scenario.name,
            max_steps=scenario.max_steps,
            timeout=scenario.timeout_seconds,
        )

        start = time.monotonic()

        try:
            # Fresh agent for each scenario.
            registry, llm = self._build_registry_and_llm()

            scenario_config = Settings(
                llm_provider=self.config.llm_provider,
                llm_api_key=self.config.llm_api_key,
                llm_model=self.config.llm_model,
                project_path=self.project_path,
                max_steps=scenario.max_steps,
            )

            agent = ProbeAgent(
                config=scenario_config,
                registry=registry,
                llm=llm,
            )

            # Run with timeout.
            agent_result = await asyncio.wait_for(
                agent.run(scenario.task),
                timeout=scenario.timeout_seconds,
            )

            duration = time.monotonic() - start

            hit_rate = compute_tool_hit_rate(
                expected=scenario.expected_tools,
                used=agent_result.tools_used,
            )

            # Pass = completed + hit at least one expected tool (or none expected).
            passed = agent_result.success and hit_rate > 0.0

            result = EvalResult(
                scenario_name=scenario.name,
                passed=passed,
                tools_expected=scenario.expected_tools,
                tools_used=agent_result.tools_used,
                tool_hit_rate=hit_rate,
                total_steps=agent_result.steps,
                total_tokens=agent_result.total_tokens,
                duration_seconds=round(duration, 2),
                final_response_preview=agent_result.final_response[:500],
                error=None,
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            result = EvalResult(
                scenario_name=scenario.name,
                passed=False,
                tools_expected=scenario.expected_tools,
                duration_seconds=round(duration, 2),
                error=f"Timeout after {scenario.timeout_seconds}s",
            )

        except Exception as exc:
            duration = time.monotonic() - start
            result = EvalResult(
                scenario_name=scenario.name,
                passed=False,
                tools_expected=scenario.expected_tools,
                duration_seconds=round(duration, 2),
                error=f"{type(exc).__name__}: {exc}",
            )

        log.info(
            "eval_scenario_end",
            scenario=scenario.name,
            passed=result.passed,
            hit_rate=result.tool_hit_rate,
            steps=result.total_steps,
            duration_s=result.duration_seconds,
            error=result.error,
        )

        return result

    # ------------------------------------------------------------------
    # Run all scenarios
    # ------------------------------------------------------------------

    async def run_all(
        self,
        scenario_names: list[str] | None = None,
    ) -> list[EvalResult]:
        """Run all (or selected) scenarios and print a results table.

        Args:
            scenario_names: If provided, run only these scenarios.
                Otherwise run every scenario in :data:`SCENARIOS`.

        Returns:
            List of :class:`EvalResult` objects, one per scenario.
        """
        if scenario_names:
            scenarios = [s for s in SCENARIOS if s.name in scenario_names]
        else:
            scenarios = list(SCENARIOS)

        self.console.print(
            f"\n[bold cyan]📊 ProbeAgent Evaluation[/bold cyan]  •  "
            f"{len(scenarios)} scenarios  •  project={self.project_path}\n"
        )

        results: list[EvalResult] = []

        for i, scenario in enumerate(scenarios, 1):
            self.console.print(
                f"[dim][{i}/{len(scenarios)}][/dim] "
                f"[bold]{scenario.name}[/bold] — {scenario.description}"
            )

            result = await self.run_scenario(scenario)
            results.append(result)

            # Live feedback.
            status = "[green]✓ PASS[/green]" if result.passed else "[red]✗ FAIL[/red]"
            self.console.print(
                f"  {status}  hit_rate={result.tool_hit_rate:.0%}  "
                f"steps={result.total_steps}  "
                f"time={result.duration_seconds:.1f}s"
            )
            if result.error:
                self.console.print(f"  [red]{result.error}[/red]")
            self.console.print()

        # --- Summary table ---
        self._print_results_table(results)

        # --- Save results to disk ---
        self._save_results(results)

        return results

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _print_results_table(self, results: list[EvalResult]) -> None:
        """Print a rich summary table to the console."""
        table = Table(title="Evaluation Results", show_lines=True)
        table.add_column("Scenario", style="bold")
        table.add_column("Passed", justify="center")
        table.add_column("Steps", justify="right")
        table.add_column("Tool Hit Rate", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Error", style="red")

        for r in results:
            status = "✓" if r.passed else "✗"
            style = "green" if r.passed else "red"

            table.add_row(
                r.scenario_name,
                f"[{style}]{status}[/{style}]",
                str(r.total_steps),
                f"{r.tool_hit_rate:.0%}",
                f"{r.total_tokens:,}",
                f"{r.duration_seconds:.1f}s",
                r.error or "",
            )

        self.console.print(table)

        # --- Aggregate summary ---
        summary = summarize_results(results)
        self.console.print(
            f"\n[bold]Summary:[/bold]  "
            f"{summary['passed']}/{summary['total_scenarios']} passed  "
            f"({summary['pass_rate']:.0%})  •  "
            f"avg hit rate: {summary['avg_tool_hit_rate']:.0%}  •  "
            f"avg steps: {summary['avg_steps']:.1f}  •  "
            f"total tokens: {summary['total_tokens']:,}\n"
        )

    def _save_results(self, results: list[EvalResult]) -> None:
        """Save evaluation results to a JSON file on disk."""
        output_dir = Path(self.project_path) / ".probe" / "evals"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"eval_{timestamp}.json"

        summary = summarize_results(results)

        data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_path": self.project_path,
            "provider": self.config.llm_provider,
            "model": self.config.llm_model,
            "summary": summary,
            "results": [
                {
                    "scenario": r.scenario_name,
                    "passed": r.passed,
                    "tools_expected": r.tools_expected,
                    "tools_used": r.tools_used,
                    "tool_hit_rate": r.tool_hit_rate,
                    "steps": r.total_steps,
                    "tokens": r.total_tokens,
                    "duration_s": r.duration_seconds,
                    "final_response_preview": r.final_response_preview,
                    "error": r.error,
                }
                for r in results
            ],
        }

        path.write_text(json.dumps(data, indent=2, default=str))

        self.console.print(f"[dim]Results saved to {path}[/dim]\n")
