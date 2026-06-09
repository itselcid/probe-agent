
"""Evaluation scenarios for ProbeAgent.

Each :class:`Scenario` defines a task the agent should execute, along with
the tools we *expect* it to use.  The :class:`EvalHarness` uses these to
score tool-selection accuracy and overall quality.

Scenarios are sorted from simple (single-namespace) to complex (multi-step
with subagent delegation).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Scenario:
    """A single evaluation scenario.

    Attributes:
        name: Machine-readable identifier (snake_case).
        description: Short human-readable summary of the scenario.
        task: The natural-language task sent to the agent.
        expected_tools: Tools the agent *should* invoke to solve the task.
            The harness measures hit-rate against this list.
        max_steps: Per-scenario step limit (overrides config).
        timeout_seconds: Maximum wall-clock time before the run is killed.
    """

    name: str
    description: str
    task: str
    expected_tools: list[str] = field(default_factory=list)
    max_steps: int = 30
    timeout_seconds: int = 300


# ---------------------------------------------------------------------------
# Scenario catalogue — ordered from simple → complex
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    # ---- Simple (single namespace) ----------------------------------------
    Scenario(
        name="container_health",
        description="Simple: list containers and report health",
        task=(
            "List all running Docker containers and report their status "
            "and resource usage."
        ),
        expected_tools=["docker_ps", "docker_stats"],
        max_steps=8,
        timeout_seconds=60,
    ),
    Scenario(
        name="project_analysis",
        description="Analyze project structure and tech stack",
        task=(
            "Analyze this project. Tell me what technologies it uses, "
            "what services it runs, and how it's structured."
        ),
        expected_tools=["project_discover", "fs_tree"],
        max_steps=12,
        timeout_seconds=120,
    ),
    # ---- Medium (multi-namespace) -----------------------------------------
    Scenario(
        name="log_investigation",
        description="Investigate container logs for errors",
        task=(
            "Check the logs of all running containers. Are there any "
            "errors or warnings? Summarize the issues found."
        ),
        expected_tools=["docker_ps", "docker_logs"],
        max_steps=15,
        timeout_seconds=120,
    ),
    Scenario(
        name="git_health",
        description="Review recent git activity and current status",
        task=(
            "Show me the current git status and the last 10 commits. "
            "Are there any uncommitted changes?"
        ),
        expected_tools=["git_status", "git_log"],
        max_steps=8,
        timeout_seconds=60,
    ),
    Scenario(
        name="service_connectivity",
        description="Check service health endpoints",
        task=(
            "Check if the services have working health endpoints. "
            "Report which are healthy and which are not."
        ),
        expected_tools=["observe_health_check", "observe_check_endpoints"],
        max_steps=12,
        timeout_seconds=120,
    ),
    # ---- Complex (subagent delegation) ------------------------------------
    Scenario(
        name="diagnostic_subagent",
        description="Use diagnostic subagent for deep investigation",
        task=(
            "Something seems wrong with the ekyc-app service. Spawn a "
            "diagnostic subagent to investigate and report findings."
        ),
        expected_tools=["agent_spawn_diagnostic"],
        max_steps=20,
        timeout_seconds=180,
    ),
    Scenario(
        name="full_workflow",
        description="End-to-end: discover → diagnose → report",
        task=(
            "First, discover what this project is about. Then check if "
            "all services are healthy. If any issues are found, investigate "
            "them. Finally, generate a status report."
        ),
        expected_tools=[
            "project_discover",
            "agent_spawn_diagnostic",
            "agent_spawn_report",
        ],
        max_steps=30,
        timeout_seconds=300,
    ),
]


def get_scenario(name: str) -> Scenario:
    """Look up a scenario by name.

    Args:
        name: The ``name`` field of the desired scenario.

    Returns:
        The matching :class:`Scenario`.

    Raises:
        KeyError: If no scenario with that name exists.
    """
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(f"Unknown scenario: {name!r}")


def list_scenario_names() -> list[str]:
    """Return all scenario names, in order.

    Returns:
        List of scenario name strings.
    """
    return [s.name for s in SCENARIOS]
