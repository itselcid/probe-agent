"""Agent tools — spawn specialised subagents.

Three tools that let the parent agent delegate work to isolated
subagents with scoped tool access:

- :func:`spawn_diagnostic` — read-only investigation
- :func:`spawn_remediation` — can modify files and restart services
- :func:`spawn_report` — read-only report generation

Each subagent gets its own conversation history, system prompt, and a
restricted subset of tools.  The parent's context is never shared.

Register all agent tools with :func:`register_agent_tools`.
"""

from __future__ import annotations

from typing import Any

from probe_agent.llm_client import LLMProvider
from probe_agent.registry import ToolRegistry
from probe_agent.subagent import SubagentRunner

# ---------------------------------------------------------------------------
# Tool scopes — which tools each subagent can access
# ---------------------------------------------------------------------------

_DIAGNOSTIC_TOOLS = [
    "fs_read_file",
    "fs_search_content",
    "fs_tree",
    "git_log",
    "git_diff",
    "docker_ps",
    "docker_logs",
    "docker_inspect",
    "observe_health_check",
    "observe_parse_log_file",
    "project_discover",
]

_REMEDIATION_TOOLS = [
    "fs_read_file",
    "fs_write_file",
    "fs_edit_file",
    "docker_restart",
    "docker_exec_command",
    "shell_run",
    "git_commit",
]

_REPORT_TOOLS = [
    "fs_read_file",
    "fs_tree",
    "git_log",
    "docker_ps",
    "docker_stats",
    "observe_log_stats",
    "project_discover",
    "project_service_map",
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_DIAGNOSTIC_PROMPT = (
    "You are a diagnostic specialist subagent.\n\n"
    "YOUR ROLE: Investigate, gather evidence, and report findings.\n"
    "YOU MUST NOT modify any files, restart containers, or make changes.\n\n"
    "GUIDELINES:\n"
    "1. Start by understanding the project structure (fs_tree, project_discover).\n"
    "2. Check container health (docker_ps, observe_health_check).\n"
    "3. Read logs for errors (docker_logs, observe_parse_log_file).\n"
    "4. Check recent code changes (git_log, git_diff).\n"
    "5. Synthesize your findings into a clear, structured summary.\n"
)

_REMEDIATION_PROMPT = (
    "You are a remediation specialist subagent.\n\n"
    "YOUR ROLE: Apply targeted fixes based on the diagnostic findings.\n"
    "ALWAYS verify your changes worked after applying them.\n\n"
    "GUIDELINES:\n"
    "1. Read the relevant files before editing them (fs_read_file).\n"
    "2. Make minimal, focused changes (fs_write_file, fs_edit_file).\n"
    "3. Restart affected services if needed (docker_restart).\n"
    "4. Verify the fix (shell_run to re-run tests or check status).\n"
    "5. Commit working changes with a descriptive message (git_commit).\n"
)

_REPORT_PROMPT = (
    "You are a report generator subagent.\n\n"
    "YOUR ROLE: Produce a clear, comprehensive Markdown report.\n"
    "YOU MUST NOT modify any files or make changes.\n\n"
    "STRUCTURE YOUR REPORT AS:\n"
    "## Summary\n"
    "Brief overview of the system state.\n\n"
    "## Findings\n"
    "Detailed observations with evidence.\n\n"
    "## Recommendations\n"
    "Prioritized list of actions to take.\n\n"
    "## Action Items\n"
    "Concrete next steps with owners if applicable.\n"
)


# ---------------------------------------------------------------------------
# 1. spawn_diagnostic
# ---------------------------------------------------------------------------


async def spawn_diagnostic(
    task: str,
    project_path: str,
    registry: ToolRegistry,
    llm: LLMProvider,
) -> dict[str, Any]:
    """Spawn a READ-ONLY diagnostic subagent.

    The diagnostic subagent investigates a problem by reading files,
    checking container health, parsing logs, and inspecting recent
    code changes.  It cannot modify anything.

    Args:
        task: What to investigate.
        project_path: Root directory of the project.
        registry: The full parent tool registry.
        llm: The LLM provider.

    Returns:
        ``{"subagent": "diagnostic", "findings": str,
        "tools_used": [...], "steps": int}``.
    """
    full_task = (
        f"Investigate the following issue in the project at {project_path}:\n\n"
        f"{task}"
    )

    runner = SubagentRunner(
        name="diagnostic",
        system_prompt=_DIAGNOSTIC_PROMPT,
        tool_names=_DIAGNOSTIC_TOOLS,
        registry=registry,
        llm=llm,
        max_steps=15,
    )

    result = await runner.run(full_task)

    return {
        "subagent": "diagnostic",
        "findings": result.result,
        "tools_used": result.tools_used,
        "steps": result.steps,
        "success": result.success,
    }


# ---------------------------------------------------------------------------
# 2. spawn_remediation
# ---------------------------------------------------------------------------


async def spawn_remediation(
    task: str,
    project_path: str,
    registry: ToolRegistry,
    llm: LLMProvider,
) -> dict[str, Any]:
    """Spawn a subagent that CAN modify things.

    The remediation subagent applies fixes by editing files, running
    commands, restarting containers, and committing changes.

    Args:
        task: What to fix, including diagnostic context.
        project_path: Root directory of the project.
        registry: The full parent tool registry.
        llm: The LLM provider.

    Returns:
        ``{"subagent": "remediation", "actions_taken": str,
        "tools_used": [...], "steps": int}``.
    """
    full_task = (
        f"Apply fixes in the project at {project_path}:\n\n"
        f"{task}"
    )

    runner = SubagentRunner(
        name="remediation",
        system_prompt=_REMEDIATION_PROMPT,
        tool_names=_REMEDIATION_TOOLS,
        registry=registry,
        llm=llm,
        max_steps=15,
    )

    result = await runner.run(full_task)

    return {
        "subagent": "remediation",
        "actions_taken": result.result,
        "tools_used": result.tools_used,
        "steps": result.steps,
        "success": result.success,
    }


# ---------------------------------------------------------------------------
# 3. spawn_report
# ---------------------------------------------------------------------------


async def spawn_report(
    task: str,
    project_path: str,
    registry: ToolRegistry,
    llm: LLMProvider,
) -> dict[str, Any]:
    """Spawn a READ-ONLY report generator subagent.

    The report subagent gathers information and produces a structured
    Markdown report with findings, recommendations, and action items.

    Args:
        task: What to report on.
        project_path: Root directory of the project.
        registry: The full parent tool registry.
        llm: The LLM provider.

    Returns:
        ``{"subagent": "report", "report": str,
        "tools_used": [...], "steps": int}``.
    """
    full_task = (
        f"Generate a report for the project at {project_path}:\n\n"
        f"{task}"
    )

    runner = SubagentRunner(
        name="report",
        system_prompt=_REPORT_PROMPT,
        tool_names=_REPORT_TOOLS,
        registry=registry,
        llm=llm,
        max_steps=15,
    )

    result = await runner.run(full_task)

    return {
        "subagent": "report",
        "report": result.result,
        "tools_used": result.tools_used,
        "steps": result.steps,
        "success": result.success,
    }


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_agent_tools(
    registry: ToolRegistry,
    full_registry: ToolRegistry,
    llm: LLMProvider,
) -> None:
    """Register all agent-spawning tools.

    These tools close over ``full_registry`` and ``llm`` so the parent
    agent can call them with just ``task`` and ``project_path``.

    Args:
        registry: The registry to register the tools INTO.
        full_registry: The complete tool registry (passed to subagents
            for subsetting).
        llm: The LLM provider (shared with subagents).
    """

    async def _diagnostic(task: str, project_path: str) -> dict[str, Any]:
        return await spawn_diagnostic(task, project_path, full_registry, llm)

    async def _remediation(task: str, project_path: str) -> dict[str, Any]:
        return await spawn_remediation(task, project_path, full_registry, llm)

    async def _report(task: str, project_path: str) -> dict[str, Any]:
        return await spawn_report(task, project_path, full_registry, llm)

    registry.register(
        namespace="agent",
        name="spawn_diagnostic",
        fn=_diagnostic,
        description=(
            "Spawn an isolated diagnostic subagent that investigates a problem. "
            "The subagent is READ-ONLY — it can inspect files, logs, containers, "
            "and code changes but CANNOT modify anything. Use this when you need "
            "to deeply investigate an issue without risk of side effects. "
            "Returns structured findings."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What to investigate (e.g. 'find why the API is returning 500 errors')",
                },
                "project_path": {
                    "type": "string",
                    "description": "Root directory of the project",
                },
            },
            "required": ["task", "project_path"],
        },
    )

    registry.register(
        namespace="agent",
        name="spawn_remediation",
        fn=_remediation,
        description=(
            "Spawn an isolated remediation subagent that CAN modify files, "
            "restart containers, and commit changes. Use this to apply fixes "
            "after diagnosis. The subagent will verify its changes worked. "
            "Returns a summary of actions taken."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What to fix, including context from diagnosis",
                },
                "project_path": {
                    "type": "string",
                    "description": "Root directory of the project",
                },
            },
            "required": ["task", "project_path"],
        },
    )

    registry.register(
        namespace="agent",
        name="spawn_report",
        fn=_report,
        description=(
            "Spawn an isolated report generator subagent. It gathers information "
            "and produces a comprehensive Markdown report with findings, "
            "recommendations, and action items. READ-ONLY — cannot modify anything. "
            "Returns the formatted report."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What to report on (e.g. 'system health overview')",
                },
                "project_path": {
                    "type": "string",
                    "description": "Root directory of the project",
                },
            },
            "required": ["task", "project_path"],
        },
    )
