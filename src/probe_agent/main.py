"""CLI entry-point for ProbeAgent.

Uses ``click`` to expose the ``probe-agent`` command.  Wires up the
LLM provider, tool registry, and agent loop.
"""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from probe_agent import __version__
from probe_agent.config import Settings, load_settings
from probe_agent.logging_setup import get_logger, setup_logging

console = Console()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--project",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to the project the agent should operate on.",
)
@click.argument("task")
@click.version_option(version=__version__, prog_name="probe-agent")
def main(project: str, task: str) -> None:
    """ProbeAgent — autonomous DevOps/SRE agent.

    Runs TASK against the project at --project, using a pluggable LLM
    provider for reasoning and tools for interacting with infrastructure.

    \b
    Examples:
        probe-agent --project ./my-app "check container health"
        probe-agent --project /srv/api  "find the root cause of 5xx errors"
    """
    # --- Bootstrap ----------------------------------------------------------
    try:
        settings: Settings = load_settings()
    except Exception as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        sys.exit(1)

    # Override project_path from CLI flag (takes precedence over env var).
    settings.project_path = project

    setup_logging(level=settings.log_level)
    log = get_logger("probe_agent.main")

    # --- Banner -------------------------------------------------------------
    # Resolve the display model name.
    display_model = settings.llm_model or f"{settings.llm_provider} (default)"

    console.print(
        f"\n[bold cyan]🔍 ProbeAgent v{__version__}[/bold cyan]"
        f"  •  provider=[magenta]{settings.llm_provider}[/magenta]"
        f"  •  model=[green]{display_model}[/green]"
        f"  •  max_steps=[yellow]{settings.max_steps}[/yellow]\n"
    )
    console.print(f"[dim]Project:[/dim]  {settings.project_path}")
    console.print(f"[dim]Task:[/dim]     {task}\n")

    log.info(
        "agent_start",
        task=task,
        project=settings.project_path,
        provider=settings.llm_provider,
        model=display_model,
        max_steps=settings.max_steps,
    )

    # --- Wire up the agent --------------------------------------------------
    from probe_agent.agent import ProbeAgent
    from probe_agent.llm_client import create_llm_provider
    from probe_agent.registry import ToolRegistry
    from probe_agent.tools.docker_tools import register_docker_tools
    from probe_agent.tools.fs import register_fs_tools
    from probe_agent.tools.git import register_git_tools
    from probe_agent.tools.observe import register_observe_tools
    from probe_agent.tools.project import register_project_tools
    from probe_agent.tools.shell import register_shell_tools
    from probe_agent.tools.agent_tools import register_agent_tools

    # Create LLM provider.
    llm = create_llm_provider(
        provider=settings.llm_provider,
        api_key=settings.llm_api_key,
        model_name=settings.llm_model,
    )

    # Build the tool registry.
    registry = ToolRegistry()
    register_fs_tools(registry)
    register_git_tools(registry)
    register_docker_tools(registry)
    register_shell_tools(registry)
    register_observe_tools(registry)
    register_project_tools(registry)

    # Agent tools are registered last — they close over the full registry
    # and LLM so subagents can be spawned with scoped tool access.
    register_agent_tools(registry, full_registry=registry, llm=llm)

    console.print(
        f"[dim]Tools:[/dim]    {registry.count()} across "
        f"{registry.list_namespaces()}\n"
    )

    # Create and run the agent.
    agent = ProbeAgent(config=settings, registry=registry, llm=llm)

    try:
        result = asyncio.run(agent.run(task))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[bold red]Agent error:[/bold red] {exc}")
        log.error("agent_error", error=str(exc), exc_info=True)
        sys.exit(1)

    # --- Print result -------------------------------------------------------
    if result.success:
        console.print(f"\n[bold green]✅ Done[/bold green] in {result.steps} steps "
                       f"({result.total_tokens:,} tokens)")
    else:
        console.print(f"\n[bold yellow]⚠ Max steps reached[/bold yellow] "
                       f"({result.steps} steps, {result.total_tokens:,} tokens)")

    console.print(f"\n[dim]Tools used:[/dim] {', '.join(result.tools_used) or 'none'}\n")

    if result.final_response:
        console.print(result.final_response)

    log.info(
        "agent_end",
        steps=result.steps,
        total_tokens=result.total_tokens,
        success=result.success,
        tools_used=result.tools_used,
    )


if __name__ == "__main__":
    main()
