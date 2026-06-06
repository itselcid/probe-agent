"""CLI entry-point for ProbeAgent.

Uses ``click`` to expose the ``probe-agent`` command.  The heavy lifting
(agent loop, tool registry, LLM calls) will be wired up in future modules;
this module is intentionally kept thin so the CLI surface is easy to test
and extend independently.
"""

from __future__ import annotations

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

    Runs TASK against the project at --project, using Gemini for reasoning
    and a pluggable tool system for interacting with infrastructure.

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
    console.print(
        f"\n[bold cyan]🔍 ProbeAgent v{__version__}[/bold cyan]"
        f"  •  model=[green]{settings.model_name}[/green]"
        f"  •  max_steps=[yellow]{settings.max_steps}[/yellow]\n"
    )
    console.print(f"[dim]Project:[/dim]  {settings.project_path}")
    console.print(f"[dim]Task:[/dim]     {task}\n")

    log.info(
        "agent_start",
        task=task,
        project=settings.project_path,
        model=settings.model_name,
        max_steps=settings.max_steps,
    )

    # --- Agent loop (placeholder) -------------------------------------------
    # TODO: Wire up the agent loop, tool registry, and Gemini client.
    console.print("[yellow]⚠  Agent loop not yet implemented — exiting.[/yellow]")
    log.info("agent_end", reason="not_implemented")


if __name__ == "__main__":
    main()
