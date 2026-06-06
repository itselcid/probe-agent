"""Central tool registry for ProbeAgent.

Stores tool definitions, generates JSON schemas for LLM function-calling,
and executes tools by name.  Designed for 50+ tools organised into
namespaces (``fs``, ``docker``, ``git``, …).

Example::

    registry = ToolRegistry()

    async def read_file(path: str) -> str:
        return open(path).read()

    registry.register(
        namespace="fs",
        name="read_file",
        fn=read_file,
        description="Read a file and return its contents.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                },
            },
            "required": ["path"],
        },
    )

    schemas = registry.get_schemas()          # → send to LLM
    result  = await registry.execute("fs_read_file", {"path": "/etc/hosts"})
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

from probe_agent.errors import ToolNotFoundError
from probe_agent.types import ToolResult

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Metadata for a single registered tool.

    Attributes:
        namespace: Logical grouping (e.g. ``"fs"``, ``"docker"``, ``"git"``).
        name: Short tool name within the namespace (e.g. ``"read_file"``).
        full_name: Canonical name exposed to the LLM, formed as
            ``"{namespace}_{name}"`` (underscores, not dots).
        description: Human-readable description shown to the LLM.  Should
            be comprehensive — the LLM never sees the implementation, only
            this text.
        parameters: JSON Schema dict describing the tool's arguments.
        fn: The async callable that implements the tool.
    """

    namespace: str
    name: str
    full_name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]


class ToolRegistry:
    """Central registry for all agent tools.

    Tools are registered with a *namespace* and *name*.  The registry
    combines them into a ``full_name`` (``namespace_name``) that the LLM
    uses to invoke tools.

    Thread safety: this class is **not** thread-safe.  It is designed to be
    populated once at startup and then used read-only during the agent loop.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        namespace: str,
        name: str,
        fn: Callable[..., Any],
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register a tool.

        The tool's ``full_name`` is formed as ``"{namespace}_{name}"``
        (underscores, not dots).

        Args:
            namespace: Logical group (e.g. ``"fs"``, ``"docker"``).
            name: Short name within the namespace (e.g. ``"read_file"``).
            fn: Async callable that implements the tool.
            description: Description shown to the LLM.  Be comprehensive.
            parameters: JSON Schema dict for the tool's arguments.

        Raises:
            ValueError: If a tool with the same ``full_name`` is already
                registered.
        """
        full_name = f"{namespace}_{name}"

        if full_name in self._tools:
            raise ValueError(
                f"Duplicate tool name: {full_name!r} is already registered."
            )

        self._tools[full_name] = RegisteredTool(
            namespace=namespace,
            name=name,
            full_name=full_name,
            description=description,
            parameters=parameters,
            fn=fn,
        )

        log.debug(
            "tool_registered",
            full_name=full_name,
            namespace=namespace,
        )

    # ------------------------------------------------------------------
    # Schema generation
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return tool definitions in standard JSON Schema format.

        Each entry is a dict with ``name``, ``description``, and
        ``parameters`` keys — the canonical format that
        :meth:`LLMProvider.chat` expects.  Each provider converts this
        to its own wire format internally.

        Returns:
            Sorted list of tool schema dicts (sorted by name for
            deterministic ordering).
        """
        schemas: list[dict[str, Any]] = []
        for tool in sorted(self._tools.values(), key=lambda t: t.full_name):
            schemas.append(
                {
                    "name": tool.full_name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            )
        return schemas

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments.

        Steps:
            1. Look up the tool (raise :class:`ToolNotFoundError` if missing).
            2. Start a timer.
            3. Call the async function with ``**args``.
            4. On success: ``ToolResult(success=True, data=result, …)``.
            5. On exception: ``ToolResult(success=False, error=str(e), …)``.
            6. Log the execution with structlog.

        Args:
            tool_name: The ``full_name`` of the tool to execute.
            args: Keyword arguments forwarded to the tool function.

        Returns:
            A :class:`ToolResult` capturing the outcome and timing.

        Raises:
            ToolNotFoundError: If *tool_name* is not in the registry.
        """
        if tool_name not in self._tools:
            raise ToolNotFoundError(tool_name)

        tool = self._tools[tool_name]
        start = time.monotonic()

        try:
            result = await tool.fn(**args)
            duration_ms = (time.monotonic() - start) * 1000

            log.info(
                "tool_executed",
                tool=tool_name,
                success=True,
                duration_ms=round(duration_ms, 1),
            )

            return ToolResult(
                success=True,
                data=result,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000

            log.warning(
                "tool_execution_failed",
                tool=tool_name,
                success=False,
                duration_ms=round(duration_ms, 1),
                error=str(exc),
                error_type=type(exc).__name__,
            )

            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=duration_ms,
            )

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def subset(self, tool_names: list[str]) -> ToolRegistry:
        """Create a new registry containing only the specified tools.

        Useful for giving sub-agents a restricted tool set without
        modifying the parent registry.

        Args:
            tool_names: List of ``full_name`` strings to include.

        Returns:
            A new :class:`ToolRegistry` containing only the requested tools.

        Raises:
            ToolNotFoundError: If any name in *tool_names* is not registered.
        """
        child = ToolRegistry()

        for name in tool_names:
            if name not in self._tools:
                raise ToolNotFoundError(name)
            # Copy the RegisteredTool directly — it's frozen/immutable.
            child._tools[name] = self._tools[name]

        return child

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_tools(self) -> list[str]:
        """Return all registered tool full names, sorted alphabetically.

        Returns:
            Sorted list of tool ``full_name`` strings.
        """
        return sorted(self._tools.keys())

    def list_namespaces(self) -> list[str]:
        """Return unique namespace names, sorted alphabetically.

        Returns:
            Sorted list of distinct namespace strings.
        """
        return sorted({t.namespace for t in self._tools.values()})

    def count(self) -> int:
        """Return the total number of registered tools.

        Returns:
            Integer count.
        """
        return len(self._tools)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def __repr__(self) -> str:
        return (
            f"ToolRegistry(tools={self.count()}, "
            f"namespaces={self.list_namespaces()})"
        )
