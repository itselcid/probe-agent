"""Tool registry and base classes for ProbeAgent tools.

Individual tool modules (e.g. ``docker_tools``, ``log_tools``) will be added
as sub-modules of this package.  The registry will auto-discover them at
startup and expose them to the agent's LLM loop.
"""

from __future__ import annotations

from typing import Any, Callable

# Tool function type: takes keyword arguments, returns a string.
ToolFunction = Callable[..., str]

# Global tool registry — maps tool name → handler function.
_TOOL_REGISTRY: dict[str, ToolFunction] = {}


def register_tool(name: str) -> Callable[[ToolFunction], ToolFunction]:
    """Decorator that registers a function as an agent tool.

    Args:
        name: The canonical tool name exposed to the LLM.

    Returns:
        The original function, unchanged.

    Example::

        @register_tool("list_containers")
        def list_containers(project_path: str) -> str:
            ...
    """

    def decorator(func: ToolFunction) -> ToolFunction:
        _TOOL_REGISTRY[name] = func
        return func

    return decorator


def get_tool(name: str) -> ToolFunction:
    """Look up a tool by name.

    Args:
        name: Registered tool name.

    Returns:
        The tool's handler function.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    return _TOOL_REGISTRY[name]


def list_tools() -> dict[str, ToolFunction]:
    """Return a shallow copy of the full tool registry.

    Returns:
        A ``{name: function}`` mapping of all registered tools.
    """
    return dict(_TOOL_REGISTRY)


def tool_schemas() -> list[dict[str, Any]]:
    """Generate JSON-Schema-style tool descriptions for the LLM.

    Reads ``__doc__`` and type annotations from each registered tool to
    build the schema.  A proper implementation will be added once the
    first concrete tools are defined.

    Returns:
        A list of tool schema dicts suitable for the Gemini function-calling
        API.
    """
    schemas: list[dict[str, Any]] = []
    for name, func in _TOOL_REGISTRY.items():
        schemas.append(
            {
                "name": name,
                "description": (func.__doc__ or "").strip(),
            }
        )
    return schemas
