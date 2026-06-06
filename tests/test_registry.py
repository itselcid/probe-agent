"""Tests for the ToolRegistry."""

from __future__ import annotations

import pytest

from probe_agent.errors import ToolNotFoundError
from probe_agent.registry import ToolRegistry
from probe_agent.types import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _dummy_read_file(path: str) -> str:
    """Fake tool: returns a canned file content string."""
    return f"contents of {path}"


async def _dummy_docker_ps(all: bool = False) -> str:
    """Fake tool: returns a container listing."""
    return "CONTAINER_ID  IMAGE  STATUS\nabc123  nginx  Up 2h"


async def _dummy_git_log(n: int = 5) -> str:
    """Fake tool: returns fake git log."""
    return "\n".join(f"commit-{i}" for i in range(n))


async def _failing_tool() -> str:
    """Fake tool: always raises."""
    raise RuntimeError("disk on fire")


_FS_READ_PARAMS: dict = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Absolute path to the file.",
        },
    },
    "required": ["path"],
}

_DOCKER_PS_PARAMS: dict = {
    "type": "object",
    "properties": {
        "all": {
            "type": "boolean",
            "description": "Show all containers, including stopped.",
        },
    },
}

_GIT_LOG_PARAMS: dict = {
    "type": "object",
    "properties": {
        "n": {
            "type": "integer",
            "description": "Number of commits to show.",
        },
    },
}


def _make_populated_registry() -> ToolRegistry:
    """Return a registry pre-loaded with three tools across two namespaces."""
    reg = ToolRegistry()
    reg.register(
        namespace="fs",
        name="read_file",
        fn=_dummy_read_file,
        description="Read a file and return its contents.",
        parameters=_FS_READ_PARAMS,
    )
    reg.register(
        namespace="docker",
        name="ps",
        fn=_dummy_docker_ps,
        description="List running Docker containers.",
        parameters=_DOCKER_PS_PARAMS,
    )
    reg.register(
        namespace="git",
        name="log",
        fn=_dummy_git_log,
        description="Show recent git commits.",
        parameters=_GIT_LOG_PARAMS,
    )
    return reg


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:
    """Tests for registering tools."""

    def test_register_tool_appears_in_list(self) -> None:
        """Registered tool is visible via list_tools()."""
        reg = ToolRegistry()
        reg.register(
            namespace="fs",
            name="read_file",
            fn=_dummy_read_file,
            description="Read a file.",
            parameters=_FS_READ_PARAMS,
        )
        assert "fs_read_file" in reg.list_tools()
        assert reg.count() == 1

    def test_tool_names_use_underscores_not_dots(self) -> None:
        """Full names are namespace_name, not namespace.name."""
        reg = _make_populated_registry()
        for name in reg.list_tools():
            assert "." not in name, f"Tool name {name!r} contains a dot"
            assert "_" in name, f"Tool name {name!r} missing underscore"

    def test_duplicate_name_raises_value_error(self) -> None:
        """Registering the same full_name twice raises ValueError."""
        reg = ToolRegistry()
        reg.register(
            namespace="fs",
            name="read_file",
            fn=_dummy_read_file,
            description="Read a file.",
            parameters=_FS_READ_PARAMS,
        )
        with pytest.raises(ValueError, match="Duplicate tool name"):
            reg.register(
                namespace="fs",
                name="read_file",
                fn=_dummy_read_file,
                description="Read a file again.",
                parameters=_FS_READ_PARAMS,
            )

    def test_count_and_namespaces(self) -> None:
        """count() and list_namespaces() reflect all registered tools."""
        reg = _make_populated_registry()
        assert reg.count() == 3
        assert len(reg) == 3  # __len__ dunder
        assert set(reg.list_namespaces()) == {"fs", "docker", "git"}

    def test_contains_dunder(self) -> None:
        """The `in` operator works on tool names."""
        reg = _make_populated_registry()
        assert "fs_read_file" in reg
        assert "nonexistent_tool" not in reg


# ---------------------------------------------------------------------------
# Schema generation tests
# ---------------------------------------------------------------------------


class TestGetSchemas:
    """Tests for get_schemas()."""

    def test_registered_tool_appears_in_schemas(self) -> None:
        """Every registered tool has an entry in get_schemas()."""
        reg = _make_populated_registry()
        schemas = reg.get_schemas()
        names = [s["name"] for s in schemas]
        assert "fs_read_file" in names
        assert "docker_ps" in names
        assert "git_log" in names

    def test_schema_format_matches_json_schema(self) -> None:
        """Each schema dict has name, description, and parameters keys."""
        reg = _make_populated_registry()
        schemas = reg.get_schemas()

        for schema in schemas:
            assert "name" in schema, "Missing 'name' key"
            assert "description" in schema, "Missing 'description' key"
            assert "parameters" in schema, "Missing 'parameters' key"
            assert isinstance(schema["name"], str)
            assert isinstance(schema["description"], str)
            assert isinstance(schema["parameters"], dict)

    def test_schema_parameters_match_registered(self) -> None:
        """The parameters dict in the schema is the one we registered."""
        reg = ToolRegistry()
        reg.register(
            namespace="fs",
            name="read_file",
            fn=_dummy_read_file,
            description="Read a file.",
            parameters=_FS_READ_PARAMS,
        )

        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["parameters"] == _FS_READ_PARAMS

    def test_schemas_sorted_alphabetically(self) -> None:
        """Schemas are returned in deterministic alphabetical order."""
        reg = _make_populated_registry()
        schemas = reg.get_schemas()
        names = [s["name"] for s in schemas]
        assert names == sorted(names)

    def test_empty_registry_returns_empty_list(self) -> None:
        """get_schemas() on an empty registry returns []."""
        reg = ToolRegistry()
        assert reg.get_schemas() == []


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------


class TestExecute:
    """Tests for async tool execution."""

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self) -> None:
        """Successful execution returns ToolResult with data."""
        reg = _make_populated_registry()
        result = await reg.execute("fs_read_file", {"path": "/etc/hosts"})

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data == "contents of /etc/hosts"
        assert result.error is None
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_raises(self) -> None:
        """Executing an unregistered tool raises ToolNotFoundError."""
        reg = _make_populated_registry()

        with pytest.raises(ToolNotFoundError, match="nonexistent"):
            await reg.execute("nonexistent", {})

    @pytest.mark.asyncio
    async def test_execute_captures_exception(self) -> None:
        """If the tool raises, result is success=False with error string."""
        reg = ToolRegistry()
        reg.register(
            namespace="test",
            name="fail",
            fn=_failing_tool,
            description="A tool that always fails.",
            parameters={"type": "object", "properties": {}},
        )

        result = await reg.execute("test_fail", {})

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "RuntimeError" in (result.error or "")
        assert "disk on fire" in (result.error or "")
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_with_kwargs(self) -> None:
        """Arguments are forwarded to the tool function as kwargs."""
        reg = _make_populated_registry()
        result = await reg.execute("git_log", {"n": 3})

        assert result.success is True
        lines = result.data.strip().split("\n")
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Subset tests
# ---------------------------------------------------------------------------


class TestSubset:
    """Tests for creating tool subsets."""

    def test_subset_creates_smaller_registry(self) -> None:
        """subset() returns a registry with only the requested tools."""
        reg = _make_populated_registry()
        child = reg.subset(["fs_read_file", "docker_ps"])

        assert child.count() == 2
        assert "fs_read_file" in child
        assert "docker_ps" in child
        assert "git_log" not in child

    def test_subset_does_not_affect_parent(self) -> None:
        """Creating a subset leaves the original registry unchanged."""
        reg = _make_populated_registry()
        original_count = reg.count()

        child = reg.subset(["fs_read_file"])
        assert child.count() == 1

        # Parent is unmodified.
        assert reg.count() == original_count
        assert "git_log" in reg

    def test_subset_unknown_tool_raises(self) -> None:
        """subset() with an unknown tool name raises ToolNotFoundError."""
        reg = _make_populated_registry()

        with pytest.raises(ToolNotFoundError, match="nonexistent"):
            reg.subset(["fs_read_file", "nonexistent"])

    @pytest.mark.asyncio
    async def test_subset_tools_are_executable(self) -> None:
        """Tools in the subset can be executed normally."""
        reg = _make_populated_registry()
        child = reg.subset(["docker_ps"])

        result = await child.execute("docker_ps", {"all": True})
        assert result.success is True
        assert "nginx" in result.data

    def test_subset_schemas_only_contain_subset(self) -> None:
        """get_schemas() on a subset only returns the subset's tools."""
        reg = _make_populated_registry()
        child = reg.subset(["git_log"])

        schemas = child.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "git_log"
