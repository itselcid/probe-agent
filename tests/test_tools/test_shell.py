"""Tests for shell and system tools."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from probe_agent.registry import ToolRegistry
from probe_agent.tools.shell import (
    check_port,
    curl,
    disk_usage,
    env_vars,
    process_list,
    register_shell_tools,
    run,
    run_in_dir,
    system_info,
)


# ---------------------------------------------------------------------------
# 1. run
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for shell_run."""

    @pytest.mark.asyncio
    async def test_simple_command(self) -> None:
        """Run echo and verify output."""
        result = await run("echo hello")
        assert result["return_code"] == 0
        assert "hello" in result["stdout"]
        assert result["duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_failing_command(self) -> None:
        """Non-zero exit code is returned, not raised."""
        result = await run("exit 42")
        assert result["return_code"] == 42

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self) -> None:
        """Long-running command is killed after timeout."""
        result = await run("sleep 60", timeout=1)
        assert result["return_code"] == -1
        assert "timed out" in result["stderr"]

    @pytest.mark.asyncio
    async def test_truncates_long_output(self) -> None:
        """Output exceeding 10 000 chars is truncated."""
        # Emit ~20 000 chars.
        result = await run("python3 -c \"print('x' * 20000)\"")
        assert "[truncated" in result["stdout"]
        assert len(result["stdout"]) < 11_000


# ---------------------------------------------------------------------------
# 2. run_in_dir
# ---------------------------------------------------------------------------


class TestRunInDir:
    """Tests for shell_run_in_dir."""

    @pytest.mark.asyncio
    async def test_runs_in_directory(self, tmp_path: Path) -> None:
        """Command runs with cwd set to the given directory."""
        result = await run_in_dir("pwd", str(tmp_path))
        assert result["return_code"] == 0
        # On macOS /private/var vs /var — just check the basename.
        assert tmp_path.name in result["stdout"]
        assert result["directory"] == str(tmp_path)

    @pytest.mark.asyncio
    async def test_timeout_in_dir(self, tmp_path: Path) -> None:
        """Timeout works with run_in_dir too."""
        result = await run_in_dir("sleep 60", str(tmp_path), timeout=1)
        assert result["return_code"] == -1


# ---------------------------------------------------------------------------
# 3. check_port
# ---------------------------------------------------------------------------


class TestCheckPort:
    """Tests for shell_check_port."""

    @pytest.mark.asyncio
    async def test_unused_port(self) -> None:
        """An unused port returns in_use=False."""
        # Port 39999 is almost certainly unused in tests.
        result = await check_port(39999)
        assert result["port"] == 39999
        assert result["in_use"] is False
        assert result["process"] is None

    @pytest.mark.asyncio
    async def test_port_with_mock(self) -> None:
        """When lsof finds a process, return its info."""
        lsof_output = (
            b"COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n"
            b"nginx     123 root   6u  IPv4 12345      0t0  TCP *:80\n"
        )
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (lsof_output, b"")

        with patch("probe_agent.tools.shell.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await check_port(80)

        assert result["in_use"] is True
        assert result["process"] == "nginx"
        assert result["pid"] == 123


# ---------------------------------------------------------------------------
# 4. env_vars
# ---------------------------------------------------------------------------


class TestEnvVars:
    """Tests for shell_env_vars."""

    @pytest.mark.asyncio
    async def test_returns_env_variables(self) -> None:
        """Returns at least PATH."""
        result = await env_vars()
        assert result["count"] > 0
        assert "PATH" in result["variables"]

    @pytest.mark.asyncio
    async def test_filter_pattern(self) -> None:
        """Filter restricts to matching variable names."""
        result = await env_vars(filter_pattern="PATH")
        for key in result["variables"]:
            assert "PATH" in key.upper()

    @pytest.mark.asyncio
    async def test_masks_secrets(self) -> None:
        """Variables with secret-like names are masked."""
        with patch.dict(os.environ, {"MY_SECRET_KEY": "supersecretvalue123"}):
            result = await env_vars(filter_pattern="MY_SECRET_KEY")

        val = result["variables"]["MY_SECRET_KEY"]
        assert val.startswith("supe")
        assert "****" in val
        assert "supersecretvalue123" != val


# ---------------------------------------------------------------------------
# 5. curl
# ---------------------------------------------------------------------------


class TestCurl:
    """Tests for shell_curl."""

    @pytest.mark.asyncio
    async def test_successful_get(self) -> None:
        """GET request returns status, headers, and body."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"ok": true}'

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("probe_agent.tools.shell.httpx.AsyncClient", return_value=mock_client):
            result = await curl("http://example.com/api")

        assert result["status_code"] == 200
        assert result["method"] == "GET"
        assert result["url"] == "http://example.com/api"
        assert '{"ok": true}' in result["body"]
        assert result["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_post_with_body(self) -> None:
        """POST sends body and custom headers."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.headers = {}
        mock_response.text = "created"

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("probe_agent.tools.shell.httpx.AsyncClient", return_value=mock_client):
            result = await curl(
                "http://example.com/api",
                method="POST",
                headers={"Content-Type": "application/json"},
                body='{"name": "test"}',
            )

        assert result["method"] == "POST"
        assert result["status_code"] == 201

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self) -> None:
        """Timeout exception returns status_code -1."""
        mock_client = AsyncMock()
        mock_client.request.side_effect = httpx.TimeoutException("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("probe_agent.tools.shell.httpx.AsyncClient", return_value=mock_client):
            result = await curl("http://example.com/slow")

        assert result["status_code"] == -1
        assert "timed out" in result["body"]

    @pytest.mark.asyncio
    async def test_truncates_large_body(self) -> None:
        """Response body larger than 5000 chars is truncated."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.text = "a" * 10_000

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("probe_agent.tools.shell.httpx.AsyncClient", return_value=mock_client):
            result = await curl("http://example.com/big")

        assert "[truncated" in result["body"]
        assert len(result["body"]) < 6_000


# ---------------------------------------------------------------------------
# 6. process_list
# ---------------------------------------------------------------------------


class TestProcessList:
    """Tests for shell_process_list."""

    @pytest.mark.asyncio
    async def test_returns_processes(self) -> None:
        """Returns at least one process."""
        result = await process_list()
        assert result["count"] > 0
        p = result["processes"][0]
        assert "pid" in p
        assert "user" in p
        assert "command" in p

    @pytest.mark.asyncio
    async def test_filter_narrows_results(self) -> None:
        """Filtering with an impossible pattern returns zero processes."""
        result = await process_list(filter_pattern="ZZZYYYXXX_NO_SUCH_PROCESS")
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_cap_at_50(self) -> None:
        """Results are capped at 50."""
        # Generate fake ps output with 100 processes.
        header = "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND"
        lines = [header]
        for i in range(100):
            lines.append(f"root       {i+1}  0.1  0.2  12345  6789 ?        S    00:00   0:00 fake_proc_{i}")

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = ("\n".join(lines).encode(), b"")

        with patch("probe_agent.tools.shell.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await process_list()

        assert result["count"] == 50


# ---------------------------------------------------------------------------
# 7. disk_usage
# ---------------------------------------------------------------------------


class TestDiskUsage:
    """Tests for shell_disk_usage."""

    @pytest.mark.asyncio
    async def test_root_disk(self) -> None:
        """Disk usage of / returns sensible numbers."""
        result = await disk_usage("/")
        assert "error" not in result
        assert result["path"] == "/"
        assert result["total_gb"] > 0
        assert result["free_gb"] >= 0
        assert 0 <= result["percent_used"] <= 100

    @pytest.mark.asyncio
    async def test_invalid_path(self) -> None:
        """Invalid path returns an error dict."""
        result = await disk_usage("/nonexistent/path/abc123")
        assert result["error_type"] == "OSError"


# ---------------------------------------------------------------------------
# 8. system_info
# ---------------------------------------------------------------------------


class TestSystemInfo:
    """Tests for shell_system_info."""

    @pytest.mark.asyncio
    async def test_returns_system_fields(self) -> None:
        """system_info returns OS, hostname, CPU count, Python version."""
        result = await system_info()
        assert platform.system() in result["os"]
        assert result["hostname"] != ""
        assert result["cpu_count"] > 0
        assert result["python_version"] == platform.python_version()

    @pytest.mark.asyncio
    async def test_memory_is_reported(self) -> None:
        """Memory is reported on macOS and Linux."""
        result = await system_info()
        # May be None on exotic platforms but should work on macOS/Linux.
        if platform.system() in ("Darwin", "Linux"):
            assert result["memory_total_gb"] is not None
            assert result["memory_total_gb"] > 0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Tests for register_shell_tools."""

    def test_registers_8_tools(self) -> None:
        """register_shell_tools populates the registry with exactly 8 tools."""
        registry = ToolRegistry()
        register_shell_tools(registry)
        assert registry.count() == 8
        assert registry.list_namespaces() == ["shell"]

    def test_all_tool_names_start_with_shell(self) -> None:
        """Every registered tool has the 'shell_' namespace prefix."""
        registry = ToolRegistry()
        register_shell_tools(registry)
        for name in registry.list_tools():
            assert name.startswith("shell_"), f"{name} missing shell_ prefix"

    def test_expected_tool_names(self) -> None:
        """Verify all 8 expected tool names are registered."""
        registry = ToolRegistry()
        register_shell_tools(registry)
        expected = {
            "shell_run", "shell_run_in_dir", "shell_check_port",
            "shell_env_vars", "shell_curl", "shell_process_list",
            "shell_disk_usage", "shell_system_info",
        }
        assert set(registry.list_tools()) == expected

    def test_schemas_have_required_keys(self) -> None:
        """Every schema has name, description, and parameters."""
        registry = ToolRegistry()
        register_shell_tools(registry)
        for schema in registry.get_schemas():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert len(schema["description"]) > 50, (
                f"Tool {schema['name']} description is too short for the LLM"
            )
