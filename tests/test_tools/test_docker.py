"""Tests for Docker tools (fully mocked — no Docker daemon required)."""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from probe_agent.registry import ToolRegistry
from probe_agent.tools.docker_tools import (
    compose_down,
    compose_logs,
    compose_ps,
    compose_up,
    exec_command,
    inspect,
    logs,
    network_inspect,
    ps,
    register_docker_tools,
    restart,
    stats,
    stop,
)


# ---------------------------------------------------------------------------
# Helpers — mock factories
# ---------------------------------------------------------------------------


def _mock_container(
    *,
    id: str = "abc123def456",
    name: str = "web-1",
    image_tags: list[str] | None = None,
    status: str = "running",
    ports: dict | None = None,
    attrs: dict | None = None,
) -> MagicMock:
    """Build a mock Docker container object."""
    c = MagicMock()
    c.id = id
    c.name = name
    c.status = status
    c.ports = ports or {}

    mock_image = MagicMock()
    mock_image.tags = image_tags or ["nginx:latest"]
    c.image = mock_image

    c.attrs = attrs or {
        "Id": id,
        "Name": f"/{name}",
        "Created": "2025-01-01T00:00:00Z",
        "State": {
            "Status": status,
            "Running": status == "running",
            "StartedAt": "2025-01-01T00:00:00Z",
            "OOMKilled": False,
        },
        "Config": {
            "Image": "nginx:latest",
            "Env": ["NGINX_HOST=localhost", "NGINX_PORT=80"],
        },
        "NetworkSettings": {
            "IPAddress": "172.17.0.2",
            "Ports": {"80/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8080"}]},
        },
        "Mounts": [
            {
                "Source": "/data/html",
                "Destination": "/usr/share/nginx/html",
                "Mode": "ro",
                "Type": "bind",
            },
        ],
        "HostConfig": {
            "RestartPolicy": {"Name": "always"},
        },
        "RestartCount": 0,
    }

    return c


def _mock_client(containers: list[MagicMock] | None = None) -> MagicMock:
    """Build a mock Docker client."""
    client = MagicMock()
    client.ping.return_value = True
    client.containers.list.return_value = containers or []
    if containers:
        client.containers.get.side_effect = lambda name: next(
            (c for c in containers if c.name == name or c.id.startswith(name)),
            MagicMock(side_effect=Exception("not found")),
        )
    return client


# ---------------------------------------------------------------------------
# ps
# ---------------------------------------------------------------------------


class TestPs:
    """Tests for docker_ps."""

    @pytest.mark.asyncio
    async def test_lists_containers(self) -> None:
        """ps returns container info for running containers."""
        mock_c = _mock_container()
        client = _mock_client([mock_c])

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await ps()

        assert "error" not in result
        assert result["count"] == 1
        c = result["containers"][0]
        assert c["name"] == "web-1"
        assert c["id"] == "abc123def456"
        assert "nginx" in c["image"]

    @pytest.mark.asyncio
    async def test_empty_list(self) -> None:
        """ps returns empty list when no containers running."""
        client = _mock_client([])
        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await ps()
        assert result["count"] == 0
        assert result["containers"] == []


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


class TestInspect:
    """Tests for docker_inspect."""

    @pytest.mark.asyncio
    async def test_inspect_returns_curated_fields(self) -> None:
        """Inspect returns state, network, mounts, env — not raw attrs."""
        mock_c = _mock_container()
        client = _mock_client([mock_c])

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await inspect("web-1")

        assert "error" not in result
        assert result["name"] == "web-1"
        assert result["state"]["running"] is True
        assert result["state"]["oom_killed"] is False
        assert result["network"]["ip_address"] == "172.17.0.2"
        assert len(result["mounts"]) == 1
        assert result["environment"]["NGINX_HOST"] == "localhost"
        assert result["restart_policy"] == "always"

    @pytest.mark.asyncio
    async def test_inspect_not_found(self) -> None:
        """Inspect returns error for unknown container."""
        import docker.errors

        client = MagicMock()
        client.ping.return_value = True
        client.containers.get.side_effect = docker.errors.NotFound("nope")

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await inspect("ghost")

        assert result["error_type"] == "NotFound"


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


class TestLogs:
    """Tests for docker_logs."""

    @pytest.mark.asyncio
    async def test_logs_returns_text(self) -> None:
        """Logs returns decoded log text and line count."""
        mock_c = _mock_container()
        mock_c.logs.return_value = b"2025-01-01 GET /\n2025-01-01 GET /health\n"
        client = _mock_client([mock_c])

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await logs("web-1")

        assert "error" not in result
        assert result["lines"] == 2
        assert "/health" in result["logs"]

    @pytest.mark.asyncio
    async def test_logs_grep_filter(self) -> None:
        """Grep filter reduces log lines to matching ones."""
        mock_c = _mock_container()
        mock_c.logs.return_value = b"ERROR disk full\nINFO started\nERROR timeout\n"
        client = _mock_client([mock_c])

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await logs("web-1", grep="ERROR")

        assert result["lines"] == 2
        assert "INFO" not in result["logs"]


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    """Tests for docker_stats."""

    @pytest.mark.asyncio
    async def test_stats_returns_cpu_mem(self) -> None:
        """Stats calculates CPU and memory percentages."""
        mock_c = _mock_container()
        mock_c.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 200},
                "system_cpu_usage": 1000,
                "online_cpus": 2,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 100},
                "system_cpu_usage": 500,
            },
            "memory_stats": {
                "usage": 100 * 1024 * 1024,  # 100 MB
                "limit": 512 * 1024 * 1024,  # 512 MB
            },
        }
        client = _mock_client([mock_c])

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await stats("web-1")

        assert len(result["stats"]) == 1
        s = result["stats"][0]
        assert s["name"] == "web-1"
        assert s["cpu_percent"] > 0
        assert s["memory_mb"] == 100.0
        assert s["memory_limit_mb"] == 512.0
        assert s["memory_percent"] > 0


# ---------------------------------------------------------------------------
# restart / stop
# ---------------------------------------------------------------------------


class TestRestartStop:
    """Tests for docker_restart and docker_stop."""

    @pytest.mark.asyncio
    async def test_restart_success(self) -> None:
        """Restart calls container.restart() and returns status."""
        mock_c = _mock_container()
        client = _mock_client([mock_c])

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await restart("web-1")

        assert result["status"] == "restarted"
        mock_c.restart.assert_called_once_with(timeout=10)

    @pytest.mark.asyncio
    async def test_stop_success(self) -> None:
        """Stop calls container.stop() and returns status."""
        mock_c = _mock_container()
        client = _mock_client([mock_c])

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await stop("web-1")

        assert result["status"] == "stopped"
        mock_c.stop.assert_called_once_with(timeout=10)


# ---------------------------------------------------------------------------
# exec_command
# ---------------------------------------------------------------------------


class TestExecCommand:
    """Tests for docker_exec_command."""

    @pytest.mark.asyncio
    async def test_exec_success(self) -> None:
        """Exec returns stdout, stderr, and exit code."""
        mock_c = _mock_container()
        mock_c.exec_run.return_value = (0, (b"hello\n", b""))
        client = _mock_client([mock_c])

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await exec_command("web-1", "echo hello")

        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]
        assert result["command"] == "echo hello"

    @pytest.mark.asyncio
    async def test_exec_truncates_long_output(self) -> None:
        """Output longer than 5000 chars is truncated."""
        mock_c = _mock_container()
        long_output = b"x" * 10000
        mock_c.exec_run.return_value = (0, (long_output, b""))
        client = _mock_client([mock_c])

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await exec_command("web-1", "cat bigfile")

        assert "[truncated]" in result["stdout"]
        assert len(result["stdout"]) < 10000


# ---------------------------------------------------------------------------
# compose_ps
# ---------------------------------------------------------------------------


class TestComposePs:
    """Tests for docker_compose_ps."""

    @pytest.mark.asyncio
    async def test_compose_ps_parses_json(self) -> None:
        """compose_ps parses JSON output from docker compose ps."""
        json_output = '[{"Name":"web-1","State":"running","Ports":"8080->80/tcp"}]'
        with patch(
            "probe_agent.tools.docker_tools._run_compose",
            return_value=(0, json_output, ""),
        ):
            result = await compose_ps()

        assert result["count"] == 1
        assert result["services"][0]["name"] == "web-1"

    @pytest.mark.asyncio
    async def test_compose_ps_line_by_line_json(self) -> None:
        """compose_ps handles line-by-line JSON (newer docker compose)."""
        json_output = '{"Name":"web","State":"running"}\n{"Name":"db","State":"running"}\n'
        with patch(
            "probe_agent.tools.docker_tools._run_compose",
            return_value=(0, json_output, ""),
        ):
            result = await compose_ps()

        assert result["count"] == 2


# ---------------------------------------------------------------------------
# compose_up / compose_down
# ---------------------------------------------------------------------------


class TestComposeUpDown:
    """Tests for docker_compose_up and docker_compose_down."""

    @pytest.mark.asyncio
    async def test_compose_up_success(self) -> None:
        """compose_up returns action and service name."""
        with patch(
            "probe_agent.tools.docker_tools._run_compose",
            return_value=(0, "Starting web-1 ... done\n", ""),
        ):
            result = await compose_up(service="web")

        assert result["action"] == "up"
        assert result["service"] == "web"

    @pytest.mark.asyncio
    async def test_compose_up_all(self) -> None:
        """compose_up with no service starts all."""
        with patch(
            "probe_agent.tools.docker_tools._run_compose",
            return_value=(0, "Starting ...\n", ""),
        ):
            result = await compose_up()
        assert result["service"] == "all"

    @pytest.mark.asyncio
    async def test_compose_down_success(self) -> None:
        """compose_down returns action and output."""
        with patch(
            "probe_agent.tools.docker_tools._run_compose",
            return_value=(0, "Stopping ...\nRemoving ...\n", ""),
        ):
            result = await compose_down()

        assert result["action"] == "down"

    @pytest.mark.asyncio
    async def test_compose_down_failure(self) -> None:
        """compose_down returns error on failure."""
        with patch(
            "probe_agent.tools.docker_tools._run_compose",
            return_value=(1, "", "no configuration file provided"),
        ):
            result = await compose_down()

        assert "error" in result
        assert result["error_type"] == "ComposeError"


# ---------------------------------------------------------------------------
# compose_logs
# ---------------------------------------------------------------------------


class TestComposeLogs:
    """Tests for docker_compose_logs."""

    @pytest.mark.asyncio
    async def test_compose_logs_returns_text(self) -> None:
        """compose_logs returns log text and line count."""
        with patch(
            "probe_agent.tools.docker_tools._run_compose",
            return_value=(0, "web-1  | GET /\nweb-1  | GET /health\n", ""),
        ):
            result = await compose_logs(service="web")

        assert result["service"] == "web"
        assert result["lines"] == 2


# ---------------------------------------------------------------------------
# network_inspect
# ---------------------------------------------------------------------------


class TestNetworkInspect:
    """Tests for docker_network_inspect."""

    @pytest.mark.asyncio
    async def test_network_inspect_returns_containers(self) -> None:
        """network_inspect lists connected containers with IPs."""
        mock_net = MagicMock()
        mock_net.attrs = {
            "Name": "app_default",
            "Driver": "bridge",
            "Containers": {
                "abc123": {"Name": "web-1", "IPv4Address": "172.18.0.2/16"},
                "def456": {"Name": "db-1", "IPv4Address": "172.18.0.3/16"},
            },
        }

        client = MagicMock()
        client.ping.return_value = True
        client.networks.get.return_value = mock_net

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await network_inspect("app_default")

        assert result["name"] == "app_default"
        assert result["driver"] == "bridge"
        assert len(result["containers"]) == 2
        names = [c["name"] for c in result["containers"]]
        assert "web-1" in names
        assert "db-1" in names

    @pytest.mark.asyncio
    async def test_network_not_found(self) -> None:
        """network_inspect returns error for unknown network."""
        import docker.errors

        client = MagicMock()
        client.ping.return_value = True
        client.networks.get.side_effect = docker.errors.NotFound("nope")

        with patch("probe_agent.tools.docker_tools._get_docker_client", return_value=client):
            result = await network_inspect("ghost_net")

        assert result["error_type"] == "NotFound"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Tests for register_docker_tools."""

    def test_registers_12_tools(self) -> None:
        """register_docker_tools populates the registry with exactly 12 tools."""
        registry = ToolRegistry()
        register_docker_tools(registry)
        assert registry.count() == 12
        assert registry.list_namespaces() == ["docker"]

    def test_all_tool_names_start_with_docker(self) -> None:
        """Every registered tool has the 'docker_' namespace prefix."""
        registry = ToolRegistry()
        register_docker_tools(registry)
        for name in registry.list_tools():
            assert name.startswith("docker_"), f"{name} missing docker_ prefix"

    def test_schemas_have_required_keys(self) -> None:
        """Every schema has name, description, and parameters."""
        registry = ToolRegistry()
        register_docker_tools(registry)
        for schema in registry.get_schemas():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert len(schema["description"]) > 50, (
                f"Tool {schema['name']} description is too short for the LLM"
            )

    def test_expected_tool_names(self) -> None:
        """Verify all 12 expected tool names are registered."""
        registry = ToolRegistry()
        register_docker_tools(registry)
        expected = {
            "docker_ps", "docker_inspect", "docker_logs", "docker_stats",
            "docker_restart", "docker_stop", "docker_exec_command",
            "docker_compose_ps", "docker_compose_up", "docker_compose_down",
            "docker_compose_logs", "docker_network_inspect",
        }
        assert set(registry.list_tools()) == expected
