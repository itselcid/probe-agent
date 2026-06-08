"""Tests for observability tools."""

from __future__ import annotations

import socket
import ssl
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from probe_agent.registry import ToolRegistry
from probe_agent.tools.observe import (
    check_dns,
    check_endpoints,
    check_resource_usage,
    check_ssl,
    health_check,
    log_stats,
    parse_log_file,
    register_observe_tools,
    search_logs,
    trace_request,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def log_file(tmp_path: Path) -> Path:
    """Create a sample log file for testing."""
    content = "\n".join([
        "2024-01-15 10:00:01 INFO Application started",
        "2024-01-15 10:00:02 INFO Listening on port 8080",
        "2024-01-15 10:01:00 WARNING Memory usage high: 85%",
        "2024-01-15 10:02:00 ERROR Connection refused: database",
        "2024-01-15 10:02:01 ERROR Connection refused: database",
        "2024-01-15 10:03:00 ERROR Timeout waiting for response",
        "2024-01-15 10:04:00 INFO Request processed in 150ms",
        "2024-01-15 10:05:00 ERROR Connection refused: database",
        "2024-01-15 10:06:00 DEBUG Detailed trace output",
        "2024-01-15 10:07:00 CRITICAL Out of memory",
    ])
    p = tmp_path / "app.log"
    p.write_text(content)
    return p


@pytest.fixture()
def simple_log_file(tmp_path: Path) -> Path:
    """Create a log file with the simple [LEVEL] format."""
    content = "\n".join([
        "[INFO] Server starting",
        "[ERROR] Failed to bind port",
        "[WARNING] Deprecated API used",
    ])
    p = tmp_path / "simple.log"
    p.write_text(content)
    return p


def _mock_httpx_client(status: int = 200, text: str = "ok", headers: dict | None = None):
    """Create a mock httpx.AsyncClient context manager."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.text = text
    mock_resp.headers = headers or {"content-type": "text/plain"}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.request.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ===========================================================================
# 1. health_check
# ===========================================================================


class TestHealthCheck:
    """Tests for observe_health_check."""

    @pytest.mark.asyncio
    async def test_healthy_service(self) -> None:
        """200 response → healthy=True."""
        mock_client = _mock_httpx_client(200, '{"status":"ok"}')

        with patch("probe_agent.tools.observe.httpx.AsyncClient", return_value=mock_client):
            result = await health_check("http://localhost:8080/health")

        assert result["healthy"] is True
        assert result["status_code"] == 200
        assert result["response_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_unhealthy_service(self) -> None:
        """503 response → healthy=False."""
        mock_client = _mock_httpx_client(503, "Service Unavailable")

        with patch("probe_agent.tools.observe.httpx.AsyncClient", return_value=mock_client):
            result = await health_check("http://localhost:8080/health")

        assert result["healthy"] is False
        assert result["status_code"] == 503

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        """Timeout → healthy=False, status_code=-1."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("probe_agent.tools.observe.httpx.AsyncClient", return_value=mock_client):
            result = await health_check("http://localhost:8080/health", timeout=1)

        assert result["healthy"] is False
        assert result["status_code"] == -1
        assert "Timeout" in result["body"]

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        """Connection refused → healthy=False."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("probe_agent.tools.observe.httpx.AsyncClient", return_value=mock_client):
            result = await health_check("http://localhost:9999/health")

        assert result["healthy"] is False
        assert "Connection error" in result["body"]


# ===========================================================================
# 2. check_endpoints
# ===========================================================================


class TestCheckEndpoints:
    """Tests for observe_check_endpoints."""

    @pytest.mark.asyncio
    async def test_checks_default_endpoints(self) -> None:
        """Default endpoints list is checked."""
        mock_client = _mock_httpx_client(200, "ok")

        with patch("probe_agent.tools.observe.httpx.AsyncClient", return_value=mock_client):
            result = await check_endpoints("http://localhost:8080")

        assert result["base_url"] == "http://localhost:8080"
        assert len(result["results"]) == 4  # 4 default endpoints
        for r in result["results"]:
            assert r["ok"] is True

    @pytest.mark.asyncio
    async def test_custom_endpoints(self) -> None:
        """Custom endpoint list is respected."""
        mock_client = _mock_httpx_client(200, "ok")

        with patch("probe_agent.tools.observe.httpx.AsyncClient", return_value=mock_client):
            result = await check_endpoints(
                "http://localhost:8080",
                endpoints=["/v1/ping", "/v2/ping"],
            )

        assert len(result["results"]) == 2
        assert result["results"][0]["path"] == "/v1/ping"

    @pytest.mark.asyncio
    async def test_failing_endpoint(self) -> None:
        """Connection error on an endpoint returns ok=False."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("probe_agent.tools.observe.httpx.AsyncClient", return_value=mock_client):
            result = await check_endpoints(
                "http://localhost:9999",
                endpoints=["/health"],
            )

        assert result["results"][0]["ok"] is False
        assert result["results"][0]["status_code"] == -1


# ===========================================================================
# 3. parse_log_file
# ===========================================================================


class TestParseLogFile:
    """Tests for observe_parse_log_file."""

    @pytest.mark.asyncio
    async def test_parses_all_entries(self, log_file: Path) -> None:
        """All recognised log entries are parsed."""
        result = await parse_log_file(str(log_file))
        assert result["count"] == 10
        assert result["error_count"] == 5  # 4 ERROR + 1 CRITICAL

    @pytest.mark.asyncio
    async def test_filter_by_level(self, log_file: Path) -> None:
        """Level filter restricts to matching entries."""
        result = await parse_log_file(str(log_file), level="ERROR")
        assert result["count"] == 4
        for entry in result["entries"]:
            assert entry["level"] == "ERROR"

    @pytest.mark.asyncio
    async def test_filter_by_since(self, log_file: Path) -> None:
        """Since filter restricts to entries after the timestamp."""
        result = await parse_log_file(str(log_file), since="2024-01-15 10:05:00")
        # Entries at 10:05, 10:06, 10:07 should pass.
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_limit_caps_results(self, log_file: Path) -> None:
        """Limit restricts the number of returned entries."""
        result = await parse_log_file(str(log_file), limit=3)
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_file_not_found(self) -> None:
        """Missing file returns error dict."""
        result = await parse_log_file("/nonexistent/app.log")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_simple_log_format(self, simple_log_file: Path) -> None:
        """Parses [LEVEL] format logs."""
        result = await parse_log_file(str(simple_log_file))
        assert result["count"] == 3
        levels = {e["level"] for e in result["entries"]}
        assert "ERROR" in levels
        assert "WARNING" in levels


# ===========================================================================
# 4. search_logs
# ===========================================================================


class TestSearchLogs:
    """Tests for observe_search_logs."""

    @pytest.mark.asyncio
    async def test_finds_pattern(self, log_file: Path) -> None:
        """Finds lines matching the pattern."""
        result = await search_logs(str(log_file), "Connection refused")
        assert result["count"] == 3
        assert result["pattern"] == "Connection refused"

    @pytest.mark.asyncio
    async def test_case_insensitive(self, log_file: Path) -> None:
        """Search is case-insensitive."""
        result = await search_logs(str(log_file), "connection REFUSED")
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_context_lines(self, log_file: Path) -> None:
        """Context lines are included around matches."""
        result = await search_logs(str(log_file), "Timeout waiting", context_lines=1)
        assert result["count"] == 1
        match = result["matches"][0]
        assert len(match["context_before"]) <= 1
        assert len(match["context_after"]) <= 1

    @pytest.mark.asyncio
    async def test_no_matches(self, log_file: Path) -> None:
        """No matches returns empty list."""
        result = await search_logs(str(log_file), "ZZZYYYXXX")
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_file_not_found(self) -> None:
        """Missing file returns error."""
        result = await search_logs("/nonexistent/file.log", "test")
        assert "error" in result


# ===========================================================================
# 5. log_stats
# ===========================================================================


class TestLogStats:
    """Tests for observe_log_stats."""

    @pytest.mark.asyncio
    async def test_counts_by_level(self, log_file: Path) -> None:
        """Counts entries per log level."""
        result = await log_stats(str(log_file))
        assert result["by_level"]["ERROR"] == 4
        assert result["by_level"]["WARNING"] == 1
        assert result["by_level"]["INFO"] == 3
        assert result["by_level"]["CRITICAL"] == 1

    @pytest.mark.asyncio
    async def test_top_errors(self, log_file: Path) -> None:
        """Top errors are grouped and counted."""
        result = await log_stats(str(log_file))
        assert len(result["top_errors"]) > 0
        # "Connection refused: database" appears 3 times (normalised).
        top = result["top_errors"][0]
        assert top["count"] == 3

    @pytest.mark.asyncio
    async def test_total_lines(self, log_file: Path) -> None:
        """total_lines reflects the file size."""
        result = await log_stats(str(log_file))
        assert result["total_lines"] == 10

    @pytest.mark.asyncio
    async def test_file_not_found(self) -> None:
        """Missing file returns error."""
        result = await log_stats("/nonexistent/app.log")
        assert "error" in result


# ===========================================================================
# 6. check_resource_usage
# ===========================================================================


class TestCheckResourceUsage:
    """Tests for observe_check_resource_usage."""

    @pytest.mark.asyncio
    async def test_returns_resource_fields(self) -> None:
        """Returns all expected resource fields."""
        result = await check_resource_usage()
        assert "cpu_percent" in result
        assert "memory_percent" in result
        assert "memory_total_gb" in result
        assert "disk_percent" in result
        assert "load_average" in result
        assert result["disk_percent"] >= 0


# ===========================================================================
# 7. trace_request
# ===========================================================================


class TestTraceRequest:
    """Tests for observe_trace_request."""

    @pytest.mark.asyncio
    async def test_successful_trace(self) -> None:
        """Successful request returns timings."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("probe_agent.tools.observe.httpx.AsyncClient", return_value=mock_client):
            result = await trace_request("http://example.com")

        assert result["status_code"] == 200
        assert result["timings"]["total_ms"] >= 0

    @pytest.mark.asyncio
    async def test_trace_connection_error(self) -> None:
        """Connection error returns status_code=-1 with error message."""
        mock_client = AsyncMock()
        mock_client.request.side_effect = httpx.ConnectError("refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("probe_agent.tools.observe.httpx.AsyncClient", return_value=mock_client):
            result = await trace_request("http://localhost:9999")

        assert result["status_code"] == -1
        assert "error" in result


# ===========================================================================
# 8. check_dns
# ===========================================================================


class TestCheckDns:
    """Tests for observe_check_dns."""

    @pytest.mark.asyncio
    async def test_successful_resolution(self) -> None:
        """Resolves a known hostname."""
        fake_infos = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800:220:1::34", 0, 0, 0)),
        ]

        with patch("probe_agent.tools.observe.socket.getaddrinfo", return_value=fake_infos):
            result = await check_dns("example.com")

        assert result["resolved"] is True
        assert "93.184.216.34" in result["addresses"]
        assert len(result["addresses"]) == 2

    @pytest.mark.asyncio
    async def test_unresolvable_hostname(self) -> None:
        """Unresolvable hostname returns resolved=False."""
        with patch(
            "probe_agent.tools.observe.socket.getaddrinfo",
            side_effect=socket.gaierror("Name does not resolve"),
        ):
            result = await check_dns("definitely.not.a.real.hostname.test")

        assert result["resolved"] is False
        assert result["addresses"] == []

    @pytest.mark.asyncio
    async def test_deduplicates_addresses(self) -> None:
        """Duplicate addresses are removed."""
        fake_infos = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("5.6.7.8", 0)),
        ]

        with patch("probe_agent.tools.observe.socket.getaddrinfo", return_value=fake_infos):
            result = await check_dns("example.com")

        assert len(result["addresses"]) == 2


# ===========================================================================
# 9. check_ssl
# ===========================================================================


class TestCheckSsl:
    """Tests for observe_check_ssl."""

    @pytest.mark.asyncio
    async def test_valid_certificate(self) -> None:
        """Valid certificate returns valid=True with issuer and expiry."""
        fake_cert = {
            "notAfter": "Dec 31 23:59:59 2030 GMT",
            "issuer": ((("organizationName", "Let's Encrypt"),),),
        }

        mock_ssock = MagicMock()
        mock_ssock.getpeercert.return_value = fake_cert
        mock_ssock.__enter__ = MagicMock(return_value=mock_ssock)
        mock_ssock.__exit__ = MagicMock(return_value=False)

        mock_sock = MagicMock()
        mock_sock.__enter__ = MagicMock(return_value=mock_sock)
        mock_sock.__exit__ = MagicMock(return_value=False)

        mock_ctx = MagicMock()
        mock_ctx.wrap_socket.return_value = mock_ssock

        with (
            patch("probe_agent.tools.observe.socket.create_connection", return_value=mock_sock),
            patch("probe_agent.tools.observe.ssl.create_default_context", return_value=mock_ctx),
        ):
            result = await check_ssl("example.com")

        assert result["valid"] is True
        assert result["issuer"] == "Let's Encrypt"
        assert result["days_remaining"] > 0

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        """Connection error returns valid=False."""
        with patch(
            "probe_agent.tools.observe.socket.create_connection",
            side_effect=socket.error("Connection refused"),
        ):
            result = await check_ssl("localhost", port=443)

        assert result["valid"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_no_certificate(self) -> None:
        """No certificate from peer returns valid=False."""
        mock_ssock = MagicMock()
        mock_ssock.getpeercert.return_value = None
        mock_ssock.__enter__ = MagicMock(return_value=mock_ssock)
        mock_ssock.__exit__ = MagicMock(return_value=False)

        mock_sock = MagicMock()
        mock_sock.__enter__ = MagicMock(return_value=mock_sock)
        mock_sock.__exit__ = MagicMock(return_value=False)

        mock_ctx = MagicMock()
        mock_ctx.wrap_socket.return_value = mock_ssock

        with (
            patch("probe_agent.tools.observe.socket.create_connection", return_value=mock_sock),
            patch("probe_agent.tools.observe.ssl.create_default_context", return_value=mock_ctx),
        ):
            result = await check_ssl("example.com")

        assert result["valid"] is False


# ===========================================================================
# Registration
# ===========================================================================


class TestRegistration:
    """Tests for register_observe_tools."""

    def test_registers_9_tools(self) -> None:
        """register_observe_tools populates the registry with exactly 9 tools."""
        registry = ToolRegistry()
        register_observe_tools(registry)
        assert registry.count() == 9
        assert registry.list_namespaces() == ["observe"]

    def test_all_names_start_with_observe(self) -> None:
        """Every registered tool has the 'observe_' namespace prefix."""
        registry = ToolRegistry()
        register_observe_tools(registry)
        for name in registry.list_tools():
            assert name.startswith("observe_"), f"{name} missing observe_ prefix"

    def test_expected_tool_names(self) -> None:
        """Verify all 9 expected tool names are registered."""
        registry = ToolRegistry()
        register_observe_tools(registry)
        expected = {
            "observe_health_check",
            "observe_check_endpoints",
            "observe_parse_log_file",
            "observe_search_logs",
            "observe_log_stats",
            "observe_check_resource_usage",
            "observe_trace_request",
            "observe_check_dns",
            "observe_check_ssl",
        }
        assert set(registry.list_tools()) == expected

    def test_schemas_have_required_keys(self) -> None:
        """Every schema has name, description, and parameters."""
        registry = ToolRegistry()
        register_observe_tools(registry)
        for schema in registry.get_schemas():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert len(schema["description"]) > 50
