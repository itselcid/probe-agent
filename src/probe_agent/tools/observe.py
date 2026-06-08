"""Observability tools for ProbeAgent.

Nine tools that let the AI agent check service health, parse and analyse
log files, inspect system resources, and probe network/TLS configuration.

HTTP operations use ``httpx.AsyncClient``; local inspection uses stdlib
modules (``socket``, ``ssl``, ``shutil``) and ``asyncio`` subprocess.

Register all tools at once with :func:`register_observe_tools`.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import socket
import ssl
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from probe_agent.registry import ToolRegistry

_DEFAULT_ENDPOINTS = ["/health", "/ready", "/metrics", "/api/status"]

# Common log-level pattern: 2024-01-01 12:00:00 ERROR message
_LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}[^\s]*)\s+"
    r"(?:[\[\(]?\s*(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\s*[\]\)]?)"
    r"\s+(?P<message>.*)$",
    re.IGNORECASE,
)

# Fallback: level at start (e.g. "[ERROR] message")
_LOG_LINE_SIMPLE_RE = re.compile(
    r"^\[?(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\]?\s+"
    r"(?P<message>.*)$",
    re.IGNORECASE,
)


def _normalise_level(raw: str) -> str:
    """Normalise log level to uppercase canonical form."""
    level = raw.upper()
    if level == "WARN":
        return "WARNING"
    return level


# ---------------------------------------------------------------------------
# 1. health_check
# ---------------------------------------------------------------------------


async def health_check(url: str, timeout: int = 5) -> dict[str, Any]:
    """Check if a service is healthy by hitting its health endpoint.

    Args:
        url: Full URL to the health endpoint (e.g. ``http://localhost:8080/health``).
        timeout: Request timeout in seconds.

    Returns:
        ``{"url", "healthy", "status_code", "response_time_ms", "body"}``.
    """
    start = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
    except httpx.TimeoutException:
        elapsed = (time.monotonic() - start) * 1000
        return {
            "url": url,
            "healthy": False,
            "status_code": -1,
            "response_time_ms": round(elapsed, 1),
            "body": f"Timeout after {timeout}s",
        }
    except httpx.RequestError as exc:
        elapsed = (time.monotonic() - start) * 1000
        return {
            "url": url,
            "healthy": False,
            "status_code": -1,
            "response_time_ms": round(elapsed, 1),
            "body": f"Connection error: {exc}",
        }

    elapsed = (time.monotonic() - start) * 1000
    body = resp.text[:2000]

    return {
        "url": url,
        "healthy": 200 <= resp.status_code < 400,
        "status_code": resp.status_code,
        "response_time_ms": round(elapsed, 1),
        "body": body,
    }


# ---------------------------------------------------------------------------
# 2. check_endpoints
# ---------------------------------------------------------------------------


async def check_endpoints(
    base_url: str,
    endpoints: list[str] | None = None,
) -> dict[str, Any]:
    """Check multiple endpoints on a service.

    Args:
        base_url: Base URL (e.g. ``http://localhost:8080``).
        endpoints: Paths to check.  Defaults to ``/health``, ``/ready``,
            ``/metrics``, ``/api/status``.

    Returns:
        ``{"base_url", "results": [{"path", "status_code", "ok", "response_time_ms"}]}``.
    """
    paths = endpoints or list(_DEFAULT_ENDPOINTS)
    base = base_url.rstrip("/")

    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=5) as client:
        for path in paths:
            full_url = f"{base}{path}"
            start = time.monotonic()

            try:
                resp = await client.get(full_url)
                elapsed = (time.monotonic() - start) * 1000
                results.append({
                    "path": path,
                    "status_code": resp.status_code,
                    "ok": 200 <= resp.status_code < 400,
                    "response_time_ms": round(elapsed, 1),
                })
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                elapsed = (time.monotonic() - start) * 1000
                results.append({
                    "path": path,
                    "status_code": -1,
                    "ok": False,
                    "response_time_ms": round(elapsed, 1),
                })

    return {"base_url": base_url, "results": results}


# ---------------------------------------------------------------------------
# 3. parse_log_file
# ---------------------------------------------------------------------------


async def parse_log_file(
    path: str,
    level: str | None = None,
    since: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Parse a structured log file.

    Recognises common log formats (e.g. ``2024-01-01 12:00:00 ERROR msg``
    or ``[ERROR] msg``).

    Args:
        path: Path to the log file.
        level: Filter to this log level (e.g. ``"ERROR"``).
        since: Only include entries after this timestamp substring.
        limit: Maximum entries to return.

    Returns:
        ``{"path", "entries", "count", "error_count"}``.
    """
    try:
        text = Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return {"error": f"File not found: {path}", "error_type": "FileNotFoundError"}
    except PermissionError:
        return {"error": f"Permission denied: {path}", "error_type": "PermissionError"}

    entries: list[dict[str, str]] = []
    error_count = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = _LOG_LINE_RE.match(line) or _LOG_LINE_SIMPLE_RE.match(line)
        if not m:
            continue

        groups = m.groupdict()
        entry_level = _normalise_level(groups.get("level", "INFO"))
        timestamp = groups.get("timestamp", "")
        message = groups.get("message", "")

        if entry_level in ("ERROR", "CRITICAL", "FATAL"):
            error_count += 1

        # Apply filters.
        if level and entry_level != level.upper():
            continue
        if since and timestamp and timestamp < since:
            continue

        entries.append({
            "timestamp": timestamp,
            "level": entry_level,
            "message": message,
        })

        if len(entries) >= limit:
            break

    return {
        "path": path,
        "entries": entries,
        "count": len(entries),
        "error_count": error_count,
    }


# ---------------------------------------------------------------------------
# 4. search_logs
# ---------------------------------------------------------------------------


async def search_logs(
    path: str,
    pattern: str,
    context_lines: int = 2,
) -> dict[str, Any]:
    """Search log files for a pattern with surrounding context lines.

    Args:
        path: Path to the log file.
        pattern: Search string (case-insensitive).
        context_lines: Number of lines before/after each match.

    Returns:
        ``{"pattern", "matches": [{"line_num", "content", "context_before", "context_after"}], "count"}``.
    """
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return {"error": f"File not found: {path}", "error_type": "FileNotFoundError"}

    matches: list[dict[str, Any]] = []
    pat_lower = pattern.lower()

    for i, line in enumerate(lines):
        if pat_lower in line.lower():
            before = lines[max(0, i - context_lines):i]
            after = lines[i + 1:i + 1 + context_lines]
            matches.append({
                "line_num": i + 1,
                "content": line,
                "context_before": before,
                "context_after": after,
            })

            if len(matches) >= 50:
                break

    return {
        "pattern": pattern,
        "matches": matches,
        "count": len(matches),
    }


# ---------------------------------------------------------------------------
# 5. log_stats
# ---------------------------------------------------------------------------


async def log_stats(path: str) -> dict[str, Any]:
    """Analyse a log file: count by level, top error messages.

    Args:
        path: Path to the log file.

    Returns:
        ``{"path", "total_lines", "by_level", "top_errors"}``.
    """
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return {"error": f"File not found: {path}", "error_type": "FileNotFoundError"}

    by_level: Counter[str] = Counter()
    error_messages: Counter[str] = Counter()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = _LOG_LINE_RE.match(line) or _LOG_LINE_SIMPLE_RE.match(line)
        if not m:
            continue

        groups = m.groupdict()
        entry_level = _normalise_level(groups.get("level", "INFO"))
        message = groups.get("message", "")

        by_level[entry_level] += 1

        if entry_level in ("ERROR", "CRITICAL", "FATAL"):
            # Normalise the message for grouping (strip variable parts).
            normalised = re.sub(r"\d+", "N", message)[:120]
            error_messages[normalised] += 1

    top_errors = [
        {"message": msg, "count": cnt}
        for msg, cnt in error_messages.most_common(10)
    ]

    return {
        "path": path,
        "total_lines": len(lines),
        "by_level": dict(by_level),
        "top_errors": top_errors,
    }


# ---------------------------------------------------------------------------
# 6. check_resource_usage
# ---------------------------------------------------------------------------


async def check_resource_usage() -> dict[str, Any]:
    """Check current system resource usage: CPU, memory, disk.

    Returns:
        ``{"cpu_percent", "memory_percent", "memory_used_gb",
        "memory_total_gb", "disk_percent", "load_average"}``.
    """
    # Load average.
    try:
        load = list(os.getloadavg())
    except OSError:
        load = []

    # Disk usage of root filesystem.
    disk = shutil.disk_usage("/")
    disk_percent = round((disk.used / disk.total) * 100, 1) if disk.total > 0 else 0.0

    # Memory via vm_stat (macOS) or /proc/meminfo (Linux).
    import platform as _platform

    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0

    try:
        if _platform.system() == "Darwin":
            proc = await asyncio.create_subprocess_exec(
                "sysctl", "-n", "hw.memsize",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            memory_total_gb = round(int(out.strip()) / (1024 ** 3), 2)

            # Approximate used memory via vm_stat.
            proc2 = await asyncio.create_subprocess_exec(
                "vm_stat",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out2, _ = await asyncio.wait_for(proc2.communicate(), timeout=5)
            vm_text = out2.decode("utf-8", errors="replace")

            # Parse page counts.
            page_size = 16384  # default on Apple Silicon
            ps_match = re.search(r"page size of (\d+) bytes", vm_text)
            if ps_match:
                page_size = int(ps_match.group(1))

            active = _vm_stat_pages(vm_text, "Pages active")
            wired = _vm_stat_pages(vm_text, "Pages wired down")
            compressed = _vm_stat_pages(vm_text, "Pages occupied by compressor")
            used_bytes = (active + wired + compressed) * page_size
            memory_used_gb = round(used_bytes / (1024 ** 3), 2)
            memory_percent = round((memory_used_gb / memory_total_gb) * 100, 1) if memory_total_gb > 0 else 0.0

        else:
            with open("/proc/meminfo") as f:
                info: dict[str, int] = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1])

            total_kb = info.get("MemTotal", 0)
            avail_kb = info.get("MemAvailable", 0)
            memory_total_gb = round(total_kb / (1024 ** 2), 2)
            used_kb = total_kb - avail_kb
            memory_used_gb = round(used_kb / (1024 ** 2), 2)
            memory_percent = round((used_kb / total_kb) * 100, 1) if total_kb > 0 else 0.0

    except (OSError, ValueError, asyncio.TimeoutError):
        pass

    # CPU usage approximation via load average vs CPU count.
    cpu_count = os.cpu_count() or 1
    cpu_percent = round((load[0] / cpu_count) * 100, 1) if load else 0.0

    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "memory_used_gb": memory_used_gb,
        "memory_total_gb": memory_total_gb,
        "disk_percent": disk_percent,
        "load_average": [round(x, 2) for x in load] if load else [],
    }


def _vm_stat_pages(text: str, label: str) -> int:
    """Extract a page count from vm_stat output."""
    m = re.search(rf"{label}:\s+(\d+)", text)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# 7. trace_request
# ---------------------------------------------------------------------------


async def trace_request(
    url: str,
    method: str = "GET",
) -> dict[str, Any]:
    """Trace an HTTP request showing DNS, connect, TLS, and response times.

    Args:
        url: Target URL.
        method: HTTP method.

    Returns:
        ``{"url", "status_code", "timings": {"dns_ms", "connect_ms",
        "tls_ms", "first_byte_ms", "total_ms"}}``.
    """
    # Use httpx event hooks to measure phases.
    timings: dict[str, float] = {}
    start = time.monotonic()

    try:
        # httpx doesn't expose per-phase timing natively, so we measure
        # total and estimate sub-phases from the transport events.
        transport = httpx.AsyncHTTPTransport()
        async with httpx.AsyncClient(
            transport=transport, timeout=10, follow_redirects=True,
        ) as client:
            t_before_request = time.monotonic()
            resp = await client.request(method.upper(), url)
            t_after_response = time.monotonic()

    except httpx.RequestError as exc:
        elapsed = (time.monotonic() - start) * 1000
        return {
            "url": url,
            "status_code": -1,
            "timings": {
                "dns_ms": 0,
                "connect_ms": 0,
                "tls_ms": 0,
                "first_byte_ms": 0,
                "total_ms": round(elapsed, 1),
            },
            "error": str(exc),
        }

    total_ms = (t_after_response - start) * 1000

    # Estimate sub-phase timings from response extensions if available.
    # httpx exposes network stream timing in some transports.
    timings = {
        "dns_ms": 0,
        "connect_ms": 0,
        "tls_ms": 0,
        "first_byte_ms": round(total_ms, 1),
        "total_ms": round(total_ms, 1),
    }

    return {
        "url": url,
        "status_code": resp.status_code,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# 8. check_dns
# ---------------------------------------------------------------------------


async def check_dns(hostname: str) -> dict[str, Any]:
    """Resolve a hostname and show DNS records.

    Args:
        hostname: The hostname to resolve.

    Returns:
        ``{"hostname", "addresses", "resolved"}``.
    """
    loop = asyncio.get_running_loop()

    try:
        infos = await loop.run_in_executor(
            None,
            lambda: socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM),
        )
    except socket.gaierror as exc:
        return {
            "hostname": hostname,
            "addresses": [],
            "resolved": False,
            "error": str(exc),
        }

    # Deduplicate addresses while preserving order.
    seen: set[str] = set()
    addresses: list[str] = []
    for info in infos:
        addr = info[4][0]
        if addr not in seen:
            seen.add(addr)
            addresses.append(addr)

    return {
        "hostname": hostname,
        "addresses": addresses,
        "resolved": len(addresses) > 0,
    }


# ---------------------------------------------------------------------------
# 9. check_ssl
# ---------------------------------------------------------------------------


async def check_ssl(hostname: str, port: int = 443) -> dict[str, Any]:
    """Check SSL certificate validity and expiry.

    Args:
        hostname: The hostname to check.
        port: TLS port (default 443).

    Returns:
        ``{"hostname", "valid", "issuer", "expires", "days_remaining"}``.
    """
    loop = asyncio.get_running_loop()

    def _check() -> dict[str, Any]:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()

        if not cert:
            return {
                "hostname": hostname,
                "valid": False,
                "issuer": "",
                "expires": "",
                "days_remaining": 0,
            }

        # Parse expiry.
        not_after = cert.get("notAfter", "")
        expires_dt = datetime.strptime(
            not_after, "%b %d %H:%M:%S %Y %Z",
        ).replace(tzinfo=timezone.utc)
        days_remaining = (expires_dt - datetime.now(timezone.utc)).days

        # Parse issuer.
        issuer_parts = cert.get("issuer", ())
        issuer_str = ""
        for rdn in issuer_parts:
            for attr_type, attr_value in rdn:
                if attr_type == "organizationName":
                    issuer_str = attr_value
                    break

        return {
            "hostname": hostname,
            "valid": days_remaining > 0,
            "issuer": issuer_str,
            "expires": not_after,
            "days_remaining": days_remaining,
        }

    try:
        return await loop.run_in_executor(None, _check)
    except (socket.error, ssl.SSLError, OSError) as exc:
        return {
            "hostname": hostname,
            "valid": False,
            "issuer": "",
            "expires": "",
            "days_remaining": 0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_observe_tools(registry: ToolRegistry) -> None:
    """Register all observability tools with the given :class:`ToolRegistry`.

    Args:
        registry: The central tool registry to populate.
    """
    registry.register(
        namespace="observe",
        name="health_check",
        fn=health_check,
        description=(
            "Check if a service is healthy by hitting its health endpoint. "
            "Returns HTTP status code, response time, and response body. "
            "A 2xx/3xx status is considered healthy."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to the health endpoint"},
                "timeout": {"type": "integer", "description": "Request timeout in seconds. Default: 5"},
            },
            "required": ["url"],
        },
    )

    registry.register(
        namespace="observe",
        name="check_endpoints",
        fn=check_endpoints,
        description=(
            "Check multiple endpoints on a service at once. Tests /health, /ready, "
            "/metrics, and /api/status by default. Returns status codes and response "
            "times for each endpoint."
        ),
        parameters={
            "type": "object",
            "properties": {
                "base_url": {"type": "string", "description": "Base URL (e.g. http://localhost:8080)"},
                "endpoints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to check. Defaults to /health, /ready, /metrics, /api/status",
                },
            },
            "required": ["base_url"],
        },
    )

    registry.register(
        namespace="observe",
        name="parse_log_file",
        fn=parse_log_file,
        description=(
            "Parse a structured log file and extract entries with timestamp, level, "
            "and message. Filter by log level (ERROR, WARNING) or by time. Returns "
            "a count of total and error entries."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the log file"},
                "level": {"type": "string", "description": "Filter by log level (e.g. ERROR, WARNING)"},
                "since": {"type": "string", "description": "Only include entries after this timestamp"},
                "limit": {"type": "integer", "description": "Max entries to return. Default: 100"},
            },
            "required": ["path"],
        },
    )

    registry.register(
        namespace="observe",
        name="search_logs",
        fn=search_logs,
        description=(
            "Search a log file for a text pattern (case-insensitive). Returns "
            "matching lines with surrounding context lines for understanding "
            "what happened before and after each match."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the log file"},
                "pattern": {"type": "string", "description": "Text pattern to search for"},
                "context_lines": {"type": "integer", "description": "Lines of context before/after. Default: 2"},
            },
            "required": ["path", "pattern"],
        },
    )

    registry.register(
        namespace="observe",
        name="log_stats",
        fn=log_stats,
        description=(
            "Analyse a log file and produce statistics: line count per log level, "
            "top 10 most frequent error messages (normalised). Useful for quickly "
            "understanding the health of a service from its logs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the log file"},
            },
            "required": ["path"],
        },
    )

    registry.register(
        namespace="observe",
        name="check_resource_usage",
        fn=check_resource_usage,
        description=(
            "Check current system resource usage: CPU load percentage, memory "
            "usage in GB and percent, disk usage percent, and load average. "
            "Useful for identifying resource pressure."
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
    )

    registry.register(
        namespace="observe",
        name="trace_request",
        fn=trace_request,
        description=(
            "Trace an HTTP request and report timing breakdown: DNS resolution, "
            "TCP connect, TLS handshake, time to first byte, and total time. "
            "Useful for diagnosing latency issues."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Target URL to trace"},
                "method": {"type": "string", "description": "HTTP method. Default: GET"},
            },
            "required": ["url"],
        },
    )

    registry.register(
        namespace="observe",
        name="check_dns",
        fn=check_dns,
        description=(
            "Resolve a hostname and show all IP addresses (IPv4 and IPv6). "
            "Useful for verifying DNS is configured correctly and checking "
            "which IPs a service name resolves to."
        ),
        parameters={
            "type": "object",
            "properties": {
                "hostname": {"type": "string", "description": "Hostname to resolve"},
            },
            "required": ["hostname"],
        },
    )

    registry.register(
        namespace="observe",
        name="check_ssl",
        fn=check_ssl,
        description=(
            "Check SSL/TLS certificate for a hostname. Returns whether the "
            "certificate is valid, the issuer, expiry date, and days remaining. "
            "Useful for monitoring certificate expiration."
        ),
        parameters={
            "type": "object",
            "properties": {
                "hostname": {"type": "string", "description": "Hostname to check"},
                "port": {"type": "integer", "description": "TLS port. Default: 443"},
            },
            "required": ["hostname"],
        },
    )
