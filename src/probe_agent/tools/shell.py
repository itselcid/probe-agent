"""Shell and system tools for ProbeAgent.

Eight tools that let the AI agent run shell commands, make HTTP requests,
and inspect the host system.  All commands enforce a timeout and truncate
output to prevent overwhelming the LLM context.

Safety rules enforced everywhere:
- Every subprocess has a configurable timeout (default 30 s).
- stdout/stderr are truncated to 10 000 characters.
- HTTP response bodies are truncated to 5 000 characters.
- Environment variables containing secrets are automatically masked.

Register all tools at once with :func:`register_shell_tools`.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import time
from typing import Any

import httpx

from probe_agent.registry import ToolRegistry

_OUTPUT_MAX = 10_000
_BODY_MAX = 5_000

# Patterns in env-var names that trigger masking.
_SECRET_PATTERNS = re.compile(
    r"(SECRET|KEY|PASSWORD|TOKEN|CREDENTIAL|PRIVATE)", re.IGNORECASE,
)


def _truncate(text: str, limit: int) -> str:
    """Truncate *text* to *limit* characters, appending a notice."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated — {len(text)} chars total]"


def _mask_value(key: str, value: str) -> str:
    """Mask secret values, showing only the first 4 characters."""
    if _SECRET_PATTERNS.search(key):
        return value[:4] + "****" if len(value) > 4 else "****"
    return value


# ---------------------------------------------------------------------------
# 1. run
# ---------------------------------------------------------------------------


async def run(command: str, timeout: int = 30) -> dict[str, Any]:
    """Run a shell command and return its output.

    Args:
        command: Shell command string (passed to ``sh -c``).
        timeout: Maximum seconds before the process is killed.

    Returns:
        ``{"command", "stdout", "stderr", "return_code", "duration_ms"}``.
        stdout and stderr are truncated to 10 000 characters.
    """
    start = time.monotonic()

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()  # type: ignore[union-attr]
        await proc.communicate()  # type: ignore[union-attr]
        duration_ms = (time.monotonic() - start) * 1000
        return {
            "command": command,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "return_code": -1,
            "duration_ms": round(duration_ms, 1),
        }

    duration_ms = (time.monotonic() - start) * 1000
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    return {
        "command": command,
        "stdout": _truncate(stdout, _OUTPUT_MAX),
        "stderr": _truncate(stderr, _OUTPUT_MAX),
        "return_code": proc.returncode or 0,
        "duration_ms": round(duration_ms, 1),
    }


# ---------------------------------------------------------------------------
# 2. run_in_dir
# ---------------------------------------------------------------------------


async def run_in_dir(
    command: str,
    directory: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Run a shell command in a specific directory.

    Args:
        command: Shell command string.
        directory: Working directory for the command.
        timeout: Maximum seconds before the process is killed.

    Returns:
        Same shape as :func:`run`, plus ``"directory"``.
    """
    start = time.monotonic()

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=directory,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()  # type: ignore[union-attr]
        await proc.communicate()  # type: ignore[union-attr]
        duration_ms = (time.monotonic() - start) * 1000
        return {
            "command": command,
            "directory": directory,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "return_code": -1,
            "duration_ms": round(duration_ms, 1),
        }

    duration_ms = (time.monotonic() - start) * 1000
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    return {
        "command": command,
        "directory": directory,
        "stdout": _truncate(stdout, _OUTPUT_MAX),
        "stderr": _truncate(stderr, _OUTPUT_MAX),
        "return_code": proc.returncode or 0,
        "duration_ms": round(duration_ms, 1),
    }


# ---------------------------------------------------------------------------
# 3. check_port
# ---------------------------------------------------------------------------


async def check_port(port: int) -> dict[str, Any]:
    """Check if a port is in use and what process is using it.

    Args:
        port: TCP port number to check.

    Returns:
        ``{"port", "in_use", "process", "pid"}``.
    """
    proc = await asyncio.create_subprocess_exec(
        "lsof", "-i", f":{port}", "-P", "-n",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {"port": port, "in_use": False, "process": None, "pid": None}

    output = stdout.decode("utf-8", errors="replace")
    lines = output.strip().splitlines()

    if len(lines) < 2:
        return {"port": port, "in_use": False, "process": None, "pid": None}

    # Parse the first data line (skip header).
    parts = lines[1].split()
    process_name = parts[0] if len(parts) > 0 else None
    pid = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None

    return {
        "port": port,
        "in_use": True,
        "process": process_name,
        "pid": pid,
    }


# ---------------------------------------------------------------------------
# 4. env_vars
# ---------------------------------------------------------------------------


async def env_vars(filter_pattern: str | None = None) -> dict[str, Any]:
    """List environment variables.  Optionally filter by pattern.

    Variables whose names contain SECRET, KEY, PASSWORD, TOKEN, CREDENTIAL,
    or PRIVATE are automatically masked (only the first 4 characters shown).

    Args:
        filter_pattern: Case-insensitive substring to filter variable names.

    Returns:
        ``{"variables": {"KEY": "VALUE"}, "count"}``.
    """
    all_vars = dict(os.environ)

    if filter_pattern:
        pat = filter_pattern.upper()
        all_vars = {k: v for k, v in all_vars.items() if pat in k.upper()}

    masked: dict[str, str] = {}
    for k, v in sorted(all_vars.items()):
        masked[k] = _mask_value(k, v)

    return {"variables": masked, "count": len(masked)}


# ---------------------------------------------------------------------------
# 5. curl
# ---------------------------------------------------------------------------


async def curl(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    """Make an HTTP request to a URL using ``httpx``.

    Args:
        url: Target URL.
        method: HTTP method (GET, POST, PUT, DELETE, etc.).
        headers: Optional request headers.
        body: Optional request body (for POST/PUT).
        timeout: Request timeout in seconds.

    Returns:
        ``{"url", "method", "status_code", "headers", "body", "duration_ms"}``.
        Response body is truncated to 5 000 characters.
    """
    start = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.request(
                method=method.upper(),
                url=url,
                headers=headers,
                content=body,
            )
    except httpx.TimeoutException:
        duration_ms = (time.monotonic() - start) * 1000
        return {
            "url": url,
            "method": method.upper(),
            "status_code": -1,
            "headers": {},
            "body": f"Request timed out after {timeout}s",
            "duration_ms": round(duration_ms, 1),
        }
    except httpx.RequestError as exc:
        duration_ms = (time.monotonic() - start) * 1000
        return {
            "url": url,
            "method": method.upper(),
            "status_code": -1,
            "headers": {},
            "body": f"Request failed: {exc}",
            "duration_ms": round(duration_ms, 1),
        }

    duration_ms = (time.monotonic() - start) * 1000
    resp_headers = dict(response.headers)
    resp_body = response.text

    return {
        "url": url,
        "method": method.upper(),
        "status_code": response.status_code,
        "headers": resp_headers,
        "body": _truncate(resp_body, _BODY_MAX),
        "duration_ms": round(duration_ms, 1),
    }


# ---------------------------------------------------------------------------
# 6. process_list
# ---------------------------------------------------------------------------


async def process_list(filter_pattern: str | None = None) -> dict[str, Any]:
    """List running processes.  Optionally filter by name.

    Args:
        filter_pattern: Case-insensitive substring to filter command names.

    Returns:
        ``{"processes": [{"pid", "user", "cpu_percent", "mem_percent", "command"}], "count"}``.
        Capped at 50 results.
    """
    proc = await asyncio.create_subprocess_exec(
        "ps", "aux",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {"processes": [], "count": 0}

    lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
    if len(lines) < 2:
        return {"processes": [], "count": 0}

    processes: list[dict[str, Any]] = []
    for line in lines[1:]:  # skip header
        parts = line.split(None, 10)
        if len(parts) < 11:
            continue

        user = parts[0]
        pid = parts[1]
        cpu = parts[2]
        mem = parts[3]
        command = parts[10]

        if filter_pattern and filter_pattern.lower() not in command.lower():
            continue

        processes.append({
            "pid": int(pid) if pid.isdigit() else pid,
            "user": user,
            "cpu_percent": float(cpu),
            "mem_percent": float(mem),
            "command": command,
        })

        if len(processes) >= 50:
            break

    return {"processes": processes, "count": len(processes)}


# ---------------------------------------------------------------------------
# 7. disk_usage
# ---------------------------------------------------------------------------


async def disk_usage(path: str = "/") -> dict[str, Any]:
    """Check disk usage for a path.

    Args:
        path: Filesystem path to check.

    Returns:
        ``{"path", "total_gb", "used_gb", "free_gb", "percent_used"}``.
    """
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return {"error": str(exc), "error_type": "OSError"}

    total_gb = round(usage.total / (1024 ** 3), 2)
    used_gb = round(usage.used / (1024 ** 3), 2)
    free_gb = round(usage.free / (1024 ** 3), 2)
    percent = round((usage.used / usage.total) * 100, 1) if usage.total > 0 else 0.0

    return {
        "path": path,
        "total_gb": total_gb,
        "used_gb": used_gb,
        "free_gb": free_gb,
        "percent_used": percent,
    }


# ---------------------------------------------------------------------------
# 8. system_info
# ---------------------------------------------------------------------------


async def system_info() -> dict[str, Any]:
    """Get system information: OS, CPU, memory, hostname.

    Returns:
        ``{"os", "hostname", "cpu_count", "memory_total_gb", "python_version"}``.
    """
    # Memory: read from sysctl on macOS, /proc/meminfo on Linux.
    memory_gb: float | None = None
    try:
        if platform.system() == "Darwin":
            proc = await asyncio.create_subprocess_exec(
                "sysctl", "-n", "hw.memsize",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            memory_gb = round(int(stdout.strip()) / (1024 ** 3), 2)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        memory_gb = round(kb / (1024 ** 2), 2)
                        break
    except (OSError, ValueError, asyncio.TimeoutError):
        pass

    return {
        "os": f"{platform.system()} {platform.release()}",
        "hostname": platform.node(),
        "cpu_count": os.cpu_count() or 0,
        "memory_total_gb": memory_gb,
        "python_version": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_shell_tools(registry: ToolRegistry) -> None:
    """Register all shell/system tools with the given :class:`ToolRegistry`.

    Args:
        registry: The central tool registry to populate.
    """
    registry.register(
        namespace="shell",
        name="run",
        fn=run,
        description=(
            "Run a shell command and return stdout, stderr, and the exit code. "
            "All commands have a timeout (default 30s) and output is truncated "
            "to 10000 characters. Use for ad-hoc system commands."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Max seconds before kill. Default: 30"},
            },
            "required": ["command"],
        },
    )

    registry.register(
        namespace="shell",
        name="run_in_dir",
        fn=run_in_dir,
        description=(
            "Run a shell command in a specific directory. Same as shell_run but "
            "with a working directory set. Useful for running build commands or "
            "project-specific scripts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "directory": {"type": "string", "description": "Working directory for the command"},
                "timeout": {"type": "integer", "description": "Max seconds before kill. Default: 30"},
            },
            "required": ["command", "directory"],
        },
    )

    registry.register(
        namespace="shell",
        name="check_port",
        fn=check_port,
        description=(
            "Check if a TCP port is in use and identify which process is "
            "listening on it. Returns process name and PID if the port is active."
        ),
        parameters={
            "type": "object",
            "properties": {
                "port": {"type": "integer", "description": "TCP port number to check"},
            },
            "required": ["port"],
        },
    )

    registry.register(
        namespace="shell",
        name="env_vars",
        fn=env_vars,
        description=(
            "List environment variables, optionally filtered by a pattern. "
            "Variables containing SECRET, KEY, PASSWORD, or TOKEN in their name "
            "are automatically masked for safety — only the first 4 characters are shown."
        ),
        parameters={
            "type": "object",
            "properties": {
                "filter_pattern": {"type": "string", "description": "Case-insensitive substring to filter variable names"},
            },
        },
    )

    registry.register(
        namespace="shell",
        name="curl",
        fn=curl,
        description=(
            "Make an HTTP request to a URL. Supports GET, POST, PUT, DELETE "
            "with custom headers and body. Returns status code, response headers, "
            "and body (truncated to 5000 chars). Uses httpx, not subprocess curl."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Target URL"},
                "method": {"type": "string", "description": "HTTP method. Default: GET"},
                "headers": {
                    "type": "object",
                    "description": "Optional request headers as key-value pairs",
                },
                "body": {"type": "string", "description": "Optional request body (for POST/PUT)"},
                "timeout": {"type": "integer", "description": "Request timeout in seconds. Default: 10"},
            },
            "required": ["url"],
        },
    )

    registry.register(
        namespace="shell",
        name="process_list",
        fn=process_list,
        description=(
            "List running processes with CPU and memory usage. Optionally filter "
            "by command name. Returns up to 50 processes with PID, user, CPU%, "
            "memory%, and command."
        ),
        parameters={
            "type": "object",
            "properties": {
                "filter_pattern": {"type": "string", "description": "Filter processes by command name (case-insensitive)"},
            },
        },
    )

    registry.register(
        namespace="shell",
        name="disk_usage",
        fn=disk_usage,
        description=(
            "Check disk usage for a filesystem path. Returns total, used, and "
            "free space in GB plus the percentage used. Defaults to the root "
            "filesystem '/'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Filesystem path to check. Default: '/'"},
            },
        },
    )

    registry.register(
        namespace="shell",
        name="system_info",
        fn=system_info,
        description=(
            "Get system information: operating system, hostname, CPU count, "
            "total memory in GB, and Python version. Useful for understanding "
            "the host environment."
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
    )
