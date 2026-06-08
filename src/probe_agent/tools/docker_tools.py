"""Docker tools for ProbeAgent.

Twelve tools that let the AI agent inspect and manage Docker containers
and Docker Compose services.  Container operations use the ``docker``
Python SDK; Compose operations use ``docker compose`` via subprocess.

Register all tools at once with :func:`register_docker_tools`.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import docker
import docker.errors

from probe_agent.errors import ToolExecutionError
from probe_agent.registry import ToolRegistry

_COMPOSE_TIMEOUT = 60  # seconds for compose subprocesses
_OUTPUT_MAX_CHARS = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_docker_client() -> docker.DockerClient:
    """Return a Docker client connected to the local daemon.

    Raises:
        ToolExecutionError: If Docker is not available.
    """
    try:
        client = docker.from_env()
        client.ping()
        return client
    except docker.errors.DockerException as exc:
        raise ToolExecutionError(
            tool_name="docker",
            cause=exc,
            context={"hint": "Is the Docker daemon running?"},
        ) from exc


def _short_id(container_id: str) -> str:
    """Return the first 12 characters of a container ID."""
    return container_id[:12]


def _format_ports(ports: dict[str, Any] | None) -> str:
    """Format a Docker ports dict into a human-readable string."""
    if not ports:
        return ""
    parts: list[str] = []
    for container_port, host_bindings in ports.items():
        if host_bindings:
            for binding in host_bindings:
                host = binding.get("HostIp", "0.0.0.0")
                hp = binding.get("HostPort", "?")
                parts.append(f"{host}:{hp}->{container_port}")
        else:
            parts.append(container_port)
    return ", ".join(parts)


async def _run_compose(
    *args: str,
    project_dir: str | None = None,
    timeout: int = _COMPOSE_TIMEOUT,
) -> tuple[int, str, str]:
    """Run a ``docker compose`` command and return (rc, stdout, stderr)."""
    cmd: list[str] = ["docker", "compose"]
    if project_dir:
        cmd += ["-f", f"{project_dir}/docker-compose.yml"]
    cmd.extend(args)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=project_dir,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return 1, "", f"docker compose command timed out after {timeout}s"

    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


# ---------------------------------------------------------------------------
# 1. ps
# ---------------------------------------------------------------------------


async def ps(all: bool = False) -> dict[str, Any]:
    """List Docker containers.  Set ``all=True`` to include stopped ones.

    Args:
        all: Include stopped containers.

    Returns:
        ``{"containers": [{"id", "name", "image", "status", "ports", "created"}], "count"}``.
    """
    try:
        client = _get_docker_client()
        containers = client.containers.list(all=all)
    except ToolExecutionError as exc:
        return {"error": str(exc), "error_type": "DockerError"}

    result: list[dict[str, Any]] = []
    for c in containers:
        result.append({
            "id": _short_id(c.id),
            "name": c.name,
            "image": ",".join(c.image.tags) if c.image and c.image.tags else str(c.image),
            "status": c.status,
            "ports": _format_ports(c.ports),
            "created": c.attrs.get("Created", ""),
        })

    return {"containers": result, "count": len(result)}


# ---------------------------------------------------------------------------
# 2. inspect
# ---------------------------------------------------------------------------


async def inspect(container: str) -> dict[str, Any]:
    """Get detailed info about a container (by name or ID).

    Returns the most useful fields — not the entire raw inspection.

    Args:
        container: Container name or ID.
    """
    try:
        client = _get_docker_client()
        c = client.containers.get(container)
    except docker.errors.NotFound:
        return {"error": f"Container not found: {container}", "error_type": "NotFound"}
    except ToolExecutionError as exc:
        return {"error": str(exc), "error_type": "DockerError"}

    attrs = c.attrs
    state = attrs.get("State", {})
    net_settings = attrs.get("NetworkSettings", {})
    host_config = attrs.get("HostConfig", {})

    # Extract mounts.
    mounts: list[dict[str, str]] = []
    for m in attrs.get("Mounts", []):
        mounts.append({
            "source": m.get("Source", ""),
            "destination": m.get("Destination", ""),
            "mode": m.get("Mode", ""),
            "type": m.get("Type", ""),
        })

    # Extract env vars.
    config = attrs.get("Config", {})
    env_list = config.get("Env", [])
    environment: dict[str, str] = {}
    for entry in env_list:
        if "=" in entry:
            k, v = entry.split("=", 1)
            environment[k] = v

    return {
        "id": attrs.get("Id", "")[:12],
        "name": attrs.get("Name", "").lstrip("/"),
        "image": config.get("Image", ""),
        "state": {
            "status": state.get("Status", ""),
            "running": state.get("Running", False),
            "started_at": state.get("StartedAt", ""),
            "oom_killed": state.get("OOMKilled", False),
        },
        "network": {
            "ip_address": net_settings.get("IPAddress", ""),
            "ports": _format_ports(net_settings.get("Ports")),
        },
        "mounts": mounts,
        "environment": environment,
        "restart_count": attrs.get("RestartCount", 0),
        "restart_policy": host_config.get("RestartPolicy", {}).get("Name", ""),
    }


# ---------------------------------------------------------------------------
# 3. logs
# ---------------------------------------------------------------------------


async def logs(
    container: str,
    tail: int = 100,
    since: str | None = None,
    grep: str | None = None,
) -> dict[str, Any]:
    """Read container logs.

    Args:
        container: Container name or ID.
        tail: Number of lines from the end.
        since: Only return logs after this timestamp (ISO 8601 or relative).
        grep: Filter log lines to those containing this pattern.

    Returns:
        ``{"container", "logs", "lines"}``.
    """
    try:
        client = _get_docker_client()
        c = client.containers.get(container)
    except docker.errors.NotFound:
        return {"error": f"Container not found: {container}", "error_type": "NotFound"}
    except ToolExecutionError as exc:
        return {"error": str(exc), "error_type": "DockerError"}

    kwargs: dict[str, Any] = {"tail": tail, "timestamps": True}
    if since:
        kwargs["since"] = since

    try:
        raw = c.logs(**kwargs)
    except docker.errors.APIError as exc:
        return {"error": f"Failed to read logs: {exc}", "error_type": "APIError"}

    log_text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

    if grep:
        log_text = "\n".join(
            line for line in log_text.splitlines() if grep in line
        )

    line_count = len(log_text.splitlines())

    return {
        "container": container,
        "logs": log_text,
        "lines": line_count,
    }


# ---------------------------------------------------------------------------
# 4. stats
# ---------------------------------------------------------------------------


async def stats(container: str | None = None) -> dict[str, Any]:
    """Get live CPU and memory usage for one or all running containers.

    Args:
        container: Container name/ID, or ``None`` for all running containers.

    Returns:
        ``{"stats": [{"name", "cpu_percent", "memory_mb", "memory_limit_mb", "memory_percent"}]}``.
    """
    try:
        client = _get_docker_client()
    except ToolExecutionError as exc:
        return {"error": str(exc), "error_type": "DockerError"}

    targets: list[Any] = []
    if container:
        try:
            targets = [client.containers.get(container)]
        except docker.errors.NotFound:
            return {"error": f"Container not found: {container}", "error_type": "NotFound"}
    else:
        targets = client.containers.list()

    results: list[dict[str, Any]] = []
    for c in targets:
        try:
            s = c.stats(stream=False)
        except docker.errors.APIError:
            continue

        # CPU calculation.
        cpu_delta = (
            s.get("cpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0)
            - s.get("precpu_stats", {}).get("cpu_usage", {}).get("total_usage", 0)
        )
        system_delta = (
            s.get("cpu_stats", {}).get("system_cpu_usage", 0)
            - s.get("precpu_stats", {}).get("system_cpu_usage", 0)
        )
        num_cpus = s.get("cpu_stats", {}).get("online_cpus", 1) or 1
        cpu_percent = (cpu_delta / system_delta) * num_cpus * 100.0 if system_delta > 0 else 0.0

        # Memory calculation.
        mem_usage = s.get("memory_stats", {}).get("usage", 0)
        mem_limit = s.get("memory_stats", {}).get("limit", 1)
        mem_mb = mem_usage / (1024 * 1024)
        mem_limit_mb = mem_limit / (1024 * 1024)
        mem_percent = (mem_usage / mem_limit) * 100.0 if mem_limit > 0 else 0.0

        results.append({
            "name": c.name,
            "cpu_percent": round(cpu_percent, 2),
            "memory_mb": round(mem_mb, 1),
            "memory_limit_mb": round(mem_limit_mb, 1),
            "memory_percent": round(mem_percent, 2),
        })

    return {"stats": results}


# ---------------------------------------------------------------------------
# 5. restart
# ---------------------------------------------------------------------------


async def restart(container: str, timeout: int = 10) -> dict[str, Any]:
    """Restart a container.

    Args:
        container: Container name or ID.
        timeout: Seconds to wait for graceful stop before kill.

    Returns:
        ``{"container", "status": "restarted"}``.
    """
    try:
        client = _get_docker_client()
        c = client.containers.get(container)
    except docker.errors.NotFound:
        return {"error": f"Container not found: {container}", "error_type": "NotFound"}
    except ToolExecutionError as exc:
        return {"error": str(exc), "error_type": "DockerError"}

    try:
        c.restart(timeout=timeout)
    except docker.errors.APIError as exc:
        return {"error": f"Restart failed: {exc}", "error_type": "APIError"}

    return {"container": container, "status": "restarted"}


# ---------------------------------------------------------------------------
# 6. stop
# ---------------------------------------------------------------------------


async def stop(container: str, timeout: int = 10) -> dict[str, Any]:
    """Stop a container.

    Args:
        container: Container name or ID.
        timeout: Seconds to wait for graceful stop before kill.

    Returns:
        ``{"container", "status": "stopped"}``.
    """
    try:
        client = _get_docker_client()
        c = client.containers.get(container)
    except docker.errors.NotFound:
        return {"error": f"Container not found: {container}", "error_type": "NotFound"}
    except ToolExecutionError as exc:
        return {"error": str(exc), "error_type": "DockerError"}

    try:
        c.stop(timeout=timeout)
    except docker.errors.APIError as exc:
        return {"error": f"Stop failed: {exc}", "error_type": "APIError"}

    return {"container": container, "status": "stopped"}


# ---------------------------------------------------------------------------
# 7. exec_command
# ---------------------------------------------------------------------------


async def exec_command(
    container: str,
    command: str,
    workdir: str | None = None,
) -> dict[str, Any]:
    """Execute a command inside a running container.

    Args:
        container: Container name or ID.
        command: Shell command to run.
        workdir: Working directory inside the container.

    Returns:
        ``{"container", "command", "stdout", "stderr", "exit_code"}``.
        Output is truncated to 5000 chars.
    """
    try:
        client = _get_docker_client()
        c = client.containers.get(container)
    except docker.errors.NotFound:
        return {"error": f"Container not found: {container}", "error_type": "NotFound"}
    except ToolExecutionError as exc:
        return {"error": str(exc), "error_type": "DockerError"}

    kwargs: dict[str, Any] = {
        "cmd": ["sh", "-c", command],
        "demux": True,
    }
    if workdir:
        kwargs["workdir"] = workdir

    try:
        exit_code, output = c.exec_run(**kwargs)
    except docker.errors.APIError as exc:
        return {"error": f"Exec failed: {exc}", "error_type": "APIError"}

    stdout_text = ""
    stderr_text = ""
    if isinstance(output, tuple):
        stdout_text = (output[0] or b"").decode("utf-8", errors="replace")
        stderr_text = (output[1] or b"").decode("utf-8", errors="replace")
    elif isinstance(output, bytes):
        stdout_text = output.decode("utf-8", errors="replace")

    # Truncate.
    if len(stdout_text) > _OUTPUT_MAX_CHARS:
        stdout_text = stdout_text[:_OUTPUT_MAX_CHARS] + "\n... [truncated]"
    if len(stderr_text) > _OUTPUT_MAX_CHARS:
        stderr_text = stderr_text[:_OUTPUT_MAX_CHARS] + "\n... [truncated]"

    return {
        "container": container,
        "command": command,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "exit_code": exit_code,
    }


# ---------------------------------------------------------------------------
# 8. compose_ps
# ---------------------------------------------------------------------------


async def compose_ps(project_dir: str | None = None) -> dict[str, Any]:
    """List Docker Compose services and their status.

    Args:
        project_dir: Directory containing ``docker-compose.yml``.

    Returns:
        ``{"services": [{"name", "status", "ports"}], "count"}``.
    """
    rc, out, err = await _run_compose(
        "ps", "--format", "json", project_dir=project_dir,
    )
    if rc != 0:
        return {"error": err.strip() or "docker compose ps failed", "error_type": "ComposeError"}

    services: list[dict[str, str]] = []
    # docker compose ps --format json may output one JSON object per line
    # or a JSON array depending on version.
    text = out.strip()
    if not text:
        return {"services": [], "count": 0}

    try:
        parsed = json.loads(text)
        items = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        # Try line-by-line JSON.
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    for item in items:
        services.append({
            "name": item.get("Name") or item.get("Service") or item.get("name", ""),
            "status": item.get("State") or item.get("Status") or item.get("status", ""),
            "ports": item.get("Ports") or item.get("Publishers") or "",
        })

    return {"services": services, "count": len(services)}


# ---------------------------------------------------------------------------
# 9. compose_up
# ---------------------------------------------------------------------------


async def compose_up(
    service: str | None = None,
    project_dir: str | None = None,
) -> dict[str, Any]:
    """Start Docker Compose services.

    Args:
        service: Specific service to start, or ``None`` for all.
        project_dir: Directory containing ``docker-compose.yml``.

    Returns:
        ``{"action": "up", "service", "output"}``.
    """
    cmd = ["up", "-d"]
    if service:
        cmd.append(service)

    rc, out, err = await _run_compose(*cmd, project_dir=project_dir)
    output = (out + err).strip()

    if rc != 0:
        return {"error": output or "docker compose up failed", "error_type": "ComposeError"}

    return {
        "action": "up",
        "service": service or "all",
        "output": output,
    }


# ---------------------------------------------------------------------------
# 10. compose_down
# ---------------------------------------------------------------------------


async def compose_down(
    project_dir: str | None = None,
    remove_volumes: bool = False,
) -> dict[str, Any]:
    """Stop Docker Compose services.

    Args:
        project_dir: Directory containing ``docker-compose.yml``.
        remove_volumes: If ``True``, also remove volumes.

    Returns:
        ``{"action": "down", "output"}``.
    """
    cmd = ["down"]
    if remove_volumes:
        cmd.append("-v")

    rc, out, err = await _run_compose(*cmd, project_dir=project_dir)
    output = (out + err).strip()

    if rc != 0:
        return {"error": output or "docker compose down failed", "error_type": "ComposeError"}

    return {"action": "down", "output": output}


# ---------------------------------------------------------------------------
# 11. compose_logs
# ---------------------------------------------------------------------------


async def compose_logs(
    service: str | None = None,
    tail: int = 100,
    project_dir: str | None = None,
) -> dict[str, Any]:
    """Read Docker Compose logs for a service.

    Args:
        service: Service name, or ``None`` for all services.
        tail: Number of lines from the end.
        project_dir: Directory containing ``docker-compose.yml``.

    Returns:
        ``{"service", "logs", "lines"}``.
    """
    cmd = ["logs", "--tail", str(tail)]
    if service:
        cmd.append(service)

    rc, out, err = await _run_compose(*cmd, project_dir=project_dir)
    log_text = (out + err).strip()

    if rc != 0 and not log_text:
        return {"error": "docker compose logs failed", "error_type": "ComposeError"}

    return {
        "service": service or "all",
        "logs": log_text,
        "lines": len(log_text.splitlines()),
    }


# ---------------------------------------------------------------------------
# 12. network_inspect
# ---------------------------------------------------------------------------


async def network_inspect(network: str) -> dict[str, Any]:
    """Inspect a Docker network — show connected containers and their IPs.

    Args:
        network: Network name or ID.

    Returns:
        ``{"name", "driver", "containers": [{"name", "ipv4_address"}]}``.
    """
    try:
        client = _get_docker_client()
        net = client.networks.get(network)
    except docker.errors.NotFound:
        return {"error": f"Network not found: {network}", "error_type": "NotFound"}
    except ToolExecutionError as exc:
        return {"error": str(exc), "error_type": "DockerError"}

    attrs = net.attrs
    connected: list[dict[str, str]] = []
    for cid, info in (attrs.get("Containers") or {}).items():
        connected.append({
            "name": info.get("Name", cid[:12]),
            "ipv4_address": info.get("IPv4Address", ""),
        })

    return {
        "name": attrs.get("Name", network),
        "driver": attrs.get("Driver", ""),
        "containers": connected,
    }


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_docker_tools(registry: ToolRegistry) -> None:
    """Register all Docker tools with the given :class:`ToolRegistry`.

    Args:
        registry: The central tool registry to populate.
    """
    registry.register(
        namespace="docker",
        name="ps",
        fn=ps,
        description=(
            "List Docker containers. Shows container ID, name, image, status, "
            "and port mappings. Set all=true to include stopped containers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "all": {"type": "boolean", "description": "Include stopped containers. Default: false"},
            },
        },
    )

    registry.register(
        namespace="docker",
        name="inspect",
        fn=inspect,
        description=(
            "Get detailed info about a Docker container: state (running, OOM killed), "
            "network settings (IP, ports), mounts, environment variables, and restart "
            "policy. Returns a curated summary, not the entire raw inspection."
        ),
        parameters={
            "type": "object",
            "properties": {
                "container": {"type": "string", "description": "Container name or ID"},
            },
            "required": ["container"],
        },
    )

    registry.register(
        namespace="docker",
        name="logs",
        fn=logs,
        description=(
            "Read container logs. Returns the most recent lines (default 100). "
            "Use 'since' to filter by time, and 'grep' to search for specific "
            "patterns in the log output."
        ),
        parameters={
            "type": "object",
            "properties": {
                "container": {"type": "string", "description": "Container name or ID"},
                "tail": {"type": "integer", "description": "Number of lines from the end. Default: 100"},
                "since": {"type": "string", "description": "Only return logs after this timestamp (ISO 8601)"},
                "grep": {"type": "string", "description": "Filter lines containing this pattern"},
            },
            "required": ["container"],
        },
    )

    registry.register(
        namespace="docker",
        name="stats",
        fn=stats,
        description=(
            "Get live CPU and memory usage for containers. If no container is "
            "specified, returns stats for all running containers. Shows CPU "
            "percent, memory in MB, memory limit, and memory percent."
        ),
        parameters={
            "type": "object",
            "properties": {
                "container": {"type": "string", "description": "Container name or ID. Omit for all running containers"},
            },
        },
    )

    registry.register(
        namespace="docker",
        name="restart",
        fn=restart,
        description=(
            "Restart a Docker container. Sends SIGTERM and waits for the timeout "
            "before killing. Useful for applying config changes or recovering "
            "from errors."
        ),
        parameters={
            "type": "object",
            "properties": {
                "container": {"type": "string", "description": "Container name or ID"},
                "timeout": {"type": "integer", "description": "Seconds to wait for graceful stop. Default: 10"},
            },
            "required": ["container"],
        },
    )

    registry.register(
        namespace="docker",
        name="stop",
        fn=stop,
        description=(
            "Stop a running Docker container. Sends SIGTERM and waits for the "
            "timeout before sending SIGKILL."
        ),
        parameters={
            "type": "object",
            "properties": {
                "container": {"type": "string", "description": "Container name or ID"},
                "timeout": {"type": "integer", "description": "Seconds to wait for graceful stop. Default: 10"},
            },
            "required": ["container"],
        },
    )

    registry.register(
        namespace="docker",
        name="exec_command",
        fn=exec_command,
        description=(
            "Execute a command inside a running Docker container. Runs the command "
            "via 'sh -c' and returns stdout, stderr, and the exit code. Output is "
            "truncated to 5000 characters."
        ),
        parameters={
            "type": "object",
            "properties": {
                "container": {"type": "string", "description": "Container name or ID"},
                "command": {"type": "string", "description": "Shell command to execute"},
                "workdir": {"type": "string", "description": "Working directory inside the container"},
            },
            "required": ["container", "command"],
        },
    )

    registry.register(
        namespace="docker",
        name="compose_ps",
        fn=compose_ps,
        description=(
            "List Docker Compose services and their status. Shows service name, "
            "state, and port mappings for the compose project."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_dir": {"type": "string", "description": "Directory containing docker-compose.yml"},
            },
        },
    )

    registry.register(
        namespace="docker",
        name="compose_up",
        fn=compose_up,
        description=(
            "Start Docker Compose services in detached mode. Specify a service "
            "name to start only that service, or omit to start all services."
        ),
        parameters={
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service name to start. Omit for all"},
                "project_dir": {"type": "string", "description": "Directory containing docker-compose.yml"},
            },
        },
    )

    registry.register(
        namespace="docker",
        name="compose_down",
        fn=compose_down,
        description=(
            "Stop and remove Docker Compose services. Set remove_volumes=true "
            "to also remove named volumes declared in the compose file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_dir": {"type": "string", "description": "Directory containing docker-compose.yml"},
                "remove_volumes": {"type": "boolean", "description": "Also remove volumes. Default: false"},
            },
        },
    )

    registry.register(
        namespace="docker",
        name="compose_logs",
        fn=compose_logs,
        description=(
            "Read Docker Compose logs. Returns the most recent lines from one "
            "or all services. Useful for diagnosing startup failures or runtime errors."
        ),
        parameters={
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service name. Omit for all services"},
                "tail": {"type": "integer", "description": "Number of lines from the end. Default: 100"},
                "project_dir": {"type": "string", "description": "Directory containing docker-compose.yml"},
            },
        },
    )

    registry.register(
        namespace="docker",
        name="network_inspect",
        fn=network_inspect,
        description=(
            "Inspect a Docker network. Shows the network driver and all connected "
            "containers with their IPv4 addresses. Useful for debugging connectivity "
            "issues between containers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "network": {"type": "string", "description": "Network name or ID"},
            },
            "required": ["network"],
        },
    )
