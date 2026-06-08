"""Project intelligence tools for ProbeAgent.

Six tools that help the AI agent understand any software project by scanning
its directory structure, parsing configuration files, running tests and
linters, and mapping service dependencies.

The most important tool is :func:`discover` — it builds a comprehensive
profile of a project's technology stack in a single call.

Register all tools at once with :func:`register_project_tools`.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml

from probe_agent.registry import ToolRegistry

_OUTPUT_MAX = 5_000

# Patterns that signal hardcoded secrets in config files.
_SECRET_PATTERNS = re.compile(
    r"(?:password|secret|api_key|token|private_key)\s*[=:]\s*\S+",
    re.IGNORECASE,
)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated — {len(text)} chars total]"


def _safe_read(path: Path) -> str | None:
    """Read a file, returning ``None`` on failure."""
    try:
        return path.read_text(errors="replace")
    except (OSError, PermissionError):
        return None


def _parse_compose(path: Path) -> list[dict[str, Any]]:
    """Parse a docker-compose YAML and return service dicts."""
    text = _safe_read(path)
    if not text:
        return []

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return []

    if not isinstance(data, dict):
        return []

    services_block = data.get("services", {})
    if not isinstance(services_block, dict):
        return []

    services: list[dict[str, Any]] = []
    for name, svc in services_block.items():
        if not isinstance(svc, dict):
            continue

        # Parse ports — can be strings ("8080:80") or ints.
        raw_ports = svc.get("ports", [])
        ports: list[str] = [str(p) for p in raw_ports] if raw_ports else []

        services.append({
            "name": name,
            "image": svc.get("image", ""),
            "build": svc.get("build", ""),
            "ports": ports,
            "depends_on": list(svc.get("depends_on", []))
                          if isinstance(svc.get("depends_on"), (list, dict))
                          else [],
            "networks": list(svc.get("networks", []))
                        if isinstance(svc.get("networks"), (list, dict))
                        else [],
            "environment": svc.get("environment", []),
        })

    return services


# ---------------------------------------------------------------------------
# 1. discover
# ---------------------------------------------------------------------------


async def discover(project_path: str) -> dict[str, Any]:
    """Scan a project directory and detect its technology stack.

    Detection rules:
    - ``docker-compose*.yml`` → parse services
    - ``Dockerfile*`` → Docker support
    - ``requirements.txt`` / ``pyproject.toml`` → Python
    - ``package.json`` → Node.js, scan for frameworks
    - ``Makefile`` → parse available targets
    - ``pytest.ini`` or ``[tool.pytest]`` → test command = ``pytest``
    - ``package.json`` ``scripts.test`` → test command = ``npm test``
    - ``.env`` / ``.env.example`` → config files

    Args:
        project_path: Root directory of the project.

    Returns:
        Comprehensive project profile dict.
    """
    root = Path(project_path)

    if not root.is_dir():
        return {"error": f"Not a directory: {project_path}", "error_type": "NotADirectory"}

    languages: set[str] = set()
    frameworks: set[str] = set()
    services: list[dict[str, Any]] = []
    config_files: list[str] = []
    entry_points: list[str] = []
    has_docker = False
    has_tests = False
    test_command: str | None = None

    # --- Docker ---
    for pattern in ("docker-compose.yml", "docker-compose.yaml",
                    "docker-compose*.yml", "docker-compose*.yaml",
                    "compose.yml", "compose.yaml"):
        for compose_path in root.glob(pattern):
            has_docker = True
            config_files.append(compose_path.name)
            services.extend(_parse_compose(compose_path))

    for df in root.glob("Dockerfile*"):
        has_docker = True
        entry_points.append(df.name)

    # --- Python ---
    pyproject = root / "pyproject.toml"
    requirements = root / "requirements.txt"

    if pyproject.exists():
        languages.add("python")
        config_files.append("pyproject.toml")
        content = _safe_read(pyproject) or ""

        # Detect frameworks from dependencies.
        lower = content.lower()
        for fw in ("fastapi", "flask", "django", "celery", "starlette", "aiohttp"):
            if fw in lower:
                frameworks.add(fw)

        # Detect test config.
        if "[tool.pytest" in content:
            has_tests = True
            test_command = "pytest"

        if "main.py" in content or "app.py" in content:
            entry_points.append("pyproject.toml (scripts)")

    if requirements.exists():
        languages.add("python")
        config_files.append("requirements.txt")
        content = _safe_read(requirements) or ""
        lower = content.lower()
        for fw in ("fastapi", "flask", "django", "celery"):
            if fw in lower:
                frameworks.add(fw)

    if (root / "pytest.ini").exists():
        has_tests = True
        test_command = "pytest"
        config_files.append("pytest.ini")

    if (root / "setup.py").exists():
        languages.add("python")
        config_files.append("setup.py")

    # Common Python entry points.
    for ep in ("main.py", "app.py", "manage.py", "wsgi.py"):
        if (root / ep).exists() or (root / "src" / ep).exists():
            entry_points.append(ep)

    # --- Node.js ---
    pkg_json = root / "package.json"
    if pkg_json.exists():
        languages.add("javascript")
        config_files.append("package.json")

        content = _safe_read(pkg_json) or ""
        try:
            import json
            pkg = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            pkg = {}

        # Detect frameworks.
        all_deps = {}
        all_deps.update(pkg.get("dependencies", {}))
        all_deps.update(pkg.get("devDependencies", {}))

        for fw in ("next", "react", "vue", "express", "nestjs", "nuxt",
                    "angular", "svelte", "fastify"):
            if fw in all_deps or f"@{fw}" in str(all_deps):
                frameworks.add(fw)

        if "typescript" in all_deps:
            languages.add("typescript")

        # Test command.
        scripts = pkg.get("scripts", {})
        if "test" in scripts:
            has_tests = True
            test_command = test_command or "npm test"

    # --- Go ---
    if (root / "go.mod").exists():
        languages.add("go")
        config_files.append("go.mod")

    # --- Rust ---
    if (root / "Cargo.toml").exists():
        languages.add("rust")
        config_files.append("Cargo.toml")

    # --- Makefile ---
    makefile = root / "Makefile"
    if makefile.exists():
        config_files.append("Makefile")
        content = _safe_read(makefile) or ""
        # Parse targets.
        for m in re.finditer(r"^([a-zA-Z_][\w-]*)\s*:", content, re.MULTILINE):
            entry_points.append(f"make {m.group(1)}")

    # --- Config files ---
    for env_file in (".env", ".env.example", ".env.sample"):
        if (root / env_file).exists():
            config_files.append(env_file)

    if (root / ".github" / "workflows").is_dir():
        config_files.append(".github/workflows/")

    # --- Project name ---
    name = root.name

    return {
        "name": name,
        "languages": sorted(languages),
        "frameworks": sorted(frameworks),
        "services": services,
        "has_docker": has_docker,
        "has_tests": has_tests,
        "test_command": test_command,
        "config_files": sorted(set(config_files)),
        "entry_points": sorted(set(entry_points)),
    }


# ---------------------------------------------------------------------------
# 2. dependency_check
# ---------------------------------------------------------------------------


async def dependency_check(project_path: str) -> dict[str, Any]:
    """List project dependencies.

    Parses ``requirements.txt`` or ``pyproject.toml`` (Python) or
    ``package.json`` (Node.js).

    Args:
        project_path: Root directory of the project.

    Returns:
        ``{"manager", "dependencies": [{"name", "version"}], "count"}``.
    """
    root = Path(project_path)

    # Try Python first.
    req_file = root / "requirements.txt"
    if req_file.exists():
        content = _safe_read(req_file) or ""
        deps = _parse_requirements(content)
        return {"manager": "pip", "dependencies": deps, "count": len(deps)}

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        content = _safe_read(pyproject) or ""
        deps = _parse_pyproject_deps(content)
        return {"manager": "pip", "dependencies": deps, "count": len(deps)}

    # Try Node.js.
    pkg_file = root / "package.json"
    if pkg_file.exists():
        content = _safe_read(pkg_file) or ""
        deps = _parse_package_json_deps(content)
        return {"manager": "npm", "dependencies": deps, "count": len(deps)}

    return {"manager": "unknown", "dependencies": [], "count": 0}


def _parse_requirements(content: str) -> list[dict[str, str]]:
    """Parse requirements.txt into name/version pairs."""
    deps: list[dict[str, str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        # Handle: package==1.0, package>=1.0, package~=1.0, package
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*([><=~!]+\s*[\d.*]+)?", line)
        if m:
            name = m.group(1)
            version = (m.group(2) or "").strip()
            deps.append({"name": name, "version": version})

    return deps


def _parse_pyproject_deps(content: str) -> list[dict[str, str]]:
    """Parse dependencies from pyproject.toml (rough regex approach)."""
    deps: list[dict[str, str]] = []

    # Match lines in dependencies = [...] sections.
    in_deps = False
    for line in content.splitlines():
        stripped = line.strip()

        if stripped.startswith("dependencies") and "=" in stripped:
            in_deps = True
            continue
        if in_deps:
            if stripped == "]":
                in_deps = False
                continue
            # Parse "package>=1.0" strings.
            m = re.match(r'["\']([A-Za-z0-9_.-]+)\s*([><=~!]+\s*[\d.*]+)?', stripped)
            if m:
                deps.append({"name": m.group(1), "version": (m.group(2) or "").strip()})

    return deps


def _parse_package_json_deps(content: str) -> list[dict[str, str]]:
    """Parse dependencies from package.json."""
    import json

    try:
        pkg = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return []

    deps: list[dict[str, str]] = []
    for section in ("dependencies", "devDependencies"):
        for name, version in pkg.get(section, {}).items():
            deps.append({"name": name, "version": version})

    return deps


# ---------------------------------------------------------------------------
# 3. test_runner
# ---------------------------------------------------------------------------


async def test_runner(
    project_path: str,
    test_path: str | None = None,
) -> dict[str, Any]:
    """Run the project's test suite and return results.

    Auto-detects pytest (Python) or npm test (Node.js).

    Args:
        project_path: Root directory of the project.
        test_path: Optional specific test path/file.

    Returns:
        ``{"command", "passed", "failed", "errors", "output", "duration_ms"}``.
    """
    root = Path(project_path)
    start = time.monotonic()

    # Determine test command.
    command: str | None = None

    if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists():
        parts = ["python", "-m", "pytest", "-v"]
        if test_path:
            parts.append(test_path)
        command = " ".join(parts)
    elif (root / "package.json").exists():
        command = "npm test"
        if test_path:
            command = f"npm test -- {test_path}"

    if not command:
        return {
            "command": None,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "output": "Could not detect test framework",
            "duration_ms": 0,
        }

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(root),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        duration_ms = (time.monotonic() - start) * 1000
        return {
            "command": command,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "output": "Test run timed out after 120s",
            "duration_ms": round(duration_ms, 1),
        }

    duration_ms = (time.monotonic() - start) * 1000
    output = stdout.decode("utf-8", errors="replace")
    err_output = stderr.decode("utf-8", errors="replace")
    combined = output + ("\n" + err_output if err_output else "")

    # Parse pytest output for counts.
    passed = failed = errors = 0
    # Pytest summary: "5 passed, 2 failed, 1 error"
    m = re.search(r"(\d+) passed", combined)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", combined)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) error", combined)
    if m:
        errors = int(m.group(1))

    return {
        "command": command,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "output": _truncate(combined, _OUTPUT_MAX),
        "duration_ms": round(duration_ms, 1),
    }


# ---------------------------------------------------------------------------
# 4. lint_check
# ---------------------------------------------------------------------------


async def lint_check(project_path: str) -> dict[str, Any]:
    """Run linting on the project.

    Auto-detects ruff (Python) or eslint (Node.js).

    Args:
        project_path: Root directory of the project.

    Returns:
        ``{"tool", "issues": [{"file", "line", "severity", "message"}], "count"}``.
    """
    root = Path(project_path)

    # Determine linter.
    tool: str | None = None
    command: str | None = None

    if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists():
        tool = "ruff"
        command = "python -m ruff check --output-format=text ."
    elif (root / "package.json").exists():
        tool = "eslint"
        command = "npx eslint . --format compact"

    if not command or not tool:
        return {"tool": "none", "issues": [], "count": 0}

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(root),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    except asyncio.TimeoutError:
        return {"tool": tool, "issues": [], "count": 0, "error": "Lint timed out after 60s"}

    output = stdout.decode("utf-8", errors="replace")
    issues: list[dict[str, str]] = []

    if tool == "ruff":
        # Parse ruff output: "file.py:10:5: E501 Line too long"
        for line in output.splitlines():
            m = re.match(r"(.+?):(\d+):\d+:\s+(\w+)\s+(.*)", line)
            if m:
                issues.append({
                    "file": m.group(1),
                    "line": m.group(2),
                    "severity": "error" if m.group(3).startswith("E") else "warning",
                    "message": f"{m.group(3)} {m.group(4)}",
                })
    elif tool == "eslint":
        # Parse eslint compact: "file.js: line 10, col 5, Error - msg (rule)"
        for line in output.splitlines():
            m = re.match(r"(.+?):\s+line\s+(\d+),.+?(Error|Warning)\s+-\s+(.*)", line)
            if m:
                issues.append({
                    "file": m.group(1),
                    "line": m.group(2),
                    "severity": m.group(3).lower(),
                    "message": m.group(4),
                })

    return {"tool": tool, "issues": issues, "count": len(issues)}


# ---------------------------------------------------------------------------
# 5. config_audit
# ---------------------------------------------------------------------------


async def config_audit(project_path: str) -> dict[str, Any]:
    """Audit project configuration for common issues.

    Checks:
    - ``.env`` not in ``.gitignore``
    - ``DEBUG=True`` or ``DEBUG=1`` in config
    - Hardcoded secrets in config files
    - Missing health endpoints in docker-compose
    - No resource limits in docker-compose

    Args:
        project_path: Root directory of the project.

    Returns:
        ``{"issues": [{"severity", "category", "message", "file"}], "score"}``.
    """
    root = Path(project_path)
    issues: list[dict[str, str]] = []

    # --- .env not in .gitignore ---
    gitignore = root / ".gitignore"
    env_file = root / ".env"

    if env_file.exists():
        gitignore_content = _safe_read(gitignore) or ""
        if ".env" not in gitignore_content:
            issues.append({
                "severity": "high",
                "category": "security",
                "message": ".env file exists but is not listed in .gitignore",
                "file": ".gitignore",
            })

    # --- DEBUG mode ---
    for cfg_name in (".env", ".env.example", "settings.py", "config.py"):
        cfg_path = root / cfg_name
        if cfg_path.exists():
            content = _safe_read(cfg_path) or ""
            if re.search(r"DEBUG\s*=\s*(True|1|true|yes)", content):
                issues.append({
                    "severity": "medium",
                    "category": "security",
                    "message": f"DEBUG mode is enabled in {cfg_name}",
                    "file": cfg_name,
                })

    # --- Hardcoded secrets ---
    for cfg_name in (".env", ".env.example", "docker-compose.yml",
                     "docker-compose.yaml"):
        cfg_path = root / cfg_name
        if cfg_path.exists():
            content = _safe_read(cfg_path) or ""
            for m in _SECRET_PATTERNS.finditer(content):
                # Don't flag placeholders like ${VAR} or <CHANGE_ME>.
                value_part = m.group(0).split("=", 1)[-1].strip() if "=" in m.group(0) else ""
                if value_part and not value_part.startswith(("${", "<", "\"${", "'${")):
                    issues.append({
                        "severity": "high",
                        "category": "security",
                        "message": f"Possible hardcoded secret: {m.group(0)[:40]}...",
                        "file": cfg_name,
                    })

    # --- Docker-compose checks ---
    for compose_name in ("docker-compose.yml", "docker-compose.yaml",
                         "compose.yml", "compose.yaml"):
        compose_path = root / compose_name
        if compose_path.exists():
            content = _safe_read(compose_path) or ""
            try:
                data = yaml.safe_load(content) or {}
            except yaml.YAMLError:
                continue

            services_block = data.get("services", {})
            if not isinstance(services_block, dict):
                continue

            for svc_name, svc in services_block.items():
                if not isinstance(svc, dict):
                    continue

                # No health check.
                if "healthcheck" not in svc:
                    issues.append({
                        "severity": "medium",
                        "category": "reliability",
                        "message": f"Service '{svc_name}' has no healthcheck configured",
                        "file": compose_name,
                    })

                # No resource limits.
                deploy = svc.get("deploy", {})
                resources = deploy.get("resources", {}) if isinstance(deploy, dict) else {}
                if not resources:
                    # Also check top-level mem_limit (v2 format).
                    if "mem_limit" not in svc and "cpus" not in svc:
                        issues.append({
                            "severity": "low",
                            "category": "reliability",
                            "message": f"Service '{svc_name}' has no resource limits",
                            "file": compose_name,
                        })

    # --- Score ---
    # Start at 100, deduct per issue.
    deductions = {"high": 15, "medium": 8, "low": 3}
    score = 100
    for issue in issues:
        score -= deductions.get(issue["severity"], 5)
    score = max(0, score)

    return {"issues": issues, "score": score}


# ---------------------------------------------------------------------------
# 6. service_map
# ---------------------------------------------------------------------------


async def service_map(project_path: str) -> dict[str, Any]:
    """Build a map of services and their connections from docker-compose.

    Args:
        project_path: Root directory of the project.

    Returns:
        ``{"services": [{"name", "depends_on", "ports", "networks"}],
        "connections": [{"from", "to"}]}``.
    """
    root = Path(project_path)
    all_services: list[dict[str, Any]] = []

    for compose_name in ("docker-compose.yml", "docker-compose.yaml",
                         "compose.yml", "compose.yaml"):
        compose_path = root / compose_name
        if compose_path.exists():
            parsed = _parse_compose(compose_path)
            for svc in parsed:
                all_services.append({
                    "name": svc["name"],
                    "depends_on": svc["depends_on"],
                    "ports": svc["ports"],
                    "networks": svc["networks"],
                })

    # Build connection list from depends_on.
    connections: list[dict[str, str]] = []
    for svc in all_services:
        for dep in svc["depends_on"]:
            connections.append({"from": svc["name"], "to": dep})

    return {"services": all_services, "connections": connections}


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_project_tools(registry: ToolRegistry) -> None:
    """Register all project intelligence tools with the given registry.

    Args:
        registry: The central tool registry to populate.
    """
    registry.register(
        namespace="project",
        name="discover",
        fn=discover,
        description=(
            "Scan a project directory and detect its technology stack, "
            "languages, frameworks, Docker services, test commands, config "
            "files, and entry points. This is usually the FIRST tool you "
            "should call to understand a project."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_path": {"type": "string", "description": "Root directory of the project"},
            },
            "required": ["project_path"],
        },
    )

    registry.register(
        namespace="project",
        name="dependency_check",
        fn=dependency_check,
        description=(
            "List all project dependencies from requirements.txt, "
            "pyproject.toml (Python), or package.json (Node.js). Returns "
            "package names and version constraints."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_path": {"type": "string", "description": "Root directory of the project"},
            },
            "required": ["project_path"],
        },
    )

    registry.register(
        namespace="project",
        name="test_runner",
        fn=test_runner,
        description=(
            "Run the project's test suite. Auto-detects pytest (Python) or "
            "npm test (Node.js). Returns pass/fail/error counts and output. "
            "Use after making changes to verify nothing broke."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_path": {"type": "string", "description": "Root directory of the project"},
                "test_path": {"type": "string", "description": "Specific test file or directory to run"},
            },
            "required": ["project_path"],
        },
    )

    registry.register(
        namespace="project",
        name="lint_check",
        fn=lint_check,
        description=(
            "Run linting on the project. Auto-detects ruff (Python) or "
            "eslint (Node.js). Returns a list of issues with file, line, "
            "severity, and message."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_path": {"type": "string", "description": "Root directory of the project"},
            },
            "required": ["project_path"],
        },
    )

    registry.register(
        namespace="project",
        name="config_audit",
        fn=config_audit,
        description=(
            "Audit project configuration for common issues: .env not in "
            ".gitignore, DEBUG mode enabled, hardcoded secrets, missing "
            "healthchecks, no resource limits. Returns a list of issues "
            "and a score from 0-100."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_path": {"type": "string", "description": "Root directory of the project"},
            },
            "required": ["project_path"],
        },
    )

    registry.register(
        namespace="project",
        name="service_map",
        fn=service_map,
        description=(
            "Build a map of services from docker-compose showing names, "
            "dependencies, ports, and networks. Returns a connection graph "
            "showing which services depend on which."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_path": {"type": "string", "description": "Root directory of the project"},
            },
            "required": ["project_path"],
        },
    )
