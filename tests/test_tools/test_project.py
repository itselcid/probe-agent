"""Tests for project intelligence tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from probe_agent.registry import ToolRegistry
from probe_agent.tools.project import (
    config_audit,
    dependency_check,
    discover,
    lint_check,
    register_project_tools,
    service_map,
    test_runner as _test_runner,
)


# ---------------------------------------------------------------------------
# Fixtures — realistic mini-projects
# ---------------------------------------------------------------------------


@pytest.fixture()
def python_project(tmp_path: Path) -> Path:
    """A minimal Python/FastAPI project with Docker."""
    # pyproject.toml
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["hatchling"]\n\n'
        "[project]\n"
        'name = "my-api"\n'
        "dependencies = [\n"
        '    "fastapi>=0.100",\n'
        '    "uvicorn>=0.20",\n'
        '    "pydantic>=2.0",\n'
        "]\n\n"
        "[tool.pytest.ini_options]\n"
        'testpaths = ["tests"]\n'
    )

    # docker-compose.yml
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n"
        "  api:\n"
        '    build: "."  \n'
        "    ports:\n"
        '      - "8080:8000"\n'
        "    depends_on:\n"
        "      - postgres\n"
        "    networks:\n"
        "      - backend\n"
        "  postgres:\n"
        '    image: "postgres:16"\n'
        "    ports:\n"
        '      - "5432:5432"\n'
        "    networks:\n"
        "      - backend\n"
        "networks:\n"
        "  backend:\n"
    )

    # Dockerfile
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")

    # .env
    (tmp_path / ".env").write_text("DEBUG=True\nDATABASE_URL=postgres://...\n")

    # .gitignore (intentionally missing .env entry)
    (tmp_path / ".gitignore").write_text("__pycache__\n*.pyc\n")

    # main.py
    (tmp_path / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")

    # tests/
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")

    return tmp_path


@pytest.fixture()
def node_project(tmp_path: Path) -> Path:
    """A minimal Node.js/Express project."""
    pkg = {
        "name": "my-api",
        "version": "1.0.0",
        "scripts": {"test": "jest", "start": "node index.js"},
        "dependencies": {"express": "^4.18.0", "dotenv": "^16.0.0"},
        "devDependencies": {"jest": "^29.0.0", "typescript": "^5.0.0"},
    }
    (tmp_path / "package.json").write_text(json.dumps(pkg, indent=2))
    (tmp_path / "index.js").write_text("const express = require('express');\n")
    return tmp_path


@pytest.fixture()
def requirements_project(tmp_path: Path) -> Path:
    """A Python project using requirements.txt."""
    (tmp_path / "requirements.txt").write_text(
        "flask==3.0.0\n"
        "gunicorn>=21.0\n"
        "redis~=5.0\n"
        "# comment\n"
        "-r requirements-dev.txt\n"
    )
    (tmp_path / "app.py").write_text("from flask import Flask\n")
    return tmp_path


# ===========================================================================
# 1. discover
# ===========================================================================


class TestDiscover:
    """Tests for project_discover."""

    @pytest.mark.asyncio
    async def test_python_project(self, python_project: Path) -> None:
        """Detects Python, FastAPI, Docker, and test config."""
        result = await discover(str(python_project))

        assert result["name"] == python_project.name
        assert "python" in result["languages"]
        assert "fastapi" in result["frameworks"]
        assert result["has_docker"] is True
        assert result["has_tests"] is True
        assert result["test_command"] == "pytest"
        assert "pyproject.toml" in result["config_files"]
        assert "docker-compose.yml" in result["config_files"]

    @pytest.mark.asyncio
    async def test_python_services(self, python_project: Path) -> None:
        """Docker-compose services are parsed correctly."""
        result = await discover(str(python_project))

        svc_names = {s["name"] for s in result["services"]}
        assert "api" in svc_names
        assert "postgres" in svc_names

        pg_svc = next(s for s in result["services"] if s["name"] == "postgres")
        assert "postgres:16" in pg_svc["image"]

    @pytest.mark.asyncio
    async def test_node_project(self, node_project: Path) -> None:
        """Detects Node.js, Express, TypeScript, and npm test."""
        result = await discover(str(node_project))

        assert "javascript" in result["languages"]
        assert "typescript" in result["languages"]
        assert "express" in result["frameworks"]
        assert result["has_tests"] is True
        assert result["test_command"] == "npm test"

    @pytest.mark.asyncio
    async def test_requirements_txt_project(self, requirements_project: Path) -> None:
        """Detects Python and Flask from requirements.txt."""
        result = await discover(str(requirements_project))

        assert "python" in result["languages"]
        assert "flask" in result["frameworks"]
        assert "app.py" in result["entry_points"]

    @pytest.mark.asyncio
    async def test_invalid_path(self) -> None:
        """Non-existent path returns error."""
        result = await discover("/nonexistent/project/path")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns empty but valid result."""
        result = await discover(str(tmp_path))
        assert result["languages"] == []
        assert result["has_docker"] is False
        assert result["has_tests"] is False

    @pytest.mark.asyncio
    async def test_go_project(self, tmp_path: Path) -> None:
        """Detects Go from go.mod."""
        (tmp_path / "go.mod").write_text("module example.com/myapp\ngo 1.21\n")
        result = await discover(str(tmp_path))
        assert "go" in result["languages"]

    @pytest.mark.asyncio
    async def test_rust_project(self, tmp_path: Path) -> None:
        """Detects Rust from Cargo.toml."""
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "myapp"\n')
        result = await discover(str(tmp_path))
        assert "rust" in result["languages"]

    @pytest.mark.asyncio
    async def test_makefile_targets(self, tmp_path: Path) -> None:
        """Parses Makefile targets into entry_points."""
        (tmp_path / "Makefile").write_text(
            "build:\n\tgo build ./...\n\ntest:\n\tgo test ./...\n\nclean:\n\trm -rf bin/\n"
        )
        result = await discover(str(tmp_path))
        assert "make build" in result["entry_points"]
        assert "make test" in result["entry_points"]


# ===========================================================================
# 2. dependency_check
# ===========================================================================


class TestDependencyCheck:
    """Tests for project_dependency_check."""

    @pytest.mark.asyncio
    async def test_requirements_txt(self, requirements_project: Path) -> None:
        """Parses requirements.txt dependencies."""
        result = await dependency_check(str(requirements_project))

        assert result["manager"] == "pip"
        assert result["count"] == 3  # flask, gunicorn, redis (skip comment + -r)
        names = {d["name"] for d in result["dependencies"]}
        assert "flask" in names
        assert "gunicorn" in names
        assert "redis" in names

    @pytest.mark.asyncio
    async def test_pyproject_toml(self, python_project: Path) -> None:
        """Parses pyproject.toml dependencies (when no requirements.txt)."""
        result = await dependency_check(str(python_project))

        assert result["manager"] == "pip"
        assert result["count"] == 3  # fastapi, uvicorn, pydantic
        names = {d["name"] for d in result["dependencies"]}
        assert "fastapi" in names

    @pytest.mark.asyncio
    async def test_package_json(self, node_project: Path) -> None:
        """Parses package.json dependencies."""
        result = await dependency_check(str(node_project))

        assert result["manager"] == "npm"
        assert result["count"] == 4  # express, dotenv, jest, typescript
        names = {d["name"] for d in result["dependencies"]}
        assert "express" in names
        assert "jest" in names

    @pytest.mark.asyncio
    async def test_empty_project(self, tmp_path: Path) -> None:
        """Project with no manifest returns unknown."""
        result = await dependency_check(str(tmp_path))
        assert result["manager"] == "unknown"
        assert result["count"] == 0


# ===========================================================================
# 3. test_runner
# ===========================================================================


class TestTestRunner:
    """Tests for project_test_runner."""

    @pytest.mark.asyncio
    async def test_detects_pytest(self, python_project: Path) -> None:
        """Auto-detects pytest from pyproject.toml."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b"tests/test_api.py::test_health PASSED\n"
            b"tests/test_api.py::test_create PASSED\n"
            b"tests/test_api.py::test_fail FAILED\n"
            b"===== 2 passed, 1 failed in 0.5s =====\n",
            b"",
        )

        with patch("probe_agent.tools.project.asyncio.create_subprocess_shell", return_value=mock_proc):
            result = await _test_runner(str(python_project))

        assert "pytest" in result["command"]
        assert result["passed"] == 2
        assert result["failed"] == 1

    @pytest.mark.asyncio
    async def test_detects_npm(self, node_project: Path) -> None:
        """Auto-detects npm test from package.json."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"Tests: 5 passed, 0 failed\n", b"")

        with patch("probe_agent.tools.project.asyncio.create_subprocess_shell", return_value=mock_proc):
            result = await _test_runner(str(node_project))

        assert result["command"] == "npm test"

    @pytest.mark.asyncio
    async def test_no_framework(self, tmp_path: Path) -> None:
        """Project with no test framework returns helpful message."""
        result = await _test_runner(str(tmp_path))
        assert result["command"] is None
        assert "Could not detect" in result["output"]

    @pytest.mark.asyncio
    async def test_specific_test_path(self, python_project: Path) -> None:
        """test_path is appended to the command."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"1 passed\n", b"")

        with patch("probe_agent.tools.project.asyncio.create_subprocess_shell", return_value=mock_proc):
            result = await _test_runner(str(python_project), test_path="tests/test_api.py")

        assert "tests/test_api.py" in result["command"]


# ===========================================================================
# 4. lint_check
# ===========================================================================


class TestLintCheck:
    """Tests for project_lint_check."""

    @pytest.mark.asyncio
    async def test_detects_ruff(self, python_project: Path) -> None:
        """Auto-detects ruff for Python projects."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b"src/app.py:10:1: E501 Line too long (120 > 88)\n"
            b"src/app.py:15:1: W291 Trailing whitespace\n",
            b"",
        )

        with patch("probe_agent.tools.project.asyncio.create_subprocess_shell", return_value=mock_proc):
            result = await lint_check(str(python_project))

        assert result["tool"] == "ruff"
        assert result["count"] == 2
        assert result["issues"][0]["file"] == "src/app.py"
        assert result["issues"][0]["severity"] == "error"
        assert result["issues"][1]["severity"] == "warning"

    @pytest.mark.asyncio
    async def test_no_linter(self, tmp_path: Path) -> None:
        """Project with no recognisable language returns no tool."""
        result = await lint_check(str(tmp_path))
        assert result["tool"] == "none"
        assert result["count"] == 0


# ===========================================================================
# 5. config_audit
# ===========================================================================


class TestConfigAudit:
    """Tests for project_config_audit."""

    @pytest.mark.asyncio
    async def test_env_not_in_gitignore(self, python_project: Path) -> None:
        """Flags .env missing from .gitignore."""
        result = await config_audit(str(python_project))

        env_issues = [i for i in result["issues"]
                      if ".env" in i["message"] and "gitignore" in i["message"]]
        assert len(env_issues) == 1
        assert env_issues[0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_debug_mode(self, python_project: Path) -> None:
        """Flags DEBUG=True in .env."""
        result = await config_audit(str(python_project))

        debug_issues = [i for i in result["issues"] if "DEBUG" in i["message"]]
        assert len(debug_issues) >= 1

    @pytest.mark.asyncio
    async def test_missing_healthcheck(self, python_project: Path) -> None:
        """Flags services without healthcheck in docker-compose."""
        result = await config_audit(str(python_project))

        health_issues = [i for i in result["issues"] if "healthcheck" in i["message"]]
        assert len(health_issues) == 2  # api + postgres

    @pytest.mark.asyncio
    async def test_missing_resource_limits(self, python_project: Path) -> None:
        """Flags services without resource limits."""
        result = await config_audit(str(python_project))

        resource_issues = [i for i in result["issues"] if "resource limits" in i["message"]]
        assert len(resource_issues) == 2

    @pytest.mark.asyncio
    async def test_score_deduction(self, python_project: Path) -> None:
        """Score is deducted for each issue."""
        result = await config_audit(str(python_project))
        assert result["score"] < 100
        assert result["score"] >= 0

    @pytest.mark.asyncio
    async def test_clean_project(self, tmp_path: Path) -> None:
        """Project with no issues scores 100."""
        result = await config_audit(str(tmp_path))
        assert result["score"] == 100
        assert result["issues"] == []

    @pytest.mark.asyncio
    async def test_hardcoded_secrets(self, tmp_path: Path) -> None:
        """Flags hardcoded secrets in .env."""
        (tmp_path / ".env").write_text("password=mysecretpassword123\nNORMAL_VAR=hello\n")
        (tmp_path / ".gitignore").write_text(".env\n")

        result = await config_audit(str(tmp_path))

        secret_issues = [i for i in result["issues"] if "secret" in i["message"].lower()]
        assert len(secret_issues) >= 1


# ===========================================================================
# 6. service_map
# ===========================================================================


class TestServiceMap:
    """Tests for project_service_map."""

    @pytest.mark.asyncio
    async def test_maps_services(self, python_project: Path) -> None:
        """Parses services and dependencies from docker-compose."""
        result = await service_map(str(python_project))

        svc_names = {s["name"] for s in result["services"]}
        assert "api" in svc_names
        assert "postgres" in svc_names

    @pytest.mark.asyncio
    async def test_connections(self, python_project: Path) -> None:
        """depends_on produces connection edges."""
        result = await service_map(str(python_project))

        assert len(result["connections"]) == 1
        conn = result["connections"][0]
        assert conn["from"] == "api"
        assert conn["to"] == "postgres"

    @pytest.mark.asyncio
    async def test_ports_and_networks(self, python_project: Path) -> None:
        """Ports and networks are parsed."""
        result = await service_map(str(python_project))

        api_svc = next(s for s in result["services"] if s["name"] == "api")
        assert "8080:8000" in api_svc["ports"]
        assert "backend" in api_svc["networks"]

    @pytest.mark.asyncio
    async def test_no_compose(self, tmp_path: Path) -> None:
        """Project without docker-compose returns empty."""
        result = await service_map(str(tmp_path))
        assert result["services"] == []
        assert result["connections"] == []


# ===========================================================================
# Registration
# ===========================================================================


class TestRegistration:
    """Tests for register_project_tools."""

    def test_registers_6_tools(self) -> None:
        """register_project_tools populates the registry with exactly 6 tools."""
        registry = ToolRegistry()
        register_project_tools(registry)
        assert registry.count() == 6
        assert registry.list_namespaces() == ["project"]

    def test_all_names_start_with_project(self) -> None:
        """Every registered tool has the 'project_' namespace prefix."""
        registry = ToolRegistry()
        register_project_tools(registry)
        for name in registry.list_tools():
            assert name.startswith("project_"), f"{name} missing project_ prefix"

    def test_expected_tool_names(self) -> None:
        """Verify all 6 expected tool names are registered."""
        registry = ToolRegistry()
        register_project_tools(registry)
        expected = {
            "project_discover",
            "project_dependency_check",
            "project_test_runner",
            "project_lint_check",
            "project_config_audit",
            "project_service_map",
        }
        assert set(registry.list_tools()) == expected

    def test_schemas_have_required_keys(self) -> None:
        """Every schema has name, description, and parameters."""
        registry = ToolRegistry()
        register_project_tools(registry)
        for schema in registry.get_schemas():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert len(schema["description"]) > 50
