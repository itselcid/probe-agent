"""Tests for the evaluation framework (scenarios, metrics, harness)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from probe_agent.eval.metrics import (
    EvalResult,
    compute_tool_hit_rate,
    summarize_results,
)
from probe_agent.eval.scenarios import (
    SCENARIOS,
    Scenario,
    get_scenario,
    list_scenario_names,
)


# ===========================================================================
# Scenarios
# ===========================================================================


class TestScenario:
    """Tests for Scenario dataclass."""

    def test_scenario_fields(self) -> None:
        """Scenario has all required fields."""
        s = Scenario(
            name="test",
            description="A test scenario",
            task="do something",
            expected_tools=["fs_read_file"],
            max_steps=10,
            timeout_seconds=60,
        )
        assert s.name == "test"
        assert s.description == "A test scenario"
        assert s.task == "do something"
        assert s.expected_tools == ["fs_read_file"]
        assert s.max_steps == 10
        assert s.timeout_seconds == 60

    def test_scenario_defaults(self) -> None:
        """Scenario uses sensible defaults for max_steps and timeout."""
        s = Scenario(name="t", description="d", task="task")
        assert s.max_steps == 30
        assert s.timeout_seconds == 300
        assert s.expected_tools == []

    def test_scenario_is_frozen(self) -> None:
        """Scenario is immutable (frozen dataclass)."""
        s = Scenario(name="t", description="d", task="task")
        with pytest.raises(AttributeError):
            s.name = "changed"  # type: ignore[misc]


class TestScenarioCatalogue:
    """Tests for the built-in scenario list."""

    def test_scenarios_not_empty(self) -> None:
        """SCENARIOS contains at least 5 scenarios."""
        assert len(SCENARIOS) >= 5

    def test_all_scenarios_have_names(self) -> None:
        """Every scenario has a non-empty name."""
        for s in SCENARIOS:
            assert s.name, f"Scenario missing name: {s}"

    def test_all_names_unique(self) -> None:
        """No duplicate scenario names."""
        names = [s.name for s in SCENARIOS]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_all_have_tasks(self) -> None:
        """Every scenario has a non-empty task."""
        for s in SCENARIOS:
            assert s.task, f"Scenario {s.name} missing task"

    def test_all_have_expected_tools(self) -> None:
        """Every scenario has at least one expected tool."""
        for s in SCENARIOS:
            assert len(s.expected_tools) >= 1, (
                f"Scenario {s.name} has no expected tools"
            )

    def test_container_health_exists(self) -> None:
        """The container_health scenario exists with correct tools."""
        s = get_scenario("container_health")
        assert "docker_ps" in s.expected_tools
        assert "docker_stats" in s.expected_tools

    def test_diagnostic_subagent_exists(self) -> None:
        """The diagnostic_subagent scenario uses agent_spawn_diagnostic."""
        s = get_scenario("diagnostic_subagent")
        assert "agent_spawn_diagnostic" in s.expected_tools

    def test_full_workflow_exists(self) -> None:
        """The full_workflow scenario covers discovery + diagnosis + report."""
        s = get_scenario("full_workflow")
        assert "project_discover" in s.expected_tools
        assert "agent_spawn_diagnostic" in s.expected_tools
        assert "agent_spawn_report" in s.expected_tools


class TestGetScenario:
    """Tests for get_scenario lookup."""

    def test_valid_name(self) -> None:
        """get_scenario returns correct scenario by name."""
        s = get_scenario("project_analysis")
        assert s.name == "project_analysis"

    def test_invalid_name_raises(self) -> None:
        """get_scenario raises KeyError for unknown names."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_scenario("nonexistent")


class TestListScenarioNames:
    """Tests for list_scenario_names."""

    def test_returns_all_names(self) -> None:
        """list_scenario_names returns all scenario names in order."""
        names = list_scenario_names()
        assert len(names) == len(SCENARIOS)
        assert names[0] == SCENARIOS[0].name

    def test_returns_strings(self) -> None:
        """All returned names are strings."""
        for name in list_scenario_names():
            assert isinstance(name, str)


# ===========================================================================
# Metrics — tool hit rate
# ===========================================================================


class TestToolHitRate:
    """Tests for compute_tool_hit_rate."""

    def test_perfect_hit_rate(self) -> None:
        """All expected tools used → 1.0."""
        rate = compute_tool_hit_rate(
            expected=["docker_ps", "docker_stats"],
            used=["docker_ps", "docker_stats", "fs_tree"],
        )
        assert rate == 1.0

    def test_partial_hit_rate(self) -> None:
        """Half of expected tools used → 0.5."""
        rate = compute_tool_hit_rate(
            expected=["docker_ps", "docker_stats"],
            used=["docker_ps"],
        )
        assert rate == 0.5

    def test_zero_hit_rate(self) -> None:
        """No expected tools used → 0.0."""
        rate = compute_tool_hit_rate(
            expected=["docker_ps", "docker_stats"],
            used=["fs_tree", "git_log"],
        )
        assert rate == 0.0

    def test_empty_expected(self) -> None:
        """No expectations → 1.0 (vacuously true)."""
        rate = compute_tool_hit_rate(expected=[], used=["docker_ps"])
        assert rate == 1.0

    def test_empty_both(self) -> None:
        """No expectations and no usage → 1.0."""
        rate = compute_tool_hit_rate(expected=[], used=[])
        assert rate == 1.0

    def test_one_of_three(self) -> None:
        """1/3 expected tools → ~0.333."""
        rate = compute_tool_hit_rate(
            expected=["a", "b", "c"],
            used=["a"],
        )
        assert abs(rate - 1 / 3) < 0.001


# ===========================================================================
# Metrics — EvalResult
# ===========================================================================


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_default_values(self) -> None:
        """EvalResult has sensible defaults."""
        r = EvalResult(scenario_name="test", passed=True)
        assert r.tools_expected == []
        assert r.tools_used == []
        assert r.tool_hit_rate == 0.0
        assert r.total_steps == 0
        assert r.total_tokens == 0
        assert r.duration_seconds == 0.0
        assert r.final_response_preview == ""
        assert r.error is None

    def test_full_result(self) -> None:
        """EvalResult stores all fields correctly."""
        r = EvalResult(
            scenario_name="container_health",
            passed=True,
            tools_expected=["docker_ps", "docker_stats"],
            tools_used=["docker_ps", "docker_stats"],
            tool_hit_rate=1.0,
            total_steps=3,
            total_tokens=500,
            duration_seconds=2.5,
            final_response_preview="All containers healthy.",
            error=None,
        )
        assert r.scenario_name == "container_health"
        assert r.passed is True
        assert r.tool_hit_rate == 1.0

    def test_failed_result(self) -> None:
        """EvalResult records errors."""
        r = EvalResult(
            scenario_name="test",
            passed=False,
            error="TimeoutError: exceeded 300s",
        )
        assert r.passed is False
        assert r.error is not None


# ===========================================================================
# Metrics — summarize_results
# ===========================================================================


class TestSummarizeResults:
    """Tests for summarize_results."""

    def test_empty_results(self) -> None:
        """Empty results list → zeroed summary."""
        s = summarize_results([])
        assert s["total_scenarios"] == 0
        assert s["pass_rate"] == 0.0

    def test_all_passed(self) -> None:
        """All passed → 100% pass rate."""
        results = [
            EvalResult(scenario_name="a", passed=True, tool_hit_rate=1.0,
                       total_steps=3, total_tokens=100, duration_seconds=1.0),
            EvalResult(scenario_name="b", passed=True, tool_hit_rate=0.5,
                       total_steps=5, total_tokens=200, duration_seconds=2.0),
        ]
        s = summarize_results(results)
        assert s["total_scenarios"] == 2
        assert s["passed"] == 2
        assert s["failed"] == 0
        assert s["pass_rate"] == 1.0
        assert s["avg_tool_hit_rate"] == 0.75
        assert s["avg_steps"] == 4.0
        assert s["avg_tokens"] == 150.0
        assert s["total_tokens"] == 300

    def test_mixed_results(self) -> None:
        """1 pass + 1 fail → 50% pass rate."""
        results = [
            EvalResult(scenario_name="a", passed=True, tool_hit_rate=1.0,
                       total_steps=3, total_tokens=100, duration_seconds=1.0),
            EvalResult(scenario_name="b", passed=False, tool_hit_rate=0.0,
                       total_steps=10, total_tokens=500, duration_seconds=5.0),
        ]
        s = summarize_results(results)
        assert s["passed"] == 1
        assert s["failed"] == 1
        assert s["pass_rate"] == 0.5

    def test_all_failed(self) -> None:
        """All failed → 0% pass rate."""
        results = [
            EvalResult(scenario_name="a", passed=False, error="timeout"),
            EvalResult(scenario_name="b", passed=False, error="crash"),
        ]
        s = summarize_results(results)
        assert s["pass_rate"] == 0.0
        assert s["failed"] == 2


# ===========================================================================
# Harness — unit tests (mocked LLM)
# ===========================================================================


class TestEvalHarness:
    """Tests for EvalHarness with mocked LLM."""

    @pytest.mark.asyncio
    async def test_run_scenario_success(self, tmp_path) -> None:
        """run_scenario returns a passing EvalResult when agent succeeds."""
        from probe_agent.config import Settings
        from probe_agent.eval.harness import EvalHarness
        from probe_agent.types import LLMResponse, ToolCall

        settings = Settings(
            llm_provider="gemini",
            llm_api_key="fake-key",
            project_path=str(tmp_path),
        )

        scenario = Scenario(
            name="test_pass",
            description="Test passing scenario",
            task="list containers",
            expected_tools=["docker_ps"],
            max_steps=5,
            timeout_seconds=10,
        )

        # Mock the LLM to call docker_ps then respond.
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(name="docker_ps", id="c1", arguments={})],
                usage={"total_tokens": 30},
            ),
            LLMResponse(
                content="2 containers running.",
                tool_calls=[],
                usage={"total_tokens": 20},
            ),
        ]

        harness = EvalHarness(project_path=str(tmp_path), config=settings)

        with patch.object(harness, "_build_registry_and_llm") as mock_build:
            # Build a registry with a mock docker_ps tool.
            from probe_agent.registry import ToolRegistry

            registry = ToolRegistry()
            registry.register(
                namespace="docker",
                name="ps",
                fn=AsyncMock(return_value={"containers": [], "count": 0}),
                description="List containers",
                parameters={"type": "object", "properties": {}},
            )
            mock_build.return_value = (registry, mock_llm)

            result = await harness.run_scenario(scenario)

        assert result.passed is True
        assert result.tool_hit_rate == 1.0
        assert "docker_ps" in result.tools_used
        assert result.error is None

    @pytest.mark.asyncio
    async def test_run_scenario_timeout(self, tmp_path) -> None:
        """run_scenario returns failed EvalResult on timeout."""
        from probe_agent.config import Settings
        from probe_agent.eval.harness import EvalHarness
        from probe_agent.types import LLMResponse, ToolCall

        settings = Settings(
            llm_provider="gemini",
            llm_api_key="fake-key",
            project_path=str(tmp_path),
        )

        scenario = Scenario(
            name="test_timeout",
            description="Times out",
            task="slow task",
            expected_tools=["docker_ps"],
            max_steps=100,
            timeout_seconds=1,  # 1 second timeout
        )

        # Mock LLM that hangs forever.
        async def slow_chat(**kwargs):
            import asyncio
            await asyncio.sleep(10)
            return LLMResponse(content="done", tool_calls=[], usage={})

        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = slow_chat

        harness = EvalHarness(project_path=str(tmp_path), config=settings)

        with patch.object(harness, "_build_registry_and_llm") as mock_build:
            from probe_agent.registry import ToolRegistry
            mock_build.return_value = (ToolRegistry(), mock_llm)

            result = await harness.run_scenario(scenario)

        assert result.passed is False
        assert "Timeout" in (result.error or "")

    @pytest.mark.asyncio
    async def test_run_scenario_exception(self, tmp_path) -> None:
        """run_scenario returns failed EvalResult on exception."""
        from probe_agent.config import Settings
        from probe_agent.eval.harness import EvalHarness

        settings = Settings(
            llm_provider="gemini",
            llm_api_key="fake-key",
            project_path=str(tmp_path),
        )

        scenario = Scenario(
            name="test_crash",
            description="Crashes",
            task="crash task",
            expected_tools=["docker_ps"],
        )

        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = RuntimeError("LLM exploded")

        harness = EvalHarness(project_path=str(tmp_path), config=settings)

        with patch.object(harness, "_build_registry_and_llm") as mock_build:
            from probe_agent.registry import ToolRegistry
            mock_build.return_value = (ToolRegistry(), mock_llm)

            result = await harness.run_scenario(scenario)

        assert result.passed is False
        assert "RuntimeError" in (result.error or "")
