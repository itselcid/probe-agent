
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from probe_agent.registry import ToolRegistry
from probe_agent.tools.git import (
    blame,
    branch_list,
    checkout,
    commit,
    diff,
    log,
    register_git_tools,
    remote_info,
    show,
    stash,
    status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repo with one initial commit."""
    subprocess.run(
        ["git", "init", str(tmp_path)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@probe.dev"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test User"],
        check=True, capture_output=True,
    )
    (tmp_path / "test.py").write_text("print('hello')\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "init"],
        check=True, capture_output=True,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    """Tests for the git status tool."""

    @pytest.mark.asyncio
    async def test_clean_repo(self, git_repo: Path) -> None:
        """A freshly committed repo reports clean."""
        result = await status(str(git_repo))
        assert "error" not in result
        assert result["clean"] is True
        assert result["staged"] == []
        assert result["unstaged"] == []
        assert result["untracked"] == []

    @pytest.mark.asyncio
    async def test_untracked_file(self, git_repo: Path) -> None:
        """A new file shows as untracked."""
        (git_repo / "new.txt").write_text("new file\n")
        result = await status(str(git_repo))
        assert result["clean"] is False
        assert "new.txt" in result["untracked"]

    @pytest.mark.asyncio
    async def test_staged_file(self, git_repo: Path) -> None:
        """A file added to the index shows as staged."""
        (git_repo / "staged.txt").write_text("staged\n")
        subprocess.run(
            ["git", "-C", str(git_repo), "add", "staged.txt"],
            check=True, capture_output=True,
        )
        result = await status(str(git_repo))
        assert "staged.txt" in result["staged"]

    @pytest.mark.asyncio
    async def test_unstaged_modification(self, git_repo: Path) -> None:
        """Modifying a tracked file without staging shows as unstaged."""
        (git_repo / "test.py").write_text("print('modified')\n")
        result = await status(str(git_repo))
        assert "test.py" in result["unstaged"]

    @pytest.mark.asyncio
    async def test_branch_name(self, git_repo: Path) -> None:
        """Status reports the current branch name."""
        result = await status(str(git_repo))
        assert result["branch"] != ""


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


class TestDiff:
    """Tests for the git diff tool."""

    @pytest.mark.asyncio
    async def test_diff_with_changes(self, git_repo: Path) -> None:
        """Diff shows modifications to tracked files."""
        (git_repo / "test.py").write_text("print('changed')\n")
        result = await diff(str(git_repo))
        assert "error" not in result
        assert "changed" in result["diff"]

    @pytest.mark.asyncio
    async def test_diff_staged(self, git_repo: Path) -> None:
        """Diff with staged=True shows index changes."""
        (git_repo / "test.py").write_text("print('staged_change')\n")
        subprocess.run(
            ["git", "-C", str(git_repo), "add", "test.py"],
            check=True, capture_output=True,
        )
        result = await diff(str(git_repo), staged=True)
        assert "staged_change" in result["diff"]


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------


class TestLog:
    """Tests for the git log tool."""

    @pytest.mark.asyncio
    async def test_log_returns_commits(self, git_repo: Path) -> None:
        """Log returns the initial commit."""
        result = await log(str(git_repo))
        assert "error" not in result
        assert result["count"] >= 1
        first = result["commits"][0]
        assert "hash" in first
        assert "author" in first
        assert "date" in first
        assert first["message"] == "init"

    @pytest.mark.asyncio
    async def test_log_max_count(self, git_repo: Path) -> None:
        """max_count limits the number of returned commits."""
        # Add a second commit.
        (git_repo / "second.txt").write_text("second\n")
        subprocess.run(["git", "-C", str(git_repo), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(git_repo), "commit", "-m", "second"], check=True, capture_output=True)

        result = await log(str(git_repo), max_count=1)
        assert result["count"] == 1
        assert result["commits"][0]["message"] == "second"


# ---------------------------------------------------------------------------
# blame
# ---------------------------------------------------------------------------


class TestBlame:
    """Tests for the git blame tool."""

    @pytest.mark.asyncio
    async def test_blame_returns_lines(self, git_repo: Path) -> None:
        """Blame returns line-level authorship info."""
        result = await blame(str(git_repo), "test.py")
        assert "error" not in result
        assert len(result["lines"]) >= 1
        line = result["lines"][0]
        assert "line_num" in line
        assert "commit" in line
        assert "author" in line
        assert "content" in line
        assert "hello" in line["content"]


# ---------------------------------------------------------------------------
# branch_list
# ---------------------------------------------------------------------------


class TestBranchList:
    """Tests for the branch_list tool."""

    @pytest.mark.asyncio
    async def test_lists_branches(self, git_repo: Path) -> None:
        """Lists at least the current branch."""
        result = await branch_list(str(git_repo))
        assert "error" not in result
        assert len(result["local"]) >= 1
        assert result["current"] != ""
        assert result["current"] in result["local"]


# ---------------------------------------------------------------------------
# checkout
# ---------------------------------------------------------------------------


class TestCheckout:
    """Tests for the checkout tool."""

    @pytest.mark.asyncio
    async def test_create_and_switch(self, git_repo: Path) -> None:
        """Creates a new branch and switches to it."""
        result = await checkout(str(git_repo), branch="feature-x", create=True)
        assert "error" not in result
        assert result["branch"] == "feature-x"
        assert result["created"] is True
        assert result["previous_branch"] != ""

        # Verify we're on the new branch.
        st = await status(str(git_repo))
        assert st["branch"] == "feature-x"

    @pytest.mark.asyncio
    async def test_switch_back(self, git_repo: Path) -> None:
        """Switches back to the previous branch."""
        original = (await status(str(git_repo)))["branch"]
        await checkout(str(git_repo), branch="temp-branch", create=True)
        result = await checkout(str(git_repo), branch=original)
        assert result["branch"] == original


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------


class TestCommit:
    """Tests for the commit tool."""

    @pytest.mark.asyncio
    async def test_commit_all(self, git_repo: Path) -> None:
        """Commit stages all changes and creates a commit."""
        (git_repo / "new_file.py").write_text("x = 1\n")
        result = await commit(str(git_repo), message="add new_file")
        assert "error" not in result
        assert result["hash"] != ""
        assert result["message"] == "add new_file"
        assert result["files_committed"] >= 1

    @pytest.mark.asyncio
    async def test_commit_specific_files(self, git_repo: Path) -> None:
        """Commit only stages specified files."""
        (git_repo / "a.txt").write_text("a\n")
        (git_repo / "b.txt").write_text("b\n")
        result = await commit(str(git_repo), message="add a only", files=["a.txt"])
        assert "error" not in result

        # b.txt should still be untracked.
        st = await status(str(git_repo))
        assert "b.txt" in st["untracked"]


# ---------------------------------------------------------------------------
# stash
# ---------------------------------------------------------------------------


class TestStash:
    """Tests for the stash tool."""

    @pytest.mark.asyncio
    async def test_stash_push_and_pop(self, git_repo: Path) -> None:
        """Stash push saves changes; pop restores them."""
        (git_repo / "test.py").write_text("print('stashed')\n")
        push_result = await stash(str(git_repo), action="push", message="wip")
        assert "error" not in push_result

        # Working dir should be clean after stash push.
        st = await status(str(git_repo))
        assert st["clean"] is True

        # Pop brings changes back.
        pop_result = await stash(str(git_repo), action="pop")
        assert "error" not in pop_result

        content = (git_repo / "test.py").read_text()
        assert "stashed" in content

    @pytest.mark.asyncio
    async def test_stash_list_empty(self, git_repo: Path) -> None:
        """Stash list on clean repo returns empty result."""
        result = await stash(str(git_repo), action="list")
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_stash_invalid_action(self, git_repo: Path) -> None:
        """Invalid stash action returns an error."""
        result = await stash(str(git_repo), action="invalid")
        assert "error" in result


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


class TestShow:
    """Tests for the show tool."""

    @pytest.mark.asyncio
    async def test_show_head(self, git_repo: Path) -> None:
        """Show HEAD returns the initial commit details."""
        result = await show(str(git_repo), commit_ref="HEAD")
        assert "error" not in result
        assert result["message"] == "init"
        assert result["author"] == "Test User"
        assert len(result["hash"]) == 40
        assert "test.py" in result["files"]


# ---------------------------------------------------------------------------
# remote_info
# ---------------------------------------------------------------------------


class TestRemoteInfo:
    """Tests for the remote_info tool."""

    @pytest.mark.asyncio
    async def test_no_remotes(self, git_repo: Path) -> None:
        """A local-only repo has no remotes."""
        result = await remote_info(str(git_repo))
        assert "error" not in result
        assert result["remotes"] == []


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Tests for register_git_tools."""

    def test_registers_10_tools(self) -> None:
        """register_git_tools populates the registry with exactly 10 tools."""
        registry = ToolRegistry()
        register_git_tools(registry)
        assert registry.count() == 10
        assert registry.list_namespaces() == ["git"]

    def test_all_tool_names_start_with_git(self) -> None:
        """Every registered tool has the 'git_' namespace prefix."""
        registry = ToolRegistry()
        register_git_tools(registry)
        for name in registry.list_tools():
            assert name.startswith("git_"), f"{name} missing git_ prefix"

    def test_schemas_have_required_keys(self) -> None:
        """Every schema has name, description, and parameters."""
        registry = ToolRegistry()
        register_git_tools(registry)
        for schema in registry.get_schemas():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            # Every git tool requires repo_path.
            assert "repo_path" in schema["parameters"]["properties"]
