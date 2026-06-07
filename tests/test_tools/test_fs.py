
from __future__ import annotations

from pathlib import Path

import pytest

from probe_agent.registry import ToolRegistry
from probe_agent.tools.fs import (
    diff_files,
    edit_file,
    file_info,
    find_files,
    list_dir,
    read_file,
    register_fs_tools,
    search_content,
    tail,
    tree,
    write_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """Create a small fake project tree for testing."""
    # src/app.py
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text(
        "from flask import Flask\n"
        "app = Flask(__name__)\n"
        "\n"
        "@app.route('/')\n"
        "def index():\n"
        "    return 'hello world'\n"
    )

    # src/utils.py
    (src / "utils.py").write_text("def helper():\n    return 42\n")

    # README.md
    (tmp_path / "README.md").write_text("# My Project\n\nA test project.\n")

    # logs/app.log (20 lines)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "app.log").write_text(
        "\n".join(f"[INFO] line {i}" for i in range(1, 21)) + "\n"
    )

    # .git dir (should be excluded)
    gitdir = tmp_path / ".git"
    gitdir.mkdir()
    (gitdir / "config").write_text("[core]\n")

    # __pycache__ (should be excluded)
    cache = src / "__pycache__"
    cache.mkdir()
    (cache / "app.cpython-312.pyc").write_bytes(b"\x00\x00")

    return tmp_path


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    """Tests for the read_file tool."""

    @pytest.mark.asyncio
    async def test_read_entire_file(self, project: Path) -> None:
        """Read a small file without line range."""
        result = await read_file(str(project / "README.md"))
        assert "error" not in result
        assert "# My Project" in result["content"]
        assert result["total_lines"] == 3

    @pytest.mark.asyncio
    async def test_read_line_range(self, project: Path) -> None:
        """Read specific lines from a file."""
        result = await read_file(str(project / "src" / "app.py"), start_line=4, end_line=6)
        assert "error" not in result
        assert "@app.route" in result["content"]
        assert "hello world" in result["content"]

    @pytest.mark.asyncio
    async def test_read_truncates_large_file(self, tmp_path: Path) -> None:
        """Files >500 lines without a range get truncated."""
        big = tmp_path / "big.txt"
        big.write_text("\n".join(f"line {i}" for i in range(800)))

        result = await read_file(str(big))
        assert "truncated" in result["content"]
        assert result["total_lines"] == 800

    @pytest.mark.asyncio
    async def test_read_missing_file(self) -> None:
        """Missing file returns an error dict, not an exception."""
        result = await read_file("/nonexistent/path.txt")
        assert "error" in result
        assert result["error_type"] == "FileNotFoundError"


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    """Tests for the write_file tool."""

    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path: Path) -> None:
        """Write to a new file."""
        target = tmp_path / "out.txt"
        result = await write_file(str(target), "hello")
        assert result["bytes_written"] == 5
        assert target.read_text() == "hello"

    @pytest.mark.asyncio
    async def test_write_creates_dirs(self, tmp_path: Path) -> None:
        """write_file with create_dirs=True creates parents."""
        target = tmp_path / "a" / "b" / "c.txt"
        result = await write_file(str(target), "deep", create_dirs=True)
        assert "error" not in result
        assert target.read_text() == "deep"


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


class TestEditFile:
    """Tests for the edit_file tool."""

    @pytest.mark.asyncio
    async def test_successful_edit(self, project: Path) -> None:
        """Edit replaces text and returns replacement count + preview."""
        path = str(project / "src" / "app.py")
        result = await edit_file(path, "hello world", "goodbye world")

        assert "error" not in result
        assert result["replacements"] == 1
        assert "goodbye world" in result["preview"]

        # Verify the file was actually changed.
        updated = (project / "src" / "app.py").read_text()
        assert "goodbye world" in updated
        assert "hello world" not in updated

    @pytest.mark.asyncio
    async def test_edit_text_not_found(self, project: Path) -> None:
        """Edit returns error when search text doesn't exist."""
        path = str(project / "src" / "app.py")
        result = await edit_file(path, "DOES NOT EXIST", "replacement")
        assert "error" in result
        assert result["error_type"] == "NotFound"

    @pytest.mark.asyncio
    async def test_edit_missing_file(self) -> None:
        """Edit returns error for missing file."""
        result = await edit_file("/nonexistent.py", "a", "b")
        assert result["error_type"] == "FileNotFoundError"


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


class TestListDir:
    """Tests for the list_dir tool."""

    @pytest.mark.asyncio
    async def test_list_top_level(self, project: Path) -> None:
        """Lists files and dirs, excluding .git and __pycache__."""
        result = await list_dir(str(project))
        names = [e["name"] for e in result["entries"]]
        assert "README.md" in names
        assert ".git" not in names

    @pytest.mark.asyncio
    async def test_entries_have_type_and_size(self, project: Path) -> None:
        """Each entry has name, type, and size_bytes."""
        result = await list_dir(str(project))
        for entry in result["entries"]:
            assert "name" in entry
            assert "type" in entry
            assert "size_bytes" in entry
            assert entry["type"] in ("file", "dir")


# ---------------------------------------------------------------------------
# find_files
# ---------------------------------------------------------------------------


class TestFindFiles:
    """Tests for the find_files tool."""

    @pytest.mark.asyncio
    async def test_find_python_files(self, project: Path) -> None:
        """Find *.py files in the project."""
        result = await find_files(str(project), "**/*.py")
        assert result["count"] >= 2
        py_files = result["matches"]
        assert any("app.py" in f for f in py_files)
        assert any("utils.py" in f for f in py_files)


# ---------------------------------------------------------------------------
# search_content
# ---------------------------------------------------------------------------


class TestSearchContent:
    """Tests for the search_content tool."""

    @pytest.mark.asyncio
    async def test_search_finds_matches(self, project: Path) -> None:
        """Search finds text across files with file/line info."""
        result = await search_content(str(project), "Flask")
        assert "error" not in result
        assert result["count"] >= 1
        match = result["matches"][0]
        assert "file" in match
        assert "line" in match
        assert "content" in match
        assert "Flask" in match["content"]

    @pytest.mark.asyncio
    async def test_search_no_matches(self, project: Path) -> None:
        """Search with no hits returns empty matches list."""
        result = await search_content(str(project), "ZZZYYYXXX_UNIQUE_STRING")
        assert result["count"] == 0
        assert result["matches"] == []

    @pytest.mark.asyncio
    async def test_search_with_file_pattern(self, project: Path) -> None:
        """file_pattern restricts search to matching files."""
        result = await search_content(str(project), "return", file_pattern="*.py")
        assert result["count"] >= 1
        for m in result["matches"]:
            assert m["file"].endswith(".py")


# ---------------------------------------------------------------------------
# file_info
# ---------------------------------------------------------------------------


class TestFileInfo:
    """Tests for the file_info tool."""

    @pytest.mark.asyncio
    async def test_info_on_file(self, project: Path) -> None:
        """file_info returns metadata for a regular file."""
        result = await file_info(str(project / "README.md"))
        assert result["type"] == "file"
        assert result["size_bytes"] > 0
        assert "modified" in result
        assert "permissions" in result

    @pytest.mark.asyncio
    async def test_info_missing_path(self) -> None:
        """file_info returns error for nonexistent path."""
        result = await file_info("/nonexistent/thing")
        assert result["error_type"] == "FileNotFoundError"


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------


class TestTree:
    """Tests for the tree tool."""

    @pytest.mark.asyncio
    async def test_tree_output(self, project: Path) -> None:
        """Tree produces a multi-line text representation."""
        result = await tree(str(project))
        assert "error" not in result
        tree_text = result["tree"]

        # Should contain project entries.
        assert "src/" in tree_text
        assert "README.md" in tree_text

        # Should exclude .git and __pycache__.
        assert ".git" not in tree_text
        assert "__pycache__" not in tree_text

    @pytest.mark.asyncio
    async def test_tree_uses_connectors(self, project: Path) -> None:
        """Tree uses ├── and └── box-drawing connectors."""
        result = await tree(str(project))
        tree_text = result["tree"]
        assert "├──" in tree_text or "└──" in tree_text


# ---------------------------------------------------------------------------
# diff_files
# ---------------------------------------------------------------------------


class TestDiffFiles:
    """Tests for the diff_files tool."""

    @pytest.mark.asyncio
    async def test_diff_identical_files(self, tmp_path: Path) -> None:
        """Identical files produce empty diff."""
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("same\n")
        b.write_text("same\n")
        result = await diff_files(str(a), str(b))
        assert result["changed"] is False
        assert result["diff"] == ""

    @pytest.mark.asyncio
    async def test_diff_different_files(self, tmp_path: Path) -> None:
        """Different files produce a non-empty unified diff."""
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("alpha\nbeta\n")
        b.write_text("alpha\ngamma\n")
        result = await diff_files(str(a), str(b))
        assert result["changed"] is True
        assert "-beta" in result["diff"]
        assert "+gamma" in result["diff"]


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------


class TestTail:
    """Tests for the tail tool."""

    @pytest.mark.asyncio
    async def test_tail_last_5_lines(self, project: Path) -> None:
        """Tail returns the last N lines."""
        result = await tail(str(project / "logs" / "app.log"), lines=5)
        assert "error" not in result
        assert result["lines_returned"] == 5
        assert "line 20" in result["content"]
        assert "line 16" in result["content"]

    @pytest.mark.asyncio
    async def test_tail_more_than_file_length(self, project: Path) -> None:
        """Requesting more lines than the file has returns the whole file."""
        result = await tail(str(project / "README.md"), lines=9999)
        assert result["lines_returned"] == 3


# ---------------------------------------------------------------------------
# Registration integration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Tests for register_fs_tools."""

    def test_registers_10_tools(self) -> None:
        """register_fs_tools populates the registry with exactly 10 tools."""
        registry = ToolRegistry()
        register_fs_tools(registry)
        assert registry.count() == 10
        assert registry.list_namespaces() == ["fs"]

    def test_all_tool_names_start_with_fs(self) -> None:
        """Every registered tool has the 'fs_' namespace prefix."""
        registry = ToolRegistry()
        register_fs_tools(registry)
        for name in registry.list_tools():
            assert name.startswith("fs_"), f"{name} missing fs_ prefix"

    def test_schemas_have_required_keys(self) -> None:
        """Every schema has name, description, and parameters."""
        registry = ToolRegistry()
        register_fs_tools(registry)
        for schema in registry.get_schemas():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert len(schema["description"]) > 50, (
                f"Tool {schema['name']} description is too short for the LLM"
            )
