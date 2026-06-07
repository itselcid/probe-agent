"""Filesystem tools for ProbeAgent.

Ten tools that let the AI agent read, write, search, and inspect files
in a software project.  Each function is async, takes explicit typed
parameters, and returns a dict with structured data.

Register all tools at once with :func:`register_fs_tools`.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from probe_agent.registry import ToolRegistry

# Directories to skip when listing / searching / building trees.
_EXCLUDED_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", "venv", ".venv",
    ".mypy_cache", ".pytest_cache", ".tox", ".eggs", "dist", "build",
})

_MAX_READ_LINES = 500
_MAX_FIND_RESULTS = 100
_MAX_SEARCH_MATCHES = 50
_MAX_LINE_LENGTH = 200


# ---------------------------------------------------------------------------
# 1. read_file
# ---------------------------------------------------------------------------


async def read_file(
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict[str, Any]:
    """Read the contents of a file, optionally a specific line range.

    If the file exceeds 500 lines and no range is specified, only the
    first 500 lines are returned with a truncation note.

    Args:
        path: Absolute or relative path to the file.
        start_line: Start line number (1-indexed, inclusive).
        end_line: End line number (inclusive).  ``None`` means read to EOF.

    Returns:
        ``{"path", "content", "total_lines"}`` on success, or
        ``{"error", "error_type"}`` on failure.
    """
    try:
        p = Path(path).resolve()
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"error": f"File not found: {path}", "error_type": "FileNotFoundError"}
    except PermissionError:
        return {"error": f"Permission denied: {path}", "error_type": "PermissionError"}
    except UnicodeDecodeError:
        return {"error": f"Cannot read binary file: {path}", "error_type": "UnicodeDecodeError"}

    all_lines = text.splitlines(keepends=True)
    total_lines = len(all_lines)

    # Determine the slice.
    s = max(start_line - 1, 0)
    e = end_line if end_line is not None else total_lines

    truncated = False
    if end_line is None and total_lines > _MAX_READ_LINES and start_line == 1:
        e = _MAX_READ_LINES
        truncated = True

    selected = all_lines[s:e]
    content = "".join(selected)

    if truncated:
        content += (
            f"\n... [truncated — showing {_MAX_READ_LINES} of {total_lines} lines. "
            f"Use start_line/end_line to read more.]\n"
        )

    return {
        "path": str(p),
        "content": content,
        "total_lines": total_lines,
    }


# ---------------------------------------------------------------------------
# 2. write_file
# ---------------------------------------------------------------------------


async def write_file(
    path: str,
    content: str,
    create_dirs: bool = False,
) -> dict[str, Any]:
    """Write content to a file.  Optionally create parent directories.

    Args:
        path: Absolute or relative path to the file.
        content: The text to write.
        create_dirs: If ``True``, create missing parent directories.

    Returns:
        ``{"path", "bytes_written"}`` on success.
    """
    p = Path(path).resolve()

    if create_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)

    try:
        p.write_text(content, encoding="utf-8")
    except FileNotFoundError:
        return {"error": f"Parent directory does not exist: {p.parent}", "error_type": "FileNotFoundError"}
    except PermissionError:
        return {"error": f"Permission denied: {path}", "error_type": "PermissionError"}

    return {
        "path": str(p),
        "bytes_written": len(content.encode("utf-8")),
    }


# ---------------------------------------------------------------------------
# 3. edit_file
# ---------------------------------------------------------------------------


async def edit_file(
    path: str,
    search: str,
    replace: str,
) -> dict[str, Any]:
    """Find exact text in a file and replace it.

    Args:
        path: Path to the file.
        search: Exact text to find.
        replace: Text to substitute.

    Returns:
        ``{"path", "replacements", "preview"}`` on success, or an error dict
        if the text was not found.
    """
    p = Path(path).resolve()

    try:
        original = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"error": f"File not found: {path}", "error_type": "FileNotFoundError"}
    except PermissionError:
        return {"error": f"Permission denied: {path}", "error_type": "PermissionError"}

    count = original.count(search)
    if count == 0:
        return {"error": f"Search text not found in {path}", "error_type": "NotFound"}

    updated = original.replace(search, replace)
    p.write_text(updated, encoding="utf-8")

    # Build a preview: 3 lines of context around the first replacement.
    lines = updated.splitlines()
    preview_lines: list[str] = []
    for idx, line in enumerate(lines):
        if replace in line:
            start = max(0, idx - 1)
            end = min(len(lines), idx + 2)
            preview_lines = [
                f"{i + 1}: {lines[i]}" for i in range(start, end)
            ]
            break

    return {
        "path": str(p),
        "replacements": count,
        "preview": "\n".join(preview_lines),
    }


# ---------------------------------------------------------------------------
# 4. list_dir
# ---------------------------------------------------------------------------


async def list_dir(
    path: str,
    recursive: bool = False,
    max_depth: int = 3,
) -> dict[str, Any]:
    """List directory contents.

    Excludes ``.git``, ``__pycache__``, ``node_modules``, ``venv``, and
    ``.venv`` by default.

    Args:
        path: Path to the directory.
        recursive: If ``True``, recurse into subdirectories.
        max_depth: Maximum recursion depth (only used when *recursive* is True).

    Returns:
        ``{"path", "entries": [{"name", "type", "size_bytes"}], "total"}``.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        return {"error": f"Not a directory: {path}", "error_type": "NotADirectory"}

    entries: list[dict[str, Any]] = []

    def _scan(directory: Path, depth: int) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return

        for child in children:
            if child.name in _EXCLUDED_DIRS:
                continue

            rel = str(child.relative_to(root))
            if child.is_dir():
                entries.append({"name": rel, "type": "dir", "size_bytes": 0})
                if recursive and depth < max_depth:
                    _scan(child, depth + 1)
            elif child.is_file():
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                entries.append({"name": rel, "type": "file", "size_bytes": size})

    _scan(root, 1)

    return {
        "path": str(root),
        "entries": entries,
        "total": len(entries),
    }


# ---------------------------------------------------------------------------
# 5. find_files
# ---------------------------------------------------------------------------


async def find_files(
    path: str,
    pattern: str,
) -> dict[str, Any]:
    """Find files matching a glob pattern.

    Args:
        path: Root directory to search from.
        pattern: Glob pattern (e.g. ``"*.py"``, ``"**/test_*.py"``).

    Returns:
        ``{"pattern", "matches": [str], "count"}``.  Capped at 100 results.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        return {"error": f"Not a directory: {path}", "error_type": "NotADirectory"}

    matches: list[str] = []
    for match in root.glob(pattern):
        if any(part in _EXCLUDED_DIRS for part in match.parts):
            continue
        if match.is_file():
            matches.append(str(match.relative_to(root)))
            if len(matches) >= _MAX_FIND_RESULTS:
                break

    return {
        "pattern": pattern,
        "matches": sorted(matches),
        "count": len(matches),
    }


# ---------------------------------------------------------------------------
# 6. search_content
# ---------------------------------------------------------------------------


async def search_content(
    path: str,
    query: str,
    file_pattern: str = "*",
) -> dict[str, Any]:
    """Search for text inside files, like ``grep -rn``.

    Uses ``grep`` as a subprocess for speed.  Falls back to a pure-Python
    scan if ``grep`` is not available.

    Args:
        path: Root directory to search.
        query: Text to search for (literal string, not regex).
        file_pattern: Glob to filter files (e.g. ``"*.py"``).

    Returns:
        ``{"query", "matches": [{"file", "line", "content"}], "count"}``.
        Capped at 50 matches.  Long lines are truncated to 200 characters.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        return {"error": f"Not a directory: {path}", "error_type": "NotADirectory"}

    matches: list[dict[str, Any]] = []

    try:
        # Build grep command.
        cmd = [
            "grep", "-rn", "--include", file_pattern,
            "-F",  # fixed string (not regex)
            "-m", str(_MAX_SEARCH_MATCHES),
            query,
            str(root),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)

        for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
            if len(matches) >= _MAX_SEARCH_MATCHES:
                break
            # grep output: /abs/path/file.py:42:matched line content
            parts = raw_line.split(":", 2)
            if len(parts) < 3:
                continue
            file_path = parts[0]
            try:
                line_no = int(parts[1])
            except ValueError:
                continue
            content = parts[2]
            if len(content) > _MAX_LINE_LENGTH:
                content = content[:_MAX_LINE_LENGTH] + "…"

            # Make paths relative to root.
            try:
                rel = str(Path(file_path).relative_to(root))
            except ValueError:
                rel = file_path

            matches.append({"file": rel, "line": line_no, "content": content})

    except (FileNotFoundError, asyncio.TimeoutError):
        # grep not available or timed out — fall back to Python scan.
        for file_match in root.rglob(file_pattern):
            if len(matches) >= _MAX_SEARCH_MATCHES:
                break
            if any(part in _EXCLUDED_DIRS for part in file_match.parts):
                continue
            if not file_match.is_file():
                continue
            try:
                lines = file_match.read_text(encoding="utf-8", errors="replace").splitlines()
            except (PermissionError, OSError):
                continue
            for line_no, line in enumerate(lines, 1):
                if query in line:
                    content = line if len(line) <= _MAX_LINE_LENGTH else line[:_MAX_LINE_LENGTH] + "…"
                    matches.append({
                        "file": str(file_match.relative_to(root)),
                        "line": line_no,
                        "content": content,
                    })
                    if len(matches) >= _MAX_SEARCH_MATCHES:
                        break

    return {
        "query": query,
        "matches": matches,
        "count": len(matches),
    }


# ---------------------------------------------------------------------------
# 7. file_info
# ---------------------------------------------------------------------------


async def file_info(path: str) -> dict[str, Any]:
    """Get file metadata: size, last modified, type, permissions.

    Args:
        path: Path to the file or directory.

    Returns:
        ``{"path", "size_bytes", "modified", "type", "permissions", "owner"}``.
    """
    p = Path(path).resolve()
    if not p.exists():
        return {"error": f"Path does not exist: {path}", "error_type": "FileNotFoundError"}

    try:
        st = p.stat()
    except PermissionError:
        return {"error": f"Permission denied: {path}", "error_type": "PermissionError"}

    file_type = "file" if p.is_file() else "dir" if p.is_dir() else "other"
    perms = stat.filemode(st.st_mode)
    modified = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()

    try:
        owner = p.owner()
    except (OSError, KeyError):
        owner = str(st.st_uid)

    return {
        "path": str(p),
        "size_bytes": st.st_size,
        "modified": modified,
        "type": file_type,
        "permissions": perms,
        "owner": owner,
    }


# ---------------------------------------------------------------------------
# 8. tree
# ---------------------------------------------------------------------------


async def tree(
    path: str,
    max_depth: int = 3,
) -> dict[str, Any]:
    """Show directory structure as a text tree.

    Excludes ``.git``, ``node_modules``, ``__pycache__``, ``venv``, and
    ``.venv``.

    Args:
        path: Root directory.
        max_depth: Maximum depth to recurse.

    Returns:
        ``{"path", "tree"}`` where ``tree`` is a multi-line string.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        return {"error": f"Not a directory: {path}", "error_type": "NotADirectory"}

    lines: list[str] = [f"{root.name}/"]

    def _build(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(
                [c for c in directory.iterdir() if c.name not in _EXCLUDED_DIRS],
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            return

        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{prefix}{connector}{child.name}{suffix}")

            if child.is_dir():
                extension = "    " if is_last else "│   "
                _build(child, prefix + extension, depth + 1)

    _build(root, "", 1)

    return {
        "path": str(root),
        "tree": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# 9. diff_files
# ---------------------------------------------------------------------------


async def diff_files(
    file_a: str,
    file_b: str,
) -> dict[str, Any]:
    """Show unified diff between two files.

    Args:
        file_a: Path to the first file.
        file_b: Path to the second file.

    Returns:
        ``{"file_a", "file_b", "diff", "changed"}``.
    """
    pa = Path(file_a).resolve()
    pb = Path(file_b).resolve()

    for p in (pa, pb):
        if not p.is_file():
            return {"error": f"File not found: {p}", "error_type": "FileNotFoundError"}

    try:
        lines_a = pa.read_text(encoding="utf-8").splitlines(keepends=True)
        lines_b = pb.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        return {"error": f"Cannot read binary file: {exc}", "error_type": "UnicodeDecodeError"}

    diff = "".join(difflib.unified_diff(
        lines_a, lines_b,
        fromfile=str(pa),
        tofile=str(pb),
    ))

    return {
        "file_a": str(pa),
        "file_b": str(pb),
        "diff": diff,
        "changed": bool(diff),
    }


# ---------------------------------------------------------------------------
# 10. tail
# ---------------------------------------------------------------------------


async def tail(
    path: str,
    lines: int = 50,
) -> dict[str, Any]:
    """Read the last N lines of a file.  Useful for log files.

    Args:
        path: Path to the file.
        lines: Number of lines from the end.

    Returns:
        ``{"path", "content", "lines_returned"}``.
    """
    p = Path(path).resolve()

    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"error": f"File not found: {path}", "error_type": "FileNotFoundError"}
    except PermissionError:
        return {"error": f"Permission denied: {path}", "error_type": "PermissionError"}
    except UnicodeDecodeError:
        return {"error": f"Cannot read binary file: {path}", "error_type": "UnicodeDecodeError"}

    all_lines = text.splitlines(keepends=True)
    selected = all_lines[-lines:] if lines < len(all_lines) else all_lines

    return {
        "path": str(p),
        "content": "".join(selected),
        "lines_returned": len(selected),
    }


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_fs_tools(registry: ToolRegistry) -> None:
    """Register all filesystem tools with the given :class:`ToolRegistry`.

    Args:
        registry: The central tool registry to populate.
    """
    registry.register(
        namespace="fs",
        name="read_file",
        fn=read_file,
        description=(
            "Read the contents of a file, optionally a specific line range. "
            "If the file is longer than 500 lines and no range is given, only "
            "the first 500 lines are returned. Use start_line/end_line to page "
            "through large files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path to the file"},
                "start_line": {"type": "integer", "description": "Start line (1-indexed, inclusive). Default: 1"},
                "end_line": {"type": "integer", "description": "End line (inclusive). Default: end of file"},
            },
            "required": ["path"],
        },
    )

    registry.register(
        namespace="fs",
        name="write_file",
        fn=write_file,
        description=(
            "Write content to a file. Overwrites existing content. "
            "Set create_dirs=true to create missing parent directories."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write to"},
                "content": {"type": "string", "description": "Text content to write"},
                "create_dirs": {"type": "boolean", "description": "Create parent directories if missing. Default: false"},
            },
            "required": ["path", "content"],
        },
    )

    registry.register(
        namespace="fs",
        name="edit_file",
        fn=edit_file,
        description=(
            "Find exact text in a file and replace it. Returns an error if "
            "the search text is not found. All occurrences are replaced. "
            "Returns a preview of the change with surrounding context."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "search": {"type": "string", "description": "Exact text to find (must match exactly)"},
                "replace": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "search", "replace"],
        },
    )

    registry.register(
        namespace="fs",
        name="list_dir",
        fn=list_dir,
        description=(
            "List directory contents. Each entry shows name, type (file/dir), "
            "and size in bytes. Excludes .git, __pycache__, node_modules, "
            "venv, .venv by default."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the directory"},
                "recursive": {"type": "boolean", "description": "Recurse into subdirectories. Default: false"},
                "max_depth": {"type": "integer", "description": "Maximum recursion depth. Default: 3"},
            },
            "required": ["path"],
        },
    )

    registry.register(
        namespace="fs",
        name="find_files",
        fn=find_files,
        description=(
            "Find files matching a glob pattern (e.g. '*.py', '**/test_*.py'). "
            "Returns up to 100 matching file paths relative to the search root."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Root directory to search from"},
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py' or '**/test_*.py'"},
            },
            "required": ["path", "pattern"],
        },
    )

    registry.register(
        namespace="fs",
        name="search_content",
        fn=search_content,
        description=(
            "Search for text inside files, like grep. Returns matching lines "
            "with file paths and line numbers. Capped at 50 matches. Use "
            "file_pattern to restrict to specific file types (e.g. '*.py')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Root directory to search"},
                "query": {"type": "string", "description": "Text to search for (literal match, not regex)"},
                "file_pattern": {"type": "string", "description": "Glob to filter files, e.g. '*.py'. Default: '*'"},
            },
            "required": ["path", "query"],
        },
    )

    registry.register(
        namespace="fs",
        name="file_info",
        fn=file_info,
        description=(
            "Get file metadata: size in bytes, last modified timestamp, "
            "type (file/dir), permissions, and owner."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file or directory"},
            },
            "required": ["path"],
        },
    )

    registry.register(
        namespace="fs",
        name="tree",
        fn=tree,
        description=(
            "Show directory structure as a text tree (like the Unix 'tree' command). "
            "Excludes .git, node_modules, __pycache__, venv, .venv. "
            "Useful for understanding project layout."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Root directory"},
                "max_depth": {"type": "integer", "description": "Maximum depth to recurse. Default: 3"},
            },
            "required": ["path"],
        },
    )

    registry.register(
        namespace="fs",
        name="diff_files",
        fn=diff_files,
        description=(
            "Show unified diff between two files. Returns the diff as text "
            "and a boolean indicating whether the files differ."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_a": {"type": "string", "description": "Path to the first file"},
                "file_b": {"type": "string", "description": "Path to the second file"},
            },
            "required": ["file_a", "file_b"],
        },
    )

    registry.register(
        namespace="fs",
        name="tail",
        fn=tail,
        description=(
            "Read the last N lines of a file. Useful for reading log files "
            "or checking recent output without loading the entire file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "lines": {"type": "integer", "description": "Number of lines from the end. Default: 50"},
            },
            "required": ["path"],
        },
    )
