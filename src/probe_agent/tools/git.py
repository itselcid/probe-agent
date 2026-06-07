"""Git tools for ProbeAgent.

Ten tools that let the AI agent inspect and modify a git repository.
All git operations use ``asyncio.create_subprocess_exec`` to call the
git CLI — no Python git library required.

Every function takes ``repo_path`` as its first parameter (the path to
the git repository root).

Register all tools at once with :func:`register_git_tools`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from probe_agent.registry import ToolRegistry

_TIMEOUT = 30  # seconds for every git subprocess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_git(
    repo_path: str,
    *args: str,
    timeout: int = _TIMEOUT,
) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr).

    Args:
        repo_path: Path to the git repository root.
        *args: Arguments to pass after ``git -C <repo_path>``.
        timeout: Seconds before the process is killed.

    Returns:
        Tuple of (return_code, stdout_text, stderr_text).
    """
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", repo_path, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return 1, "", f"git command timed out after {timeout}s"

    return (
        proc.returncode or 0,
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
    )


# ---------------------------------------------------------------------------
# 1. status
# ---------------------------------------------------------------------------


async def status(repo_path: str) -> dict[str, Any]:
    """Get git status: current branch, staged/unstaged/untracked files.

    Args:
        repo_path: Path to the git repository root.

    Returns:
        ``{"branch", "clean", "staged", "unstaged", "untracked"}``.
    """
    rc, out, err = await _run_git(repo_path, "status", "--porcelain", "-b")
    if rc != 0:
        return {"error": err.strip(), "error_type": "GitError"}

    branch = ""
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []

    for line in out.splitlines():
        if line.startswith("## "):
            # ## main...origin/main  or  ## main
            branch_part = line[3:].split("...")[0]
            branch = branch_part.strip()
            continue

        if len(line) < 2:
            continue

        index_status = line[0]
        worktree_status = line[1]
        filename = line[3:].strip()

        if index_status == "?":
            untracked.append(filename)
        else:
            if index_status not in (" ", "?"):
                staged.append(filename)
            if worktree_status not in (" ", "?"):
                unstaged.append(filename)

    return {
        "branch": branch,
        "clean": len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
    }


# ---------------------------------------------------------------------------
# 2. diff
# ---------------------------------------------------------------------------


async def diff(
    repo_path: str,
    target: str = "HEAD",
    staged: bool = False,
) -> dict[str, Any]:
    """Show changes.  Use ``staged=True`` for staged (index) changes.

    Args:
        repo_path: Path to the git repository root.
        target: Diff target ref (e.g. ``"HEAD"``, ``"main"``).
        staged: If ``True``, show staged changes (``--cached``).

    Returns:
        ``{"diff", "files_changed"}``.
    """
    cmd = ["diff"]
    if staged:
        cmd.append("--cached")
    cmd.append(target)

    rc, out, err = await _run_git(repo_path, *cmd)
    if rc != 0:
        return {"error": err.strip(), "error_type": "GitError"}

    # Count files from diff --stat.
    stat_rc, stat_out, _ = await _run_git(
        repo_path, "diff", *(["--cached"] if staged else []), target, "--stat",
    )
    files_changed = 0
    if stat_rc == 0 and stat_out.strip():
        # Last line: "N files changed, ..."
        last = stat_out.strip().splitlines()[-1]
        for token in last.split():
            if token.isdigit():
                files_changed = int(token)
                break

    return {
        "diff": out,
        "files_changed": files_changed,
    }


# ---------------------------------------------------------------------------
# 3. log
# ---------------------------------------------------------------------------


async def log(
    repo_path: str,
    max_count: int = 20,
    since: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Show recent commits.

    Args:
        repo_path: Path to the git repository root.
        max_count: Maximum number of commits to return.
        since: Only show commits after this date (e.g. ``"2024-01-01"``).
        path: Only show commits touching this file/directory.

    Returns:
        ``{"commits": [{"hash", "author", "date", "message"}], "count"}``.
    """
    # Use a unique separator that won't appear in commit messages.
    sep = "---PROBE_SEP---"
    fmt = f"%H{sep}%an{sep}%ai{sep}%s"

    cmd: list[str] = ["log", f"--max-count={max_count}", f"--pretty=format:{fmt}"]
    if since:
        cmd.append(f"--since={since}")
    if path:
        cmd += ["--", path]

    rc, out, err = await _run_git(repo_path, *cmd)
    if rc != 0:
        return {"error": err.strip(), "error_type": "GitError"}

    commits: list[dict[str, str]] = []
    for line in out.strip().splitlines():
        parts = line.split(sep)
        if len(parts) >= 4:
            commits.append({
                "hash": parts[0],
                "author": parts[1],
                "date": parts[2],
                "message": parts[3],
            })

    return {
        "commits": commits,
        "count": len(commits),
    }


# ---------------------------------------------------------------------------
# 4. blame
# ---------------------------------------------------------------------------


async def blame(
    repo_path: str,
    file_path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict[str, Any]:
    """Show who last modified each line of a file.

    Args:
        repo_path: Path to the git repository root.
        file_path: Path to the file (relative to repo root).
        start_line: First line to show (1-indexed).
        end_line: Last line to show (inclusive).  ``None`` = end of file.

    Returns:
        ``{"file", "lines": [{"line_num", "commit", "author", "content"}]}``.
    """
    cmd = ["blame", "--porcelain"]
    if end_line is not None:
        cmd += [f"-L{start_line},{end_line}"]
    elif start_line > 1:
        cmd += [f"-L{start_line},"]
    cmd.append(file_path)

    rc, out, err = await _run_git(repo_path, *cmd)
    if rc != 0:
        return {"error": err.strip(), "error_type": "GitError"}

    lines: list[dict[str, Any]] = []
    current_commit = ""
    current_author = ""
    current_line_num = 0

    for raw in out.splitlines():
        # Lines starting with a 40-char hex hash start a new blame entry.
        if len(raw) >= 40 and all(c in "0123456789abcdef" for c in raw[:40]):
            parts = raw.split()
            current_commit = parts[0]
            if len(parts) >= 3:
                current_line_num = int(parts[2])
        elif raw.startswith("author "):
            current_author = raw[7:]
        elif raw.startswith("\t"):
            lines.append({
                "line_num": current_line_num,
                "commit": current_commit[:8],
                "author": current_author,
                "content": raw[1:],  # strip leading tab
            })

    return {
        "file": file_path,
        "lines": lines,
    }


# ---------------------------------------------------------------------------
# 5. branch_list
# ---------------------------------------------------------------------------


async def branch_list(repo_path: str) -> dict[str, Any]:
    """List all branches (local and remote).

    Args:
        repo_path: Path to the git repository root.

    Returns:
        ``{"current", "local": [...], "remote": [...]}``.
    """
    rc, out, err = await _run_git(repo_path, "branch", "-a")
    if rc != 0:
        return {"error": err.strip(), "error_type": "GitError"}

    current = ""
    local: list[str] = []
    remote: list[str] = []

    for line in out.splitlines():
        line = line.strip()
        is_current = line.startswith("* ")
        name = line.lstrip("* ").strip()

        if "->" in name:
            continue  # skip HEAD -> origin/main pointers

        if name.startswith("remotes/"):
            remote.append(name.removeprefix("remotes/"))
        else:
            local.append(name)
            if is_current:
                current = name

    return {
        "current": current,
        "local": local,
        "remote": remote,
    }


# ---------------------------------------------------------------------------
# 6. checkout
# ---------------------------------------------------------------------------


async def checkout(
    repo_path: str,
    branch: str,
    create: bool = False,
) -> dict[str, Any]:
    """Switch to a branch.  If ``create=True``, create it first.

    Args:
        repo_path: Path to the git repository root.
        branch: Branch name to switch to.
        create: If ``True``, create a new branch (``git checkout -b``).

    Returns:
        ``{"branch", "created", "previous_branch"}``.
    """
    # Get current branch before switching.
    prev_rc, prev_out, _ = await _run_git(repo_path, "branch", "--show-current")
    previous = prev_out.strip() if prev_rc == 0 else ""

    cmd = ["checkout"]
    if create:
        cmd.append("-b")
    cmd.append(branch)

    rc, out, err = await _run_git(repo_path, *cmd)
    if rc != 0:
        return {"error": err.strip(), "error_type": "GitError"}

    return {
        "branch": branch,
        "created": create,
        "previous_branch": previous,
    }


# ---------------------------------------------------------------------------
# 7. commit
# ---------------------------------------------------------------------------


async def commit(
    repo_path: str,
    message: str,
    files: list[str] | None = None,
) -> dict[str, Any]:
    """Stage files and commit.

    If ``files`` is ``None``, all changed files are staged (``git add -A``).

    Args:
        repo_path: Path to the git repository root.
        message: Commit message.
        files: Specific files to stage, or ``None`` to stage all.

    Returns:
        ``{"hash", "message", "files_committed"}``.
    """
    # Stage.
    if files:
        rc, _, err = await _run_git(repo_path, "add", *files)
    else:
        rc, _, err = await _run_git(repo_path, "add", "-A")

    if rc != 0:
        return {"error": f"git add failed: {err.strip()}", "error_type": "GitError"}

    # Commit.
    rc, out, err = await _run_git(repo_path, "commit", "-m", message)
    if rc != 0:
        return {"error": f"git commit failed: {err.strip()}", "error_type": "GitError"}

    # Get the hash of the new commit.
    hash_rc, hash_out, _ = await _run_git(
        repo_path, "rev-parse", "--short", "HEAD",
    )
    commit_hash = hash_out.strip() if hash_rc == 0 else ""

    # Count files committed.
    stat_rc, stat_out, _ = await _run_git(
        repo_path, "diff", "--name-only", "HEAD~1", "HEAD",
    )
    files_committed = len(stat_out.strip().splitlines()) if stat_rc == 0 else 0

    return {
        "hash": commit_hash,
        "message": message,
        "files_committed": files_committed,
    }


# ---------------------------------------------------------------------------
# 8. stash
# ---------------------------------------------------------------------------


async def stash(
    repo_path: str,
    action: str = "push",
    message: str | None = None,
) -> dict[str, Any]:
    """Stash operations: ``push`` (save), ``pop`` (restore), ``list``.

    Args:
        repo_path: Path to the git repository root.
        action: One of ``"push"``, ``"pop"``, ``"list"``.
        message: Optional message for ``push``.

    Returns:
        ``{"action", "result"}``.
    """
    if action not in ("push", "pop", "list"):
        return {"error": f"Invalid stash action: {action!r}", "error_type": "ValueError"}

    cmd = ["stash", action]
    if action == "push" and message:
        cmd += ["-m", message]

    rc, out, err = await _run_git(repo_path, *cmd)
    if rc != 0:
        return {"error": err.strip(), "error_type": "GitError"}

    return {
        "action": action,
        "result": out.strip() or "Done",
    }


# ---------------------------------------------------------------------------
# 9. show
# ---------------------------------------------------------------------------


async def show(
    repo_path: str,
    commit_ref: str = "HEAD",
) -> dict[str, Any]:
    """Show details of a specific commit.

    Args:
        repo_path: Path to the git repository root.
        commit_ref: Commit hash or ref (default ``"HEAD"``).

    Returns:
        ``{"hash", "author", "date", "message", "diff", "files"}``.
    """
    sep = "---PROBE_SEP---"
    fmt = f"%H{sep}%an{sep}%ai{sep}%s"

    # Get metadata.
    rc, out, err = await _run_git(
        repo_path, "log", "-1", f"--pretty=format:{fmt}", commit_ref,
    )
    if rc != 0:
        return {"error": err.strip(), "error_type": "GitError"}

    parts = out.strip().split(sep)
    if len(parts) < 4:
        return {"error": f"Could not parse commit {commit_ref}", "error_type": "ParseError"}

    commit_hash, author, date, message = parts[0], parts[1], parts[2], parts[3]

    # Get diff (handle root commit which has no parent).
    diff_rc, diff_out, _ = await _run_git(
        repo_path, "diff", f"{commit_ref}~1", commit_ref,
    )
    if diff_rc != 0:
        # Likely root commit — use diff-tree --root instead.
        diff_rc, diff_out, _ = await _run_git(
            repo_path, "diff-tree", "--root", "-p", commit_ref,
        )
    diff_text = diff_out if diff_rc == 0 else ""

    # Get list of changed files (handle root commit).
    files_rc, files_out, _ = await _run_git(
        repo_path, "diff", "--name-only", f"{commit_ref}~1", commit_ref,
    )
    if files_rc != 0:
        # Root commit — use diff-tree --root --name-only.
        files_rc, files_out, _ = await _run_git(
            repo_path, "diff-tree", "--root", "--name-only", "-r", commit_ref,
        )
    files = [f for f in files_out.strip().splitlines() if f and not f.startswith(commit_hash[:8])] if files_rc == 0 else []

    return {
        "hash": commit_hash,
        "author": author,
        "date": date,
        "message": message,
        "diff": diff_text,
        "files": files,
    }


# ---------------------------------------------------------------------------
# 10. remote_info
# ---------------------------------------------------------------------------


async def remote_info(repo_path: str) -> dict[str, Any]:
    """Show remote repository info.

    Args:
        repo_path: Path to the git repository root.

    Returns:
        ``{"remotes": [{"name", "url"}], "default_branch"}``.
    """
    rc, out, err = await _run_git(repo_path, "remote", "-v")
    if rc != 0:
        return {"error": err.strip(), "error_type": "GitError"}

    seen: dict[str, str] = {}
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            url = parts[1]
            if name not in seen:
                seen[name] = url

    remotes = [{"name": n, "url": u} for n, u in seen.items()]

    # Try to detect default branch.
    default_branch = "main"
    ref_rc, ref_out, _ = await _run_git(
        repo_path, "symbolic-ref", "refs/remotes/origin/HEAD",
    )
    if ref_rc == 0 and ref_out.strip():
        # refs/remotes/origin/main → main
        default_branch = ref_out.strip().split("/")[-1]

    return {
        "remotes": remotes,
        "default_branch": default_branch,
    }


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_git_tools(registry: ToolRegistry) -> None:
    """Register all git tools with the given :class:`ToolRegistry`.

    Args:
        registry: The central tool registry to populate.
    """
    registry.register(
        namespace="git",
        name="status",
        fn=status,
        description=(
            "Get git status: current branch, staged files, unstaged changes, "
            "and untracked files. Shows whether the working directory is clean."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the git repository root"},
            },
            "required": ["repo_path"],
        },
    )

    registry.register(
        namespace="git",
        name="diff",
        fn=diff,
        description=(
            "Show changes in the working directory or staging area. "
            "Use staged=true to see staged changes. Target can be a branch "
            "name or commit ref to compare against."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the git repository root"},
                "target": {"type": "string", "description": "Ref to diff against (default: HEAD)"},
                "staged": {"type": "boolean", "description": "Show staged changes instead of unstaged. Default: false"},
            },
            "required": ["repo_path"],
        },
    )

    registry.register(
        namespace="git",
        name="log",
        fn=log,
        description=(
            "Show recent git commits with hash, author, date, and message. "
            "Use max_count to limit results, since to filter by date, and "
            "path to show commits for a specific file or directory."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the git repository root"},
                "max_count": {"type": "integer", "description": "Maximum number of commits. Default: 20"},
                "since": {"type": "string", "description": "Only show commits after this date, e.g. '2024-01-01'"},
                "path": {"type": "string", "description": "Only show commits touching this file/directory"},
            },
            "required": ["repo_path"],
        },
    )

    registry.register(
        namespace="git",
        name="blame",
        fn=blame,
        description=(
            "Show who last modified each line of a file (git blame). "
            "Returns commit hash, author, and content for each line. "
            "Use start_line/end_line to inspect a specific range."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the git repository root"},
                "file_path": {"type": "string", "description": "Path to the file (relative to repo root)"},
                "start_line": {"type": "integer", "description": "First line number (1-indexed). Default: 1"},
                "end_line": {"type": "integer", "description": "Last line number (inclusive). Default: end of file"},
            },
            "required": ["repo_path", "file_path"],
        },
    )

    registry.register(
        namespace="git",
        name="branch_list",
        fn=branch_list,
        description=(
            "List all local and remote branches. Shows which branch is "
            "currently checked out."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the git repository root"},
            },
            "required": ["repo_path"],
        },
    )

    registry.register(
        namespace="git",
        name="checkout",
        fn=checkout,
        description=(
            "Switch to a different branch. Set create=true to create a new "
            "branch from the current HEAD. Returns the previous branch name "
            "so you can switch back."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the git repository root"},
                "branch": {"type": "string", "description": "Branch name to switch to"},
                "create": {"type": "boolean", "description": "Create the branch if it doesn't exist. Default: false"},
            },
            "required": ["repo_path", "branch"],
        },
    )

    registry.register(
        namespace="git",
        name="commit",
        fn=commit,
        description=(
            "Stage files and create a commit. If files is not provided, all "
            "changed files are staged (git add -A). Returns the commit hash "
            "and number of files committed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the git repository root"},
                "message": {"type": "string", "description": "Commit message"},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific files to stage. Omit to stage all changes.",
                },
            },
            "required": ["repo_path", "message"],
        },
    )

    registry.register(
        namespace="git",
        name="stash",
        fn=stash,
        description=(
            "Stash operations: 'push' to save current changes, 'pop' to "
            "restore the most recent stash, 'list' to show all stashes. "
            "Use message to label a stash."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the git repository root"},
                "action": {"type": "string", "enum": ["push", "pop", "list"], "description": "Stash action. Default: push"},
                "message": {"type": "string", "description": "Label for the stash (push only)"},
            },
            "required": ["repo_path"],
        },
    )

    registry.register(
        namespace="git",
        name="show",
        fn=show,
        description=(
            "Show details of a specific commit: hash, author, date, message, "
            "diff, and list of changed files. Defaults to HEAD."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the git repository root"},
                "commit_ref": {"type": "string", "description": "Commit hash or ref. Default: HEAD"},
            },
            "required": ["repo_path"],
        },
    )

    registry.register(
        namespace="git",
        name="remote_info",
        fn=remote_info,
        description=(
            "Show remote repository info: remote names, URLs, and the "
            "default branch. Useful for understanding the upstream configuration."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to the git repository root"},
            },
            "required": ["repo_path"],
        },
    )
