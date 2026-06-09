# ProbeAgent

**An autonomous AI agent that diagnoses, remediates, and reports on software infrastructure.**

Give ProbeAgent a task in plain English — it figures out what tools to use, investigates the problem, and either fixes it or tells you what's wrong. No runbooks, no playbooks, no manual steps.

```
$ probe-agent run --project ./my-app "The API is returning 500 errors. Find out why."

🔍 ProbeAgent v0.1.0  •  gemini-2.5-flash  •  58 tools

Step 1: project_discover        ✅ (42ms)
Step 2: docker_ps               ✅ (180ms)
Step 3: docker_logs ekyc-app    ✅ (95ms)
Step 4: observe_health_check    ✅ (210ms)
Step 5: agent_spawn_diagnostic  ✅ (4.2s)

─────────────────────────────────
Root cause: The ekyc-app container is OOM-killed (exit code 137).
Memory limit is 256M but the app uses ~380M under load.
Recommendation: Increase mem_limit to 512M in docker-compose.yml.
─────────────────────────────────

✅ Done in 5 steps (18,432 tokens)
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tools (58 across 7 namespaces)](#tools)
- [Subagent System](#subagent-system)
- [Example Tasks](#example-tasks)
- [Evaluation](#evaluation)
- [Session Recording](#session-recording)
- [Configuration](#configuration)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Quick Start

### Prerequisites

- Python 3.12+
- An API key for at least one LLM provider (see below)
- Docker (optional — needed for container tools)

### Install

```bash
git clone https://github.com/itselcid/probe-agent.git
cd probe-agent
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

### Run with Gemini (free)

```bash
# Get a free key at https://aistudio.google.com/apikey
export LLM_PROVIDER=gemini
export LLM_API_KEY=your-gemini-key

probe-agent run --project /path/to/your/project "What is this project?"
```

### Run with OpenRouter (free models available)

```bash
# Get a free key at https://openrouter.ai/keys
export LLM_PROVIDER=openai
export LLM_API_KEY=your-openrouter-key
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export LLM_MODEL=google/gemini-2.0-flash-exp:free

probe-agent run --project /path/to/your/project "What is this project?"
```

### Run with OpenAI

```bash
export LLM_PROVIDER=openai
export LLM_API_KEY=sk-your-openai-key
export LLM_MODEL=gpt-4o

probe-agent run --project /path/to/your/project "What is this project?"
```

### Run with Anthropic Claude

```bash
export LLM_PROVIDER=anthropic
export LLM_API_KEY=sk-ant-your-anthropic-key
export LLM_MODEL=claude-sonnet-4-20250514

probe-agent run --project /path/to/your/project "What is this project?"
```

---

## How It Works

ProbeAgent operates in a **tool-calling loop**. You give it a task, and it autonomously decides which tools to call until the task is complete:

```
User: "Check if all services are healthy"
  │
  ▼
┌─────────────────────────────────────────────┐
│  1. Agent sends task + 58 tool schemas      │
│     to the LLM (Gemini)                     │
│                                             │
│  2. LLM decides: "I need to call docker_ps" │
│                                             │
│  3. Agent executes docker_ps, sends result  │
│     back to LLM                             │
│                                             │
│  4. LLM decides: "Now I need docker_logs"   │
│                                             │
│  5. Repeat until LLM has enough info        │
│                                             │
│  6. LLM produces a final text answer        │
└─────────────────────────────────────────────┘
```

The agent **never uses hardcoded logic** — the LLM reads the tool descriptions and decides dynamically. This means it can handle tasks it's never seen before, as long as the tools exist.

### Key Capabilities

| Capability | Description |
|-----------|-------------|
| **Investigate** | Read files, parse logs, check health endpoints, inspect containers |
| **Diagnose** | Correlate evidence across multiple sources to find root cause |
| **Remediate** | Edit config files, restart containers, run commands |
| **Report** | Generate structured Markdown reports with findings and recommendations |
| **Delegate** | Spawn isolated subagents for complex multi-phase investigations |

---

## Architecture

```
probe-agent run --project ./my-app "task description"
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  main.py (CLI)                                          │
│  ├── Settings ← env vars / .env                         │
│  ├── create_llm_provider() ← Gemini / OpenAI            │
│  ├── ToolRegistry ← 58 tools across 7 namespaces        │
│  └── ProbeAgent.run(task)                               │
│        │                                                │
│        ├── ContextManager.get_messages()                │
│        │   └── sliding window (20 msgs) + rolling       │
│        │       summary for long conversations           │
│        │                                                │
│        ├── LLMProvider.chat(messages, tool_schemas)     │
│        │   └── provider-agnostic interface              │
│        │                                                │
│        ├── if tool_calls → ToolRegistry.execute()       │
│        │   └── loop back ↑                              │
│        │                                                │
│        ├── if text → return final answer                │
│        │                                                │
│        └── SessionRecorder.save()                       │
│            └── .probe/sessions/{uuid}.json              │
└─────────────────────────────────────────────────────────┘
```

### Design Principles

- **Model-driven**: The LLM decides which tools to call — no if/elif routing
- **Provider-agnostic**: Switch between Gemini, OpenAI, or Anthropic with one env var
- **Context windowing**: Keeps conversations coherent over 30+ steps by summarising old messages
- **Graceful failures**: Every tool catches exceptions and returns structured errors — the agent never crashes

---

## Tools

58 tools across 7 namespaces. The LLM sees all tool schemas and picks the right ones.

### Filesystem (`fs_*`) — 10 tools

| Tool | Description |
|------|-------------|
| `fs_read_file` | Read file contents with line range support |
| `fs_write_file` | Write or overwrite a file |
| `fs_edit_file` | Apply targeted edits to specific lines |
| `fs_list_dir` | List directory contents with metadata |
| `fs_tree` | Recursive directory tree (depth-limited) |
| `fs_search_content` | Grep-like content search with context |
| `fs_find_files` | Find files by name pattern |
| `fs_file_info` | File metadata (size, permissions, timestamps) |
| `fs_mkdir` | Create directories |
| `fs_delete` | Delete files or directories |

### Git (`git_*`) — 10 tools

| Tool | Description |
|------|-------------|
| `git_status` | Working tree status (staged, modified, untracked) |
| `git_log` | Commit history with diffs |
| `git_diff` | Show changes between commits or working tree |
| `git_show` | Show a specific commit's details |
| `git_blame` | Line-by-line authorship |
| `git_branch` | List or create branches |
| `git_stash` | Stash/unstash changes |
| `git_commit` | Stage and commit changes |
| `git_checkout` | Switch branches or restore files |
| `git_remote` | Remote repository information |

### Docker (`docker_*`) — 12 tools

| Tool | Description |
|------|-------------|
| `docker_ps` | List containers with health and resource info |
| `docker_logs` | Container log retrieval with filtering |
| `docker_inspect` | Detailed container configuration |
| `docker_stats` | Live CPU/memory/network statistics |
| `docker_restart` | Restart a container |
| `docker_stop` | Stop a running container |
| `docker_start` | Start a stopped container |
| `docker_exec_command` | Execute a command inside a container |
| `docker_images` | List available images |
| `docker_networks` | List Docker networks |
| `docker_volumes` | List Docker volumes |
| `docker_compose_status` | Docker Compose service status |

### Shell (`shell_*`) — 8 tools

| Tool | Description |
|------|-------------|
| `shell_run` | Execute shell commands with timeout |
| `shell_curl` | HTTP requests with headers and body |
| `shell_env` | Read environment variables |
| `shell_which` | Find executables in PATH |
| `shell_disk_usage` | Disk space analysis |
| `shell_system_info` | OS, CPU, memory, architecture |
| `shell_ports` | List listening ports |
| `shell_processes` | List running processes |

### Observability (`observe_*`) — 9 tools

| Tool | Description |
|------|-------------|
| `observe_health_check` | Hit a health endpoint, check response |
| `observe_check_endpoints` | Probe multiple endpoints (/health, /ready, /metrics) |
| `observe_parse_log_file` | Parse structured logs with level/time filters |
| `observe_search_logs` | Grep logs with context lines |
| `observe_log_stats` | Count by log level, top error messages |
| `observe_check_resource_usage` | CPU, memory, disk percentages |
| `observe_trace_request` | HTTP timing breakdown (DNS/connect/TLS/total) |
| `observe_check_dns` | DNS resolution check |
| `observe_check_ssl` | SSL certificate validation and expiry |

### Project Intelligence (`project_*`) — 6 tools

| Tool | Description |
|------|-------------|
| `project_discover` | Scan a directory → languages, frameworks, Docker services, config |
| `project_dependency_check` | Parse dependency files → version lists |
| `project_test_runner` | Auto-detect and run tests (pytest/npm test) |
| `project_lint_check` | Auto-detect and run linters (ruff/eslint) |
| `project_config_audit` | Check for secrets, debug mode, missing healthchecks → score 0-100 |
| `project_service_map` | Parse docker-compose → service dependency graph |

### Agent (`agent_*`) — 3 tools

| Tool | Description |
|------|-------------|
| `agent_spawn_diagnostic` | Spawn a read-only diagnostic subagent |
| `agent_spawn_remediation` | Spawn a subagent that can modify files and restart services |
| `agent_spawn_report` | Spawn a read-only report generator subagent |

---

## Subagent System

For complex tasks, the parent agent can **delegate work to specialised subagents**. Each subagent is fully isolated:

- **Own conversation history** — starts from scratch, parent context not shared
- **Scoped tool access** — only sees the tools it needs
- **Bounded steps** — max 15 steps, can't run forever

```
Parent Agent (58 tools, full context)
  │
  ├── agent_spawn_diagnostic
  │   └── SubagentRunner("diagnostic")
  │       ├── 11 READ-ONLY tools (fs_read, docker_logs, etc.)
  │       ├── Cannot modify files or restart services
  │       └── Returns: structured findings
  │
  ├── agent_spawn_remediation
  │   └── SubagentRunner("remediation")
  │       ├── 7 tools INCLUDING write/restart/commit
  │       ├── CAN modify files, restart containers, run commands
  │       └── Returns: actions taken + verification
  │
  └── agent_spawn_report
      └── SubagentRunner("report")
          ├── 8 READ-ONLY tools (stats, service map, etc.)
          ├── Cannot modify anything
          └── Returns: structured Markdown report
```

### Why Isolation Matters

The diagnostic subagent **cannot accidentally** modify production files. The remediation subagent **cannot** access the full tool set. This separation of concerns is enforced at the registry level — not by prompting alone.

---

## Example Tasks

### Simple — Single Namespace

```bash
# Understand a project
probe-agent run --project ./my-app "What is this project? What languages and frameworks does it use?"

# Check git status
probe-agent run --project ./my-app "Show me the last 10 commits and any uncommitted changes"

# System info
probe-agent run --project ./my-app "Check disk space and system resources"
```

### Medium — Cross-Namespace

```bash
# Container investigation
probe-agent run --project ./my-app "List all running containers and check their logs for errors"

# Security audit
probe-agent run --project ./my-app "Check for hardcoded secrets, debug mode, and missing .gitignore entries"

# Service health
probe-agent run --project ./my-app "Check if all services have working health endpoints"
```

### Complex — Subagent Delegation

```bash
# Full diagnostic workflow
probe-agent run --project ./my-app "The ekyc-app keeps crashing. Investigate and fix it."

# End-to-end analysis
probe-agent run --project ./my-app "Discover what this project does, check all services, investigate any issues, and generate a status report."
```

---

## Evaluation

Built-in evaluation harness with 7 scenarios to measure agent quality:

```bash
# Run all scenarios
probe-agent eval --project ./my-app

# Run specific scenarios
probe-agent eval --project ./my-app -s container_health -s project_analysis
```

| Scenario | Complexity | Expected Tools | Max Steps |
|----------|-----------|----------------|-----------|
| `container_health` | Simple | docker_ps, docker_stats | 8 |
| `project_analysis` | Simple | project_discover, fs_tree | 12 |
| `log_investigation` | Medium | docker_ps, docker_logs | 15 |
| `git_health` | Medium | git_status, git_log | 8 |
| `service_connectivity` | Medium | observe_health_check, observe_check_endpoints | 12 |
| `diagnostic_subagent` | Complex | agent_spawn_diagnostic | 20 |
| `full_workflow` | Complex | project_discover, agent_spawn_diagnostic, agent_spawn_report | 30 |

**Scoring**: `tool_hit_rate = |expected ∩ used| / |expected|` — a perfect score means the agent used all the tools we expected.

Results saved to `{project}/.probe/evals/eval_{timestamp}.json`.

---

## Session Recording

Every agent run is recorded to JSON for debugging and replay:

```
{project}/.probe/sessions/{session-id}.json
```

Each session file contains:

```json
{
  "session_id": "a1b2c3d4-...",
  "start_time": "2026-06-09T17:15:00+00:00",
  "end_time": "2026-06-09T17:15:12+00:00",
  "total_duration_s": 12.34,
  "total_steps": 5,
  "tools_used": ["docker_ps", "fs_read_file"],
  "steps": [
    {"type": "llm_call", "step": 1, "usage": {"total_tokens": 5991}},
    {"type": "tool_call", "tool": "docker_ps", "success": true, "duration_ms": 180},
    {"type": "subagent", "subagent": "diagnostic", "steps": 4, "success": true}
  ]
}
```

---

## Configuration

All settings via environment variables (powered by `pydantic-settings`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | LLM backend (`gemini`, `openai`, `anthropic`) |
| `LLM_API_KEY` | *(required)* | API key for the selected provider |
| `LLM_MODEL` | provider default | Model override (e.g. `gemini-2.0-flash`) |
| `PROJECT_PATH` | `.` | Path to the target project |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_STEPS` | `30` | Max tool calls before the agent stops |

Or use a `.env` file:

```env
LLM_PROVIDER=gemini
LLM_API_KEY=your-key-here
LLM_MODEL=gemini-2.5-flash
LOG_LEVEL=INFO
MAX_STEPS=30
```

---

## Testing

```bash
source .venv/bin/activate
pytest
```

**301 tests**, all passing. Tests cover:

- Tool registry operations and schema generation
- All 58 tools (mocked external dependencies)
- Agent loop (mocked LLM responses)
- Context manager windowing and summarisation
- Subagent isolation and scoped registries
- Session recording and JSON persistence
- Evaluation harness scenarios and metrics

---

## Project Structure

```
probe-agent/
├── pyproject.toml              # Package config, dependencies
├── Dockerfile                  # Container deployment
├── README.md                   # This file
├── MEMO.md                     # Architecture reflection
├── src/
│   └── probe_agent/
│       ├── __init__.py
│       ├── main.py             # CLI entry point (click)
│       ├── agent.py            # Core agent loop
│       ├── config.py           # pydantic-settings config
│       ├── context.py          # Sliding window + rolling summary
│       ├── errors.py           # Typed error hierarchy
│       ├── types.py            # Pydantic data models
│       ├── registry.py         # Tool registry + schema generation
│       ├── llm_client.py       # Abstract LLM provider interface
│       ├── subagent.py         # Isolated subagent runner
│       ├── session.py          # Session recording
│       ├── logging_setup.py    # structlog JSON logging
│       ├── retry.py            # Exponential backoff decorator
│       ├── rate_limiter.py     # Token bucket rate limiter
│       ├── providers/
│       │   ├── gemini.py       # Google Gemini provider
│       │   ├── openai_provider.py
│       │   └── anthropic_provider.py
│       ├── tools/
│       │   ├── fs.py           # 10 filesystem tools
│       │   ├── git.py          # 10 git tools
│       │   ├── docker_tools.py # 12 Docker tools
│       │   ├── shell.py        # 8 shell tools
│       │   ├── observe.py      # 9 observability tools
│       │   ├── project.py      # 6 project intelligence tools
│       │   └── agent_tools.py  # 3 subagent spawners
│       └── eval/
│           ├── scenarios.py    # 7 evaluation scenarios
│           ├── harness.py      # Evaluation runner
│           └── metrics.py      # Scoring and results
└── tests/
    ├── test_registry.py
    ├── test_llm.py
    ├── test_agent.py
    ├── test_fs.py
    ├── test_git.py
    ├── test_docker.py
    ├── test_shell.py
    ├── test_observe.py
    ├── test_project.py
    ├── test_subagent.py
    ├── test_session.py
    └── test_eval.py
```

---

## License

MIT
