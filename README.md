# ProbeAgent

**Autonomous DevOps/SRE agent powered by Gemini.**

ProbeAgent is an AI-driven agent that observes, diagnoses, and remediates
issues across containerised infrastructure.  It uses Google's Gemini models
for reasoning and a pluggable tool system for interacting with Docker,
Kubernetes, logs, metrics, and more.

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Set your Gemini API key
export GOOGLE_API_KEY="your-key-here"

# Run the agent
probe-agent --project /path/to/your/project "check container health"
```

## Configuration

All settings are loaded from environment variables (powered by
`pydantic-settings`):

| Variable          | Default              | Description                         |
| ----------------- | -------------------- | ----------------------------------- |
| `GOOGLE_API_KEY`  | *(required)*         | Gemini API key                      |
| `PROJECT_PATH`    | `.`                  | Path to the target project          |
| `LOG_LEVEL`       | `INFO`               | Logging level (DEBUG, INFO, …)      |
| `MAX_STEPS`       | `30`                 | Max tool calls before the agent stops |
| `MODEL_NAME`      | `gemini-2.5-flash`   | Gemini model to use                 |

## Project Structure

```
probe-agent/
├── pyproject.toml
├── Dockerfile
├── README.md
├── src/
│   └── probe_agent/
│       ├── __init__.py
│       ├── main.py          # Click CLI entry-point
│       ├── config.py         # pydantic-settings configuration
│       ├── errors.py         # Typed error hierarchy
│       ├── types.py          # Pydantic data models
│       ├── logging_setup.py  # structlog JSON logging
│       └── tools/
│           └── __init__.py   # Tool registry (future)
└── tests/
    ├── conftest.py
    └── __init__.py
```

## Development

```bash
# Run tests
pytest

# Type-check
mypy src/

# Lint
ruff check src/ tests/
```

## Docker

```bash
docker build -t probe-agent .
docker run --rm \
  -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  probe-agent --project /app "diagnose failing containers"
```

## License

MIT
