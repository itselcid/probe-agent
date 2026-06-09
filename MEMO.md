# MEMO — ProbeAgent

## What I Built

ProbeAgent is an autonomous DevOps/SRE agent: you give it a natural-language task like "why is my container crashing?" and it investigates the problem by calling tools, reading the results, and deciding what to do next — without human intervention. It has 58 tools across 7 namespaces, 3 specialised subagents, and a context manager that keeps it coherent over 30+ step investigations.

**Stats:** ~8,400 lines of source, ~5,400 lines of tests, 301 tests passing, 3 LLM providers supported.

---

## Architecture Decisions

### Plugin Registry Over Hardcoded Routing

The ToolRegistry is a flat dictionary mapping tool names (e.g. `docker_ps`) to async functions. When the LLM says "call docker_ps", the registry looks it up and runs it. The alternative — a giant if/elif chain — would be unmaintainable at 58 tools and violates the brief's requirement for model-driven selection. The registry also generates JSON schemas that the LLM reads to understand each tool's parameters, and supports `subset()` for creating scoped registries for subagents.

### Context Windowing With Rolling Summary

After 20+ tool calls, the conversation history becomes enormous. The LLM loses focus and may exceed its context window. The ContextManager solves this with a sliding window: it keeps the last 20 messages in full detail and periodically asks the LLM to compress older messages into a one-paragraph summary. This lets the agent maintain "memory" over long investigations without losing coherence.

### Isolated Subagents With Scoped Tools

A subagent gets its own fresh conversation history and a restricted set of tools. The diagnostic subagent can only read — it cannot modify files or restart containers. The remediation subagent can write, but only to specific tools. This isolation is enforced at the registry level via `subset()`, which creates a new ToolRegistry containing only the named tools. The parent's conversation and registry are never modified by subagent execution.

### Provider-Agnostic LLM Interface

The `LLMProvider` abstract class defines a single `chat()` method. Gemini, OpenAI, and Anthropic each have their own concrete implementation that handles message format conversion internally. Switching providers is one environment variable: `LLM_PROVIDER=openai`. This was a deliberate choice over coupling to Google's SDK directly — it keeps the agent portable.

---

## What I'd Do Differently

1. **OpenTelemetry tracing** — I added structured logging and session recording, but proper distributed traces with spans per tool call would make debugging multi-step investigations much easier. The dependency is already in `pyproject.toml` but unused.

2. **Smarter retry on rate limits** — The current retry decorator uses exponential backoff, but the Gemini free tier has a hard 5-requests-per-minute limit. A token bucket that respects the provider's `retry_delay` header would be more effective than blind retries.

3. **Tool result caching** — `project_discover` scans the same directory every time it's called. Caching expensive, idempotent tool results for the duration of a session would reduce both latency and token usage.

---

## AI Collaboration

I used AI throughout the entire build. Here is exactly how:

**Planning phase:** I used an AI assistant (Antigravity) to design the full architecture — the 5-day plan, the file structure, the tool specifications, and the implementation prompts. Every prompt was a detailed specification: function signatures, return types, error handling patterns, and test requirements.

**Implementation phase:** I fed those prompts into separate AI coding sessions. Each session produced one component: the registry, the filesystem tools, the git tools, etc. The AI generated the implementations and tests. I reviewed the output, verified test counts, and committed.

**Where I drove the decisions:**

- **Rejected LangGraph three times.** The AI suggested using it for the agent loop. I chose to build from scratch because the brief values original design and the loop is only ~50 lines.
- **Chose local observe tools over Prometheus/Elasticsearch wrappers.** The AI's original plan had tools that query Prometheus and ELK. I switched to tools that parse local log files, check SSL certs, and trace HTTP requests — things evaluators can actually run without infrastructure.
- **Refined the evaluation scenarios.** The AI generated 9 scenarios. I trimmed it to 7, adjusted timeouts, and reorganised from simple → complex.
- **Tested against a real project.** I ran the agent against my own eKYC platform (bio-verify) and verified it correctly identified the tech stack, services, and Docker networks.


**My approach:** AI was my implementation engine — but building a working agent required more than generated code. I brought the domain knowledge (what tools a DevOps agent actually needs), made architectural decisions (rejecting LangGraph for a custom loop, choosing local observability over Prometheus wrappers, designing subagent isolation boundaries), managed the build across 4 days in a structured sequence, and validated the final system against my own production eKYC platform. The AI could generate tools and tests, but it couldn't decide *which* tools matter, *why* subagents need isolation, or *whether* the agent's output was actually correct. That judgement was mine.
