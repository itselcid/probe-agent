"""LLM provider implementations.

Each sub-module implements :class:`~probe_agent.llm_client.LLMProvider` for a
specific vendor API.  The agent loop never imports these directly — it uses
:func:`~probe_agent.llm_client.create_llm_provider` instead.

Available providers:

- :mod:`.gemini` — Google Gemini (default, free tier available)
- :mod:`.openai_provider` — OpenAI / OpenRouter (stub)
- :mod:`.anthropic_provider` — Anthropic Claude (stub)
"""
