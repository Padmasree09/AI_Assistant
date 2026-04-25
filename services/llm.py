from __future__ import annotations

import json
import logging
import threading
from typing import Generator

import requests

from core.config import get_settings

logger = logging.getLogger("ai_research_assistant.llm")

# ---------------------------------------------------------------------------
# phi-3 mini 4k context budget
# ---------------------------------------------------------------------------
_CONTEXT_LIMIT = 4096


def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~1 token per 4 chars for English text."""
    return len(text) // 4


def _truncate_prompt(prompt: str, max_tokens: int) -> str:
    """Trim the prompt so prompt + max_tokens fits inside the context window.

    Strategy: keep the first section (system instructions / prompt header) and
    the last section (user query / final instructions), cut from the middle.
    """
    prompt_budget = _CONTEXT_LIMIT - max_tokens - 100  # safety margin
    if _estimate_tokens(prompt) <= prompt_budget:
        return prompt

    char_budget = prompt_budget * 4  # inverse of estimate
    if char_budget <= 0:
        return prompt[:200]

    half = char_budget // 2
    return prompt[:half] + "\n...(trimmed)...\n" + prompt[-half:]


# ---------------------------------------------------------------------------
# Circuit breaker: avoid hammering a dead LLM server
# ---------------------------------------------------------------------------
_consecutive_failures = 0
_MAX_FAILURES = 3
_failure_lock = threading.Lock()


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM server has failed repeatedly."""


def _record_failure() -> None:
    global _consecutive_failures
    with _failure_lock:
        _consecutive_failures += 1


def _reset_failures() -> None:
    global _consecutive_failures
    with _failure_lock:
        _consecutive_failures = 0


def _check_circuit() -> None:
    if _consecutive_failures >= _MAX_FAILURES:
        raise LLMUnavailableError(
            f"LLM service unavailable after {_MAX_FAILURES} consecutive failures. "
            "Restart the llama.cpp server and retry."
        )


# ---------------------------------------------------------------------------
# Response cleaner
# ---------------------------------------------------------------------------
_REMOVABLE_PREFIXES = [
    "Answer:",
    "Final Answer:",
    "Final answer:",
    "Response:",
    "Analysis:",
    "Summary:",
]


def _clean_response_text(text: str) -> str:
    cleaned = text.strip()
    for prefix in _REMOVABLE_PREFIXES:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Synchronous LLM call (with token budget + circuit breaker)
# ---------------------------------------------------------------------------
def call_llm(prompt: str, max_tokens: int = 512) -> str:
    _check_circuit()

    settings = get_settings()
    prompt = _truncate_prompt(prompt, max_tokens)

    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.7,
        "stop": ["</s>", "User: ", "Human: "],
    }

    try:
        response = requests.post(
            settings.llama_url, json=payload, timeout=settings.llama_timeout_seconds
        )
        response.raise_for_status()
        _reset_failures()
        return _clean_response_text(response.json()["content"])
    except requests.exceptions.ConnectionError:
        _record_failure()
        logger.error("llm.connection_error", extra={"failures": _consecutive_failures})
        raise
    except requests.exceptions.Timeout:
        _record_failure()
        logger.error("llm.timeout", extra={"failures": _consecutive_failures})
        raise


# ---------------------------------------------------------------------------
# Streaming LLM call: yields tokens as they arrive (for SSE endpoints)
# ---------------------------------------------------------------------------
def call_llm_stream(prompt: str, max_tokens: int = 512) -> Generator[str, None, None]:
    """Stream tokens from llama.cpp using ``"stream": true``."""
    _check_circuit()

    settings = get_settings()
    prompt = _truncate_prompt(prompt, max_tokens)

    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.7,
        "stop": ["</s>", "User: ", "Human: "],
        "stream": True,
    }

    try:
        with requests.post(
            settings.llama_url,
            json=payload,
            stream=True,
            timeout=settings.llama_timeout_seconds,
        ) as resp:
            resp.raise_for_status()
            _reset_failures()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                # llama.cpp streams JSON objects prefixed with "data: "
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    data = json.loads(line)
                    token = data.get("content", "")
                    if token:
                        yield token
                    if data.get("stop", False):
                        return
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.ConnectionError:
        _record_failure()
        logger.error("llm.stream_connection_error", extra={"failures": _consecutive_failures})
        raise
    except requests.exceptions.Timeout:
        _record_failure()
        logger.error("llm.stream_timeout", extra={"failures": _consecutive_failures})
        raise
