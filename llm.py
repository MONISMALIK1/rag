"""OpenRouter LLM wrapper for RAG.

The same proven, dependency-free pattern used across this series of paper
implementations: one ``chat`` call over stdlib ``urllib``. Set
``OPENROUTER_API_KEY`` in your environment before use — the key is read only
from there and never logged or written to disk.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = os.environ.get("RAG_MODEL", "openai/gpt-oss-120b:free")
_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_MAX_RETRIES = 5
_RETRY_BASE = 2.0

# Strip control characters that break json.loads (e.g. NUL bytes).
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(text: str) -> str:
    return _CTRL.sub("", text)


def chat(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    stop: list[str] | None = None,
    timeout: int = 120,
) -> str:
    """Single chat completion via OpenRouter. Returns the assistant text."""
    if not _API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Export it before running: export OPENROUTER_API_KEY=sk-or-..."
        )

    payload: dict = {
        "model": model or DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop

    body = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/MONISMALIK1/rag",
        "X-Title": "RAG: Retrieval-Augmented Generation",
    }

    for attempt in range(_MAX_RETRIES):
        req = urllib.request.Request(_API_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = _clean(resp.read().decode())
                data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = float(exc.headers.get("Retry-After", _RETRY_BASE ** attempt))
                time.sleep(wait)
                continue
            raise

        # Inline error in a 200 response (some free-tier models do this).
        if "error" in data and "choices" not in data:
            raise RuntimeError(f"API error: {data['error']}")

        msg = data["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning") or ""
        return text.strip()

    raise RuntimeError("Exceeded max retries calling OpenRouter API")
