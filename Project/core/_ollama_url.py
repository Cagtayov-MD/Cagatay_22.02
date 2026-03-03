"""_ollama_url.py — Shared Ollama base-URL normalization helper.

Converts any of:
    http://host:11434
    http://host:11434/
    http://host:11434/api
    http://host:11434/api/chat
    http://host:11434/api/generate
    http://host:11434/api/tags
into the bare base URL:
    http://host:11434

This prevents double-path bugs like /api/api/generate when callers
append their own /api/<endpoint> suffix.
"""

import re

_API_SUFFIX_RE = re.compile(
    r"/api(?:/(?:chat|generate|tags))?/?$",
    re.IGNORECASE,
)


def normalize_ollama_url(url: str) -> str:
    """Return bare Ollama base URL (e.g. http://host:11434).

    Strips any ``/api``, ``/api/chat``, ``/api/generate``, or
    ``/api/tags`` suffix so callers can safely append their own
    endpoint path without producing invalid double-path URLs.
    """
    url = url.strip()
    url = _API_SUFFIX_RE.sub("", url)
    return url.rstrip("/")
