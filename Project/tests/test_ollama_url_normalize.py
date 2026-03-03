"""test_ollama_url_normalize.py — Unit tests for normalize_ollama_url helper.

URL-NORM-01: Bare base URL is unchanged (trailing slash stripped)
URL-NORM-02: /api suffix is stripped
URL-NORM-03: /api/generate suffix is stripped
URL-NORM-04: /api/chat suffix is stripped
URL-NORM-05: /api/tags suffix is stripped
URL-NORM-06: LLMCastFilter normalizes incoming URL
URL-NORM-07: MatchParser normalizes incoming URL
URL-NORM-08: QwenVerifier normalizes incoming URL (self.url ends with /api/chat)
URL-NORM-09: VLMReader normalizes incoming URL (self.url ends with /api/chat)
"""

import os
import sys

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core._ollama_url import normalize_ollama_url
from core.llm_cast_filter import LLMCastFilter
from core.match_parser import MatchParser
from core.qwen_verifier import QwenVerifier
from core.vlm_reader import VLMReader


# ═══════════════════════════════════════════════════════════════════════════════
# URL-NORM-01 … 05: normalize_ollama_url helper
# ═══════════════════════════════════════════════════════════════════════════════

def test_bare_url_unchanged():
    """URL-NORM-01: Bare base URL with no path returns as-is."""
    assert normalize_ollama_url("http://localhost:11434") == "http://localhost:11434"


def test_trailing_slash_stripped():
    """URL-NORM-01b: Trailing slash is stripped."""
    assert normalize_ollama_url("http://localhost:11434/") == "http://localhost:11434"


def test_api_suffix_stripped():
    """URL-NORM-02: /api suffix is stripped."""
    assert normalize_ollama_url("http://localhost:11434/api") == "http://localhost:11434"


def test_api_generate_suffix_stripped():
    """URL-NORM-03: /api/generate suffix is stripped."""
    assert normalize_ollama_url("http://localhost:11434/api/generate") == "http://localhost:11434"


def test_api_chat_suffix_stripped():
    """URL-NORM-04: /api/chat suffix is stripped."""
    assert normalize_ollama_url("http://localhost:11434/api/chat") == "http://localhost:11434"


def test_api_tags_suffix_stripped():
    """URL-NORM-05: /api/tags suffix is stripped."""
    assert normalize_ollama_url("http://localhost:11434/api/tags") == "http://localhost:11434"


def test_custom_host_and_port():
    """URL-NORM-01c: Custom host/port with /api/chat is normalized correctly."""
    assert normalize_ollama_url("http://192.168.1.10:11434/api/chat") == "http://192.168.1.10:11434"


def test_https_scheme():
    """URL-NORM-01d: HTTPS URLs are normalized correctly."""
    assert normalize_ollama_url("https://ollama.example.com/api/generate") == "https://ollama.example.com"


# ═══════════════════════════════════════════════════════════════════════════════
# URL-NORM-06: LLMCastFilter
# ═══════════════════════════════════════════════════════════════════════════════

def test_llm_cast_filter_normalizes_url():
    """URL-NORM-06: LLMCastFilter stores bare base URL regardless of suffix."""
    for url_in in [
        "http://localhost:11434",
        "http://localhost:11434/",
        "http://localhost:11434/api",
        "http://localhost:11434/api/generate",
        "http://localhost:11434/api/chat",
    ]:
        f = LLMCastFilter(ollama_url=url_in)
        assert f.ollama_url == "http://localhost:11434", (
            f"Expected bare base URL for input {url_in!r}, got {f.ollama_url!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# URL-NORM-07: MatchParser
# ═══════════════════════════════════════════════════════════════════════════════

def test_match_parser_normalizes_url():
    """URL-NORM-07: MatchParser stores bare base URL regardless of suffix."""
    for url_in in [
        "http://localhost:11434",
        "http://localhost:11434/api",
        "http://localhost:11434/api/generate",
    ]:
        mp = MatchParser(ollama_url=url_in)
        assert mp.ollama_url == "http://localhost:11434", (
            f"Expected bare base URL for input {url_in!r}, got {mp.ollama_url!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# URL-NORM-08: QwenVerifier
# ═══════════════════════════════════════════════════════════════════════════════

def test_qwen_verifier_normalizes_url():
    """URL-NORM-08: QwenVerifier.url ends with /api/chat after normalization."""
    for url_in in [
        "http://localhost:11434",
        "http://localhost:11434/api",
        "http://localhost:11434/api/chat",
        "http://localhost:11434/api/generate",
    ]:
        v = QwenVerifier(ollama_url=url_in)
        assert v.url == "http://localhost:11434/api/chat", (
            f"Expected /api/chat endpoint for input {url_in!r}, got {v.url!r}"
        )
        assert v._base_url == "http://localhost:11434", (
            f"Expected bare base URL for input {url_in!r}, got {v._base_url!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# URL-NORM-09: VLMReader
# ═══════════════════════════════════════════════════════════════════════════════

def test_vlm_reader_normalizes_url():
    """URL-NORM-09: VLMReader.url ends with /api/chat after normalization."""
    for url_in in [
        "http://localhost:11434",
        "http://localhost:11434/api",
        "http://localhost:11434/api/chat",
        "http://localhost:11434/api/generate",
    ]:
        r = VLMReader(ollama_url=url_in)
        assert r.url == "http://localhost:11434/api/chat", (
            f"Expected /api/chat endpoint for input {url_in!r}, got {r.url!r}"
        )
        assert r._base_url == "http://localhost:11434", (
            f"Expected bare base URL for input {url_in!r}, got {r._base_url!r}"
        )
