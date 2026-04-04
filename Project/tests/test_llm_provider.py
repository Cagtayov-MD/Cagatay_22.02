"""test_llm_provider.py — Unit tests for the provider-agnostic LLM client.

Covers:
  PROV-01: get_provider() returns "gemini" by default
  PROV-02: get_provider() returns "ollama" when LLM_PROVIDER=ollama
  PROV-03: generate() routes to Gemini when provider="gemini"
  PROV-04: generate() routes to Ollama when provider="ollama"
  PROV-05: Gemini payload is built correctly (contents list)
  PROV-06: Gemini response text is extracted from candidates[0].content.parts
  PROV-07: Gemini returns None when GEMINI_API_KEY is missing
  PROV-08: Gemini returns None on HTTP error
  PROV-09: Ollama payload is built correctly (messages list)
  PROV-10: Ollama response text is extracted from message.content
  PROV-11: Gemini strips <think>…</think> tags from response
  PROV-12: LLMCastFilter uses provider abstraction (_query_llm)
  PROV-13: LLMCastFilter._check_availability returns True for Gemini provider
  PROV-14: MatchParser routes to llm_provider.generate
  PROV-15: check_ollama_availability returns False when Ollama is unreachable
"""

import json
import os
import sys
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

# Ensure the Project directory is on the path
_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import core.llm_provider as _llm
from core.llm_cast_filter import LLMCastFilter
from core.match_parser import MatchParser


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fake_urlopen_gemini(response_text: str):
    """Return a context-manager mock for urllib.request.urlopen (Gemini)."""
    body = json.dumps({
        "candidates": [{
            "content": {
                "parts": [{"text": response_text}]
            }
        }]
    }).encode("utf-8")

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=body)))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _fake_urlopen_ollama(response_text: str):
    """Return a context-manager mock for urllib.request.urlopen (Ollama /api/chat)."""
    body = json.dumps({
        "message": {"content": response_text}
    }).encode("utf-8")

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=body)))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ─────────────────────────────────────────────────────────────────────────────
# PROV-01 / PROV-02: get_provider()
# ─────────────────────────────────────────────────────────────────────────────

def test_get_provider_default_is_gemini(monkeypatch):
    """PROV-01: default provider is gemini when LLM_PROVIDER is unset."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert _llm.get_provider() == "gemini"


def test_get_provider_ollama(monkeypatch):
    """PROV-02: provider is ollama when LLM_PROVIDER=ollama."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert _llm.get_provider() == "ollama"


# ─────────────────────────────────────────────────────────────────────────────
# PROV-03 / PROV-04: generate() routing
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_routes_to_gemini(monkeypatch):
    """PROV-03: generate(provider='gemini') calls _gemini_generate."""
    called = {}

    def fake_gemini(prompt, **kw):
        called["provider"] = "gemini"
        called["prompt"] = prompt
        return "gemini-answer"

    monkeypatch.setattr(_llm, "_gemini_generate", fake_gemini)
    result = _llm.generate("hello", provider="gemini")
    assert result == "gemini-answer"
    assert called["provider"] == "gemini"


def test_generate_routes_to_ollama(monkeypatch):
    """PROV-04: generate(provider='ollama') calls _ollama_generate."""
    called = {}

    def fake_ollama(prompt, **kw):
        called["provider"] = "ollama"
        return "ollama-answer"

    monkeypatch.setattr(_llm, "_ollama_generate", fake_ollama)
    result = _llm.generate("hello", provider="ollama")
    assert result == "ollama-answer"
    assert called["provider"] == "ollama"


# ─────────────────────────────────────────────────────────────────────────────
# PROV-05 / PROV-06: Gemini payload and response parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_payload_contents_structure(monkeypatch):
    """PROV-05: Gemini request includes a 'contents' list with user turn."""
    captured = {}
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_urlopen(req, timeout=None):
        import json as _json
        body = _json.loads(req.data.decode("utf-8"))
        captured["contents"] = body.get("contents", [])
        return _fake_urlopen_gemini("ok")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    _llm._gemini_generate("test-prompt", model="gemini-2.5-flash")

    contents = captured["contents"]
    assert any(c["role"] == "user" for c in contents), "Expected a user turn in contents"
    last_user = [c for c in contents if c["role"] == "user"][-1]
    assert "test-prompt" in last_user["parts"][0]["text"]


def test_gemini_response_extraction(monkeypatch):
    """PROV-06: Gemini text is extracted from candidates[0].content.parts."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: _fake_urlopen_gemini("ISIM: 3\n5"),
    )
    result = _llm._gemini_generate("prompt", model="gemini-2.5-flash")
    assert result == "ISIM: 3\n5"


def test_gemini_system_prompt_prepended(monkeypatch):
    """PROV-05b: system prompt is prepended as first user+model exchange."""
    captured = {}
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_urlopen(req, timeout=None):
        import json as _json
        body = _json.loads(req.data.decode("utf-8"))
        captured["contents"] = body.get("contents", [])
        return _fake_urlopen_gemini("ok")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    _llm._gemini_generate("user msg", system="system instruction", model="gemini-2.5-flash")

    contents = captured["contents"]
    assert len(contents) >= 3, "Expected system user/model + user turn"
    assert contents[0]["role"] == "user"
    assert "system instruction" in contents[0]["parts"][0]["text"]
    assert contents[1]["role"] == "model"


# ─────────────────────────────────────────────────────────────────────────────
# PROV-07: Gemini returns None when API key missing
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_no_api_key_returns_none(monkeypatch):
    """PROV-07: _gemini_generate returns None when GEMINI_API_KEY is absent."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = _llm._gemini_generate("hello", api_key=None)
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# PROV-08: Gemini returns None on HTTP error
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_http_error_returns_none(monkeypatch):
    """PROV-08: _gemini_generate returns None on HTTP error."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(_llm, "_GEMINI_MODEL_COOLDOWNS", {})

    def raise_http_error(req, timeout=None):
        raise urllib.error.HTTPError(
            url="http://x", code=403, msg="Forbidden",
            hdrs=None, fp=BytesIO(b"")
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)
    result = _llm._gemini_generate("hello", model="gemini-2.5-flash")
    assert result is None


def test_gemini_high_demand_sets_model_cooldown(monkeypatch):
    """PROV-08b: 503 high demand sonrası model cooldown'a alınır."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(_llm, "_GEMINI_MODEL_COOLDOWNS", {})
    monkeypatch.setattr(_llm.time, "sleep", lambda _sec: None)

    attempts = {"count": 0}

    def raise_http_error(req, timeout=None):
        attempts["count"] += 1
        raise urllib.error.HTTPError(
            url="http://x",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=BytesIO(
                b'{"error":{"code":503,"message":"This model is currently experiencing high demand.","status":"UNAVAILABLE"}}'
            ),
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)
    result = _llm._gemini_generate("hello", model="gemini-2.5-pro")

    assert result is None
    assert attempts["count"] == _llm._GEMINI_MAX_RETRIES
    assert _llm._get_gemini_cooldown_remaining("gemini-2.5-pro") >= 590


def test_gemini_skips_request_while_model_is_in_cooldown(monkeypatch):
    """PROV-08c: cooldown aktifken yeni HTTP isteği atılmaz."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(_llm, "_GEMINI_MODEL_COOLDOWNS", {})
    _llm._set_gemini_model_cooldown("gemini-2.5-pro", 120, now_ts=1000.0)
    monkeypatch.setattr(_llm.time, "time", lambda: 1001.0)

    def fail_if_called(req, timeout=None):
        raise AssertionError("Cooldown varken ağ çağrısı yapılmamalı")

    monkeypatch.setattr("urllib.request.urlopen", fail_if_called)
    result = _llm._gemini_generate("hello", model="gemini-2.5-pro")

    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# PROV-09 / PROV-10: Ollama payload and response parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_ollama_payload_messages_structure(monkeypatch):
    """PROV-09: Ollama request uses messages list with system+user roles."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        import json as _json
        body = _json.loads(req.data.decode("utf-8"))
        captured["messages"] = body.get("messages", [])
        return _fake_urlopen_ollama("ok")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    _llm._ollama_generate(
        "user text", system="sys text",
        ollama_url="http://localhost:11434", model="llama3.1:8b"
    )

    msgs = captured["messages"]
    assert msgs[0] == {"role": "system", "content": "sys text"}
    assert msgs[-1]["role"] == "user"
    assert "user text" in msgs[-1]["content"]


def test_ollama_response_extraction(monkeypatch):
    """PROV-10: Ollama text is extracted from message.content."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: _fake_urlopen_ollama("some response"),
    )
    result = _llm._ollama_generate(
        "prompt", ollama_url="http://localhost:11434", model="llama3.1:8b"
    )
    assert result == "some response"


# ─────────────────────────────────────────────────────────────────────────────
# PROV-11: Think tag stripping
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_strips_think_tags(monkeypatch):
    """PROV-11: <think>…</think> tags are stripped from Gemini response."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    raw = "<think>internal reasoning</think>ISIM: 1"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: _fake_urlopen_gemini(raw),
    )
    result = _llm._gemini_generate("prompt", model="gemini-2.5-flash")
    assert "<think>" not in (result or "")
    assert "ISIM: 1" in (result or "")


# ─────────────────────────────────────────────────────────────────────────────
# PROV-12 / PROV-13: LLMCastFilter provider integration
# ─────────────────────────────────────────────────────────────────────────────

def test_llm_cast_filter_uses_query_llm():
    """PROV-12: LLMCastFilter._query_ollama delegates to _query_llm."""
    f = LLMCastFilter(enabled=True, provider="gemini")
    # monkey-patch _query_llm
    f._query_llm = lambda prompt: "ISIM: 1"
    f._check_availability = lambda: True

    cast = [{"actor_name": "Nisa Serezli", "confidence": 0.7}]
    result = f.filter_cast(cast)
    assert len(result) == 1
    assert result[0]["is_llm_verified"] is True


def test_llm_cast_filter_availability_true_for_gemini():
    """PROV-13: LLMCastFilter._check_availability returns True for Gemini provider."""
    f = LLMCastFilter(enabled=True, provider="gemini")
    assert f._check_availability() is True


# ─────────────────────────────────────────────────────────────────────────────
# PROV-14: MatchParser routes to llm_provider.generate
# ─────────────────────────────────────────────────────────────────────────────

def test_match_parser_uses_llm_provider(monkeypatch):
    """PROV-14: MatchParser._llm_parse uses llm_provider.generate."""
    called = {}

    def fake_generate(prompt, **kw):
        called["invoked"] = True
        return '{"spor_turu": "futbol", "lig": "", "sehir": "", "takimlar": [], "teknik_direktorler": [], "olaylar": []}'

    monkeypatch.setattr(_llm, "generate", fake_generate)
    mp = MatchParser(provider="gemini")
    result = mp._llm_parse("Galatasaray 2-1 Fenerbahçe")
    assert called.get("invoked") is True
    assert result["spor_turu"] == "futbol"


# ─────────────────────────────────────────────────────────────────────────────
# PROV-15: check_ollama_availability returns False when unreachable
# ─────────────────────────────────────────────────────────────────────────────

def test_check_ollama_availability_returns_false_on_error(monkeypatch):
    """PROV-15: check_ollama_availability returns False when Ollama is unreachable."""
    def raise_url_error(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", raise_url_error)
    result = _llm.check_ollama_availability(
        ollama_url="http://localhost:11434", model="llama3.1:8b"
    )
    assert result is False
