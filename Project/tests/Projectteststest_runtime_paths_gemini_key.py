import os
import sys

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from config import runtime_paths


def test_get_gemini_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setattr(runtime_paths, "load_api_keys", lambda: {"gemini_api_key": "file-key"})
    assert runtime_paths.get_gemini_api_key() == "env-key"


def test_get_gemini_api_key_falls_back_to_api_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(runtime_paths, "load_api_keys", lambda: {"gemini_api_key": "file-key"})
    assert runtime_paths.get_gemini_api_key() == "file-key"


def test_get_gemini_film_credit_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("GEMINI_FILM_CREDIT_API_KEY", "shadow-env-key")
    monkeypatch.setattr(
        runtime_paths,
        "load_api_keys",
        lambda: {"gemini_film_credit_api_key": "shadow-file-key"},
    )
    assert runtime_paths.get_gemini_film_credit_api_key() == "shadow-env-key"


def test_get_gemini_film_credit_api_key_falls_back_to_api_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_FILM_CREDIT_API_KEY", raising=False)
    monkeypatch.setattr(
        runtime_paths,
        "load_api_keys",
        lambda: {"gemini_film_credit_api_key": "shadow-file-key"},
    )
    assert runtime_paths.get_gemini_film_credit_api_key() == "shadow-file-key"


def test_gemini_film_credit_shadow_enabled_prefers_env(monkeypatch):
    monkeypatch.setenv("GEMINI_FILM_CREDIT_SHADOW_ENABLED", "1")
    monkeypatch.setattr(
        runtime_paths,
        "load_api_keys",
        lambda: {"gemini_film_credit_shadow_enabled": False},
    )
    assert runtime_paths.is_gemini_film_credit_shadow_enabled() is True


def test_gemini_film_credit_shadow_enabled_falls_back_to_api_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_FILM_CREDIT_SHADOW_ENABLED", raising=False)
    monkeypatch.setattr(
        runtime_paths,
        "load_api_keys",
        lambda: {"gemini_film_credit_shadow_enabled": True},
    )
    assert runtime_paths.is_gemini_film_credit_shadow_enabled() is True


def test_gemini_film_credit_shadow_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GEMINI_FILM_CREDIT_SHADOW_ENABLED", raising=False)
    monkeypatch.setattr(runtime_paths, "load_api_keys", lambda: {})
    assert runtime_paths.is_gemini_film_credit_shadow_enabled() is False
