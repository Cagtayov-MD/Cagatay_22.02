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
