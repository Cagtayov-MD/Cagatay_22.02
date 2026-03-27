import os
import sys
from types import SimpleNamespace

import pytest

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def test_normalize_subprocess_text_tolerates_non_utf8_bytes():
    from core.sport_pipeline import _normalize_subprocess_text

    text = _normalize_subprocess_text(b"prefix \xfd suffix")
    assert isinstance(text, str)
    assert "prefix" in text


def test_run_whisper_empty_stdout_returns_empty(monkeypatch):
    from core import sport_pipeline

    captured = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return SimpleNamespace(stdout=b"", stderr=b"", returncode=0)

    fake_subprocess = SimpleNamespace(
        run=fake_run,
        PIPE=-1,
        TimeoutExpired=Exception,
    )
    monkeypatch.setattr(sport_pipeline, "subprocess", fake_subprocess)

    result = sport_pipeline._run_whisper("dummy.wav", {})
    assert result == ""
    assert captured["env"]["PYTHONUTF8"] == "1"
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"


def test_run_whisper_nonzero_logs_stderr_safely(monkeypatch, caplog):
    from core import sport_pipeline

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout=b"", stderr=b"fatal \xfd whisper error", returncode=1)

    fake_subprocess = SimpleNamespace(
        run=fake_run,
        PIPE=-1,
        TimeoutExpired=Exception,
    )
    monkeypatch.setattr(sport_pipeline, "subprocess", fake_subprocess)

    with caplog.at_level("ERROR", logger="VITOS.pipeline_runner"):
        result = sport_pipeline._run_whisper("dummy.wav", {})

    assert result == ""
    assert any("[Whisper] Subprocess hatas" in rec.message for rec in caplog.records)


def test_namedb_without_config_logs_soft_fallback(monkeypatch):
    def _raise():
        raise RuntimeError("IMDB not configured")

    monkeypatch.setattr("config.runtime_paths.get_imdb_db_path", _raise)

    from core.turkish_name_db import TurkishNameDB

    logs = []
    TurkishNameDB(log_cb=logs.append)

    assert any("DB yapılandırılmadı" in msg for msg in logs)
    assert all("Beklenen:" not in msg for msg in logs)


def test_namedb_with_missing_db_path_logs_explicit_path():
    from core.turkish_name_db import TurkishNameDB

    logs = []
    TurkishNameDB(db_path=r"F:\missing\names.db", log_cb=logs.append)

    assert any(r"DB bulunamadı: F:\missing\names.db" in msg for msg in logs)


def test_namedb_with_missing_sql_seed_logs_seed_message():
    from core.turkish_name_db import TurkishNameDB

    logs = []
    TurkishNameDB(db_path=r"F:\missing\names.sql", log_cb=logs.append)

    assert any(r"SQL seed bulunamadı: F:\missing\names.sql" in msg for msg in logs)


def test_main_routes_cli_video_argument_to_ui(monkeypatch):
    import main as app_main

    calls = []

    monkeypatch.setattr(app_main, "_run_headless", lambda argv=None, environ=None: calls.append("headless") or 0)
    monkeypatch.setattr(app_main, "_run_ui", lambda argv=None: calls.append("ui") or 0)

    rc = app_main.main(argv=["main.py", "video.mp4"], environ={})
    assert rc == 0
    assert calls == ["ui"]


def test_main_routes_interactive_launch_to_ui(monkeypatch):
    import main as app_main

    calls = []

    monkeypatch.setattr(app_main, "_run_headless", lambda argv=None, environ=None: calls.append("headless") or 0)
    monkeypatch.setattr(app_main, "_run_ui", lambda argv=None: calls.append("ui") or 0)

    rc = app_main.main(argv=["main.py"], environ={})
    assert rc == 0
    assert calls == ["ui"]


def test_main_routes_scope_env_to_headless(monkeypatch):
    import main as app_main

    calls = []

    monkeypatch.setattr(app_main, "_run_headless", lambda argv=None, environ=None: calls.append("headless") or 0)
    monkeypatch.setattr(app_main, "_run_ui", lambda argv=None: calls.append("ui") or 0)

    rc = app_main.main(argv=["main.py"], environ={"SCOPE": "video+audio"})
    assert rc == 0
    assert calls == ["headless"]


def test_main_routes_headless_flag_to_headless(monkeypatch):
    import main as app_main

    calls = []

    monkeypatch.setattr(app_main, "_run_headless", lambda argv=None, environ=None: calls.append("headless") or 0)
    monkeypatch.setattr(app_main, "_run_ui", lambda argv=None: calls.append("ui") or 0)

    rc = app_main.main(argv=["main.py", "--headless", "video.mp4"], environ={})
    assert rc == 0
    assert calls == ["headless"]