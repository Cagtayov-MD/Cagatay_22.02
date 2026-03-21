"""test_pipeline_modes.py — SCOPE ortam değişkenine göre pipeline dallanma testleri.

Kullanıcı senaryoları:
  MODE-01: SCOPE=video_only  → sadece video işlenir, ses analizi atlanır
  MODE-02: SCOPE=audio_only  → sadece ses işlenir, video (OCR) atlanır
  MODE-03: SCOPE=video+audio → video ve ses birlikte çalışır

Gerçek dosya/model çalıştırılmaz; sadece hangi kod dalının tetiklendiği
mock ile doğrulanır.

Pipeline dallanma mantığı (pipeline_runner.py'den çoğaltılmıştır):
  - scope == "audio_only"   → _run_audio() seri çağrılır, OCR başlatılmaz
  - scope == "video+audio"  → _run_audio() arka planda (Thread), OCR çalışır
  - else (video_only)       → _run_audio() hiç çağrılmaz, OCR çalışır
"""

import concurrent.futures
import os
import sys
from unittest.mock import MagicMock, call, patch

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ── İnline yardımcılar (pipeline_runner.py mantığını birebir çoğaltır) ────────


def _get_scope_from_env(default="video+audio"):
    """main.py satır 26: scope = os.environ.get("SCOPE", "video+audio")"""
    return os.environ.get("SCOPE", default)


def _audio_branch(scope, run_audio_fn, video_path, work_dir):
    """pipeline_runner.py satır 427-442: ses dallanma mantığı.

    Döndürür: (audio_result, audio_future)
      - video_only  → future=None, result["status"]=="skipped"
      - audio_only  → future=None, result seri çağrıdan gelir
      - video+audio → future bekletilir, result paralel çağrıdan gelir
    """
    audio_result = None
    audio_future = None
    executor = None

    if scope == "audio_only":
        # Seri çalıştır — video işleme bloklanır
        audio_result = run_audio_fn(video_path, work_dir)
    elif scope == "video+audio":
        # Paralel çalıştır — arka plan thread
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        audio_future = executor.submit(run_audio_fn, video_path, work_dir)
    else:
        # video_only — ses tamamen atlanıyor
        audio_result = {"status": "skipped", "reason": f"scope={scope}"}

    if audio_future is not None:
        audio_result = audio_future.result()
        executor.shutdown(wait=False)

    return audio_result, audio_future


def _ocr_should_init(scope):
    """pipeline_runner.py satır 354: if scope != 'audio_only': OCR başlat."""
    return scope != "audio_only"


def _video_processing_should_run(scope, ocr_enabled=True):
    """pipeline_runner.py satır 454: if scope != 'audio_only' and ocr_enabled."""
    return scope != "audio_only" and ocr_enabled


# ── MODE-01: SCOPE=video_only ─────────────────────────────────────────────────


def test_mode_video_env_var_is_read_correctly():
    """MODE-01: SCOPE=video_only env var doğru okunur."""
    with patch.dict(os.environ, {"SCOPE": "video_only"}):
        scope = _get_scope_from_env()
    assert scope == "video_only", f"Beklenen 'video_only', alınan: '{scope}'"


def test_mode_video_audio_branch_not_called():
    """MODE-01: SCOPE=video_only → _run_audio() hiç çağrılmaz."""
    run_audio = MagicMock(return_value={"status": "ok", "transcript": []})

    with patch.dict(os.environ, {"SCOPE": "video_only"}):
        scope = _get_scope_from_env()
        audio_result, audio_future = _audio_branch(scope, run_audio, "film.mp4", "/tmp")

    run_audio.assert_not_called()
    assert audio_result["status"] == "skipped", (
        f"video_only: audio_result 'skipped' olmalı, alınan: {audio_result}"
    )
    assert audio_future is None, "video_only: future oluşmamalı"


def test_mode_video_ocr_enabled():
    """MODE-01: SCOPE=video_only → OCR motoru başlatılır."""
    with patch.dict(os.environ, {"SCOPE": "video_only"}):
        scope = _get_scope_from_env()
    assert _ocr_should_init(scope), "video_only: OCR motoru aktif olmalı"


def test_mode_video_video_processing_enabled():
    """MODE-01: SCOPE=video_only → video frame extraction ve OCR işleme çalışır."""
    with patch.dict(os.environ, {"SCOPE": "video_only"}):
        scope = _get_scope_from_env()
    assert _video_processing_should_run(scope), (
        "video_only: video işleme (frame extraction + OCR) aktif olmalı"
    )


def test_mode_video_ocr_init_mock():
    """MODE-01: SCOPE=video_only → OCR init mock'u tetiklenir, ses init edilmez."""
    init_ocr = MagicMock(name="init_ocr")
    init_audio = MagicMock(name="_run_audio")

    with patch.dict(os.environ, {"SCOPE": "video_only"}):
        scope = _get_scope_from_env()
        if _ocr_should_init(scope):
            init_ocr()
        _audio_branch(scope, init_audio, "film.mp4", "/tmp")

    init_ocr.assert_called_once()
    init_audio.assert_not_called()


# ── MODE-02: SCOPE=audio_only ─────────────────────────────────────────────────


def test_mode_audio_env_var_is_read_correctly():
    """MODE-02: SCOPE=audio_only env var doğru okunur."""
    with patch.dict(os.environ, {"SCOPE": "audio_only"}):
        scope = _get_scope_from_env()
    assert scope == "audio_only", f"Beklenen 'audio_only', alınan: '{scope}'"


def test_mode_audio_branch_called_serially():
    """MODE-02: SCOPE=audio_only → _run_audio() seri (blocking) çağrılır."""
    run_audio = MagicMock(return_value={"status": "ok", "transcript": ["Selam"]})

    with patch.dict(os.environ, {"SCOPE": "audio_only"}):
        scope = _get_scope_from_env()
        audio_result, audio_future = _audio_branch(scope, run_audio, "film.mp4", "/work")

    run_audio.assert_called_once_with("film.mp4", "/work")
    assert audio_future is None, "audio_only: future olmamalı (seri çalışır)"
    assert audio_result["status"] == "ok", (
        f"audio_only: audio_result['status'] 'ok' olmalı, alınan: {audio_result}"
    )


def test_mode_audio_ocr_not_initialized():
    """MODE-02: SCOPE=audio_only → OCR motoru başlatılmaz."""
    with patch.dict(os.environ, {"SCOPE": "audio_only"}):
        scope = _get_scope_from_env()
    assert not _ocr_should_init(scope), (
        "audio_only: OCR motoru başlatılmamalı (gereksiz ve pahalı)"
    )


def test_mode_audio_video_processing_skipped():
    """MODE-02: SCOPE=audio_only → video frame extraction ve OCR işleme atlanır."""
    with patch.dict(os.environ, {"SCOPE": "audio_only"}):
        scope = _get_scope_from_env()
    assert not _video_processing_should_run(scope), (
        "audio_only: video işleme (frame extraction + OCR) atlanmalı"
    )


def test_mode_audio_ocr_init_mock_not_called():
    """MODE-02: SCOPE=audio_only → OCR init mock'u tetiklenmez, ses init edilir."""
    init_ocr = MagicMock(name="init_ocr")
    run_audio = MagicMock(name="_run_audio", return_value={"status": "ok"})

    with patch.dict(os.environ, {"SCOPE": "audio_only"}):
        scope = _get_scope_from_env()
        if _ocr_should_init(scope):
            init_ocr()
        _audio_branch(scope, run_audio, "film.mp4", "/tmp")

    init_ocr.assert_not_called()
    run_audio.assert_called_once()


# ── MODE-03: SCOPE=video+audio ────────────────────────────────────────────────


def test_mode_video_plus_audio_env_var_is_read_correctly():
    """MODE-03: SCOPE=video+audio env var doğru okunur."""
    with patch.dict(os.environ, {"SCOPE": "video+audio"}):
        scope = _get_scope_from_env()
    assert scope == "video+audio", f"Beklenen 'video+audio', alınan: '{scope}'"


def test_mode_video_plus_audio_audio_called_in_background():
    """MODE-03: SCOPE=video+audio → _run_audio() arka plan thread'inde çalışır."""
    run_audio = MagicMock(return_value={"status": "ok", "transcript": ["Merhaba"]})

    with patch.dict(os.environ, {"SCOPE": "video+audio"}):
        scope = _get_scope_from_env()
        audio_result, audio_future = _audio_branch(scope, run_audio, "film.mp4", "/work")

    run_audio.assert_called_once_with("film.mp4", "/work")
    assert audio_result is not None, "video+audio: audio_result None olmamalı"
    assert audio_result["status"] == "ok", (
        f"video+audio: audio_result['status'] 'ok' olmalı, alınan: {audio_result}"
    )


def test_mode_video_plus_audio_ocr_enabled():
    """MODE-03: SCOPE=video+audio → OCR motoru başlatılır."""
    with patch.dict(os.environ, {"SCOPE": "video+audio"}):
        scope = _get_scope_from_env()
    assert _ocr_should_init(scope), "video+audio: OCR motoru aktif olmalı"


def test_mode_video_plus_audio_video_processing_enabled():
    """MODE-03: SCOPE=video+audio → video frame extraction ve OCR işleme çalışır."""
    with patch.dict(os.environ, {"SCOPE": "video+audio"}):
        scope = _get_scope_from_env()
    assert _video_processing_should_run(scope), (
        "video+audio: video işleme (frame extraction + OCR) aktif olmalı"
    )


def test_mode_video_plus_audio_both_branches_active():
    """MODE-03: SCOPE=video+audio → hem OCR init hem ses analizi tetiklenir."""
    init_ocr = MagicMock(name="init_ocr")
    run_audio = MagicMock(name="_run_audio", return_value={"status": "ok"})

    with patch.dict(os.environ, {"SCOPE": "video+audio"}):
        scope = _get_scope_from_env()
        if _ocr_should_init(scope):
            init_ocr()
        _audio_branch(scope, run_audio, "film.mp4", "/tmp")

    init_ocr.assert_called_once()
    run_audio.assert_called_once()


# ── VARSAYILAN (SCOPE env var yok) ───────────────────────────────────────────


def test_default_scope_when_env_var_absent():
    """SCOPE env var set edilmemişse varsayılan 'video+audio' kullanılır."""
    env_without_scope = {k: v for k, v in os.environ.items() if k != "SCOPE"}
    with patch.dict(os.environ, env_without_scope, clear=True):
        scope = _get_scope_from_env()
    assert scope == "video+audio", (
        f"SCOPE env var yokken varsayılan 'video+audio' olmalı, alınan: '{scope}'"
    )
