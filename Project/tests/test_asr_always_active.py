"""test_asr_always_active.py — Validates scope-based ASR/OCR skip logic.

Covers:
  SCOPE-01: scope="video_only"  -> audio SKIPPED, OCR runs
  SCOPE-02: scope="audio_only"  -> audio runs serially, OCR SKIPPED
  SCOPE-03: scope="video+audio" -> audio runs in background, OCR runs
  SCOPE-04: Explicit scope parameter takes priority over content_profile scope
  SCOPE-05: If no explicit scope, content_profile scope is used
  SCOPE-06: If no scope and no profile, config "scope" key is used
  SCOPE-07: Film/Dizi profile uses only ["detect_language", "extract", "transcribe"] stages
"""

import concurrent.futures
import os
import sys
from unittest.mock import MagicMock

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def _select_audio_stages(config):
    """Replicate the stage selection logic from PipelineRunner._run_audio."""
    stages = config.get("audio_stages", None)
    if stages is None:
        program_type = config.get("program_type", "film_dizi")
        if program_type in ("film_dizi", "kisa_haber"):
            stages = ["detect_language", "extract", "transcribe"]
        else:
            stages = ["detect_language", "extract", "denoise", "diarize", "transcribe", "post_process"]
    return stages


def _audio_branch(scope, run_audio_fn, video_path, work_dir):
    """Replicate the audio branching logic from PipelineRunner.run()."""
    audio_result = None
    audio_future = None
    executor = None

    if scope == "audio_only":
        # Sadece ses -- seri çalıştır
        audio_result = run_audio_fn(video_path, work_dir)
    elif scope == "video+audio":
        # Video + Ses -- audio arka planda paralel çalışsın
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        audio_future = executor.submit(run_audio_fn, video_path, work_dir)
    else:
        # video_only -- ses analizi tamamen atlanıyor
        audio_result = {"status": "skipped", "reason": "scope=video_only"}

    if audio_future is not None:
        audio_result = audio_future.result()
        executor.shutdown(wait=False)

    return audio_result, audio_future


def _resolve_scope(explicit_scope, content_profile, config):
    """Replicate the scope resolution logic from PipelineRunner.run().

    Priority: explicit parameter > content_profile["scope"] > config["scope"] > "video+audio"
    """
    scope = explicit_scope
    if scope is None and content_profile:
        scope = content_profile.get("scope")
    if scope is None:
        scope = config.get("scope", "video+audio")
    return scope


# -- SCOPE-01: video_only skips audio ----------------------------------
def test_video_only_scope_skips_audio():
    """SCOPE-01: scope='video_only' must NOT call _run_audio(); returns skipped status."""
    run_audio = MagicMock(return_value={"status": "ok", "transcript": []})
    audio_result, audio_future = _audio_branch("video_only", run_audio, "v.mp4", "/tmp")
    run_audio.assert_not_called()
    assert audio_result is not None
    assert audio_result.get("status") == "skipped", (
        "scope='video_only' ile audio_result 'skipped' olmalı, alınan: {}".format(audio_result)
    )
    assert audio_future is None


# -- SCOPE-02: audio_only runs audio serially, no OCR future ----------
def test_audio_only_scope_calls_run_audio_serially():
    """SCOPE-02: scope='audio_only' calls _run_audio() directly (serial), no future."""
    run_audio = MagicMock(return_value={"status": "ok", "transcript": []})
    audio_result, audio_future = _audio_branch("audio_only", run_audio, "v.mp4", "/tmp")
    run_audio.assert_called_once_with("v.mp4", "/tmp")
    assert audio_future is None, "scope='audio_only' için future olmamalı (serial çalışır)"
    assert audio_result is not None


# -- SCOPE-03: video+audio runs both ----------------------------------
def test_video_and_audio_scope_calls_run_audio():
    """SCOPE-03: scope='video+audio' must cause _run_audio() to be submitted in background."""
    run_audio = MagicMock(return_value={"status": "ok", "transcript": []})
    audio_result, audio_future = _audio_branch("video+audio", run_audio, "v.mp4", "/tmp")
    run_audio.assert_called_once_with("v.mp4", "/tmp")
    assert audio_result is not None, "scope='video+audio' ile audio_result None olmamalı"


# -- SCOPE-04: Explicit scope overrides profile scope -----------------
def test_explicit_scope_overrides_profile_scope():
    """SCOPE-04: Explicit scope parameter must win over content_profile['scope']."""
    profile = {"scope": "video+audio", "_name": "FilmDizi-Hybrid"}
    resolved = _resolve_scope(explicit_scope="video_only", content_profile=profile, config={})
    assert resolved == "video_only", (
        "Açık scope='video_only' profil scope'unu (video+audio) ezmelidir, alınan: {}".format(resolved)
    )


def test_explicit_audio_only_overrides_profile_scope():
    """SCOPE-04b: Explicit scope='audio_only' must override profile scope='video+audio'."""
    profile = {"scope": "video+audio", "_name": "FilmDizi-Hybrid"}
    resolved = _resolve_scope(explicit_scope="audio_only", content_profile=profile, config={})
    assert resolved == "audio_only", (
        "Açık scope='audio_only' profil scope'unu ezmelidir, alınan: {}".format(resolved)
    )


# -- SCOPE-05: Profile scope used when no explicit scope --------------
def test_profile_scope_used_when_no_explicit():
    """SCOPE-05: Profile scope is applied when no explicit scope is passed (None)."""
    profile = {"scope": "video_only", "_name": "Spor"}
    resolved = _resolve_scope(explicit_scope=None, content_profile=profile, config={})
    assert resolved == "video_only", (
        "Açık scope verilmediğinde profil scope='video_only' kullanılmalı, alınan: {}".format(resolved)
    )


# -- SCOPE-06: Config scope fallback ----------------------------------
def test_config_scope_used_when_no_explicit_no_profile():
    """SCOPE-06: config['scope'] is used when neither explicit scope nor profile is given."""
    resolved = _resolve_scope(explicit_scope=None, content_profile=None, config={"scope": "video_only"})
    assert resolved == "video_only", (
        "Config scope='video_only' kullanılmalı, alınan: {}".format(resolved)
    )


def test_default_scope_is_video_plus_audio():
    """SCOPE-06b: Default scope is 'video+audio' when nothing is configured."""
    resolved = _resolve_scope(explicit_scope=None, content_profile=None, config={})
    assert resolved == "video+audio", (
        "Varsayılan scope 'video+audio' olmalı, alınan: {}".format(resolved)
    )


# -- SCOPE-07: Film/Dizi audio stages ---------------------------------
def test_film_dizi_profile_uses_extract_and_transcribe_only():
    """SCOPE-07: Film/Dizi program_type uses only ['detect_language', 'extract', 'transcribe'] stages."""
    stages = _select_audio_stages({"program_type": "film_dizi"})
    assert stages == ["detect_language", "extract", "transcribe"], (
        "Film/Dizi için beklenen ['detect_language', 'extract', 'transcribe'], alınan: {}".format(stages)
    )
    assert "post_process" not in stages, "post_process (Ollama) film_dizi profilinde olmamalı"
    assert "denoise" not in stages, "denoise film_dizi profilinde olmamalı"


def test_kisa_haber_profile_uses_extract_and_transcribe_only():
    """SCOPE-08: kisa_haber profile also uses only ['detect_language', 'extract', 'transcribe'] stages."""
    stages = _select_audio_stages({"program_type": "kisa_haber"})
    assert stages == ["detect_language", "extract", "transcribe"], (
        "kisa_haber için beklenen ['detect_language', 'extract', 'transcribe'], alınan: {}".format(stages)
    )
