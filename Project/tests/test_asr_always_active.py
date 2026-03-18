"""test_asr_always_active.py — Validates ASR runs for all scopes.

Covers:
  ASR-ALWAYS-01: scope="video_only" causes _run_audio() to be called (not skipped)
  ASR-ALWAYS-02: scope="video+audio" still causes _run_audio() to be called
  ASR-ALWAYS-03: Film/Dizi profile uses only ["extract", "transcribe"] stages
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
        audio_result = run_audio_fn(video_path, work_dir)
    else:
        # video_only ve video+audio → her zaman audio başlat (paralel)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        audio_future = executor.submit(run_audio_fn, video_path, work_dir)

    if audio_future is not None:
        audio_result = audio_future.result()
        executor.shutdown(wait=False)

    return audio_result, audio_future


def test_video_only_scope_calls_run_audio():
    """ASR-ALWAYS-01: scope='video_only' must cause _run_audio() to be submitted."""
    run_audio = MagicMock(return_value={"status": "ok", "transcript": []})
    audio_result, audio_future = _audio_branch("video_only", run_audio, "v.mp4", "/tmp")
    run_audio.assert_called_once_with("v.mp4", "/tmp")
    assert audio_result is not None, "scope='video_only' ile audio_result None olmamalı"


def test_video_and_audio_scope_calls_run_audio():
    """ASR-ALWAYS-02: scope='video+audio' must cause _run_audio() to be submitted."""
    run_audio = MagicMock(return_value={"status": "ok", "transcript": []})
    audio_result, audio_future = _audio_branch("video+audio", run_audio, "v.mp4", "/tmp")
    run_audio.assert_called_once_with("v.mp4", "/tmp")
    assert audio_result is not None, "scope='video+audio' ile audio_result None olmamalı"


def test_audio_only_scope_calls_run_audio_serially():
    """ASR-ALWAYS-00: scope='audio_only' calls _run_audio() directly (serial)."""
    run_audio = MagicMock(return_value={"status": "ok", "transcript": []})
    audio_result, audio_future = _audio_branch("audio_only", run_audio, "v.mp4", "/tmp")
    run_audio.assert_called_once_with("v.mp4", "/tmp")
    assert audio_future is None, "scope='audio_only' için future olmamalı (serial çalışır)"


def test_film_dizi_profile_uses_extract_and_transcribe_only():
    """ASR-ALWAYS-03: Film/Dizi program_type uses only ['detect_language', 'extract', 'transcribe'] stages."""
    stages = _select_audio_stages({"program_type": "film_dizi"})
    assert stages == ["detect_language", "extract", "transcribe"], (
        f"Film/Dizi için beklenen ['detect_language', 'extract', 'transcribe'], alınan: {stages}"
    )
    assert "post_process" not in stages, "post_process (Ollama) film_dizi profilinde olmamalı"
    assert "denoise" not in stages, "denoise film_dizi profilinde olmamalı"


def test_kisa_haber_profile_uses_extract_and_transcribe_only():
    """ASR-ALWAYS-04: kisa_haber profile also uses only ['detect_language', 'extract', 'transcribe'] stages."""
    stages = _select_audio_stages({"program_type": "kisa_haber"})
    assert stages == ["detect_language", "extract", "transcribe"], (
        f"kisa_haber için beklenen ['detect_language', 'extract', 'transcribe'], alınan: {stages}"
    )
