import io
import json
import sys
import types
from pathlib import Path

from audio.stages.transcribe import TranscribeStage
from core import sport_pipeline
from core.audio_bridge import AudioBridge


class _Info:
    def __init__(self, language="tr", prob=0.99):
        self.language = language
        self.language_probability = prob


class _Seg:
    def __init__(self, start, end, text, words=None):
        self.start = start
        self.end = end
        self.text = text
        self.words = words or []


def test_content_profile_film_dizi_uses_large_v3():
    profiles_path = Path(__file__).resolve().parent.parent / "config" / "content_profiles.json"
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    assert profiles["FilmDizi-Hybrid"]["whisper_model"] == "large-v3"


def test_audio_bridge_normalizes_whisper_model_before_writing_config(monkeypatch, tmp_path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake-video")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    result_path = work_dir / "audio_result.json"

    bridge = AudioBridge(
        log_cb=lambda *_args, **_kwargs: None,
        config={
            "venv_audio_python": "fake-python.exe",
            "audio_worker_script": "fake-worker.py",
        },
    )
    monkeypatch.setattr(bridge, "_check_prerequisites", lambda *_args, **_kwargs: True)

    class _FakeProc:
        def __init__(self):
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout=None):
            payload = {"status": "ok", "transcript": [], "speakers": {}, "stages": {}}
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            return 0

        def kill(self):
            return None

    monkeypatch.setattr("core.audio_bridge.subprocess.Popen", lambda *args, **kwargs: _FakeProc())

    result = bridge.run(
        str(video_path),
        str(work_dir),
        {"options": {"whisper_model": "large-v3-turbo"}},
    )

    written_config = json.loads((work_dir / "audio_config.json").read_text(encoding="utf-8"))
    assert written_config["options"]["whisper_model"] == "large-v3"
    assert result["status"] == "ok"


def test_audio_bridge_filters_pkg_resources_warning_from_stderr():
    stderr_tail, known_notes = AudioBridge._summarize_stderr([
        r"F:\Root\venv_audio\lib\site-packages\ctranslate2\__init__.py:8: UserWarning: pkg_resources is deprecated as an API.",
        "  import pkg_resources",
    ])

    assert stderr_tail == ""
    assert "ctranslate2/setuptools pkg_resources uyarısı" in known_notes


def test_audio_bridge_returns_ok_on_benign_nonzero_exit_with_result(monkeypatch, tmp_path):
    logs = []
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake-video")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    result_path = work_dir / "audio_result.json"

    bridge = AudioBridge(
        log_cb=logs.append,
        config={
            "venv_audio_python": "fake-python.exe",
            "audio_worker_script": "fake-worker.py",
        },
    )
    monkeypatch.setattr(bridge, "_check_prerequisites", lambda *_args, **_kwargs: True)

    class _FakeProc:
        def __init__(self):
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO(
                "F:\\Root\\venv_audio\\lib\\site-packages\\ctranslate2\\__init__.py:8: "
                "UserWarning: pkg_resources is deprecated as an API.\n"
                "  import pkg_resources\n"
            )
            self.returncode = 3221226505

        def wait(self, timeout=None):
            payload = {"status": "ok", "transcript": [{"text": "ok"}], "speakers": {}, "stages": {}}
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            return self.returncode

        def kill(self):
            return None

    monkeypatch.setattr("core.audio_bridge.subprocess.Popen", lambda *args, **kwargs: _FakeProc())

    result = bridge.run(
        str(video_path),
        str(work_dir),
        {"options": {"whisper_model": "large-v3"}},
    )

    assert result["status"] == "ok"
    assert any("benign shutdown" in line for line in logs)
    assert not any("pkg_resources is deprecated as an API" in line for line in logs)


def test_transcribe_stage_normalizes_alias_model(monkeypatch, tmp_path):
    captured = {}

    class _FakeWhisperModel:
        def __init__(self, model_name, device=None, compute_type=None):
            captured["model_name"] = model_name

        def transcribe(self, audio_path, **kwargs):
            return iter([_Seg(0.0, 1.0, "test satiri")]), _Info("tr", 0.97)

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr("audio.stages.transcribe.VRAMManager.get_device", lambda: "cpu")
    monkeypatch.setattr("audio.stages.transcribe.VRAMManager.release", lambda: None)

    wav = tmp_path / "dummy.wav"
    wav.write_bytes(b"RIFF....WAVE")

    stage = TranscribeStage(log_cb=lambda *_args, **_kwargs: None)
    result = stage._transcribe(
        audio_path=str(wav),
        opts={
            "whisper_model": "v3",
            "whisper_language": "tr",
            "beam_size": 1,
            "cache_model": False,
        },
        diarization=[],
    )

    assert captured["model_name"] == "large-v3"
    assert result["status"] == "ok"


def test_sport_pipeline_normalizes_asr_model_alias(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, capture_output, text, env, timeout):
        captured["cmd"] = cmd
        payload = {
            "transcript": "test",
            "segments": [],
            "speech_duration": 1.0,
            "avg_logprob": -0.5,
            "avg_no_speech_prob": 0.1,
            "avg_compression_ratio": 1.2,
            "detected_language": "tr",
            "language_probability": 0.98,
        }
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload).encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr("core.sport_pipeline.subprocess.run", fake_run)
    monkeypatch.setattr("core.sport_pipeline._get_audio_duration", lambda *_args, **_kwargs: 12.0)
    monkeypatch.setattr("core.sport_pipeline._build_subprocess_env", lambda: {})

    audio_path = tmp_path / "sport.wav"
    result = sport_pipeline._run_whisper_detailed(
        str(audio_path),
        {
            "asr_model": "large-v3-turbo",
            "asr_device": "cpu",
            "asr_language": "tr",
            "beam_size": 1,
        },
        _log=lambda *_args, **_kwargs: None,
    )

    script = captured["cmd"][2]
    assert "large-v3-turbo" not in script
    assert "large-v3" in script
    assert result["transcript"] == "test"
