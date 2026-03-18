import sys
import types

from audio.stages.transcribe import TranscribeStage


class _Info:
    def __init__(self, language="tr", prob=0.99):
        self.language = language
        self.language_probability = prob


class _Word:
    def __init__(self, word, start, end, probability):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class _Seg:
    def __init__(self, start, end, text, words=None):
        self.start = start
        self.end = end
        self.text = text
        self.words = words or []


class _FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("word_timestamps") is True:
            def _broken_gen():
                yield _Seg(0.0, 1.0, "ilk", words=[_Word("ilk", 0.0, 1.0, 0.9)])
                raise RuntimeError("simulated generator failure")

            return _broken_gen(), _Info("tr", 0.93)

        def _ok_gen():
            yield _Seg(0.0, 1.0, "fallback satiri")

        return _ok_gen(), _Info("tr", 0.91)


def test_transcribe_falls_back_when_segment_generator_crashes(monkeypatch, tmp_path):
    """Generator iterasyonu patlarsa word_timestamps=False fallback devreye girmeli."""
    # faster_whisper modülünü sahteleyelim
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    # GPU/VRAM yan etkilerini izole et
    monkeypatch.setattr("audio.stages.transcribe.VRAMManager.get_device", lambda: "cpu")
    monkeypatch.setattr("audio.stages.transcribe.VRAMManager.release", lambda: None)

    stage = TranscribeStage(log_cb=lambda *_args, **_kwargs: None)
    wav = tmp_path / "dummy.wav"
    wav.write_bytes(b"RIFF....WAVE")

    result = stage._transcribe(
        audio_path=str(wav),
        opts={"whisper_model": "large-v3", "whisper_language": "tr", "beam_size": 1},
        diarization=[],
    )

    assert result["status"] == "ok"
    assert result["total_segments"] == 1
    assert result["segments"][0]["text"] == "fallback satiri"
    assert result["segments"][0]["words"] == []
