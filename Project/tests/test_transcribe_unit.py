"""
TranscribeStage birim testleri.
Gerçek Whisper modeli yüklenmeden çalışır (mock).
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure Project/ directory is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestTranscribeStageUnit(unittest.TestCase):
    """TranscribeStage'in hata senaryolarını test eder."""

    def setUp(self):
        # Reset cached model before each test
        from audio.stages.transcribe import TranscribeStage
        TranscribeStage._cached_model = None
        TranscribeStage._cached_model_key = None

    def test_missing_audio_path_returns_error(self):
        """Audio path olmadan çağrıldığında error döner."""
        from audio.stages.transcribe import TranscribeStage
        stage = TranscribeStage()
        result = stage._transcribe(audio_path=None, opts={}, diarization=None)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "missing_audio_path")
        self.assertEqual(result["segments"], [])

    def test_empty_audio_path_returns_error(self):
        """Boş audio path ile çağrıldığında error döner."""
        from audio.stages.transcribe import TranscribeStage
        stage = TranscribeStage()
        result = stage._transcribe(audio_path="", opts={}, diarization=None)
        self.assertEqual(result["status"], "error")

    def test_assign_speakers_with_none(self):
        """None diarization ile çökme olmaz."""
        from audio.stages.transcribe import TranscribeStage
        stage = TranscribeStage()
        segments = [{"start": 0.0, "end": 1.0, "text": "test", "speaker": ""}]
        stage._assign_speakers(segments, None)
        self.assertEqual(segments[0]["speaker"], "")

    def test_assign_speakers_with_dict(self):
        """Dict diarization (yanlış format) ile çökme olmaz."""
        from audio.stages.transcribe import TranscribeStage
        stage = TranscribeStage()
        segments = [{"start": 0.0, "end": 1.0, "text": "test", "speaker": ""}]
        stage._assign_speakers(segments, {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPK_00"}]})
        self.assertEqual(segments[0]["speaker"], "SPK_00")

    def test_assign_speakers_overlap(self):
        """Overlap-based speaker assignment doğru çalışır."""
        from audio.stages.transcribe import TranscribeStage
        stage = TranscribeStage()
        segments = [{"start": 0.0, "end": 5.0, "text": "test", "speaker": ""}]
        diar = [
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 1.0, "end": 5.0, "speaker": "B"},
        ]
        stage._assign_speakers(segments, diar)
        # B has more overlap (4s vs 2s)
        self.assertEqual(segments[0]["speaker"], "B")

    def test_resolve_compute_type_cpu(self):
        """CPU device ile float16 -> int8 dönüşümü."""
        from audio.stages.transcribe import TranscribeStage
        stage = TranscribeStage()
        self.assertEqual(stage._resolve_compute_type("float16", "cpu"), "int8")
        self.assertEqual(stage._resolve_compute_type(None, "cpu"), "int8")

    def test_resolve_compute_type_cuda(self):
        """CUDA device ile float16 korunur."""
        from audio.stages.transcribe import TranscribeStage
        stage = TranscribeStage()
        self.assertEqual(stage._resolve_compute_type(None, "cuda"), "float16")
        self.assertEqual(stage._resolve_compute_type("float16", "cuda"), "float16")

    def test_run_dispatches_context_mode(self):
        """run() iki dict ile çağrıldığında _run_from_context'e yönlendirir."""
        from audio.stages.transcribe import TranscribeStage
        stage = TranscribeStage()
        # Context with no audio path -> error
        result = stage.run({}, {"options": {}})
        self.assertEqual(result["status"], "error")

    def test_run_dispatches_legacy_mode(self):
        """run() string ile çağrıldığında _run_legacy'ye yönlendirir."""
        from audio.stages.transcribe import TranscribeStage
        stage = TranscribeStage()
        result = stage.run(None)  # No audio path
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()
