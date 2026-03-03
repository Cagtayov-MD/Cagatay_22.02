"""
TranscribeStage unit testleri.
faster-whisper gerektirmez — sadece yardimci fonksiyonlari test eder.
"""
import sys
import os
import unittest

# Proje kokunu path'e ekle
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestAssignSpeakers(unittest.TestCase):
    """_assign_speakers metodunun dogru calistigini test eder."""

    def _make_stage(self):
        from audio.stages.transcribe import TranscribeStage
        return TranscribeStage(log_cb=lambda *a, **kw: None)

    def test_basic_speaker_assignment(self):
        stage = self._make_stage()
        segments = [{"start": 0.0, "end": 2.0, "text": "Merhaba", "speaker": ""}]
        diarization = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
        stage._assign_speakers(segments, diarization)
        self.assertEqual(segments[0]["speaker"], "SPEAKER_00")

    def test_empty_diarization(self):
        stage = self._make_stage()
        segments = [{"start": 0.0, "end": 2.0, "text": "Test", "speaker": ""}]
        stage._assign_speakers(segments, [])
        self.assertEqual(segments[0]["speaker"], "")

    def test_overlapping_speakers(self):
        stage = self._make_stage()
        # Segment 1.0-3.0, iki konusmaci: 0-2 ve 1-4; ikincisi daha fazla overlap
        segments = [{"start": 1.0, "end": 3.0, "text": "Test", "speaker": ""}]
        diarization = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},  # overlap: 1.0s
            {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_01"},  # overlap: 2.0s
        ]
        stage._assign_speakers(segments, diarization)
        self.assertEqual(segments[0]["speaker"], "SPEAKER_01")

    def test_none_diarization(self):
        stage = self._make_stage()
        segments = [{"start": 0.0, "end": 2.0, "text": "Test", "speaker": ""}]
        # None gecilince hata vermemeli
        stage._assign_speakers(segments, None)
        self.assertEqual(segments[0]["speaker"], "")

    def test_dict_diarization_fallback(self):
        stage = self._make_stage()
        segments = [{"start": 0.0, "end": 2.0, "text": "Test", "speaker": ""}]
        # Dict gecilince segments listesini cikarip kullanmali
        diarization = {"segments": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]}
        stage._assign_speakers(segments, diarization)
        self.assertEqual(segments[0]["speaker"], "SPEAKER_00")


class TestResolveComputeType(unittest.TestCase):

    def _make_stage(self):
        from audio.stages.transcribe import TranscribeStage
        return TranscribeStage(log_cb=lambda *a, **kw: None)

    def test_cuda_default(self):
        stage = self._make_stage()
        self.assertEqual(stage._resolve_compute_type(None, "cuda"), "float16")

    def test_cpu_default(self):
        stage = self._make_stage()
        self.assertEqual(stage._resolve_compute_type(None, "cpu"), "int8")

    def test_cpu_float16_fallback(self):
        stage = self._make_stage()
        self.assertEqual(stage._resolve_compute_type("float16", "cpu"), "int8")

    def test_auto_value(self):
        stage = self._make_stage()
        self.assertEqual(stage._resolve_compute_type("auto", "cuda"), "float16")
        self.assertEqual(stage._resolve_compute_type("auto", "cpu"), "int8")

    def test_valid_value_kept(self):
        stage = self._make_stage()
        self.assertEqual(stage._resolve_compute_type("int8", "cpu"), "int8")
        self.assertEqual(stage._resolve_compute_type("int8_float16", "cuda"), "int8_float16")


class TestFmtHms(unittest.TestCase):

    def _fmt(self, seconds, with_ms=True):
        from utils.time_utils import fmt_hms
        return fmt_hms(seconds, with_ms=with_ms)

    def test_zero(self):
        self.assertEqual(self._fmt(0.0), "00:00:00.000")
        self.assertEqual(self._fmt(0.0, with_ms=False), "00:00:00")

    def test_one_hour(self):
        self.assertEqual(self._fmt(3600.0), "01:00:00.000")
        self.assertEqual(self._fmt(3600.0, with_ms=False), "01:00:00")

    def test_milliseconds(self):
        self.assertEqual(self._fmt(1.5), "00:00:01.500")

    def test_negative_clamped_to_zero(self):
        self.assertEqual(self._fmt(-5.0), "00:00:00.000")

    def test_complex_value(self):
        # 1h 2m 3.456s
        secs = 3600 + 2 * 60 + 3.456
        self.assertEqual(self._fmt(secs), "01:02:03.456")


if __name__ == "__main__":
    unittest.main()
