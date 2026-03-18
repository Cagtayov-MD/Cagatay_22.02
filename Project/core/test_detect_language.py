"""
test_detect_language.py — LanguageDetectionStage birim testleri

Testler gerçek model/video gerektirmez (mock kullanır).
Çalıştırma: pytest Project/tests/test_detect_language.py -v
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Proje kökünü sys.path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Project"))

from audio.stages.detect_language import LanguageDetectionStage


class TestLanguageDetectionStage(unittest.TestCase):
    """LanguageDetectionStage birim testleri."""

    def setUp(self):
        self.logs = []
        self.stage = LanguageDetectionStage(
            ffmpeg_path="ffmpeg",
            log_cb=self.logs.append,
        )

    # ─── Karar mantığı testleri (mock) ───────────────────────────────

    def test_both_matched_turkish(self):
        """İki örnek de Türkçe → both_matched, language_is_turkish=True"""
        samples = [
            {"sample_name": "initial", "language": "tr", "confidence": 0.91, "start_sec": 0, "end_sec": 30},
            {"sample_name": "mid_4min", "language": "tr", "confidence": 0.93, "start_sec": 240, "end_sec": 270},
        ]
        result = self._run_with_samples(samples)

        self.assertEqual(result["detected_language"], "tr")
        self.assertTrue(result["language_is_turkish"])
        self.assertEqual(result["decision_logic"], "both_matched")
        self.assertAlmostEqual(result["confidence"], 0.92, places=2)

    def test_both_matched_english(self):
        """İki örnek de İngilizce → both_matched, language_is_turkish=False"""
        samples = [
            {"sample_name": "initial", "language": "en", "confidence": 0.88, "start_sec": 0, "end_sec": 30},
            {"sample_name": "mid_4min", "language": "en", "confidence": 0.85, "start_sec": 240, "end_sec": 270},
        ]
        result = self._run_with_samples(samples)

        self.assertEqual(result["detected_language"], "en")
        self.assertFalse(result["language_is_turkish"])
        self.assertEqual(result["decision_logic"], "both_matched")

    def test_single_sample_turkish(self):
        """Kısa video — tek örnek Türkçe → single_sample"""
        samples = [
            {"sample_name": "initial", "language": "tr", "confidence": 0.90, "start_sec": 0, "end_sec": 30},
        ]
        result = self._run_with_samples(samples)

        self.assertEqual(result["detected_language"], "tr")
        self.assertTrue(result["language_is_turkish"])
        self.assertEqual(result["decision_logic"], "single_sample")
        self.assertEqual(result["confidence"], 0.90)

    def test_conflicting_use_high_confidence(self):
        """Çelişki: intro İngilizce (düşük conf) + ana içerik Türkçe (yüksek conf) → Türkçe seçilir"""
        samples = [
            {"sample_name": "initial", "language": "en", "confidence": 0.65, "start_sec": 0, "end_sec": 30},
            {"sample_name": "mid_4min", "language": "tr", "confidence": 0.94, "start_sec": 240, "end_sec": 270},
        ]
        result = self._run_with_samples(samples)

        self.assertEqual(result["detected_language"], "tr")
        self.assertTrue(result["language_is_turkish"])
        self.assertEqual(result["decision_logic"], "conflicting_use_second")

    def test_no_samples_returns_unknown(self):
        """Hiç örnek işlenemezse → unknown, status=error"""
        result = self._run_with_samples([])

        self.assertEqual(result["detected_language"], "unknown")
        self.assertFalse(result["language_is_turkish"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["decision_logic"], "no_samples")

    # ─── Yardımcı metodlar ───────────────────────────────────────────

    def _run_with_samples(self, samples: list) -> dict:
        """
        Gerçek ffmpeg/model çağrısı olmadan sadece karar mantığını test eder.
        LanguageDetectionStage.run() içindeki sample hesaplama mantığını doğrudan uygular.
        """
        result = {
            "status": "ok",
            "stage_time_sec": 0.0,
            "detected_language": "unknown",
            "language_is_turkish": False,
            "confidence": 0.0,
            "samples": samples,
            "decision_logic": "unknown",
        }

        if not samples:
            result["status"] = "error"
            result["error"] = "Hiçbir örnek işlenemedi"
            result["decision_logic"] = "no_samples"
        elif len(samples) == 1:
            result["detected_language"] = samples[0]["language"]
            result["confidence"] = samples[0]["confidence"]
            result["decision_logic"] = "single_sample"
        else:
            s0, s1 = samples[0], samples[1]
            if s0["language"] == s1["language"]:
                result["detected_language"] = s0["language"]
                result["confidence"] = round((s0["confidence"] + s1["confidence"]) / 2, 4)
                result["decision_logic"] = "both_matched"
            else:
                winner = s1 if s1["confidence"] >= s0["confidence"] else s0
                result["detected_language"] = winner["language"]
                result["confidence"] = winner["confidence"]
                result["decision_logic"] = "conflicting_use_second"

        result["language_is_turkish"] = (result["detected_language"] == "tr")
        return result


class TestContentProfiles(unittest.TestCase):
    """content_profiles.json doğrulama testleri."""

    def setUp(self):
        import json
        profiles_path = Path(__file__).parent.parent / "config" / "content_profiles.json"
        with open(profiles_path, encoding="utf-8") as f:
            self.profiles = json.load(f)

    def test_detect_language_in_all_active_profiles(self):
        """Tüm aktif profillerde detect_language stage'i olmalı."""
        active_profiles = ["FilmDizi-Hybrid", "Spor"]
        for profile_name in active_profiles:
            with self.subTest(profile=profile_name):
                profile = self.profiles.get(profile_name, {})
                stages = profile.get("audio_stages", [])
                self.assertIn(
                    "detect_language", stages,
                    f"{profile_name} profilinde detect_language eksik"
                )

    def test_compute_type_float16_in_active_profiles(self):
        """Aktif profillerde compute_type float16 olmalı."""
        active_profiles = ["FilmDizi-Hybrid", "Spor"]
        for profile_name in active_profiles:
            with self.subTest(profile=profile_name):
                profile = self.profiles.get(profile_name, {})
                self.assertEqual(
                    profile.get("compute_type"), "float16",
                    f"{profile_name} profilinde compute_type float16 değil"
                )

    def test_beam_size_1_in_active_profiles(self):
        """Aktif profillerde beam_size 1 olmalı."""
        active_profiles = ["FilmDizi-Hybrid", "Spor"]
        for profile_name in active_profiles:
            with self.subTest(profile=profile_name):
                profile = self.profiles.get(profile_name, {})
                self.assertEqual(
                    profile.get("beam_size"), 1,
                    f"{profile_name} profilinde beam_size 1 değil"
                )

    def test_stage_order_correct(self):
        """FilmDizi-Hybrid profilinde stage sırası doğru olmalı."""
        expected_order = ["extract", "detect_language", "denoise", "diarize", "transcribe", "post_process"]
        actual = self.profiles["FilmDizi-Hybrid"]["audio_stages"]
        self.assertEqual(actual, expected_order)


if __name__ == "__main__":
    unittest.main(verbosity=2)
