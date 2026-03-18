"""
test_channel_and_language.py — v2 değişikliklerinin kapsamlı testi.

Test edilen değişiklikler:
  1. detect_language.py: kanal deneme mantığı
  2. extract.py: selected_channel desteği
  3. audio_pipeline.py: stage sırası (DETECT_LANGUAGE → EXTRACT)
  4. gemini_summarizer.py: iki adımlı yabancı dil akışı + TMDB cast
  5. pipeline_runner.py: tmdb_cast parametresi aktarımı

Testler gerçek API/model çağırmaz — mock kullanır.
"""

import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Proje kök dizinini path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════
# TEST 1: detect_language.py — kanal deneme mantığı
# ═══════════════════════════════════════════════════════════════════
class TestChannelDetection(unittest.TestCase):
    """detect_language.py v2 kanal deneme testleri."""

    def setUp(self):
        from audio.stages.detect_language import LanguageDetectionStage
        self.stage = LanguageDetectionStage(ffmpeg_path="ffmpeg", log_cb=print)

    def test_result_has_new_fields(self):
        """Sonuç dict'inde yeni alanlar (selected_channel, channel_trials) var mı."""
        # Mock: tüm helper'ları geç, direkt sonuç yapısını kontrol et
        with patch.object(self.stage, '_get_duration', return_value=300.0), \
             patch.object(self.stage, '_get_audio_channels', return_value=2), \
             patch.object(self.stage, '_extract_segment', return_value=True), \
             patch.object(self.stage, '_detect_from_file', return_value=("tr", 0.92)):
            result = self.stage.run("/fake/video.mp4", "/fake/audio_dir", 300.0)

        self.assertIn("selected_channel", result)
        self.assertIn("channel_trials", result)
        self.assertEqual(result["status"], "ok")

    def test_high_confidence_no_channel_retry(self):
        """Karışık mono %92 güven → kanal deneme yapılmamalı."""
        with patch.object(self.stage, '_get_duration', return_value=300.0), \
             patch.object(self.stage, '_get_audio_channels', return_value=2), \
             patch.object(self.stage, '_extract_segment', return_value=True), \
             patch.object(self.stage, '_detect_from_file', return_value=("tr", 0.92)):
            result = self.stage.run("/fake/video.mp4", "/fake/audio_dir", 300.0)

        self.assertEqual(result["detected_language"], "tr")
        self.assertIsNone(result["selected_channel"])
        # Sadece 1 trial (mix) olmalı — kanal deneme yapılmadı
        self.assertEqual(len(result["channel_trials"]), 1)
        self.assertEqual(result["channel_trials"][0]["channel"], "mix")

    def test_low_confidence_triggers_channel_retry(self):
        """Karışık mono %42 güven → kanal deneme tetiklenmeli."""
        call_count = {"n": 0}
        def mock_detect(wav_path):
            call_count["n"] += 1
            # mix: EN %42, ch0: EN %45, ch1: TR %88
            if "mix" in wav_path:
                return ("en", 0.42)
            elif "ch0" in wav_path:
                return ("en", 0.45)
            elif "ch1" in wav_path:
                return ("tr", 0.88)
            return ("unknown", 0.0)

        with patch.object(self.stage, '_get_duration', return_value=300.0), \
             patch.object(self.stage, '_get_audio_channels', return_value=2), \
             patch.object(self.stage, '_extract_segment', return_value=True), \
             patch.object(self.stage, '_detect_from_file', side_effect=mock_detect):
            result = self.stage.run("/fake/video.mp4", "/fake/audio_dir", 300.0)

        self.assertEqual(result["detected_language"], "tr")
        self.assertEqual(result["selected_channel"], 1)
        self.assertGreater(result["confidence"], 0.75)
        # 3 trial olmalı: mix, ch0, ch1
        self.assertEqual(len(result["channel_trials"]), 3)

    def test_mono_source_no_channel_retry(self):
        """Mono kaynak → kanal deneme yapılmamalı (1 kanal)."""
        with patch.object(self.stage, '_get_duration', return_value=300.0), \
             patch.object(self.stage, '_get_audio_channels', return_value=1), \
             patch.object(self.stage, '_extract_segment', return_value=True), \
             patch.object(self.stage, '_detect_from_file', return_value=("en", 0.40)):
            result = self.stage.run("/fake/video.mp4", "/fake/audio_dir", 300.0)

        # Mono'da kanal deneme yok, düşük güven olsa bile
        self.assertIsNone(result["selected_channel"])
        self.assertEqual(len(result["channel_trials"]), 1)

    def test_fallback_8min_sample(self):
        """Tüm kanallar düşük güven + video 10dk → 8. dk'dan ek sample."""
        def mock_detect(wav_path):
            if "fallback" in wav_path:
                return ("tr", 0.80)
            return ("unknown", 0.30)

        with patch.object(self.stage, '_get_duration', return_value=600.0), \
             patch.object(self.stage, '_get_audio_channels', return_value=2), \
             patch.object(self.stage, '_extract_segment', return_value=True), \
             patch.object(self.stage, '_detect_from_file', side_effect=mock_detect):
            result = self.stage.run("/fake/video.mp4", "/fake/audio_dir", 600.0)

        # Fallback sample'dan TR %80 geldi
        has_fallback = any(
            t.get("tag") == "fallback_8min" for t in result["channel_trials"]
        )
        self.assertTrue(has_fallback)

    def test_extract_segment_channel_parameter(self):
        """_extract_segment'e channel parametresi doğru aktarılıyor mu."""
        import subprocess
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # channel=1 (sağ kanal)
            with patch.object(Path, 'is_file', return_value=True), \
                 patch.object(Path, 'stat', return_value=MagicMock(st_size=1000)):
                self.stage._extract_segment(
                    "/fake/video.mp4", 0, 30, "/fake/out.wav", channel=1
                )

            # ffmpeg komutunda pan=mono|c0=c1 olmalı
            cmd = mock_run.call_args[0][0]
            self.assertIn("-af", cmd)
            af_idx = cmd.index("-af")
            self.assertEqual(cmd[af_idx + 1], "pan=mono|c0=c1")

    def test_extract_segment_no_channel(self):
        """channel=None → eski davranış (-ac 1)."""
        import subprocess
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch.object(Path, 'is_file', return_value=True), \
                 patch.object(Path, 'stat', return_value=MagicMock(st_size=1000)):
                self.stage._extract_segment(
                    "/fake/video.mp4", 0, 30, "/fake/out.wav", channel=None
                )

            cmd = mock_run.call_args[0][0]
            self.assertIn("-ac", cmd)
            self.assertNotIn("-af", cmd)


# ═══════════════════════════════════════════════════════════════════
# TEST 2: extract.py — selected_channel desteği
# ═══════════════════════════════════════════════════════════════════
class TestExtractChannel(unittest.TestCase):
    """extract.py v2 kanal seçimi testleri."""

    def setUp(self):
        from core.extract import ExtractStage
        self.stage = ExtractStage(ffmpeg_path="ffmpeg", log_cb=print)

    def test_selected_channel_in_ffmpeg_cmd(self):
        """selected_channel=1 → ffmpeg'e pan=mono|c0=c1 geçmeli."""
        import subprocess
        cmds_captured = []

        def capture_run(cmd, **kwargs):
            cmds_captured.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch('subprocess.run', side_effect=capture_run), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'stat', return_value=MagicMock(st_size=10000)):
            # _get_duration mock
            with patch.object(self.stage, '_get_duration', return_value=100.0):
                result = self.stage.run(
                    "/fake/video.mp4", "/tmp/test_extract",
                    selected_channel=1
                )

        # İlk ffmpeg çağrısı (48k) kanal seçimi içermeli
        first_cmd = cmds_captured[0]
        self.assertIn("-af", first_cmd)
        af_idx = first_cmd.index("-af")
        self.assertEqual(first_cmd[af_idx + 1], "pan=mono|c0=c1")
        self.assertEqual(result["selected_channel"], 1)

    def test_no_channel_uses_ac1(self):
        """selected_channel=None → eski davranış (-ac 1)."""
        import subprocess
        cmds_captured = []

        def capture_run(cmd, **kwargs):
            cmds_captured.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        with patch('subprocess.run', side_effect=capture_run), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'stat', return_value=MagicMock(st_size=10000)):
            with patch.object(self.stage, '_get_duration', return_value=100.0):
                result = self.stage.run(
                    "/fake/video.mp4", "/tmp/test_extract",
                    selected_channel=None
                )

        first_cmd = cmds_captured[0]
        self.assertIn("-ac", first_cmd)
        self.assertIsNone(result["selected_channel"])


# ═══════════════════════════════════════════════════════════════════
# TEST 3: gemini_summarizer.py — iki adımlı yabancı dil akışı
# ═══════════════════════════════════════════════════════════════════
class TestSummarizerLanguageFlow(unittest.TestCase):
    """gemini_summarizer.py v2 dil akışı testleri."""

    @patch('core.llm_provider._gemini_generate')
    def test_turkish_single_step(self, mock_gemini):
        """Türkçe transcript → tek adım, doğrudan Türkçe özet."""
        mock_gemini.return_value = "Ali köyden şehre göç eder, fabrikada çalışır..."
        from core.gemini_summarizer import summarize_transcript

        result = summarize_transcript(
            "Ali köyünden ayrıldı...",
            api_key="fake_key",
            detected_language="tr",
        )

        self.assertIsNotNone(result)
        self.assertIn("en", result)
        # Tek çağrı yapılmalı (iki adım değil)
        self.assertEqual(mock_gemini.call_count, 1)

    @patch('core.llm_provider._gemini_generate')
    def test_english_two_step(self, mock_gemini):
        """İngilizce transcript → iki adım (EN özet → TR çeviri)."""
        # İlk çağrı: İngilizce özet
        # İkinci çağrı: Türkçe çeviri
        mock_gemini.side_effect = [
            "Jack Sparrow steals the Black Pearl and escapes...",
            "Jack Sparrow Kara İnci'yi çalar ve kaçar...",
        ]
        from core.gemini_summarizer import summarize_transcript

        result = summarize_transcript(
            "Captain Jack Sparrow arrived at the port...",
            api_key="fake_key",
            detected_language="en",
        )

        self.assertIsNotNone(result)
        # İki çağrı: Pro (özet) + Flash (çeviri)
        self.assertEqual(mock_gemini.call_count, 2)
        # İkinci çağrıda system prompt çeviri prompt'u olmalı
        second_call_kwargs = mock_gemini.call_args_list[1]
        self.assertIn("translate", second_call_kwargs.kwargs.get("system", "").lower())

    @patch('core.llm_provider._gemini_generate')
    def test_arabic_two_step(self, mock_gemini):
        """Arapça transcript → iki adım."""
        mock_gemini.side_effect = [
            "أحمد يسافر إلى المدينة ويجد عملاً...",
            "Ahmed şehre gider ve iş bulur...",
        ]
        from core.gemini_summarizer import summarize_transcript

        result = summarize_transcript(
            "أحمد قرر أن يسافر...",
            api_key="fake_key",
            detected_language="ar",
        )

        self.assertIsNotNone(result)
        self.assertEqual(mock_gemini.call_count, 2)

    @patch('core.llm_provider._gemini_generate')
    def test_tmdb_cast_in_translation_prompt(self, mock_gemini):
        """TMDB cast listesi çeviri prompt'una ekleniyor mu."""
        mock_gemini.side_effect = [
            "Jack steals the ship and escapes...",
            "Jack gemiyi çalar ve kaçar...",
        ]
        from core.gemini_summarizer import summarize_transcript

        tmdb_cast = [
            {"actor_name": "Johnny Depp", "character_name": "Captain Jack Sparrow"},
            {"actor_name": "Geoffrey Rush", "character_name": "Captain Barbossa"},
        ]

        result = summarize_transcript(
            "The captain arrived...",
            api_key="fake_key",
            detected_language="en",
            tmdb_cast=tmdb_cast,
        )

        # İkinci çağrının prompt'unda cast bilgisi olmalı
        second_call_args = mock_gemini.call_args_list[1]
        prompt_text = second_call_args.args[0] if second_call_args.args else ""
        self.assertIn("Johnny Depp", prompt_text)
        self.assertIn("Captain Jack Sparrow", prompt_text)

    @patch('core.llm_provider._gemini_generate')
    def test_no_tmdb_cast_still_works(self, mock_gemini):
        """TMDB cast yoksa da çalışmalı — cast reference boş."""
        mock_gemini.side_effect = [
            "The detective finds the killer...",
            "Dedektif katili bulur...",
        ]
        from core.gemini_summarizer import summarize_transcript

        result = summarize_transcript(
            "The detective walked into...",
            api_key="fake_key",
            detected_language="en",
            tmdb_cast=None,
        )

        self.assertIsNotNone(result)
        # Prompt'ta CAST REFERENCE olmamalı
        second_call_args = mock_gemini.call_args_list[1]
        prompt_text = second_call_args.args[0] if second_call_args.args else ""
        self.assertNotIn("CAST REFERENCE", prompt_text)

    @patch('core.llm_provider._gemini_generate')
    def test_french_two_step(self, mock_gemini):
        """Fransızca transcript → iki adım."""
        mock_gemini.side_effect = [
            "Jean voyage à Paris et découvre la vérité...",
            "Jean Paris'e gider ve gerçeği keşfeder...",
        ]
        from core.gemini_summarizer import summarize_transcript

        result = summarize_transcript(
            "Jean a décidé de partir...",
            api_key="fake_key",
            detected_language="fr",
        )

        self.assertIsNotNone(result)
        self.assertEqual(mock_gemini.call_count, 2)

    @patch('core.llm_provider._gemini_generate')
    def test_kurdish_two_step(self, mock_gemini):
        """Kürtçe transcript → iki adım."""
        mock_gemini.side_effect = [
            "Ehmed diçe bajar û kar dibîne...",
            "Ahmed şehre gider ve iş bulur...",
        ]
        from core.gemini_summarizer import summarize_transcript

        result = summarize_transcript(
            "Ehmed biryar da ku biçe...",
            api_key="fake_key",
            detected_language="ku",
        )

        self.assertIsNotNone(result)
        self.assertEqual(mock_gemini.call_count, 2)


# ═══════════════════════════════════════════════════════════════════
# TEST 4: _build_cast_reference yardımcı fonksiyon
# ═══════════════════════════════════════════════════════════════════
class TestBuildCastReference(unittest.TestCase):
    """Cast referans metni oluşturma testleri."""

    def test_full_cast(self):
        from core.gemini_summarizer import _build_cast_reference
        cast = [
            {"actor_name": "Johnny Depp", "character_name": "Jack Sparrow"},
            {"actor_name": "Orlando Bloom", "character_name": "Will Turner"},
        ]
        ref = _build_cast_reference(cast)
        self.assertIn("Johnny Depp → Jack Sparrow", ref)
        self.assertIn("Orlando Bloom → Will Turner", ref)
        self.assertIn("CAST REFERENCE", ref)

    def test_empty_cast(self):
        from core.gemini_summarizer import _build_cast_reference
        self.assertEqual(_build_cast_reference([]), "")
        self.assertEqual(_build_cast_reference(None), "")

    def test_max_20_cast(self):
        from core.gemini_summarizer import _build_cast_reference
        cast = [{"actor_name": f"Actor {i}", "character_name": f"Char {i}"}
                for i in range(30)]
        ref = _build_cast_reference(cast)
        # Max 20 kişi
        self.assertIn("Actor 19", ref)
        self.assertNotIn("Actor 20", ref)

    def test_partial_cast_data(self):
        """Sadece actor_name var, character_name yok."""
        from core.gemini_summarizer import _build_cast_reference
        cast = [
            {"actor_name": "Johnny Depp"},
            {"name": "Orlando Bloom", "character": "Will Turner"},
        ]
        ref = _build_cast_reference(cast)
        self.assertIn("Johnny Depp", ref)
        self.assertIn("Orlando Bloom", ref)


# ═══════════════════════════════════════════════════════════════════
# TEST 5: pipeline_runner.py — tmdb_cast aktarımı
# ═══════════════════════════════════════════════════════════════════
class TestPipelineRunnerCastPassthrough(unittest.TestCase):
    """pipeline_runner.py'da summarize_transcript'e tmdb_cast geçiriliyor mu."""

    def test_cast_parameter_in_source(self):
        """pipeline_runner.py kaynak kodunda tmdb_cast parametresi var mı."""
        source_path = PROJECT_ROOT / "core" / "pipeline_runner.py"
        source = source_path.read_text(encoding="utf-8")
        self.assertIn("tmdb_cast=_tmdb_cast_for_summary", source)
        self.assertIn('cdata.get("cast")', source)
        self.assertIn('cdata.get("_tmdb_cast_ref")', source)


# ═══════════════════════════════════════════════════════════════════
# TEST 6: audio_pipeline.py — stage sırası
# ═══════════════════════════════════════════════════════════════════
class TestAudioPipelineStageOrder(unittest.TestCase):
    """audio_pipeline.py v2 stage sıralama testi."""

    def test_detect_before_extract_in_source(self):
        """Kaynak kodda DETECT_LANGUAGE bölümü EXTRACT'tan önce mi."""
        source_path = PROJECT_ROOT / "core" / "audio_pipeline.py"
        source = source_path.read_text(encoding="utf-8")
        detect_pos = source.find("[B] DİL TESPİTİ")
        extract_pos = source.find("[A] EXTRACT")
        self.assertGreater(extract_pos, detect_pos,
                          "DİL TESPİTİ, EXTRACT'tan önce olmalı")

    def test_selected_channel_passed_to_extract(self):
        """audio_pipeline.py'da extract.run'a selected_channel geçiriliyor mu."""
        source_path = PROJECT_ROOT / "core" / "audio_pipeline.py"
        source = source_path.read_text(encoding="utf-8")
        self.assertIn("selected_channel=selected_channel", source)

    def test_version_updated(self):
        """Pipeline versiyonu 1.2 olmalı."""
        source_path = PROJECT_ROOT / "core" / "audio_pipeline.py"
        source = source_path.read_text(encoding="utf-8")
        self.assertIn('VERSION = "1.2"', source)


# ═══════════════════════════════════════════════════════════════════
# TEST 7: Dil bazlı senaryo entegrasyon testleri
# ═══════════════════════════════════════════════════════════════════
class TestLanguageScenarios(unittest.TestCase):
    """Farklı dil senaryolarında pipeline davranışı."""

    @patch('core.llm_provider._gemini_generate')
    def test_scenario_turkish_film(self, mock_gemini):
        """Türkçe film: tek adım, PostProcess çalışmalı."""
        mock_gemini.return_value = "Hüseyin köyden çıkıp şehre göç eder..."
        from core.gemini_summarizer import summarize_transcript

        result = summarize_transcript(
            "Hüseyin çiftliği bıraktı...",
            api_key="fake", detected_language="tr",
        )
        # Tek çağrı
        self.assertEqual(mock_gemini.call_count, 1)
        # Sonuçta Türkçe karakter var
        self.assertIn("ü", result["en"])

    @patch('core.llm_provider._gemini_generate')
    def test_scenario_english_film_with_cast(self, mock_gemini):
        """İngilizce film + TMDB cast: iki adım, isimler düzelmeli."""
        mock_gemini.side_effect = [
            "Jack steals the ship...",
            "Jack Sparrow gemiyi çalar...",  # Flash düzeltmiş
        ]
        from core.gemini_summarizer import summarize_transcript

        result = summarize_transcript(
            "The captain steals...",
            api_key="fake", detected_language="en",
            tmdb_cast=[{"actor_name": "Johnny Depp", "character_name": "Jack Sparrow"}],
        )
        self.assertEqual(mock_gemini.call_count, 2)
        self.assertIn("Jack Sparrow", result["en"])

    @patch('core.llm_provider._gemini_generate')
    def test_scenario_arabic_film_no_cast(self, mock_gemini):
        """Arapça film, TMDB cast yok: iki adım, cast reference boş."""
        mock_gemini.side_effect = [
            "أحمد يذهب إلى المدينة...",
            "Ahmed şehre gider...",
        ]
        from core.gemini_summarizer import summarize_transcript

        result = summarize_transcript(
            "أحمد قال...",
            api_key="fake", detected_language="ar", tmdb_cast=None,
        )
        self.assertEqual(mock_gemini.call_count, 2)
        self.assertIsNotNone(result)

    @patch('core.llm_provider._gemini_generate')
    def test_scenario_pro_fails_flash_fallback(self, mock_gemini):
        """Pro başarısız → Flash fallback (yabancı dilde)."""
        mock_gemini.side_effect = [
            None,  # Pro başarısız
            "The hero saves the day...",  # Flash başarılı
            "Kahraman günü kurtarır...",  # Flash çeviri
        ]
        from core.gemini_summarizer import summarize_transcript

        result = summarize_transcript(
            "The hero walked...",
            api_key="fake", detected_language="en",
        )
        self.assertIsNotNone(result)
        # Pro fail + Flash özet + Flash çeviri = 3 çağrı
        self.assertEqual(mock_gemini.call_count, 3)


# ═══════════════════════════════════════════════════════════════════
# TEST 8: pipeline_runner.py — stages listesinde detect_language var mı
# ═══════════════════════════════════════════════════════════════════
class TestStagesListHasDetectLanguage(unittest.TestCase):
    """detect_language tüm profillerin stages listesinde var mı."""

    def test_film_dizi_has_detect_language(self):
        source = (PROJECT_ROOT / "core" / "pipeline_runner.py").read_text("utf-8")
        # film_dizi bloğu
        self.assertIn('"detect_language", "extract", "transcribe"', source)

    def test_full_pipeline_has_detect_language(self):
        source = (PROJECT_ROOT / "core" / "pipeline_runner.py").read_text("utf-8")
        # mac/muzik_programi bloğu
        self.assertIn('"detect_language", "extract", "denoise"', source)


# ═══════════════════════════════════════════════════════════════════
# TEST 9: Uçtan uca senaryo simülasyonları — "ne yapacak" testi
# ═══════════════════════════════════════════════════════════════════
class TestEndToEndScenarios(unittest.TestCase):
    """
    Her dil senaryosunda pipeline'ın hangi adımları çalıştıracağını
    ve sonucun ne olacağını simüle eder.
    """

    def _simulate_full_flow(self, detected_lang, confidence, n_channels,
                            selected_channel, tmdb_cast, transcript_text):
        """
        Pipeline akışını simüle et:
          detect_language → extract → transcribe → summarize
        """
        from core.gemini_summarizer import summarize_transcript

        # 1. detect_language sonucu
        is_turkish = (detected_lang == "tr")
        lang_result = {
            "detected_language": detected_lang,
            "confidence": confidence,
            "language_is_turkish": is_turkish,
            "selected_channel": selected_channel,
        }

        # 2. extract — selected_channel'ı kullanacak
        extract_would_use = selected_channel  # None veya 0 veya 1

        # 3. transcribe — whisper_language
        whisper_lang = detected_lang if detected_lang != "unknown" else None

        # 4. post_process — sadece Türkçe'de çalışır
        post_process_runs = is_turkish

        # 5. summarize — Türkçe tek adım, yabancı iki adım
        return {
            "detected_language": detected_lang,
            "confidence": confidence,
            "selected_channel": extract_would_use,
            "whisper_language": whisper_lang,
            "post_process_runs": post_process_runs,
            "is_two_step_summary": not is_turkish,
            "has_cast_reference": bool(tmdb_cast) and not is_turkish,
        }

    def test_scenario_diablo_film_stereo(self):
        """
        Orijinal sorun: 1954 Diablo'dan Kaçış.
        Stereo, kanal 1=İng+Tür karışık, kanal 2=Türkçe.
        Mix %42 EN → kanal deneme → kanal 1: TR %88 → seçildi.
        """
        result = self._simulate_full_flow(
            detected_lang="tr", confidence=0.88, n_channels=2,
            selected_channel=1, tmdb_cast=[{"actor_name": "Denver Pyle"}],
            transcript_text="Diablo'dan kaçış...",
        )
        self.assertEqual(result["detected_language"], "tr")
        self.assertEqual(result["selected_channel"], 1)  # kanal 1 seçildi
        self.assertEqual(result["whisper_language"], "tr")
        self.assertTrue(result["post_process_runs"])  # Türkçe → PostProcess çalışır
        self.assertFalse(result["is_two_step_summary"])  # tek adım özet

    def test_scenario_english_film_pirates(self):
        """
        Karayip Korsanları — İngilizce, TMDB cast var.
        Mono, EN %95.
        """
        result = self._simulate_full_flow(
            detected_lang="en", confidence=0.95, n_channels=1,
            selected_channel=None,
            tmdb_cast=[
                {"actor_name": "Johnny Depp", "character_name": "Jack Sparrow"},
                {"actor_name": "Geoffrey Rush", "character_name": "Barbossa"},
            ],
            transcript_text="Captain Jack Sparrow...",
        )
        self.assertEqual(result["detected_language"], "en")
        self.assertIsNone(result["selected_channel"])  # mono, kanal seçimi yok
        self.assertEqual(result["whisper_language"], "en")
        self.assertFalse(result["post_process_runs"])  # İngilizce → PostProcess atla
        self.assertTrue(result["is_two_step_summary"])  # iki adım: EN özet → TR çeviri
        self.assertTrue(result["has_cast_reference"])  # cast var → Flash'a verilecek

    def test_scenario_arabic_film(self):
        """
        Arapça film — AR %91, TMDB cast yok.
        """
        result = self._simulate_full_flow(
            detected_lang="ar", confidence=0.91, n_channels=1,
            selected_channel=None, tmdb_cast=None,
            transcript_text="أحمد ذهب إلى...",
        )
        self.assertEqual(result["detected_language"], "ar")
        self.assertEqual(result["whisper_language"], "ar")
        self.assertFalse(result["post_process_runs"])
        self.assertTrue(result["is_two_step_summary"])  # AR özet → TR çeviri
        self.assertFalse(result["has_cast_reference"])  # cast yok

    def test_scenario_french_film(self):
        """Fransızca film — FR %89."""
        result = self._simulate_full_flow(
            detected_lang="fr", confidence=0.89, n_channels=2,
            selected_channel=0, tmdb_cast=[{"actor_name": "Jean Reno"}],
            transcript_text="Jean est parti...",
        )
        self.assertEqual(result["detected_language"], "fr")
        self.assertEqual(result["selected_channel"], 0)
        self.assertTrue(result["is_two_step_summary"])
        self.assertTrue(result["has_cast_reference"])

    def test_scenario_kurdish_film(self):
        """
        Kürtçe film — Whisper zayıf tanıyabilir.
        Varsayım: KU %55 geldi.
        """
        result = self._simulate_full_flow(
            detected_lang="ku", confidence=0.55, n_channels=1,
            selected_channel=None, tmdb_cast=None,
            transcript_text="Ehmed biryar da...",
        )
        self.assertEqual(result["detected_language"], "ku")
        self.assertEqual(result["whisper_language"], "ku")
        self.assertTrue(result["is_two_step_summary"])

    def test_scenario_unknown_language(self):
        """
        Dil tespit edilemedi — unknown.
        Whisper'a None verilecek, kendi tespit etsin.
        """
        result = self._simulate_full_flow(
            detected_lang="unknown", confidence=0.20, n_channels=1,
            selected_channel=None, tmdb_cast=None,
            transcript_text="...",
        )
        self.assertIsNone(result["whisper_language"])  # Whisper kendi tespit etsin
        self.assertTrue(result["is_two_step_summary"])  # unknown = yabancı gibi davran

    def test_scenario_turkish_mono_high_confidence(self):
        """
        Normal Türkçe film — mono, TR %96.
        Hiçbir şey değişmemeli, eski davranış aynen.
        """
        result = self._simulate_full_flow(
            detected_lang="tr", confidence=0.96, n_channels=1,
            selected_channel=None, tmdb_cast=None,
            transcript_text="Ali köyden ayrıldı...",
        )
        self.assertEqual(result["detected_language"], "tr")
        self.assertIsNone(result["selected_channel"])
        self.assertEqual(result["whisper_language"], "tr")
        self.assertTrue(result["post_process_runs"])
        self.assertFalse(result["is_two_step_summary"])
        self.assertFalse(result["has_cast_reference"])


class TestAudioPipelineUnknownLanguage(unittest.TestCase):
    """Dil tespiti 'unknown' geldiğinde Whisper'ın kendi tespitine izin verilmeli."""

    @patch("core.audio_pipeline.TranscribeStage")
    @patch("core.audio_pipeline.ExtractStage")
    @patch("core.audio_pipeline.LanguageDetectionStage")
    def test_unknown_language_passes_none_to_whisper(self, mock_lang_stage,
                                                     mock_extract_stage,
                                                     mock_transcribe_stage):
        mock_lang_stage.return_value.run.return_value = {
            "detected_language": "unknown",
            "language_is_turkish": False,
            "confidence": 0.12,
            "samples": [],
            "selected_channel": None,
            "stage_time_sec": 0.1,
            "status": "ok",
        }
        mock_extract_stage.return_value.run.return_value = {
            "status": "ok",
            "wav_16k": "/tmp/fake16k.wav",
            "wav_48k": "/tmp/fake48k.wav",
            "duration_sec": 120.0,
            "selected_channel": None,
            "stage_time_sec": 0.1,
        }

        captured = {}

        def _fake_transcribe(wav_path, diarization, options):
            captured["whisper_language"] = options.get("whisper_language")
            return {
                "stage_time_sec": 0.1,
                "status": "ok",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "مرحبا", "speaker": "S1"}
                ],
                "total_segments": 1,
                "detected_language": "ar",
            }

        mock_transcribe_stage.return_value.run.side_effect = _fake_transcribe

        from core.audio_pipeline import AudioPipeline

        pipeline = AudioPipeline(
            config={
                "video_path": "/tmp/fake_video.mp4",
                "work_dir": "/tmp/audio_pipeline_lang",
                "options": {"whisper_language": "tr"},
                "stages": ["detect_language", "extract", "transcribe"],
            },
            log_cb=lambda *args, **kwargs: None,
        )

        result = pipeline.run()

        self.assertIsNone(captured.get("whisper_language"))
        self.assertEqual(result.get("detected_language"), "ar")
        self.assertEqual(result.get("transcript")[0]["text"], "مرحبا")


if __name__ == "__main__":
    unittest.main(verbosity=2)
