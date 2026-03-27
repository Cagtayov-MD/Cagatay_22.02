"""test_sport_pipeline.py — Spor profil ve SportAnalyzer testleri

Testler:
  SM-01: Futbol transcript → spor dalı "futbol" tespiti
  SM-02: Basketbol transcript → spor dalı "basketbol" tespiti
  SM-03: Voleybol transcript → spor dalı "voleybol" tespiti
  SM-04: Boş transcript → spor dalı "bilinmiyor"
  SM-05: SKOR_PATTERNS regex — "GS 2 - FB 1" tipi skor bulma
  SM-06: SKOR_PATTERNS regex — "87-74" sade sayı skoru
  SM-07: GOL_PATTERNS regex — "34. dakikada Metin Oktay gol attı"
  SM-08: KART_PATTERNS regex — "sarı kart 23. dakika Alpaslan"
  SM-09: dag_definitions — SPOR_DAG PROFILE_DAGS içinde kayıtlı
  SM-10: dag_definitions — get_dag("Spor") SPOR_DAG döndürür
  SM-11: dag_definitions — is_sport_match_profile("Spor") True
  SM-12: dag_definitions — is_sport_match_profile("FilmDizi-Hybrid") False
  SM-13: content_profiles.json — Spor profili tüm zorunlu alanları içeriyor
  SM-14: SportAnalyzer.analyze() Gemini kapalı — futbol → goals/cards çalıştırılıyor
  SM-15: SportAnalyzer.analyze() Gemini kapalı — basketbol → goals/cards boş
  SM-16: SportAnalyzer._cross_validate() ASR==OCR skor → DOĞRULANDI log
  SM-17: SportAnalyzer._cross_validate() ASR≠OCR skor → OCR tercih edildi log
  SM-18: SportAnalyzer.build_report_text() — "SPOR MAÇI RAPORU" başlığı içeriyor
  SM-19: Pipeline runner — Spor profili run_sport_pipeline'a yönlendiriyor
  SM-20: SportAnalyzer.set_transcripts() + set_ocr_results() sonrası log düzgün
"""

import os
import sys
import json
import re
import unittest
from unittest.mock import patch, MagicMock

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core.sport_analyzer import SportAnalyzer, SPORT_KEYWORDS, SKOR_PATTERNS, GOL_PATTERNS, KART_PATTERNS
from core import sport_pipeline


# ── SM-01 / 02 / 03 / 04 — Spor Dalı Tespiti ────────────────────────────────

class TestSporDaliTespiti(unittest.TestCase):

    def _make_analyzer(self):
        return SportAnalyzer({"gemini_enabled": False})

    def test_sm01_futbol_tespiti(self):
        """Futbol anahtar kelimeleri içeren transcript → futbol."""
        a = self._make_analyzer()
        a.set_transcripts(
            "Galatasaray Fenerbahçe derbisi. Kaleci hazır. Korner atışı...",
            "Gol! Penaltı bölgesinden harika bir vuruş. Ofsayt pozisyonu var. Sarı kart 34. dakikada.",
        )
        self.assertEqual(a._detect_sport_type(), "futbol")

    def test_sm02_basketbol_tespiti(self):
        """Basketbol anahtar kelimeleri içeren transcript → basketbol."""
        a = self._make_analyzer()
        a.set_transcripts(
            "İlk çeyrek başladı. Rebound kazandı. Serbest atış hakkı verildi.",
            "Üç sayı! Fast break. Ribaund alındı. Periyot bitti. Overtime'a gidiyoruz.",
        )
        self.assertEqual(a._detect_sport_type(), "basketbol")

    def test_sm03_voleybol_tespiti(self):
        """Voleybol anahtar kelimeleri içeren transcript → voleybol."""
        a = self._make_analyzer()
        a.set_transcripts(
            "İlk set başlıyor. Servis hatası. Smaç geldi, blok yaptı. Libero hazır.",
            "Ace servis! Manşet kurtarış. Setter pozisyon aldı. Spike harika.",
        )
        self.assertEqual(a._detect_sport_type(), "voleybol")

    def test_sm04_bos_transcript_bilinmiyor(self):
        """Boş transcript → bilinmiyor."""
        a = self._make_analyzer()
        a.set_transcripts("", "")
        self.assertEqual(a._detect_sport_type(), "bilinmiyor")

    def test_sm04b_alakasiz_metin_bilinmiyor(self):
        """Spor kelimesi içermeyen transcript → bilinmiyor."""
        a = self._make_analyzer()
        a.set_transcripts("Hava bugün güneşli.", "Restoranlar açık mı?")
        self.assertEqual(a._detect_sport_type(), "bilinmiyor")


# ── SM-05 / 06 — SKOR_PATTERNS Regex ────────────────────────────────────────

class TestSkorPatterns(unittest.TestCase):

    def _find_scores(self, text):
        scores = []
        for pattern in SKOR_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                scores.append(m.group(0).strip())
        return scores

    def test_sm05_takim_isimli_skor(self):
        """'Galatasaray 2 - 1 Fenerbahçe' formatında skor bulunmalı."""
        results = self._find_scores("Galatasaray 2 - 1 Fenerbahçe")
        self.assertTrue(len(results) > 0, "Takım isimli skor bulunamadı")

    def test_sm06_sade_sayisal_skor(self):
        """'87-74' sade sayısal skor bulunmalı."""
        results = self._find_scores("Skor durumu 87-74 arada.")
        self.assertTrue(any("87" in r and "74" in r for r in results))

    def test_sm06b_iki_nokta_ust_uste_skor(self):
        """'2:1' formatında skor bulunmalı."""
        results = self._find_scores("Maç sonu: 2:1")
        self.assertTrue(len(results) > 0, "İki nokta üst üste skoru bulunamadı")


# ── SM-07 — GOL_PATTERNS Regex ───────────────────────────────────────────────

class TestGolPatterns(unittest.TestCase):

    def _find_goals(self, text):
        goals = []
        for pattern in GOL_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                goals.append(m.group(0).strip())
        return goals

    def test_sm07_dakika_isim_gol(self):
        """'34. dakikada Metin Oktay gol' formatı eşleşmeli."""
        results = self._find_goals("34. dakikada Metin Oktay gol attı.")
        self.assertTrue(len(results) > 0, "Gol deseni eşleşmedi")

    def test_sm07b_penalti_gol(self):
        """'penaltıdan gol Hagi' formatı eşleşmeli."""
        results = self._find_goals("Penaltıdan gol! Hagi!")
        self.assertTrue(len(results) > 0, "Penaltı gol deseni eşleşmedi")


# ── SM-08 — KART_PATTERNS Regex ──────────────────────────────────────────────

class TestKartPatterns(unittest.TestCase):

    def _find_cards(self, text):
        cards = []
        for pattern in KART_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                cards.append(m.group(0).strip())
        return cards

    def test_sm08_sari_kart_dakika_isim(self):
        """'sarı kart 23. dakika Alpaslan' formatı eşleşmeli."""
        results = self._find_cards("Sarı kart 23. dakika Alpaslan Eratlı.")
        self.assertTrue(len(results) > 0, "Sarı kart deseni eşleşmedi")

    def test_sm08b_kirmizi_kart(self):
        """'kırmızı kart' formatı eşleşmeli."""
        results = self._find_cards("Kırmızı kart! Selçuk Yula 75. dakikada.")
        self.assertTrue(len(results) > 0, "Kırmızı kart deseni eşleşmedi")


# ── SM-09 / 10 / 11 / 12 — DAG Definitions ──────────────────────────────────

class TestDagDefinitions(unittest.TestCase):

    def setUp(self):
        from core.dag_definitions import PROFILE_DAGS, SPOR_DAG, get_dag, is_sport_match_profile
        self.PROFILE_DAGS = PROFILE_DAGS
        self.SPOR_DAG = SPOR_DAG
        self.get_dag = get_dag
        self.is_sport_match_profile = is_sport_match_profile

    def test_sm09_spor_dag_kayitli(self):
        """SPOR_DAG PROFILE_DAGS içinde 'Spor' adıyla kayıtlı olmalı."""
        self.assertIn("Spor", self.PROFILE_DAGS)

    def test_sm10_get_dag_spor(self):
        """get_dag('Spor') SPOR_DAG döndürmeli."""
        dag = self.get_dag("Spor")
        self.assertIs(dag, self.SPOR_DAG)

    def test_sm11_is_sport_match_true(self):
        """is_sport_match_profile('Spor') → True."""
        self.assertTrue(self.is_sport_match_profile("Spor"))

    def test_sm12_is_sport_match_false(self):
        """is_sport_match_profile('FilmDizi-Hybrid') → False."""
        self.assertFalse(self.is_sport_match_profile("FilmDizi-Hybrid"))

    def test_sm12b_spor_dag_adimlar_var(self):
        """SPOR_DAG içinde VIDEO_INPUT, ASR_BRANCH, FRAME_BRANCH, SPORT_ANALYZE bulunmalı."""
        dag = self.SPOR_DAG
        self.assertIn("VIDEO_INPUT", dag)
        self.assertIn("ASR_BRANCH", dag)
        self.assertIn("FRAME_BRANCH", dag)
        self.assertIn("SPORT_ANALYZE", dag)


# ── SM-13 — content_profiles.json Spor Profili ───────────────────────────────

class TestContentProfilesSpor(unittest.TestCase):

    def setUp(self):
        profiles_path = os.path.join(_project_dir, "config", "content_profiles.json")
        with open(profiles_path, encoding="utf-8") as f:
            self.profiles = json.load(f)
        self.profile = self.profiles.get("Spor", {})

    def test_sm13_profil_var(self):
        """content_profiles.json'da 'Spor' profili mevcut olmalı."""
        self.assertIn("Spor", self.profiles)

    def test_sm13b_segment_minutes_15(self):
        """segment_minutes değeri 15 olmalı."""
        self.assertEqual(self.profile.get("segment_minutes"), 15)

    def test_sm13c_frame_interval_10(self):
        """frame_interval_sec değeri 10 olmalı."""
        self.assertEqual(self.profile.get("frame_interval_sec"), 10)

    def test_sm13d_tmdb_disabled(self):
        """TMDB Spor profilinde devre dışı olmalı."""
        self.assertFalse(self.profile.get("tmdb_enabled", True))

    def test_sm13e_sport_analyze_enabled(self):
        """sport_analyze_enabled açık olmalı."""
        self.assertTrue(self.profile.get("sport_analyze_enabled", False))

    def test_sm13f_sport_auto_detect(self):
        """sport_auto_detect açık olmalı."""
        self.assertTrue(self.profile.get("sport_auto_detect", False))

    def test_sm13g_zorunlu_alanlar(self):
        """Zorunlu alanların hepsi mevcut olmalı."""
        required = [
            "segment_minutes", "frame_interval_sec",
            "asr_engine", "ocr_engine",
            "gemini_enabled", "gemini_model",
        ]
        for field in required:
            self.assertIn(field, self.profile, f"Eksik alan: {field}")

    def test_sm13h_speech_tuning_defaults(self):
        """Spor profili ses ayrımı ve mix fallback ayarlarını görünür taşımalı."""
        self.assertTrue(self.profile.get("speech_separation"))
        self.assertTrue(self.profile.get("include_mix_fallback"))
        self.assertEqual(self.profile.get("sport_vad_min_silence_ms"), 450)
        self.assertEqual(self.profile.get("sport_vad_speech_pad_ms"), 200)


# ── SM-14 / 15 — analyze() Gemini Kapalı ────────────────────────────────────

class TestAnalyzeGeminiKapali(unittest.TestCase):

    def _make_analyzer(self):
        return SportAnalyzer({"gemini_enabled": False})

    def test_sm14_futbol_goals_cards_calistirilir(self):
        """Futbol transcript'inde analyze() goals ve cards çıkarma adımını çalıştırmalı."""
        a = self._make_analyzer()
        a.set_transcripts(
            "Galatasaray Fenerbahçe maçı. Korner, kaleci, gol, ofsayt.",
            "34. dakikada Metin Oktay gol attı. Sarı kart 23. dakika Alpaslan.",
        )
        a.set_ocr_results([])
        result = a.analyze()
        # Futbol tespiti bekleniyor
        self.assertEqual(result["spor_dali"], "futbol")
        # goals ve cards anahtarları result'ta olmalı (boş olabilir, ama mevcut olmalı)
        self.assertIn("goller", result)
        self.assertIn("kartlar", result)

    def test_sm15_basketbol_goals_cards_bos(self):
        """Basketbol maçında analyze() goals ve cards üretmemeli."""
        a = self._make_analyzer()
        a.set_transcripts(
            "İlk çeyrek. Serbest atış. Ribaund. Üç sayı. Quarter.",
            "Periyot sonu. Rebound alındı. Three pointer. Overtime.",
        )
        a.set_ocr_results([])
        result = a.analyze()
        self.assertEqual(result["spor_dali"], "basketbol")
        self.assertEqual(result["goller"], [])
        self.assertEqual(result["kartlar"], [])


# ── SM-16 / 17 — Çapraz Doğrulama Logu ──────────────────────────────────────

class TestCaprazDogrulama(unittest.TestCase):

    def _make_analyzer_with_scores(self, asr_score, ocr_score):
        a = SportAnalyzer({"gemini_enabled": False})
        a.sport_type = "futbol"
        a.score_info = {"asr_score": asr_score, "ocr_score": ocr_score}
        a.goals = []
        a.cards = []
        a._cross_validate()
        return a

    def test_sm16_asr_ocr_esit_dogrulandi(self):
        """ASR skoru == OCR skoru → log'da 'DOĞRULANDI' geçmeli."""
        a = self._make_analyzer_with_scores("2-1", "2-1")
        log_text = a.get_readable_log()
        self.assertIn("DOĞRULANDI", log_text)

    def test_sm17_asr_ocr_farkli_ocr_tercih(self):
        """ASR skoru ≠ OCR skoru → log'da 'OCR tercih' geçmeli."""
        a = self._make_analyzer_with_scores("2-0", "2-1")
        log_text = a.get_readable_log()
        self.assertIn("OCR tercih", log_text)


# ── SM-18 — build_report_text Formatı ───────────────────────────────────────

class TestBuildReportText(unittest.TestCase):

    def test_sm18_rapor_basligi(self):
        """build_report_text() çıktısı 'SPOR MAÇI RAPORU' başlığı içermeli."""
        a = SportAnalyzer({"gemini_enabled": False})
        result = {
            "spor_dali": "futbol",
            "mac_bilgileri": {
                "spor_dali": "futbol",
                "takimlar": "Galatasaray — Fenerbahçe",
                "lig": "Süper Lig",
                "hafta": "", "tarih": "", "sehir": "", "stadyum": "", "hava": "", "hakem": "",
            },
            "skor": {"asr_score": "2-1", "ocr_score": "2-1", "final": "2-1", "source": "ASR+OCR"},
            "goller": [{"dakika": 34, "oyuncu": "Metin Oktay", "takim": "Galatasaray"}],
            "kartlar": [],
            "spiker_notlari": [],
            "verification_log": [],
        }
        text = a.build_report_text(result)
        self.assertIn("SPOR MAÇI RAPORU", text)
        self.assertIn("FUTBOL", text)
        self.assertIn("GALATASARAY", text)

    def test_sm18b_goller_bolumu(self):
        """Futbol raporunda GOLLER bölümü ve gol satırı olmalı."""
        a = SportAnalyzer({"gemini_enabled": False})
        result = {
            "spor_dali": "futbol",
            "mac_bilgileri": {"spor_dali": "futbol", "takimlar": "", "lig": "",
                              "hafta": "", "tarih": "", "sehir": "", "stadyum": "", "hava": "", "hakem": ""},
            "skor": {},
            "goller": [{"dakika": 34, "oyuncu": "Metin Oktay", "takim": "GS"}],
            "kartlar": [],
            "spiker_notlari": [],
            "verification_log": [],
        }
        text = a.build_report_text(result)
        self.assertIn("GOLLER", text)
        self.assertIn("METİN OKTAY", text)


# ── SM-19 — Pipeline Runner Spor Yönlendirme ─────────────────────────────────

class TestPipelineRunnerSporRouting(unittest.TestCase):

    @unittest.skip("pipeline_runner import sırasında pypdfium2 crash — sistem sorunu, logic test değil")
    def test_sm19_spor_profili_sport_pipeline_cagirir(self):
        """
        PipelineRunner.run() Spor profiliyle çağrılınca
        run_sport_pipeline() çağrılmalı (mevcut FilmDizi akışı çalışmamalı).
        """
        pass  # skip


# ── SM-20 — set_transcripts + set_ocr_results Log ────────────────────────────

class TestSetDataLog(unittest.TestCase):

    def test_sm20_set_transcripts_log(self):
        """set_transcripts() sonrası log'da her iki segment geçmeli."""
        a = SportAnalyzer({"gemini_enabled": False})
        a.set_transcripts("İlk segment metni.", "Son segment metni.")
        log_text = a.get_readable_log()
        self.assertIn("İlk", log_text)
        self.assertIn("Son", log_text)

    def test_sm20b_set_ocr_results_log(self):
        """set_ocr_results() sonrası log'da frame sayısı (3) geçmeli."""
        a = SportAnalyzer({"gemini_enabled": False})
        a.set_ocr_results([(1, "GS 2 - FB 1"), (2, ""), (3, "Dakika 87")])
        log_text = a.get_readable_log()
        self.assertIn("3", log_text)


# ── SM-21 — Seçilen Kanal + Mix Fallback ────────────────────────────────────

class TestSportAudioCandidates(unittest.TestCase):

    def test_sm21_selected_channel_extracts_detected_channel_and_mix(self):
        """selected_channel=1 ise ch1 raw ve mix fallback adayı üretilmeli."""
        analyzer = SportAnalyzer({
            "gemini_enabled": False,
            "speech_separation": False,
            "selected_channel": 1,
            "selected_channel_confidence": 0.91,
            "include_mix_fallback": True,
            "ffmpeg": "ffmpeg",
            "ffprobe": "ffprobe",
        })
        calls = []

        def fake_run(cmd, capture_output=False, text=False, timeout=None, **kwargs):
            calls.append(cmd)
            mocked = MagicMock()
            mocked.returncode = 0
            mocked.stderr = ""
            if "format=duration" in cmd:
                mocked.stdout = "1200\n"
            elif "stream=channels" in cmd:
                mocked.stdout = "2\n"
            else:
                mocked.stdout = ""
            return mocked

        with patch("subprocess.run", side_effect=fake_run):
            first_path, last_path = analyzer.extract_segments("match.mp4", "C:\\temp\\spor")

        self.assertTrue(first_path.endswith("segment_first_ch1.wav"))
        self.assertTrue(last_path.endswith("segment_last_ch1.wav"))

        first_candidates = analyzer.get_segment_audio_candidates("first")
        labels = [c["label"] for c in first_candidates]
        self.assertIn("ch1_raw", labels)
        self.assertIn("mix_raw", labels)

        ffmpeg_calls = [" ".join(map(str, cmd)) for cmd in calls if cmd and cmd[0] == "ffmpeg"]
        self.assertTrue(any("pan=mono|c0=c1" in cmd for cmd in ffmpeg_calls))
        self.assertTrue(any("-ac 1 -af" in cmd for cmd in ffmpeg_calls))


# ── SM-22 / 23 — ASR Aday Seçimi ─────────────────────────────────────────────

class TestSportAsrSelection(unittest.TestCase):

    def test_sm22_asr_bundle_prefers_cleaner_vocals_candidate(self):
        """raw ve vocals arasında kalite skoru daha iyi olan aday seçilmeli."""
        candidates = [
            {"path": "raw.wav", "candidate_kind": "raw", "source_mode": "ch1", "label": "ch1_raw"},
            {"path": "vocals.wav", "candidate_kind": "vocals", "source_mode": "ch1", "label": "ch1_vocals"},
        ]

        def fake_detail(audio_path, engine, config, _log=None):
            if audio_path == "vocals.wav":
                return {
                    "transcript": "Galatasaray korner gol penaltı",
                    "segments": [{"text": "Galatasaray korner gol penaltı"}],
                    "audio_duration": 120.0,
                    "speech_duration": 70.0,
                    "avg_logprob": -0.35,
                    "avg_no_speech_prob": 0.08,
                    "avg_compression_ratio": 1.2,
                }
            return {
                "transcript": "eee eee eee tribün tribün",
                "segments": [{"text": "eee eee eee tribün tribün"}],
                "audio_duration": 120.0,
                "speech_duration": 18.0,
                "avg_logprob": -1.9,
                "avg_no_speech_prob": 0.72,
                "avg_compression_ratio": 2.8,
            }

        with patch("core.sport_pipeline._run_asr_detailed", side_effect=fake_detail):
            result = sport_pipeline._run_asr_bundle(candidates, "whisper", {}, "ilk")

        self.assertEqual(result["selected_candidate"], "ch1_vocals")
        self.assertIn("Galatasaray", result["transcript"])

    def test_sm23_quality_score_penalises_repetition(self):
        """Tekrarlı/noisy transcript, anlamlı spor transcript'inden düşük skor almalı."""
        noisy = {
            "transcript": "gol gol gol gol gol gol",
            "audio_duration": 120.0,
            "speech_duration": 20.0,
            "avg_logprob": -1.7,
            "avg_no_speech_prob": 0.7,
            "avg_compression_ratio": 3.0,
            "candidate_kind": "raw",
        }
        clean = {
            "transcript": "Galatasaray korner kazandı penaltı bekleniyor sarı kart çıktı",
            "audio_duration": 120.0,
            "speech_duration": 75.0,
            "avg_logprob": -0.4,
            "avg_no_speech_prob": 0.1,
            "avg_compression_ratio": 1.3,
            "candidate_kind": "vocals",
        }

        self.assertGreater(
            sport_pipeline._score_asr_detail(clean),
            sport_pipeline._score_asr_detail(noisy),
        )


class TestSummaryParagraphExtraction(unittest.TestCase):

    def test_sm24_summary_paragraph_contains_match_facts(self):
        """Spikerin maç sonu paragrafı skor, kırmızı kart ve golsüz bitişi taşımalı."""
        analyzer = SportAnalyzer({"gemini_enabled": False})
        analyzer.set_video_name("PTT_1._LİG_MERSİN_İDMAN_YURDU_-_BOLUSPOR")
        analyzer.set_transcripts(
            (
                "Maçın hakemi Serkan Tokat, Serhan Malkoç ve Bahtiyar birinci yardımcıları. "
                "Serkan Önsal da dördüncü hakem olarak bu karşılaşmada görev alacak. "
                "PTT 1. Lig'de Mersin'de oynanıyor."
            ),
            (
                "Mersin İdman Yurdu 10 kişi oynuyor. Son dakikaları heyecanlı geçen bu mücadelede "
                "şu ana kadar gol izleyemedik. Maç bitti. Mersin İdman Yurdu 0. Bolu Spor 0 maçın sonucu. "
                "Maçın 61. dakikasında Alitan Doğan kırmızı kart gördü. Direkten dönen toplar var. Ama gol yok."
            ),
        )
        analyzer.set_ocr_results([])

        result = analyzer.analyze()

        self.assertEqual(result["skor"]["final_score"], "0-0")
        self.assertEqual(result["goller"], [])
        self.assertTrue(any(card.get("tip") == "kırmızı kart" for card in result["kartlar"]))
        self.assertEqual(result["mac_bilgileri"]["hakem"], "Serkan Tokat")
        self.assertIn("Serhan Malkoç", result["mac_bilgileri"]["yardimci_hakemler"])
        self.assertIn("Bahtiyar", result["mac_bilgileri"]["yardimci_hakemler"])
        self.assertEqual(result["mac_bilgileri"]["dorduncu_hakem"], "Serkan Önsal")
        self.assertIn("PTT 1. Lig", result["mac_bilgileri"]["lig"])

    def test_sm24b_match_info_fallback_still_runs_when_first_transcript_is_empty(self):
        """İlk segment boş olsa bile son transcript + dosya adı fallback'i çalışmalı."""
        analyzer = SportAnalyzer({"gemini_enabled": False})
        analyzer.sport_type = "futbol"
        analyzer.set_video_name("PTT_1.LİG_MERSİN_İDMAN_YURDU-_BOLUSPOR")
        analyzer.set_transcripts(
            "",
            (
                "PTT 1. Lig'de Mersin'de oynanıyor. "
                "Maçın hakemi Serkan Tokat. "
                "Maç bitti. Mersin İdman Yurdu 0. Boluspor 0 maçın sonucu."
            ),
        )

        info = analyzer._extract_match_info()

        self.assertIn("MERSİN İDMAN YURDU", info["takimlar"])
        self.assertIn("BOLUSPOR", info["takimlar"])
        self.assertIn("PTT 1. Lig", info["lig"])
        self.assertEqual(info["sehir"], "Mersin")
        self.assertEqual(info["hakem"], "Serkan Tokat")

    def test_sm24c_filename_team_fallback_accepts_live_dash_variant(self):
        """Canlıdaki 'YURDU-_BOLUSPOR' biçimi de takım ayırıcı olarak kabul edilmeli."""
        analyzer = SportAnalyzer({"gemini_enabled": False})
        analyzer.set_video_name("PTT_1.LİG_MERSİN_İDMAN_YURDU-_BOLUSPOR")

        teams = analyzer._extract_teams_from_filename()

        self.assertEqual(teams, ("MERSİN İDMAN YURDU", "BOLUSPOR"))


class TestGeminiEvidenceExtraction(unittest.TestCase):

    def test_sm25_collect_evidence_text_filters_to_related_sentences(self):
        """Hakem kanıtı yalnızca ilgili cümleleri ve komşularını toplamalı."""
        analyzer = SportAnalyzer({"gemini_enabled": False})
        text = (
            "Stadyumda tribünler tamamen dolu. "
            "Maçın hakemi Serkan Tokat. "
            "Serhan Malkoç ve Bahtiyar birinci yardımcıları. "
            "Serkan Önsal da dördüncü hakem olarak görev alacak. "
            "Direkten dönen toplar var ama gol yok."
        )

        evidence = analyzer._collect_evidence_text(text, "match_officials", max_chars=500)

        self.assertIn("Serkan Tokat", evidence)
        self.assertIn("Serhan Malkoç", evidence)
        self.assertIn("Serkan Önsal", evidence)
        self.assertNotIn("Direkten dönen toplar", evidence)

    def test_sm25b_parse_json_helpers_accept_code_fences(self):
        """Gemini code fence ile dönerse JSON yardımcıları yine parse edebilmeli."""
        analyzer = SportAnalyzer({"gemini_enabled": False})
        obj = analyzer._parse_json_object("""```json
        {"hakem": "Serkan Tokat"}
        ```""")
        arr = analyzer._parse_json_list("""```json
        [{"tip": "kırmızı kart", "dakika": 61}]
        ```""")

        self.assertEqual(obj["hakem"], "Serkan Tokat")
        self.assertEqual(arr[0]["tip"], "kırmızı kart")

    def test_sm25c_parse_json_helpers_accept_first_json_block_with_trailing_content(self):
        """İlk geçerli JSON'dan sonra ek blok/prose olsa da parser ilk bloğu almalı."""
        analyzer = SportAnalyzer({"gemini_enabled": False})
        obj = analyzer._parse_json_object(
            '{"hakem": "Serkan Tokat"}\n{"note": "extra"}'
        )
        arr = analyzer._parse_json_list(
            '[{"tip": "kırmızı kart", "dakika": 61}]\nTeşekkürler.'
        )

        self.assertEqual(obj["hakem"], "Serkan Tokat")
        self.assertEqual(arr[0]["dakika"], 61)

    def test_sm26_match_info_gemini_uses_targeted_evidence_prompts(self):
        """Maç bilgisi için Gemini'ye tüm transcript değil alan bazlı kanıt parçaları gitmeli."""
        analyzer = SportAnalyzer({"gemini_enabled": True, "gemini_model": "gemini-test"})
        analyzer.sport_type = "futbol"
        analyzer.set_transcripts(
            (
                "Yayın başlıyor. "
                "Maçın hakemi Serkan Tokat. "
                "Serhan Malkoç ve Bahtiyar birinci yardımcıları. "
                "Serkan Önsal da dördüncü hakem olarak görev alacak. "
                "PTT 1. Lig'de Mersin'de oynanıyor."
            ),
            "Maç bitti. Mersin İdman Yurdu 0. Bolu Spor 0 maçın sonucu.",
        )

        prompts = []

        def fake_generate(prompt):
            prompts.append(prompt)
            if '"hakem": "Baş hakem adı"' in prompt:
                return json.dumps({
                    "hakem": "Serkan Tokat",
                    "yardimci_hakemler": "Serhan Malkoç, Bahtiyar",
                    "dorduncu_hakem": "Serkan Önsal",
                    "evidence": ["Maçın hakemi Serkan Tokat"],
                }, ensure_ascii=False)
            return json.dumps({
                "takimlar": "",
                "lig": "PTT 1. Lig",
                "hafta": "",
                "tarih": "",
                "sehir": "Mersin",
                "stadyum": "",
                "hava": "",
                "evidence": ["PTT 1. Lig'de Mersin'de oynanıyor"],
            }, ensure_ascii=False)

        with patch("core.gemini_client.GeminiClient.generate", side_effect=fake_generate):
            info = analyzer._extract_match_info()

        self.assertEqual(info["hakem"], "Serkan Tokat")
        self.assertEqual(info["yardimci_hakemler"], "Serhan Malkoç, Bahtiyar")
        self.assertEqual(info["dorduncu_hakem"], "Serkan Önsal")
        self.assertEqual(info["lig"], "PTT 1. Lig")
        self.assertEqual(info["sehir"], "Mersin")
        self.assertEqual(len(prompts), 2)
        self.assertTrue(any("Serkan Tokat" in prompt for prompt in prompts))
        self.assertTrue(any("PTT 1. Lig" in prompt for prompt in prompts))
        self.assertTrue(all("Maç bitti" not in prompt for prompt in prompts))

    def test_sm26b_match_info_logs_raw_response_and_uses_regex_when_parse_fails(self):
        """Gemini parse bozulursa ham cevap loglanmalı ve regex fallback devreye girmeli."""
        analyzer = SportAnalyzer({"gemini_enabled": True, "gemini_model": "gemini-test"})
        analyzer.sport_type = "futbol"
        analyzer.set_transcripts(
            (
                "Maçın hakemi Serkan Tokat. "
                "Serhan Malkoç ve Bahtiyar birinci yardımcıları. "
                "Serkan Önsal da dördüncü hakem olarak görev alacak. "
                "PTT 1. Lig'de Mersin'de oynanıyor."
            ),
            "",
        )

        def fake_generate(prompt):
            if '"hakem": "Baş hakem adı"' in prompt:
                return '{"hakem": "Serkan Tokat"'
            return json.dumps({
                "takimlar": "",
                "lig": "PTT 1. Lig",
                "hafta": "",
                "tarih": "",
                "sehir": "Mersin",
                "stadyum": "",
                "hava": "",
            }, ensure_ascii=False)

        with patch("core.gemini_client.GeminiClient.generate", side_effect=fake_generate):
            info = analyzer._extract_match_info()

        self.assertEqual(info["hakem"], "Serkan Tokat")
        self.assertIn("Serhan Malkoç", info["yardimci_hakemler"])
        self.assertEqual(info["dorduncu_hakem"], "Serkan Önsal")
        self.assertTrue(any("ham cevap" in row["message"] for row in analyzer.verification_log))

    def test_sm27_score_gemini_fallback_uses_score_evidence(self):
        """Regex skor bulamazsa Gemini kısa skor parçasından fallback yapmalı."""
        analyzer = SportAnalyzer({"gemini_enabled": True, "gemini_model": "gemini-test"})
        analyzer.set_transcripts(
            "",
            (
                "Son dakikaları heyecanlı geçen bu mücadelede skor golsüz eşitlikte kaldı. "
                "Mersin İdman Yurdu sıfır. "
                "Bolu Spor sıfır maçın sonucu. "
                "Az sonra tekrar birlikteyiz."
            ),
        )
        analyzer.set_ocr_results([])

        prompts = []

        def fake_generate(prompt):
            prompts.append(prompt)
            return json.dumps({
                "final_score": "0-0",
                "evidence": ["Bolu Spor sıfır maçın sonucu"],
            }, ensure_ascii=False)

        with patch("core.gemini_client.GeminiClient.generate", side_effect=fake_generate):
            score = analyzer._extract_score()

        self.assertEqual(score["asr_score"], "0-0")
        self.assertEqual(score["final_score"], "0-0")
        self.assertEqual(len(prompts), 1)
        self.assertIn("skor golsüz eşitlikte kaldı", prompts[0].lower())
        self.assertNotIn("Az sonra tekrar birlikteyiz", prompts[0])

    def test_sm28_goals_gemini_uses_goal_evidence_window(self):
        """Gol sorgusunda yalnızca gol ile ilgili kısa parça Gemini'ye gitmeli."""
        analyzer = SportAnalyzer({"gemini_enabled": True, "gemini_model": "gemini-test"})
        analyzer.set_transcripts(
            "",
            (
                "Orta sahada oyun sürüyor. "
                "82. dakikada Ahmet golü attı. "
                "Tribünler ayağa kalktı. "
                "Az sonra programımız devam edecek."
            ),
        )

        prompts = []

        def fake_generate(prompt):
            prompts.append(prompt)
            return json.dumps([
                {"dakika": 82, "oyuncu": "Ahmet", "takim": "?"},
            ], ensure_ascii=False)

        with patch("core.gemini_client.GeminiClient.generate", side_effect=fake_generate):
            goals = analyzer._extract_goals_gemini()

        self.assertEqual(goals[0]["dakika"], 82)
        self.assertEqual(goals[0]["oyuncu"], "Ahmet")
        self.assertEqual(len(prompts), 1)
        self.assertIn("Ahmet golü attı", prompts[0])
        self.assertNotIn("Az sonra programımız devam edecek", prompts[0])

    def test_sm29_cards_gemini_uses_card_evidence_window(self):
        """Kart sorgusunda kart ve komşu cümleler tutulmalı, alakasız kısım atılmalı."""
        analyzer = SportAnalyzer({"gemini_enabled": True, "gemini_model": "gemini-test"})
        analyzer.set_transcripts(
            "",
            (
                "Mersin İdman Yurdu 10 kişi oynuyor. "
                "Maçın 61. dakikasında Alitan Doğan kırmızı kart gördü. "
                "Heyecan son anlarda artıyor. "
                "Program bitiminde röportajlar var."
            ),
        )

        prompts = []

        def fake_generate(prompt):
            prompts.append(prompt)
            return json.dumps([
                {"tip": "kırmızı kart", "dakika": 61, "oyuncu": "Alitan Doğan", "takim": "?"},
            ], ensure_ascii=False)

        with patch("core.gemini_client.GeminiClient.generate", side_effect=fake_generate):
            cards = analyzer._extract_cards_gemini()

        self.assertEqual(cards[0]["tip"], "kırmızı kart")
        self.assertEqual(cards[0]["dakika"], 61)
        self.assertEqual(len(prompts), 1)
        self.assertIn("10 kişi oynuyor", prompts[0])
        self.assertIn("kırmızı kart", prompts[0].lower())
        self.assertNotIn("Program bitiminde röportajlar var", prompts[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
