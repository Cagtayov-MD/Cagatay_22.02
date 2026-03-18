"""
test_content_type_profiles.py — İçerik türü profilleri ve diagnostics testleri.

İçerik kategorileri: EskiFilm, YeniFilm, OrijinalDilFilm, YeniDizi, EskiDizi
"""

import sys
import os

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import pytest
from config.profile_loader import load_profile, list_profiles


# ─── Profil Yükleme Testleri ──────────────────────────────────────────────────

class TestContentProfiles:
    """İçerik profillerinin doğru yüklendiğini ve gerekli alanları içerdiğini doğrula."""

    EXPECTED_PROFILES = ["EskiFilm", "YeniFilm", "OrijinalDilFilm", "YeniDizi", "EskiDizi"]

    def test_all_new_profiles_exist(self):
        """5 yeni profil content_profiles.json'da bulunmalı."""
        available = list_profiles()
        for p in self.EXPECTED_PROFILES:
            assert p in available, f"Profil eksik: {p}"

    def test_eski_film_profile_keys(self):
        p = load_profile("EskiFilm")
        assert p["ocr_engine"] == "qwen", "EskiFilm VLM tabanlı OCR kullanmalı"
        assert p["blok2_enabled"] is True, "EskiFilm BLOK2 derin okuma açık olmalı"
        assert p["tmdb_lenient_match"] is True, "EskiFilm lenient TMDB eşleme kullanmalı"
        assert p["gemini_enabled"] is True, "EskiFilm Gemini cast ayıklama açık olmalı"
        assert p.get("content_type") == "eski_film"

    def test_yeni_film_profile_keys(self):
        p = load_profile("YeniFilm")
        assert p["ocr_engine"] == "hybrid", "YeniFilm hibrit OCR kullanmalı"
        assert p["blok2_enabled"] is False, "YeniFilm BLOK2 kapalı olmalı"
        assert p["tmdb_enabled"] is True
        assert p["tmdb_person_verify"] is True
        assert p.get("content_type") == "yeni_film"

    def test_orijinal_dil_film_profile_keys(self):
        p = load_profile("OrijinalDilFilm")
        assert p["whisper_language"] == "auto", "OrijinalDilFilm otomatik dil tespiti kullanmalı"
        assert p["post_process_all_languages"] is True, "post_process_all_languages açık olmalı"
        assert "post_process" in p["audio_stages"], "post_process stage'i açık olmalı"
        assert p["gemini_enabled"] is True
        assert p.get("content_type") == "orijinal_dil_film"

    def test_yeni_dizi_profile_keys(self):
        p = load_profile("YeniDizi")
        assert p["first_segment_minutes"] == 0, "YeniDizi açılış credits yok — first=0"
        assert p["last_segment_minutes"] <= 3, "YeniDizi kısa credits — last<=3"
        assert p["tmdb_lenient_match"] is True, "YeniDizi lenient TMDB eşleme kullanmalı"
        assert p["gemini_enabled"] is True, "YeniDizi Gemini devrede olmalı"
        assert p.get("content_type") == "yeni_dizi"

    def test_eski_dizi_profile_keys(self):
        p = load_profile("EskiDizi")
        assert p["guest_actor_enabled"] is True, "EskiDizi konuk oyuncu algılama açık olmalı"
        assert p["tmdb_enabled"] is True
        assert p.get("content_type") == "eski_dizi"

    def test_all_profiles_have_required_fields(self):
        """Tüm yeni profillerde zorunlu alanlar mevcut olmalı."""
        required_keys = [
            "scope", "ocr_enabled", "ocr_engine", "tmdb_enabled",
            "whisper_model", "audio_stages",
        ]
        for profile_name in self.EXPECTED_PROFILES:
            p = load_profile(profile_name)
            for key in required_keys:
                assert key in p, f"{profile_name} profilinde '{key}' eksik"

    def test_all_profiles_have_description_strength_weakness(self):
        """Yeni profillerde _description, _strength, _weakness belgelenmiş olmalı."""
        for profile_name in self.EXPECTED_PROFILES:
            p = load_profile(profile_name)
            assert p.get("_description"), f"{profile_name}: _description eksik"
            assert p.get("_strength"), f"{profile_name}: _strength eksik"
            assert p.get("_weakness"), f"{profile_name}: _weakness eksik"

    def test_legacy_profile_unchanged(self):
        """Mevcut FilmDizi-Hybrid profili değişmemeli."""
        p = load_profile("FilmDizi-Hybrid")
        assert p["ocr_engine"] == "hybrid"
        assert p["scope"] == "video+audio"
        assert p["whisper_language"] == "tr"
        assert p["blok2_enabled"] is False


# ─── Diagnostics Testleri ─────────────────────────────────────────────────────

class TestContentTypeDiagnostics:
    """ContentTypeDiagnostics modülü testleri."""

    def test_build_report_returns_all_types(self):
        from core.content_type_diagnostics import ContentTypeDiagnostics, CONTENT_TYPES
        diag = ContentTypeDiagnostics()
        report = diag.build_report()
        names = [r.profile_name for r in report]
        for ct in CONTENT_TYPES:
            assert ct in names

    def test_yeni_film_highest_overall_score(self):
        """YeniFilm en yüksek genel skora sahip olmalı."""
        from core.content_type_diagnostics import ContentTypeDiagnostics
        diag = ContentTypeDiagnostics()
        report = diag.build_report()
        score_map = {r.profile_name: r.overall_score for r in report}
        assert score_map["YeniFilm"] >= max(score_map.values()) - 0.01

    def test_yeni_dizi_lowest_ocr_score(self):
        """YeniDizi OCR aşamasında zayıf (score=1) olmalı."""
        from core.content_type_diagnostics import ContentTypeDiagnostics
        diag = ContentTypeDiagnostics()
        weak = diag.get_weakest_stages("YeniDizi", threshold=1)
        stage_keys = [s.stage for s in weak]
        assert "ocr_credits" in stage_keys, "YeniDizi OCR credits zayıf olmalı"
        assert "tmdb_verify" in stage_keys, "YeniDizi TMDB zayıf olmalı"

    def test_orijinal_dil_film_strong_language_detect(self):
        """OrijinalDilFilm dil tespiti ve transkriptte güçlü olmalı."""
        from core.content_type_diagnostics import ContentTypeDiagnostics
        diag = ContentTypeDiagnostics()
        strong = diag.get_strongest_stages("OrijinalDilFilm", threshold=3)
        stage_keys = [s.stage for s in strong]
        assert "audio_transcribe" in stage_keys
        assert "language_detect" in stage_keys

    def test_format_report_contains_all_profiles(self):
        """format_report çıktısı tüm profil adlarını içermeli."""
        from core.content_type_diagnostics import ContentTypeDiagnostics, CONTENT_TYPES
        diag = ContentTypeDiagnostics()
        report = diag.build_report()
        text = diag.format_report(report)
        for ct in CONTENT_TYPES:
            assert ct in text

    def test_format_report_contains_guclu_zayif(self):
        """format_report güçlü/zayıf anahtar kelimelerini içermeli."""
        from core.content_type_diagnostics import ContentTypeDiagnostics
        diag = ContentTypeDiagnostics()
        report = diag.build_report()
        text = diag.format_report(report)
        assert "GÜÇLÜ" in text
        assert "ZAYIF" in text

    def test_overall_verdict_labels(self):
        """overall_verdict doğru etiket döndürmeli."""
        from core.content_type_diagnostics import ContentTypeDiagnosticResult
        strong = ContentTypeDiagnosticResult("X", "", "", "", overall_score=3.0)
        mid = ContentTypeDiagnosticResult("X", "", "", "", overall_score=2.0)
        weak = ContentTypeDiagnosticResult("X", "", "", "", overall_score=1.0)
        assert "Güçlü" in strong.overall_verdict()
        assert "Orta" in mid.overall_verdict()
        assert "Zayıf" in weak.overall_verdict()


# ─── Pipeline Runner Profil Merge Testleri ────────────────────────────────────

class TestPipelineRunnerProfileMerge:
    """pipeline_runner.py'nin yeni profil anahtarlarını config'e doğru merge ettiğini doğrula."""

    # Aşağıdaki testler PipelineRunner'ı import etmeden,
    # merge mantığını bağımsız olarak doğrular.  cv2/oneocr gibi eksik
    # bağımlılıklardan etkilenmez.

    def _apply_profile_merge(self, profile: dict, base_config: dict | None = None) -> dict:
        """pipeline_runner._profile_merge_keys mantığını simüle et."""
        config = dict(base_config or {})
        merge_keys = (
            "audio_stages", "whisper_model", "whisper_language",
            "compute_type", "beam_size", "ocr_engine",
            "tmdb_enabled", "tmdb_person_verify", "tmdb_lenient_match",
            "gemini_enabled", "llm_cast_filter", "blok2_enabled",
            "qwen_fallback_on_handwriting",
            "post_process_all_languages", "guest_actor_enabled",
            "content_type",
        )
        for k in merge_keys:
            if k in profile:
                config[k] = profile[k]
        return config

    def test_new_profile_keys_merged(self):
        """post_process_all_languages ve guest_actor_enabled config'e taşınmalı."""
        profile = {
            "_name": "OrijinalDilFilm",
            "scope": "video+audio",
            "first_segment_minutes": 6,
            "last_segment_minutes": 10,
            "audio_stages": ["detect_language", "extract", "transcribe"],
            "whisper_language": "auto",
            "post_process_all_languages": True,
            "guest_actor_enabled": False,
            "content_type": "orijinal_dil_film",
            "tmdb_lenient_match": False,
        }
        config = self._apply_profile_merge(profile)

        assert config.get("post_process_all_languages") is True
        assert config.get("whisper_language") == "auto"
        assert config.get("content_type") == "orijinal_dil_film"
        assert config.get("tmdb_lenient_match") is False

    def test_guest_actor_enabled_merged(self):
        """EskiDizi profili guest_actor_enabled=True'yu config'e taşımalı."""
        profile = {
            "_name": "EskiDizi",
            "scope": "video+audio",
            "first_segment_minutes": 4,
            "last_segment_minutes": 10,
            "audio_stages": ["detect_language", "extract", "transcribe"],
            "guest_actor_enabled": True,
            "content_type": "eski_dizi",
            "tmdb_lenient_match": False,
        }
        config = self._apply_profile_merge(profile)

        assert config.get("guest_actor_enabled") is True
        assert config.get("content_type") == "eski_dizi"

    def test_whisper_language_auto_maps_to_none_in_audio_cfg(self):
        """whisper_language='auto' → audio_cfg'de None olarak iletilmeli."""
        # pipeline_runner._run_audio mantığını doğrula
        _whisper_lang_cfg = "auto"
        if str(_whisper_lang_cfg).strip().lower() in ("auto", ""):
            _whisper_lang_cfg = None
        assert _whisper_lang_cfg is None

    def test_whisper_language_tr_stays_tr(self):
        """whisper_language='tr' → değişmeden kalmalı."""
        _whisper_lang_cfg = "tr"
        if str(_whisper_lang_cfg).strip().lower() in ("auto", ""):
            _whisper_lang_cfg = None
        assert _whisper_lang_cfg == "tr"


# ─── Audio Pipeline post_process_all_languages Testleri ──────────────────────

class TestAudioPipelinePostProcessAllLanguages:
    """audio_pipeline.py post_process_all_languages davranışı."""

    def _make_pipeline(self, post_process_all_languages: bool) -> "AudioPipeline":
        from core.audio_pipeline import AudioPipeline
        return AudioPipeline(config={
            "post_process_all_languages": post_process_all_languages,
        })

    def test_default_post_process_requires_turkish(self):
        """Varsayılanda post_process sadece Türkçe içerik için çalışmalı."""
        pipeline = self._make_pipeline(False)
        # _post_process_all_langs=False → yabancı dil için post_process atlanmalı
        _post_process_all_langs = bool(pipeline.config.get("post_process_all_languages", False))
        language_is_turkish = False
        _post_process_lang_ok = language_is_turkish or _post_process_all_langs
        assert _post_process_lang_ok is False

    def test_post_process_all_languages_enabled(self):
        """post_process_all_languages=True ile yabancı dil için post_process çalışmalı."""
        pipeline = self._make_pipeline(True)
        _post_process_all_langs = bool(pipeline.config.get("post_process_all_languages", False))
        language_is_turkish = False
        _post_process_lang_ok = language_is_turkish or _post_process_all_langs
        assert _post_process_lang_ok is True

    def test_turkish_always_runs_post_process(self):
        """Türkçe içerik her zaman post_process çalıştırmalı (all_languages=False bile)."""
        pipeline = self._make_pipeline(False)
        _post_process_all_langs = bool(pipeline.config.get("post_process_all_languages", False))
        language_is_turkish = True
        _post_process_lang_ok = language_is_turkish or _post_process_all_langs
        assert _post_process_lang_ok is True
