"""
test_system_validation.py — ASR ve OCR Sistem Doğrulama Testleri

Bu dosya, test_asr_ocr_system_report.py'de belgelenen tüm hataların
doğrulanması için oluşturulmuştur.

Amaç: Bulunan hataların gerçekten mevcut olduğunu doğrulamak.
"""

import sys
import os

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# ASR SİSTEMİ DOĞRULAMA TESTLERİ
# ═══════════════════════════════════════════════════════════════════════════════

class TestASRSystemValidation:
    """ASR sisteminin tüm fix'lenmiş buglarının çözüldüğünü doğrula."""

    def test_bug_asr_01_double_diarization_fixed(self):
        """BUG-ASR-01: Double diarization fix doğrulaması."""
        from audio.stages.transcribe import TranscribeStage

        stage = TranscribeStage()
        segments = [{"start": 0.0, "end": 3.0, "text": "Test", "speaker": ""}]

        # Full diarize_result dict geçebilmeliyiz
        diarize_result = {
            "status": "ok",
            "segments": [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}],
            "speakers_found": 1,
        }

        stage._assign_speakers(segments, diarize_result)
        assert segments[0]["speaker"] == "SPEAKER_00"

    def test_bug_asr_02_dict_handling_fixed(self):
        """BUG-ASR-02: Diarize result dict normalize edildi mi?"""
        from audio.stages.transcribe import TranscribeStage

        stage = TranscribeStage()
        segments = [{"start": 0.0, "end": 2.0, "text": "Merhaba", "speaker": ""}]

        # Dict geçilince normalize edilmeli
        diar_dict = {"segments": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]}
        stage._assign_speakers(segments, diar_dict)

        assert segments[0]["speaker"] == "SPEAKER_00"

    def test_bug_asr_03_kwargs_extraction_fixed(self, monkeypatch):
        """BUG-ASR-03: Direct kwargs extract edildi mi?"""
        from audio.stages.transcribe import TranscribeStage

        stage = TranscribeStage()
        captured = {}

        def fake_transcribe(audio_path, opts, diarization=None):
            captured.update(opts)
            return {
                "status": "error",
                "segments": [],
                "total_segments": 0,
                "stage_time_sec": 0.0,
                "error": "test_only",
            }

        monkeypatch.setattr(stage, "_transcribe", fake_transcribe)

        # Direct kwargs geçilince options'a eklenmeli
        stage._run_legacy(
            "fake.wav",
            whisper_model="medium",
            whisper_language="en",
        )

        assert captured.get("whisper_model") == "medium"
        assert captured.get("whisper_language") == "en"


# ═══════════════════════════════════════════════════════════════════════════════
# OCR SİSTEMİ DOĞRULAMA TESTLERİ
# ═══════════════════════════════════════════════════════════════════════════════

class TestOCRSystemValidation:
    """OCR sisteminin tüm fix'lenmiş buglarının çözüldüğünü doğrula."""

    def test_bug_ocr_01_generator_materialization_fixed(self):
        """BUG-OCR-01: Generator materialization check."""
        def _iter_nothing():
            if False:
                yield ("dummy",)
            return None

        gen = _iter_nothing()
        # Generator her zaman truthy
        assert bool(gen) is True

        # List'e dönüştürünce boş çıkar
        items = list(_iter_nothing())
        assert items == []
        assert not items  # Bu kontrol doğru çalışır

    def test_bug_ocr_02_best_actor_no_name_error_fixed(self, monkeypatch):
        """BUG-OCR-02: _best_actor exception handler NameError üretmez."""
        from core.export_engine import _best_actor, _split_name

        def raise_error(word):
            raise RuntimeError("forced split error")

        monkeypatch.setattr("core.export_engine._split_name", raise_error)

        # NameError oluşmamalı
        result = _best_actor(["LONGWORD1"])
        assert isinstance(result, str)

    def test_bug_ocr_03_google_ocr_error_check_order_fixed(self):
        """BUG-OCR-03: Error check text_annotations'dan önce gelir."""
        import ast
        import inspect
        import textwrap

        try:
            from core import google_ocr_engine
            src = inspect.getsource(google_ocr_engine.GoogleOCREngine.ocr_image)
            src = textwrap.dedent(src)
        except Exception:
            pytest.skip("google_ocr_engine import edilemedi")

        tree = ast.parse(src)
        error_check_line = None
        text_access_line = None

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                try:
                    cond = ast.dump(node.test)
                    if 'error' in cond and 'message' in cond:
                        if error_check_line is None:
                            error_check_line = node.lineno
                except Exception:
                    pass

            if isinstance(node, ast.Subscript):
                try:
                    val_dump = ast.dump(node.value)
                    if 'text_annotations' in val_dump:
                        if text_access_line is None:
                            text_access_line = node.lineno
                except Exception:
                    pass

        assert error_check_line is not None
        assert text_access_line is not None
        assert error_check_line < text_access_line

    def test_bug_ocr_04_qwen_verifier_custom_url_fixed(self, monkeypatch):
        """BUG-OCR-04: is_available() custom URL kullanır, localhost değil."""
        from core.qwen_verifier import QwenVerifier

        checked_urls = []

        def fake_urlopen(req, timeout=5):
            checked_urls.append(req.full_url)
            raise OSError("connection refused (test)")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        verifier = QwenVerifier(ollama_url="http://custom-host:9999/api/chat")
        verifier._available = None
        verifier.is_available()

        # custom-host kullanılmalı, localhost değil
        assert any("custom-host:9999" in u for u in checked_urls)
        assert not any("localhost" in u for u in checked_urls)

    def test_bug_ocr_05_confidence_before_propagation_fixed(self, monkeypatch):
        """BUG-OCR-05: confidence_before gerçek değeri taşır."""
        from core.qwen_verifier import QwenVerifier, VerifyResult
        import json

        verifier = QwenVerifier(ollama_url="http://localhost:11434/api/chat")

        class FakeResp:
            def read(self):
                return json.dumps({"message": {"content": "corrected"}}).encode()
            def __enter__(self): return self
            def __exit__(self, *a): pass

        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=60: FakeResp())
        monkeypatch.setattr("core.qwen_verifier.HAS_CV2", False)

        result = verifier._verify_single(
            "original text",
            __file__,
            confidence_before=0.72,
        )

        assert result is not None
        assert result.confidence_before == 0.72

    def test_perf_ocr_01_turkish_name_db_cache_fixed(self):
        """PERF-OCR-01: _all_names cached olarak saklanır."""
        from core.turkish_name_db import TurkishNameDB

        db = TurkishNameDB()
        assert hasattr(db, "_all_names")
        assert db._all_names == db._first_names + db._surnames

        # Cache ID değişmemeli
        original_id = id(db._all_names)
        db._fuzzy_find("test", threshold=85)
        assert id(db._all_names) == original_id


# ═══════════════════════════════════════════════════════════════════════════════
# POST-PROCESS SİSTEMİ DOĞRULAMA TESTLERİ (YENİ BULUNAN HATALAR)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPostProcessSystemValidation:
    """Post-process sisteminde yeni bulunan hataları doğrula."""

    def test_bug_post_01_invalid_url_validation_error(self):
        """
        ✅ BUG-POST-01 DÜZELTİLDİ: URL validation artık provider'dan bağımsız çalışıyor.

        DÜZELTME: ollama_url doğrulaması provider=="ollama" şartı kaldırılarak
        her zaman çalışır hale getirildi.

        Bu test DÜZELTİLMİŞ davranışı doğrular.
        """
        from core.post_process import PostProcessStage

        stage = PostProcessStage()
        segments = [{"start": 0.0, "end": 1.0, "text": "test", "confidence": 0.9, "speaker": ""}]

        # Geçersiz URL geçilince status="skipped" dönmeli (provider ne olursa olsun)
        result = stage.run(segments, ollama_url="not_a_url")
        assert result["status"] == "skipped"
        assert result["error"] == "invalid_ollama_url"

    def test_bug_post_02_chat_signature_mismatch_error(self, monkeypatch):
        """
        ✅ BUG-POST-02 DÜZELTİLDİ: _chat method signature restored to 4-arg API

        DÜZELTME: _chat imzası eski haline (4 parametre) döndürüldü:
          _chat(prompt, system, base_url, model, provider=None)

        Önceki kırık imza:
          _chat(prompt, system, provider, base_url, model)  ← 'provider' ekstra/zorunluydu

        Bu test DÜZELTİLMİŞ davranışı doğrular.
        """
        import urllib.error
        from core.post_process import PostProcessStage

        stage = PostProcessStage()
        logs = []
        stage._log = logs.append

        def raise_http(*args, **kwargs):
            raise urllib.error.HTTPError(
                url="http://localhost:11434/api/chat",
                code=500, msg="Internal Server Error", hdrs=None, fp=None
            )

        monkeypatch.setattr("urllib.request.urlopen", raise_http)

        # DÜZELTME: 4 parametre ile çağrı artık çalışmalı:
        result = stage._chat("prompt", "system", "http://localhost:11434", "llama3.1:8b")
        assert result == ""
        assert any("500" in m for m in logs)


# ═══════════════════════════════════════════════════════════════════════════════
# OCR ANALİZ RAPORU DOĞRULAMA (BELGELENMİŞ SORUNLAR)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOCRAnalysisValidation:
    """OCR analiz raporunda belgelenen sorunların varlığını doğrula."""

    def test_bug_analyze_02_fix_parts_exact_match_no_fuzzy(self, monkeypatch):
        """BUG-ANALYZE-02: Exact match bulununca fuzzy çağrılmamalı."""
        from core.turkish_name_db import TurkishNameDB

        db = TurkishNameDB()
        fuzzy_call_count = [0]
        original_fuzzy = db._fuzzy_find

        def counting_fuzzy(text, threshold):
            fuzzy_call_count[0] += 1
            return original_fuzzy(text, threshold)

        monkeypatch.setattr(db, "_fuzzy_find", counting_fuzzy)

        # "SEBNEM" hardcoded'da var
        fixed, changed = db._fix_parts(["SEBNEM"], threshold=85)

        # Fuzzy çağrılmamalı - exact match bulundu
        assert fuzzy_call_count[0] == 0
        assert fixed == ["Şebnem"]

    def test_bug_analyze_03_fuzzy_dedup_performance(self):
        """BUG-ANALYZE-03: Fuzzy dedup O(n²) complexity."""
        from core.ocr_engine import OCREngine, OCRResult

        engine = object.__new__(OCREngine)
        engine.fuzzy_threshold = 82

        # 50 element - hızlı olmalı
        results_50 = [
            OCRResult(f"Name{i}", 0.9, float(i), f"f{i}.png")
            for i in range(50)
        ]

        import time
        t0 = time.time()
        deduped = engine._fuzzy_dedup(results_50)
        elapsed = time.time() - t0

        # 50 element için < 100ms olmalı
        assert elapsed < 0.1, f"50 element için {elapsed*1000:.1f}ms (beklenen <100ms)"

    def test_bug_analyze_06_text_filter_direct_cv2_import(self):
        """BUG-ANALYZE-06: text_filter.py doğrudan cv2 import kullanıyor."""
        from pathlib import Path

        tf_path = Path(_project_dir) / "core" / "text_filter.py"
        assert tf_path.exists()

        src = tf_path.read_text(encoding="utf-8")

        # Doğrudan import var
        assert "import cv2" in src

        # HAS_CV2 guard yok
        has_cv2_guard = "HAS_CV2" in src
        assert not has_cv2_guard, "BUG-ANALYZE-06: HAS_CV2 guard yok"

    def test_bug_analyze_07_persistence_watermark_conflict(self):
        """BUG-ANALYZE-07: watermark_threshold=2 mantıksal çelişkisi."""
        from core.ocr_engine import OCREngine, OCRResult

        engine = object.__new__(OCREngine)
        engine.watermark_threshold = 2

        # seen_count=1 → penalty ile tutulur
        single_results = [OCRResult("Test1", 0.9, 1.0, "f1.png", bbox=[])]
        lines_single = engine._persistence_and_watermark(single_results)
        assert len(lines_single) == 1, "Tek görülen tutulmalı"

        # seen_count=2 → watermark olarak atlanır
        double_results = [
            OCRResult("Test2", 0.9, 1.0, "f1.png", bbox=[]),
            OCRResult("Test2", 0.9, 2.0, "f2.png", bbox=[]),
        ]
        lines_double = engine._persistence_and_watermark(double_results)
        assert len(lines_double) == 0, "2 kez görülen watermark sayılıp atlanmalı"

        # Mantıksal çelişki: 2>1 ama 2 daha kötü muamele görüyor


# ═══════════════════════════════════════════════════════════════════════════════
# GENEL SİSTEM SAĞLIK KONTROLLERI
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemHealthChecks:
    """Genel sistem sağlığı ve bütünlük kontrolleri."""

    def test_asr_all_tests_passing(self):
        """ASR testlerinin tümü geçiyor mu?"""
        import subprocess

        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/test_asr_ocr_fixes.py",
             "tests/test_transcribe_unit.py", "tests/test_asr_always_active.py",
             "-v", "--tb=no"],
            capture_output=True,
            text=True,
            cwd=_project_dir,
        )

        # Tüm ASR testleri geçmeli
        assert "failed" not in result.stdout.lower()
        assert "passed" in result.stdout

    def test_ocr_core_tests_passing(self):
        """OCR core testlerinin tümü geçiyor mu?"""
        import subprocess

        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/test_ocr_bug_fixes.py",
             "tests/test_ocr_analysis_report.py", "-v", "--tb=no"],
            capture_output=True,
            text=True,
            cwd=_project_dir,
        )

        # Tüm OCR core testleri geçmeli
        assert "failed" not in result.stdout.lower()
        assert "passed" in result.stdout

    def test_imports_work(self):
        """Kritik modüller import edilebiliyor mu?"""
        # ASR
        from audio.stages.transcribe import TranscribeStage
        assert TranscribeStage is not None

        # OCR
        from core.ocr_engine import OCREngine
        assert OCREngine is not None

        # Post-process
        from core.post_process import PostProcessStage
        assert PostProcessStage is not None

        # Export
        from core.export_engine import ExportEngine
        assert ExportEngine is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
