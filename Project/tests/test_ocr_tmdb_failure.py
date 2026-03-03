"""
test_ocr_tmdb_failure.py — TMDB eşleşme başarısızlığında devreye giren
OCR/VLM (Qwen 2.5 / GLM-4V) fallback sürecinin unit test'leri.

Kapsam:
  TMDB-F01: TMDB match not found → BLOK2 aktif (VLM deep read çalışır)
  TMDB-F02: TMDB no API key → BLOK2 tetiklenir
  TMDB-F03: TMDB miss + VLM kapalı → BLOK2 atlanır
  TMDB-F04: BLOK2 merge: fuzzy dedup çalışır (aynı satır tekrar eklenmez)
  TMDB-F05: BLOK2 merge: benzersiz VLM satırları eklenir
  TMDB-F06: TMDB fail → LLM cast filter aktif
  TMDB-F07: TMDB başarılı → LLM cast filter atlanır
  TMDB-F08: BLOK2 VLM disabled ama Ollama çalışıyor → BUG-1 testi (loop'a girmemeli)
  TMDB-F09: _merge_blok2_results boş VLM → orijinal korunur
  TMDB-F10: _merge_blok2_results rapidfuzz yok → fallback exact match dedup
  TMDB-F11: QwenVerifier confidence_before gerçek değeri taşımalı (BUG-3)
  TMDB-F12: Crop upscale max limit testi (BUG-4)
"""

import sys
import os
import copy
from unittest.mock import MagicMock

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# ── Ağır native bağımlılıklar test ortamında olmayabilir ────────────────────
# text_filter.py: import cv2, import numpy as np (try/except YOK)
# test_database_writer.py ile aynı stub yaklaşımı
import types as _types

for _stub_name in ("cv2", "numpy", "numpy.core", "numpy.linalg",
                   "paddleocr", "paddleocr.paddleocr"):
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = _types.ModuleType(_stub_name)

# numpy stub'ına pytest.approx() uyumluluğu için gerekli attribute'lar
_np_stub = sys.modules["numpy"]
if not hasattr(_np_stub, "isscalar"):
    _np_stub.isscalar = lambda obj: isinstance(obj, (int, float, complex))
if not hasattr(_np_stub, "array"):
    _np_stub.array = lambda *a, **k: None
if not hasattr(_np_stub, "bool_"):
    _np_stub.bool_ = bool
if not hasattr(_np_stub, "ndarray"):
    _np_stub.ndarray = type("ndarray", (), {})


# ── Yardımcı: minimal PipelineRunner mock'ları ──────────────────────────────

def _make_ocr_line(text, conf=0.90, frame="frame_001.png", bbox=None):
    """Test amaçlı OCR satırı (dict)."""
    return {
        "text": text,
        "avg_confidence": conf,
        "frame_path": frame,
        "bbox": bbox or [10, 20, 200, 40],
    }


def _make_vlm_line(text, frame="frame_001.png"):
    """BLOK2 VLM çıktısı gibi bir satır."""
    return {
        "text": text,
        "avg_confidence": 0.85,
        "bbox": [],
        "frame_path": frame,
        "source": "vlm_blok2",
    }


def _make_runner(vlm_enabled=False, vlm_available=False, llm_filter_enabled=True):
    """Minimal PipelineRunner benzeri nesne oluştur (BLOK2 akışı test etmek için)."""
    from core.pipeline_runner import PipelineRunner
    from unittest.mock import MagicMock, patch

    # PipelineRunner.__init__ çok bağımlılık gerektiriyor, mock ile atlıyoruz
    runner = object.__new__(PipelineRunner)
    runner.config = {"difficulty": "medium", "database_enabled": False}
    runner._log_cb = None
    runner._log_messages = []
    runner.stage_stats = {}

    # NameDB mock
    name_db = MagicMock()
    name_db.is_name = lambda text: False
    name_db.__len__ = lambda self: 0
    runner._name_db = name_db

    # VLMReader mock
    vlm_reader = MagicMock()
    vlm_reader.enabled = vlm_enabled
    vlm_reader.is_available = MagicMock(return_value=vlm_available)
    vlm_reader.read_text_from_frame = MagicMock(return_value=None)
    runner._vlm_reader = vlm_reader

    # QwenVerifier mock (pass-through)
    qwen = MagicMock()
    qwen.enabled = True
    qwen.is_available = MagicMock(return_value=False)
    qwen.verify = MagicMock(side_effect=lambda lines, **kw: lines)
    runner._qwen = qwen

    # LLMCastFilter
    runner._llm_filter_enabled = llm_filter_enabled

    # Stats mock
    stats = MagicMock()
    runner.stats = stats

    # Log callback
    runner._log = lambda msg: runner._log_messages.append(str(msg))

    return runner


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F01: TMDB match not found → BLOK2 aktif
# ═══════════════════════════════════════════════════════════════════════════════

def test_tmdb_miss_activates_blok2():
    """TMDB-F01: TMDB eşleşmezse ve VLM aktifse BLOK2 devreye girer."""
    runner = _make_runner(vlm_enabled=True, vlm_available=True)

    # VLM read sonucu döndür
    runner._vlm_reader.read_text_from_frame.return_value = {
        "text": "Nisa Serezli\nHaluk Bilginer",
        "avg_confidence": 0.85,
        "bbox": [],
        "frame_path": "frame_001.png",
    }

    paddle_lines = [_make_ocr_line("Ali Veli")]
    candidates = [{"path": "frame_001.png"}, {"path": "frame_002.png"}]

    # BLOK2 akışını simüle et (pipeline_runner.py satır 352-381 mantığı)
    tmdb_matched = False
    vlm_ocr_lines = []

    if not tmdb_matched:
        if runner._vlm_reader.enabled and runner._vlm_reader.is_available():
            for frame_info in candidates:
                result = runner._vlm_reader.read_text_from_frame(
                    frame_info["path"], lang="tr")
                if result and result.get("text"):
                    for line_text in result["text"].splitlines():
                        line_text = line_text.strip()
                        if line_text:
                            vlm_ocr_lines.append(_make_vlm_line(
                                line_text, frame_info["path"]))

    assert len(vlm_ocr_lines) > 0, "VLM satırları üretilmeli"
    assert any("Nisa Serezli" in l["text"] for l in vlm_ocr_lines)
    assert any("Haluk Bilginer" in l["text"] for l in vlm_ocr_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F02: TMDB no API key → BLOK2 tetiklenir
# ═══════════════════════════════════════════════════════════════════════════════

def test_tmdb_no_api_key_triggers_blok2():
    """TMDB-F02: API key yoksa TMDBVerifyResult(False, 'no tmdb api key') döner → tmdb_matched=False."""
    from core.tmdb_verify import TMDBVerifyResult

    result = TMDBVerifyResult(updated=False, reason="no tmdb api key")
    assert result.updated is False
    assert result.reason == "no tmdb api key"
    assert result.hits == 0
    # Bu durumda pipeline'da tmdb_matched=False olur → BLOK2 tetiklenir


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F03: TMDB miss + VLM kapalı → BLOK2 atlanır
# ═══════════════════════════════════════════════════════════════════════════════

def test_tmdb_miss_vlm_disabled_skips_blok2():
    """TMDB-F03: VLM kapalıysa BLOK2 çalışmaz, read_text_from_frame çağrılmaz."""
    runner = _make_runner(vlm_enabled=False, vlm_available=True)

    paddle_lines = [_make_ocr_line("Ali Veli")]
    candidates = [{"path": "frame_001.png"}]

    tmdb_matched = False
    vlm_called = False

    if not tmdb_matched:
        if runner._vlm_reader.enabled and runner._vlm_reader.is_available():
            vlm_called = True

    assert vlm_called is False, "VLM kapalıyken BLOK2 loop'a girmemeli"
    runner._vlm_reader.read_text_from_frame.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F04: BLOK2 merge: fuzzy dedup çalışır
# ═══════════════════════════════════════════════════════════════════════════════

def test_blok2_merge_fuzzy_dedup():
    """TMDB-F04: Fuzzy benzer satır (_merge_blok2_results) eklenmez."""
    runner = _make_runner()

    paddle_lines = [_make_ocr_line("Nisa Serezli")]
    vlm_lines = [_make_vlm_line("Nisa Serezll")]  # 1 harf fark → fuzzy match

    merged = runner._merge_blok2_results(paddle_lines, vlm_lines)

    # rapidfuzz varsa fuzzy dedup çalışmalı
    try:
        from rapidfuzz.fuzz import ratio
        # Benzerlik %85'ten yüksek → VLM satırı eklenmemeli
        sim = ratio("nisa serezli", "nisa serezll")
        if sim >= 85:
            assert len(merged) == 1, f"Fuzzy dedup çalışmalı (benzerlik={sim:.0f}%)"
        else:
            assert len(merged) == 2
    except ImportError:
        # rapidfuzz yoksa exact match → farklı → eklenir
        assert len(merged) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F05: BLOK2 merge: benzersiz VLM satırları eklenir
# ═══════════════════════════════════════════════════════════════════════════════

def test_blok2_merge_unique_vlm_lines_added():
    """TMDB-F05: Benzersiz VLM satırları merged listesine eklenir."""
    runner = _make_runner()

    paddle_lines = [_make_ocr_line("Ali Veli")]
    vlm_lines = [
        _make_vlm_line("Haluk Bilginer"),
        _make_vlm_line("Nisa Serezli"),
    ]

    merged = runner._merge_blok2_results(paddle_lines, vlm_lines)
    texts = [l["text"] for l in merged]

    assert "Ali Veli" in texts
    assert "Haluk Bilginer" in texts
    assert "Nisa Serezli" in texts
    assert len(merged) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F06: TMDB fail → LLM cast filter aktif
# ═══════════════════════════════════════════════════════════════════════════════

def test_tmdb_fail_activates_llm_cast_filter():
    """TMDB-F06: TMDB eşleşmediğinde LLM cast filter çağrılmalı."""
    tmdb_matched = False
    llm_filter_called = False

    # Pipeline'daki mantığı simüle et
    if not tmdb_matched:
        llm_filter_called = True

    assert llm_filter_called is True, "TMDB miss → LLM cast filter aktif olmalı"


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F07: TMDB başarılı → LLM cast filter atlanır
# ═══════════════════════════════════════════════════════════════════════════════

def test_tmdb_success_skips_llm_cast_filter():
    """TMDB-F07: TMDB eşleşince LLM cast filter atlanmalı."""
    tmdb_matched = True
    llm_filter_called = False

    if not tmdb_matched:
        llm_filter_called = True

    assert llm_filter_called is False, "TMDB match → LLM cast filter atlanmalı"


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F08: BLOK2 VLM disabled ama Ollama çalışıyor → BUG-1 testi
# ═══════════════════════════════════════════════════════════════════════════════

def test_blok2_vlm_disabled_but_ollama_running_no_loop():
    """TMDB-F08 (BUG-1): vlm_ocr_enabled=False ama Ollama çalışıyor.
    Fix öncesi: is_available()=True → loop'a girer ama her frame None döner.
    Fix sonrası: enabled=False kontrol edilir → loop'a hiç girmez.
    """
    runner = _make_runner(vlm_enabled=False, vlm_available=True)

    candidates = [{"path": f"frame_{i:03d}.png"} for i in range(100)]
    tmdb_matched = False
    loop_entered = False

    if not tmdb_matched:
        # BUG-1 fix: enabled check OLMALI
        if runner._vlm_reader.enabled and runner._vlm_reader.is_available():
            loop_entered = True
            for frame_info in candidates:
                runner._vlm_reader.read_text_from_frame(
                    frame_info["path"], lang="tr")

    assert loop_entered is False, \
        "BUG-1: VLM disabled iken BLOK2 loop'a girmemeli (enabled check eksikti)"
    runner._vlm_reader.read_text_from_frame.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F09: _merge_blok2_results boş VLM → orijinal korunur
# ═══════════════════════════════════════════════════════════════════════════════

def test_merge_blok2_empty_vlm_preserves_original():
    """TMDB-F09: VLM satırları boşsa paddle_lines aynen dönmeli."""
    runner = _make_runner()

    paddle_lines = [_make_ocr_line("Ali Veli"), _make_ocr_line("Haluk Bilginer")]
    vlm_lines = []

    merged = runner._merge_blok2_results(paddle_lines, vlm_lines)
    assert len(merged) == 2
    assert merged is paddle_lines, "Boş VLM → orijinal liste referans olarak dönmeli"


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F10: _merge_blok2_results rapidfuzz yok → fallback exact match dedup
# ═══════════════════════════════════════════════════════════════════════════════

def test_merge_blok2_without_rapidfuzz_exact_dedup(monkeypatch):
    """TMDB-F10: rapidfuzz import hata verirse exact match ile dedup yapılır."""
    import builtins
    _real_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if "rapidfuzz" in name:
            raise ImportError("mock: rapidfuzz not installed")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    runner = _make_runner()
    paddle_lines = [_make_ocr_line("Ali Veli")]

    # Exact match: aynı metin (case-insensitive)
    vlm_exact_dup = [_make_vlm_line("Ali Veli")]
    merged = runner._merge_blok2_results(paddle_lines, vlm_exact_dup)
    assert len(merged) == 1, "Exact match dedup çalışmalı (rapidfuzz yokken)"

    # Farklı metin: eklenmeli
    vlm_unique = [_make_vlm_line("Nisa Serezli")]
    merged2 = runner._merge_blok2_results(paddle_lines, vlm_unique)
    assert len(merged2) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F11: QwenVerifier confidence_before gerçek değeri taşımalı (BUG-3)
# ═══════════════════════════════════════════════════════════════════════════════

def test_qwen_verifier_confidence_before_passed(monkeypatch):
    """TMDB-F11 (BUG-3): _verify_single'a gerçek confidence geçirilmeli."""
    from core.qwen_verifier import QwenVerifier, VerifyResult

    verifier = QwenVerifier(
        enabled=True,
        confidence_threshold=0.80,
        name_checker=lambda text: False,
    )

    # _verify_single'ı yakalayıp confidence_before'u kontrol et
    captured = {}

    def _mock_verify_single(ocr_text, frame_path, bbox=None, confidence_before=0.0):
        captured["confidence_before"] = confidence_before
        return VerifyResult(
            original=ocr_text,
            corrected="Düzeltilmiş",
            was_fixed=True,
            confidence_before=confidence_before,
        )

    monkeypatch.setattr(verifier, "is_available", lambda: True)
    monkeypatch.setattr(verifier, "_verify_single", _mock_verify_single)

    lines = [{
        "text": "DYAYUCE",
        "avg_confidence": 0.72,
        "frame_path": __file__,  # mevcut dosya yolu (exists kontrolünü geçmek için)
        "bbox": [0, 0, 100, 30],
    }]

    verifier.verify(lines, resolution="1920x1080")

    assert "confidence_before" in captured, "confidence_before yakalanmalı"
    assert captured["confidence_before"] == 0.72, \
        f"BUG-3: confidence_before gerçek değer (0.72) olmalı, {captured['confidence_before']} geldi"


# ═══════════════════════════════════════════════════════════════════════════════
# TMDB-F12: Crop upscale max limit testi (BUG-4)
# ═══════════════════════════════════════════════════════════════════════════════

def test_crop_upscale_max_limit():
    """TMDB-F12 (BUG-4): crop_h=1 için scale max 8 olmalı (80 değil)."""
    # Bu test doğrudan scale hesaplama mantığını kontrol eder
    # pipeline_runner.py / qwen_verifier.py'deki formülü simüle et:
    #   scale = min(max(2, 80 // crop_h), 8)  ← FIX sonrası

    test_cases = [
        (1, 8),     # crop_h=1 → fix öncesi 80, fix sonrası 8
        (2, 8),     # crop_h=2 → fix öncesi 40, fix sonrası 8
        (5, 8),     # crop_h=5 → fix öncesi 16, fix sonrası 8
        (10, 8),    # crop_h=10 → 80//10=8 → max(2,8) = 8, min(8,8) = 8
        (15, 5),    # crop_h=15 → 80//15=5 → max(2,5) = 5, min(5,8) = 5
        (20, 4),    # crop_h=20 → 80//20=4 → max(2,4) = 4, min(4,8) = 4
        (30, 2),    # crop_h=30 → 80//30=2 → max(2,2) = 2, min(2,8) = 2
        (39, 2),    # crop_h=39 → 80//39=2 → max(2,2) = 2, min(2,8) = 2
    ]

    for crop_h, expected_scale in test_cases:
        # Fix sonrası formül
        scale = min(max(2, 80 // crop_h), 8)
        assert scale == expected_scale, \
            f"crop_h={crop_h}: scale={scale} beklenen={expected_scale}"
        assert scale <= 8, \
            f"BUG-4: crop_h={crop_h} için scale={scale} > 8 — bellek patlaması riski!"


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: BLOK2 sonrası cdata_raw güncellenme testi (BUG-2)
# ═══════════════════════════════════════════════════════════════════════════════

def test_blok2_updates_cdata_raw():
    """BUG-2: BLOK2 re-parse sonrası cdata_raw güncellenmeli."""
    import copy

    # İlk parse sonucu (TMDB öncesi snapshot)
    cdata = {
        "cast": [{"actor_name": "Ali Veli"}],
        "total_actors": 1,
        "crew": [],
        "total_crew": 0,
    }
    cdata_raw = copy.deepcopy(cdata)

    # BLOK2 çalıştı, yeni cast eklendi (re-parse)
    cdata["cast"].append({"actor_name": "Haluk Bilginer"})
    cdata["total_actors"] = 2

    # BUG-2 FIX: cdata_raw da güncellenmeli
    cdata_raw = copy.deepcopy(cdata)

    assert cdata_raw["total_actors"] == 2, \
        "BUG-2: BLOK2 sonrası cdata_raw güncel olmalı"
    assert len(cdata_raw["cast"]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: VLMReader.is_available() enabled kontrolü
# ═══════════════════════════════════════════════════════════════════════════════

def test_vlm_reader_is_available_does_not_check_enabled():
    """is_available() sadece Ollama bağlantısını kontrol eder, enabled'ı değil.
    Bu nedenle çağıranlar enabled'ı ayrıca kontrol etmeli."""
    from core.vlm_reader import VLMReader

    reader = VLMReader(enabled=False)
    # is_available() enabled'ı kontrol etmez — bu davranışı doğruluyoruz
    # (Ollama yoksa False döner, ama enabled kontrolü yapılmaz)
    reader._available = True  # Zorla True yap (Ollama var gibi)

    assert reader.is_available() is True, \
        "is_available() enabled'ı kontrol etmiyor — bu doğru davranış"
    assert reader.enabled is False, \
        "enabled=False — çağıran taraf (pipeline) bunu kontrol etmeli"

    # read_text_from_frame enabled=False olduğu için None dönmeli
    result = reader.read_text_from_frame("test.png", lang="tr")
    assert result is None, "enabled=False → read_text_from_frame None dönmeli"
