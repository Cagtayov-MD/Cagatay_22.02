"""
test_hybrid_ocr.py — HybridOCRRouter birim testleri.

Senaryolar:
  HYBRID-01: HybridOCRRouter import edilebiliyor mu?
  HYBRID-02: Karar mekanizması — yüksek confidence → Qwen atlanıyor
  HYBRID-03: Karar mekanizması — düşük confidence + handwriting → Qwen gerek
  HYBRID-04: Karar mekanizması — NameDB eşleşme oranı yüksek → Qwen atlanıyor
  HYBRID-05: process_frames arayüzü doğru (ocr_lines, layout_pairs) döndürüyor mu?
  HYBRID-06: oneocr yok + Qwen var → Qwen tek başına çalışıyor
  HYBRID-07: qwen_fallback_on_handwriting: false → Qwen hiç çalışmıyor
  HYBRID-08: Merge — Qwen birincil, oneocr eklentisi flag ile ekleniyor
  HYBRID-09: Merge — boş Qwen → oneocr sonucu korunuyor
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core.hybrid_ocr_router import (
    HybridOCRRouter,
    CONF_HIGH,
    CONF_LOW,
    NAMEDB_MATCH_THRESHOLD,
)
from core.ocr_engine import OCRLine


# ── Yardımcı: sahte OCRLine ───────────────────────────────────────────────────

def _make_line(text="TEST LINE", conf=0.95, frame_path="/tmp/f.png", first_seen=0.0):
    return OCRLine(
        text=text,
        first_seen=first_seen,
        last_seen=first_seen + 0.5,
        seen_count=2,
        avg_confidence=conf,
        bbox=[0, 0, 100, 30],
        frame_path=frame_path,
        source="oneocr",
    )


def _make_frame(path="/tmp/f.png"):
    return {"path": path, "timecode_sec": 1.0}


def _make_meta(has_text=True, line_count=1, avg_confidence=0.95, font_type="standard"):
    return {
        "has_text": has_text,
        "line_count": line_count,
        "avg_confidence": avg_confidence,
        "font_type": font_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID-01: Import
# ─────────────────────────────────────────────────────────────────────────────

def test_hybrid_import():
    """HYBRID-01: HybridOCRRouter doğru import edilmeli."""
    from core.hybrid_ocr_router import HybridOCRRouter
    assert HybridOCRRouter is not None


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID-02: Yüksek confidence → Qwen atlanıyor
# ─────────────────────────────────────────────────────────────────────────────

def test_decide_high_confidence_skips_qwen():
    """HYBRID-02: Yüksek confidence frame'leri Qwen kuyruğuna girmemeli."""
    mock_oneocr = MagicMock()
    mock_qwen = MagicMock()

    router = object.__new__(HybridOCRRouter)
    router.cfg = {}
    router._log = lambda m: None
    router._name_db = None
    router._qwen_fallback = True
    router._oneocr = mock_oneocr
    router._qwen = mock_qwen
    router._font_cache = {}

    frames = [_make_frame("/tmp/a.png"), _make_frame("/tmp/b.png")]
    frame_meta = {
        "/tmp/a.png": _make_meta(avg_confidence=CONF_HIGH + 0.01, font_type="handwriting"),
        "/tmp/b.png": _make_meta(avg_confidence=CONF_HIGH + 0.02, font_type="handwriting"),
    }

    with patch("pathlib.Path.exists", return_value=True):
        result = router._decide_qwen_frames(frames, [], frame_meta, lambda m: None)

    mock_oneocr.estimate_font_type.assert_not_called()
    assert result == [], f"Yüksek confidence → Qwen boş olmalı, ama {result!r} döndü"


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID-03: Düşük confidence + handwriting → Qwen gerek
# ─────────────────────────────────────────────────────────────────────────────

def test_decide_low_confidence_needs_qwen():
    """HYBRID-03: Düşük confidence frame'i Qwen kuyruğuna girmeli."""
    mock_oneocr = MagicMock()
    mock_oneocr.estimate_font_type.return_value = "handwriting"

    router = object.__new__(HybridOCRRouter)
    router.cfg = {}
    router._log = lambda m: None
    router._name_db = None
    router._qwen_fallback = True
    router._oneocr = mock_oneocr
    router._qwen = MagicMock()
    router._font_cache = {}

    frames = [_make_frame("/tmp/hw.png")]
    frame_meta = {
        "/tmp/hw.png": _make_meta(avg_confidence=CONF_LOW - 0.05, font_type="standard"),
    }

    with patch("pathlib.Path.exists", return_value=True):
        result = router._decide_qwen_frames(frames, [], frame_meta, lambda m: None)

    mock_oneocr.estimate_font_type.assert_not_called()
    assert len(result) == 1, f"Düşük confidence → 1 frame Qwen'e gitmeli, {result!r}"


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID-04: NameDB eşleşme oranı yüksek → Qwen atlanıyor
# ─────────────────────────────────────────────────────────────────────────────

def test_decide_namedb_high_ratio_skips_qwen():
    """HYBRID-04: NameDB eşleşme oranı ≥ eşik → Qwen atlanmalı."""
    mock_oneocr = MagicMock()
    mock_oneocr.estimate_font_type.return_value = "decorative"

    mock_namedb = MagicMock()
    mock_namedb.is_name.return_value = True  # find() yerine is_name() kullanılıyor

    router = object.__new__(HybridOCRRouter)
    router.cfg = {}
    router._log = lambda m: None
    router._name_db = mock_namedb
    router._qwen_fallback = True
    router._oneocr = mock_oneocr
    router._qwen = MagicMock()
    router._font_cache = {}

    # Orta confidence → Katman 2 (decorative) → Katman 3
    mid_conf = (CONF_HIGH + CONF_LOW) / 2.0
    lines = [
        _make_line("Nisa Serezli", conf=mid_conf, frame_path="/tmp/c.png"),
        _make_line("Selma Türkdoğan", conf=mid_conf, frame_path="/tmp/c.png"),
    ]
    frames = [_make_frame("/tmp/c.png")]
    frame_meta = {
        "/tmp/c.png": _make_meta(avg_confidence=mid_conf, font_type="decorative"),
    }

    with patch("pathlib.Path.exists", return_value=True):
        result = router._decide_qwen_frames(frames, lines, frame_meta, lambda m: None)

    mock_oneocr.estimate_font_type.assert_not_called()
    # NameDB oranı yüksek → Qwen listesi boş
    assert result == [], f"NameDB oranı yüksek → Qwen boş olmalı, {result!r}"


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID-05: process_frames arayüzü doğru döndürüyor
# ─────────────────────────────────────────────────────────────────────────────

def test_process_frames_returns_tuple():
    """HYBRID-05: process_frames (ocr_lines, layout_pairs) demeti döndürmeli."""
    mock_oneocr = MagicMock()
    expected_lines = [_make_line("KATHLEEN HERBERT", conf=0.97)]
    expected_pairs = [MagicMock()]
    mock_oneocr.process_frames.return_value = (expected_lines, expected_pairs)

    router = object.__new__(HybridOCRRouter)
    router.cfg = {"qwen_fallback_on_handwriting": True}
    router._log = lambda m: None
    router._name_db = None
    router._qwen_fallback = True
    router._oneocr = mock_oneocr
    router._qwen = MagicMock()
    router._font_cache = {}
    mock_oneocr._last_frame_meta = {"/tmp/f.png": _make_meta(avg_confidence=CONF_HIGH + 0.05)}

    # _decide_qwen_frames'i mock ile bypass et
    router._decide_qwen_frames = MagicMock(return_value=[])

    frames = [_make_frame()]
    result = router.process_frames(frames)

    assert isinstance(result, tuple), "process_frames demet döndürmeli"
    assert len(result) == 2, "process_frames (ocr_lines, layout_pairs) döndürmeli"
    lines, pairs = result
    assert isinstance(lines, list)
    assert isinstance(pairs, list)


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID-06: oneocr yok → Qwen tek başına çalışıyor
# ─────────────────────────────────────────────────────────────────────────────

def test_no_oneocr_uses_qwen_alone():
    """HYBRID-06: oneocr kurulu değilken Qwen tek başına çalışmalı."""
    qwen_lines = [_make_line("Nisa Serezli", conf=0.92)]
    mock_qwen = MagicMock()
    mock_qwen.process_frames.return_value = (qwen_lines, [])

    router = object.__new__(HybridOCRRouter)
    router.cfg = {}
    router._log = lambda m: None
    router._name_db = None
    router._qwen_fallback = True
    router._oneocr = None
    router._qwen = mock_qwen

    frames = [_make_frame()]
    lines, pairs = router.process_frames(frames)

    mock_qwen.process_frames.assert_called_once()
    assert lines == qwen_lines


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID-07: qwen_fallback_on_handwriting: false → Qwen çalışmıyor
# ─────────────────────────────────────────────────────────────────────────────

def test_qwen_fallback_disabled():
    """HYBRID-07: qwen_fallback_on_handwriting false olunca Qwen çalışmamalı."""
    oneocr_lines = [_make_line("JACK KOSSLYN", conf=0.60)]
    mock_oneocr = MagicMock()
    mock_oneocr.process_frames.return_value = (oneocr_lines, [])

    mock_qwen = MagicMock()

    router = object.__new__(HybridOCRRouter)
    router.cfg = {"qwen_fallback_on_handwriting": False}
    router._log = lambda m: None
    router._name_db = None
    router._qwen_fallback = False   # ← devre dışı
    router._oneocr = mock_oneocr
    router._qwen = mock_qwen

    frames = [_make_frame()]
    lines, pairs = router.process_frames(frames)

    mock_qwen.process_frames.assert_not_called()
    assert lines == oneocr_lines


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID-08: Merge — Qwen birincil, oneocr eklentisi flag ile ekleniyor
# ─────────────────────────────────────────────────────────────────────────────

def test_merge_qwen_primary_oneocr_supplement():
    """HYBRID-08: Merge'de Qwen satırları birincil, oneocr'da extra satırlar eklenmeli."""
    router = object.__new__(HybridOCRRouter)
    router.cfg = {}
    router._log = lambda m: None
    router._name_db = None
    router._qwen_fallback = True
    router._oneocr = MagicMock()
    router._qwen = MagicMock()
    router._font_cache = {}

    qwen_lines = [_make_line("Nisa Serezli", conf=0.92, frame_path="/tmp/q.png")]
    # oneocr'da farklı bir satır var (Qwen'de yok)
    oneocr_lines = [
        _make_line("Nisa Serezli", conf=0.75, frame_path="/tmp/q.png"),
        _make_line("Selma Türkdoğan", conf=0.80, frame_path="/tmp/q.png"),
    ]

    merged = router._merge_results(
        oneocr_lines=oneocr_lines,
        qwen_lines=qwen_lines,
        cb=lambda m: None,
    )

    texts = [getattr(l, "text", "") for l in merged]
    assert "Nisa Serezli" in texts
    assert "Selma Türkdoğan" in texts
    # Toplam satır: 1 (Qwen) + 1 (extra oneocr)
    assert len(merged) == 2


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID-09: Merge — boş Qwen → oneocr sonucu korunuyor
# ─────────────────────────────────────────────────────────────────────────────

def test_merge_empty_qwen_preserves_oneocr():
    """HYBRID-09: Qwen boş dönerse oneocr satırları korunmalı."""
    router = object.__new__(HybridOCRRouter)
    router.cfg = {}
    router._log = lambda m: None
    router._name_db = None
    router._qwen_fallback = True
    router._oneocr = MagicMock()
    router._qwen = MagicMock()
    router._font_cache = {}

    oneocr_lines = [
        _make_line("KATHLEEN HERBERT", conf=0.97),
        _make_line("JACK KOSSLYN", conf=0.94),
    ]

    merged = router._merge_results(
        oneocr_lines=oneocr_lines,
        qwen_lines=[],
        cb=lambda m: None,
    )

    assert merged == oneocr_lines, "Qwen boş → oneocr sonucu değişmeden dönmeli"


def test_decide_fallbacks_to_estimate_when_frame_meta_missing():
    """Metadata yoksa güvenlik fallback'i olan estimate_font_type çalışmalı."""
    mock_oneocr = MagicMock()
    mock_oneocr.estimate_font_type.return_value = "handwriting"

    router = object.__new__(HybridOCRRouter)
    router.cfg = {}
    router._log = lambda m: None
    router._name_db = None
    router._qwen_fallback = True
    router._oneocr = mock_oneocr
    router._qwen = MagicMock()
    router._font_cache = {}

    frames = [_make_frame("/tmp/missing-meta.png")]
    with patch("pathlib.Path.exists", return_value=True):
        result = router._decide_qwen_frames(frames, [], {}, lambda m: None)

    mock_oneocr.estimate_font_type.assert_called_once_with("/tmp/missing-meta.png")
    assert len(result) == 1


def test_phase2_progress_logging_emits_updates():
    """50 frame üstünde Phase 2 ilerleme logları görünür olmalı."""
    mock_oneocr = MagicMock()

    router = object.__new__(HybridOCRRouter)
    router.cfg = {}
    router._log = lambda m: None
    router._name_db = None
    router._qwen_fallback = True
    router._oneocr = mock_oneocr
    router._qwen = MagicMock()
    router._font_cache = {}

    frames = [_make_frame(f"/tmp/frame_{i}.png") for i in range(51)]
    frame_meta = {
        frame["path"]: _make_meta(
            avg_confidence=(CONF_LOW - 0.01) if i == 0 else (CONF_HIGH + 0.05),
            font_type="standard",
        )
        for i, frame in enumerate(frames)
    }
    logs = []

    with patch("pathlib.Path.exists", return_value=True):
        result = router._decide_qwen_frames(frames, [], frame_meta, logs.append)

    assert len(result) == 1
    assert any("Phase 2 ilerleme: 50/51" in msg for msg in logs)
    assert any("Phase 2 ilerleme: 51/51" in msg for msg in logs)
