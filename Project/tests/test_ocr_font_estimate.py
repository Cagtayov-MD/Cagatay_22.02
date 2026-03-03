"""
test_ocr_font_estimate.py — OCREngine.estimate_font_type() unit tests.

Senaryolar:
  FONT-01: Boş liste → "unknown"
  FONT-02: Tek girdi (1 frame, 1 bbox) → "unknown" (< 2 bbox)
  FONT-03: cv2 işleme hatası → "unknown"
  FONT-04: Geçersiz frame yolu → "unknown"
  FONT-05: Tüm bbox'lar boşsa → "unknown"

Not: cv2 bu test ortamında kurulu olmayabilir. cv2 yoksa OCREngine import
edilemez; bu durumda tüm testler atlanır (pytest.importorskip).
"""

import sys
import os

import pytest

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# cv2 yoksa OCREngine import edilemez — testleri atla
cv2 = pytest.importorskip("cv2", reason="cv2 kurulu değil — OCREngine testleri atlanıyor")

try:
    from core.ocr_engine import OCREngine
    _HAS_OCR_ENGINE = True
except Exception:
    _HAS_OCR_ENGINE = False

pytestmark = pytest.mark.skipif(
    not _HAS_OCR_ENGINE,
    reason="OCREngine import edilemedi (PaddleOCR eksik olabilir)",
)


# ─────────────────────────────────────────────────────────────────────────────
# FONT-01: Boş liste → "unknown"
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_list_returns_unknown():
    """FONT-01: Boş frame listesi → 'unknown'."""
    result = OCREngine.estimate_font_type([])
    assert result == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# FONT-02: Tek bbox → "unknown" (yetersiz veri)
# ─────────────────────────────────────────────────────────────────────────────

def test_single_bbox_returns_unknown():
    """FONT-02: Yalnızca 1 bbox varsa yetersiz veri → 'unknown'."""
    # Var olmayan frame yolu + tek bbox → heights listesi 0 veya 1 eleman
    result = OCREngine.estimate_font_type([("/nonexistent/frame.png", [[0, 0, 50, 20]])])
    assert result == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# FONT-03: cv2.imread hata verirse → "unknown"
# ─────────────────────────────────────────────────────────────────────────────

def test_imread_failure_returns_unknown(monkeypatch):
    """FONT-03: cv2.imread hata verirse 'unknown' döner (güvenli taraf)."""
    if not hasattr(cv2, "imread"):
        pytest.skip("cv2.imread mevcut değil (stub cv2) — test atlanıyor")

    def fake_imread(*args, **kwargs):
        raise RuntimeError("Simulated imread failure")

    monkeypatch.setattr(cv2, "imread", fake_imread)

    result = OCREngine.estimate_font_type([("/nonexistent/frame.png", [[0, 0, 50, 20]])])
    # imread başarısız olursa frame yüklenemez → heights < 2 → "unknown"
    assert result == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# FONT-04: Geçersiz frame yolu → "unknown"
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_frame_path_returns_unknown():
    """FONT-04: Var olmayan frame dosyaları → 'unknown'."""
    data = [
        ("/nonexistent/a.png", [[0, 0, 100, 30], [0, 40, 100, 70]]),
        ("/nonexistent/b.png", [[0, 0, 100, 30], [0, 40, 100, 70]]),
    ]
    result = OCREngine.estimate_font_type(data)
    assert result == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# FONT-05: Tüm bbox'lar boşsa → "unknown"
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_bboxes_returns_unknown():
    """FONT-05: Tüm bbox listesi boşsa → 'unknown'."""
    result = OCREngine.estimate_font_type([
        ("/nonexistent/frame.png", []),
        ("/nonexistent/frame2.png", []),
    ])
    assert result == "unknown"

