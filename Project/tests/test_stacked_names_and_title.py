"""
test_stacked_names_and_title.py — Tests for two new features.

STACK-01..STACK-04: layout_analyzer._detect_stacked_name_rows
TITLE-01..TITLE-05: pipeline_runner._extract_film_title_from_filename
"""

import sys
import os

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ─────────────────────────────────────────────────────────────────────────────
# STACK-01..STACK-04: _detect_stacked_name_rows
# ─────────────────────────────────────────────────────────────────────────────

def _make_bbox(x1, y1, x2, y2):
    return [x1, y1, x2, y2]


def _make_ocr_results_3col():
    """
    Simulates 3-column stacked name layout:
        Tommy   Murvyn   Douglas   (y≈310, x≈80/280/480)
        Rettig  Vye      Spencer   (y≈360, x≈80/280/480)
    """
    frame_width = 640
    frame_height = 480
    h = 30  # character height
    results = [
        # Row 1 (y≈310)
        {"text": "Tommy",   "bbox": _make_bbox(60, 295, 120, 325),  "confidence": 0.9},
        {"text": "Murvyn",  "bbox": _make_bbox(260, 295, 330, 325), "confidence": 0.9},
        {"text": "Douglas", "bbox": _make_bbox(460, 295, 540, 325), "confidence": 0.9},
        # Row 2 (y≈360)
        {"text": "Rettig",  "bbox": _make_bbox(60, 345, 120, 375),  "confidence": 0.9},
        {"text": "Vye",     "bbox": _make_bbox(270, 345, 310, 375), "confidence": 0.9},
        {"text": "Spencer", "bbox": _make_bbox(455, 345, 545, 375), "confidence": 0.9},
    ]
    return results, frame_width, frame_height


def test_stacked_names_detected_3col():
    """STACK-01: 3-column stacked names produce correct merged full names."""
    from core.layout_analyzer import LayoutAnalyzer

    analyzer = LayoutAnalyzer()
    results, fw, fh = _make_ocr_results_3col()
    output = analyzer.analyze_frame_results(results, fw, fh)

    stacked = output.get("stacked_names", [])
    assert len(stacked) == 3, f"Expected 3 merged names, got: {stacked}"
    assert "Tommy Rettig" in stacked
    assert "Murvyn Vye" in stacked
    assert "Douglas Spencer" in stacked


def test_stacked_names_key_present_in_output():
    """STACK-02: analyze_frame_results always returns 'stacked_names' key."""
    from core.layout_analyzer import LayoutAnalyzer

    analyzer = LayoutAnalyzer()
    output = analyzer.analyze_frame_results([], 640, 480)
    assert "stacked_names" in output


def test_stacked_names_not_triggered_for_2col():
    """STACK-03: 2-column layout (character↔actor) does NOT produce stacked names."""
    from core.layout_analyzer import LayoutAnalyzer

    # Two-column layout: left=character, right=actor — only 2 items per row
    results = [
        {"text": "SHERIFF",    "bbox": _make_bbox(50, 100, 200, 130),  "confidence": 0.9},
        {"text": "John Wayne",  "bbox": _make_bbox(350, 100, 550, 130), "confidence": 0.9},
        {"text": "DEPUTY",     "bbox": _make_bbox(50, 150, 200, 180),  "confidence": 0.9},
        {"text": "Dean Martin", "bbox": _make_bbox(350, 150, 560, 180), "confidence": 0.9},
    ]
    analyzer = LayoutAnalyzer()
    output = analyzer.analyze_frame_results(results, 640, 480)
    stacked = output.get("stacked_names", [])
    # 2 items per row → should not be treated as stacked (N < 3 check)
    assert stacked == [], f"Expected no stacked names for 2-col layout, got: {stacked}"


def test_stacked_names_empty_for_single_line():
    """STACK-04: Single row with multiple words → no stacked names."""
    from core.layout_analyzer import LayoutAnalyzer

    results = [
        {"text": "Tommy",   "bbox": _make_bbox(60, 295, 120, 325),  "confidence": 0.9},
        {"text": "Murvyn",  "bbox": _make_bbox(260, 295, 330, 325), "confidence": 0.9},
        {"text": "Douglas", "bbox": _make_bbox(460, 295, 540, 325), "confidence": 0.9},
    ]
    analyzer = LayoutAnalyzer()
    output = analyzer.analyze_frame_results(results, 640, 480)
    stacked = output.get("stacked_names", [])
    # Only one row → no pair of consecutive rows → no stacked names
    assert stacked == [], f"Expected no stacked names for single row, got: {stacked}"


# ─────────────────────────────────────────────────────────────────────────────
# TITLE-01..TITLE-05: _extract_film_title_from_filename
# (Tested directly without importing PipelineRunner, which requires cv2/paddle.
#  This mirrors the pattern used in test_hardening.py CONF-01 tests.
#  The helper below is a faithful copy of the static method logic.)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re


def _extract_film_title_from_filename(stem: str) -> str:
    """Mirror of PipelineRunner._extract_film_title_from_filename for testing."""
    m = _re.search(r'\d{4}-\d{4}-\d-\d{4}-\d{2}-\d-(.+)$', stem)
    if m:
        title_raw = m.group(1)
        title = title_raw.replace('_', ' ').strip()
        # Turkish rules: İ(U+0130)→i, I→ı (not regular i)
        def _tr_capitalize(w: str) -> str:
            if not w:
                return w
            rest = w[1:].replace('\u0130', 'i').replace('I', 'ı').lower()
            return w[0] + rest
        title = ' '.join(_tr_capitalize(w) for w in title.split())
        return title
    return stem


def test_extract_title_simple():
    """TITLE-01: Standard evoArcadmin format with simple title."""
    stem = "evoArcadmin_TEST1_1955-0019-1-0000-00-1-KÜL_KEDİSİ"
    result = _extract_film_title_from_filename(stem)
    assert result == "Kül Kedisi", f"Got: {result}"


def test_extract_title_multi_word():
    """TITLE-02: Multi-word title with underscores."""
    stem = "evoArcadmin_TEST2_2015-0114-0-0055-88-0-AŞKIN_YOLCULUĞU_YUNUS_EMRE"
    result = _extract_film_title_from_filename(stem)
    assert result == "Aşkın Yolculuğu Yunus Emre", f"Got: {result}"


def test_extract_title_with_dash_in_title():
    """TITLE-03: Title containing a dash."""
    stem = "evoArcadmin_FUTBOLMAÇ_2013-0311-1-0600-00-1-PTT_1._LİG_MERSİN_İDMAN_YURDU_-_BOLUSPOR"
    result = _extract_film_title_from_filename(stem)
    assert result == "Ptt 1. Lig Mersin İdman Yurdu - Boluspor", f"Got: {result}"


def test_extract_title_repeated_in_stem():
    """TITLE-04: Title appears both in program code and at the end (should use end part)."""
    stem = "evoArcadmin_DÖNÜŞÜ OLMAYAN NEHİR_1989-0624-1-0000-00-1-DÖNÜŞÜ_OLMAYAN_NEHİR"
    result = _extract_film_title_from_filename(stem)
    assert result == "Dönüşü Olmayan Nehir", f"Got: {result}"


def test_extract_title_fallback_on_unknown_format():
    """TITLE-05: Unknown format → original stem returned as fallback."""
    stem = "some_random_filename_without_pattern"
    result = _extract_film_title_from_filename(stem)
    assert result == stem, f"Expected original stem, got: {result}"
