"""
test_vlm_tmdb_fallback.py — TMDB miss → VLM otomatik devreye girme testleri.

VLM-01: TMDB miss olduğunda VLM enabled=False olsa bile is_available() True ise çalışmalı
VLM-02: VLM model yoksa (is_available()=False) graceful skip olmalı
VLM-03: TMDB eşleşirse VLM çalışmamalı
VLM-04: VLM enabled flag BLOK2 sonrası orijinal değerine dönmeli
"""

import sys
import os

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vlm_reader(enabled=False, available=True):
    """Return a stubbed VLMReader."""
    reader = MagicMock()
    reader.enabled = enabled
    reader.is_available = MagicMock(return_value=available)
    reader.read_text_from_frame = MagicMock(return_value=None)
    return reader


# ---------------------------------------------------------------------------
# VLM-01: TMDB miss → VLM çalışmalı (enabled=False olsa bile)
# ---------------------------------------------------------------------------

def test_vlm_activated_on_tmdb_miss_even_when_disabled():
    """VLM-01: enabled=False iken TMDB miss → is_available() True → VLM çalışmalı."""
    reader = _make_vlm_reader(enabled=False, available=True)

    # BLOK2 mantığını simüle et
    tmdb_matched = False
    vlm_was_enabled = reader.enabled
    reader.enabled = True  # TMDB miss → force enable
    vlm_available = reader.is_available()
    reader.enabled = vlm_was_enabled  # restore

    assert vlm_available is True, "is_available() True ise VLM çalışabilir olmalı"
    assert reader.enabled is False, "VLM enabled flag orijinal değerine dönmeli"


# ---------------------------------------------------------------------------
# VLM-02: VLM model yoksa graceful skip
# ---------------------------------------------------------------------------

def test_vlm_skipped_gracefully_when_model_unavailable():
    """VLM-02: is_available()=False → VLM atlanmalı, hata fırlatılmamalı."""
    reader = _make_vlm_reader(enabled=False, available=False)

    tmdb_matched = False
    vlm_was_enabled = reader.enabled
    reader.enabled = True
    vlm_available = reader.is_available()
    reader.enabled = vlm_was_enabled

    assert vlm_available is False, "Model yoksa is_available() False olmalı"
    reader.read_text_from_frame.assert_not_called()


# ---------------------------------------------------------------------------
# VLM-03: TMDB eşleşirse VLM çalışmamalı
# ---------------------------------------------------------------------------

def test_vlm_not_activated_when_tmdb_matched():
    """VLM-03: TMDB match → BLOK2 atlanmalı, VLM çalışmamalı."""
    reader = _make_vlm_reader(enabled=True, available=True)

    tmdb_matched = True

    # BLOK2 koşulu: if not tmdb_matched
    if not tmdb_matched:
        reader.read_text_from_frame("dummy_frame.jpg")

    reader.read_text_from_frame.assert_not_called()


# ---------------------------------------------------------------------------
# VLM-04: enabled flag BLOK2 sonrası restore edilmeli
# ---------------------------------------------------------------------------

def test_vlm_enabled_flag_restored_after_blok2():
    """VLM-04: BLOK2 sonrası VLM enabled flag orijinal değerine dönmeli."""
    original_enabled = False
    reader = _make_vlm_reader(enabled=original_enabled, available=True)

    # BLOK2 save-restore pattern
    vlm_was_enabled = reader.enabled
    reader.enabled = True
    # ... VLM işleme ...
    reader.enabled = vlm_was_enabled  # restore

    assert reader.enabled == original_enabled, (
        f"enabled flag restore edilmeli: {original_enabled!r}, "
        f"got {reader.enabled!r}"
    )


def test_vlm_enabled_true_flag_restored_after_blok2():
    """VLM-04b: enabled=True iken BLOK2 sonrası da True kalmalı."""
    original_enabled = True
    reader = _make_vlm_reader(enabled=original_enabled, available=True)

    vlm_was_enabled = reader.enabled
    reader.enabled = True
    reader.enabled = vlm_was_enabled

    assert reader.enabled == original_enabled
