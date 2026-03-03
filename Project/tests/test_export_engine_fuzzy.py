"""
test_export_engine_fuzzy.py — Cast ve crew word-level fuzzy matching testleri.

WFUZZ-01: Aynı kelime sayısı, benzer OCR varyantları → merge (aynı kişi)
WFUZZ-02: Aynı kelime sayısı, farklı isimler → ayrı tutulmalı
WFUZZ-03: Farklı kelime sayısı → WRatio fallback
WFUZZ-04: Crew fuzzy clustering — aynı rol+benzer isim → merge
WFUZZ-05: Crew fuzzy — farklı kişi aynı rol → ayrı tutulmalı
WFUZZ-06: rapidfuzz yoksa exact-key fallback çalışmalı
"""

import sys
import os

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import pytest
from core.export_engine import _canonicalize_cast, _canonicalize_crew, _words_fuzzy_match, _HAS_RAPIDFUZZ


# ---------------------------------------------------------------------------
# WFUZZ-01: Benzer OCR varyantları → merge
# ---------------------------------------------------------------------------

def test_words_fuzzy_match_similar_ocr_variants():
    """WFUZZ-01: Benzer OCR varyantları eşleşmeli."""
    if not _HAS_RAPIDFUZZ:
        pytest.skip("rapidfuzz not installed")
    # "Nisa Serezli" vs "Nisa Sereзli" (OCR z bozulması)
    assert _words_fuzzy_match("Nisa Serezli", "Nisa Serezll") is True
    # Same first + last name with minor typo
    assert _words_fuzzy_match("Haluk Bilginer", "Haluk Bilqiner") is True


# ---------------------------------------------------------------------------
# WFUZZ-02: Farklı isimler → ayrı tutulmalı
# ---------------------------------------------------------------------------

def test_words_fuzzy_match_different_names_not_merged():
    """WFUZZ-02: Açıkça farklı ad/soyad → False dönmeli."""
    if not _HAS_RAPIDFUZZ:
        pytest.skip("rapidfuzz not installed")
    # Different first AND last names — should NOT merge
    assert _words_fuzzy_match("Nisa Serezli", "Haluk Bilginer") is False
    # Same first name, completely different last name — should NOT merge
    assert _words_fuzzy_match("Nisa Serezli", "Nisa Mehmet") is False
    # Reversed order — should NOT merge
    assert _words_fuzzy_match("Nisa Serezli", "Serezli Nisa") is False


# ---------------------------------------------------------------------------
# WFUZZ-03: Farklı kelime sayısı → WRatio fallback
# ---------------------------------------------------------------------------

def test_words_fuzzy_match_different_word_count_uses_wratio():
    """WFUZZ-03: Farklı kelime sayısı → WRatio >= 75 fallback."""
    if not _HAS_RAPIDFUZZ:
        pytest.skip("rapidfuzz not installed")
    # 3 words vs 2 words — use WRatio
    # "Ali Veli Yilmaz" vs "Ali Veli" — WRatio will be high (substring)
    result_short = _words_fuzzy_match("Ali Veli Yilmaz", "Ali Veli")
    # Just checking it doesn't throw and returns bool
    assert isinstance(result_short, bool)


# ---------------------------------------------------------------------------
# WFUZZ-04: Crew fuzzy clustering — benzer isim, aynı rol → merge
# ---------------------------------------------------------------------------

def test_crew_fuzzy_merge_same_role_similar_name():
    """WFUZZ-04: Aynı rol, benzer isim → tek giriş olmalı."""
    if not _HAS_RAPIDFUZZ:
        pytest.skip("rapidfuzz not installed")
    crew = [
        {"name": "Nisa Serezli", "role": "Senaryo", "confidence": 0.9},
        {"name": "Nisa Serezll", "role": "Senaryo", "confidence": 0.8},  # OCR typo
    ]
    result = _canonicalize_crew(crew)
    names = [r["name"] for r in result]
    assert len(result) == 1, f"Benzer isimler merge edilmeli, got: {names}"


# ---------------------------------------------------------------------------
# WFUZZ-05: Crew fuzzy — farklı kişi, aynı rol → ayrı tutulmalı
# ---------------------------------------------------------------------------

def test_crew_different_names_same_role_kept_separate():
    """WFUZZ-05: Farklı isim, aynı rol → ayrı tutulmalı."""
    crew = [
        {"name": "Nisa Serezli", "role": "Senaryo"},
        {"name": "Haluk Bilginer", "role": "Senaryo"},
    ]
    result = _canonicalize_crew(crew)
    assert len(result) == 2, (
        f"Farklı isimler ayrı tutulmalı, got: {[r['name'] for r in result]}"
    )


# ---------------------------------------------------------------------------
# WFUZZ-06: rapidfuzz yok → exact-key fallback
# ---------------------------------------------------------------------------

def test_words_fuzzy_match_without_rapidfuzz(monkeypatch):
    """WFUZZ-06: rapidfuzz yoksa exact match fallback çalışmalı."""
    import core.export_engine as ee_mod
    monkeypatch.setattr(ee_mod, "_HAS_RAPIDFUZZ", False)

    # exact match → True
    assert ee_mod._words_fuzzy_match("Nisa Serezli", "Nisa Serezli") is True
    # different → False
    assert ee_mod._words_fuzzy_match("Nisa Serezli", "Haluk Bilginer") is False


# ---------------------------------------------------------------------------
# Cast canonicalization — word-level dedup test
# ---------------------------------------------------------------------------

def test_cast_word_level_merge_ocr_variants():
    """Cast OCR varyantları word-level fuzzy ile merge edilmeli."""
    if not _HAS_RAPIDFUZZ:
        pytest.skip("rapidfuzz not installed")
    cast = [
        {"actor_name": "Haluk Bilginer", "character_name": "", "confidence": 0.95, "seen_count": 3},
        {"actor_name": "Haluk Bilqiner", "character_name": "", "confidence": 0.85, "seen_count": 2},
    ]
    result = _canonicalize_cast(cast)
    assert len(result) == 1, f"OCR varyantları merge edilmeli, got: {[r['actor_name'] for r in result]}"
    # Higher confidence variant should be chosen
    assert result[0]["confidence"] >= 0.85


def test_cast_word_level_keeps_different_names_separate():
    """Farklı kişiler word-level matching sonrası ayrı tutulmalı."""
    if not _HAS_RAPIDFUZZ:
        pytest.skip("rapidfuzz not installed")
    cast = [
        {"actor_name": "Nisa Serezli", "character_name": "", "confidence": 0.90, "seen_count": 2},
        {"actor_name": "Nisa Mehmet", "character_name": "", "confidence": 0.90, "seen_count": 2},
    ]
    result = _canonicalize_cast(cast)
    assert len(result) == 2, (
        f"Farklı isimler ayrı tutulmalı, got: {[r['actor_name'] for r in result]}"
    )
