"""Tests for word-level fuzzy comparison and crew fuzzy dedup in export_engine.

Covers:
  CREW-FUZZY-01: _canonicalize_crew() merges similar name variants within same role
  CREW-FUZZY-02: _canonicalize_crew() keeps distinct persons separate (different surname)
  CAST-WORD-01: _canonicalize_cast() word-level comparison merges OCR variants correctly
  CAST-WORD-02: _canonicalize_cast() word-level comparison separates different surnames
  OLLAMA-PASS-01: Ollama None → entries preserved (pass-through fail-safe)
"""

import sys
import os

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CREW-FUZZY-01: Crew fuzzy dedup — similar variants merge
# ─────────────────────────────────────────────────────────────────────────────

def test_crew_fuzzy_dedup_merges_similar_variants():
    """CREW-FUZZY-01: 'Abraham Jimenex' ve 'Abrahom Jimenez' aynı rol altında birleşmeli."""
    from core.export_engine import _canonicalize_crew, _HAS_RAPIDFUZZ
    if not _HAS_RAPIDFUZZ:
        import pytest
        pytest.skip("rapidfuzz not installed")

    crew = [
        {"name": "Abraham Jimenex", "role": "Director"},
        {"name": "Abrahom Jimenez", "role": "Director"},
    ]
    result = _canonicalize_crew(crew)
    director_entries = [r for r in result if r.get("role") == "Director"]
    assert len(director_entries) == 1, (
        f"Expected 1 director, got {len(director_entries)}: "
        f"{[r['name'] for r in director_entries]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CREW-FUZZY-02: Crew fuzzy dedup — distinct surnames stay separate
# ─────────────────────────────────────────────────────────────────────────────

def test_crew_fuzzy_dedup_keeps_distinct_persons_separate():
    """CREW-FUZZY-02: 'Abraham Jimenez' ve 'Abraham Ramirez' farklı kişi — birleşmemeli."""
    from core.export_engine import _canonicalize_crew, _HAS_RAPIDFUZZ
    if not _HAS_RAPIDFUZZ:
        import pytest
        pytest.skip("rapidfuzz not installed")

    crew = [
        {"name": "Abraham Jimenez", "role": "Editor"},
        {"name": "Abraham Ramirez", "role": "Editor"},
    ]
    result = _canonicalize_crew(crew)
    editor_entries = [r for r in result if r.get("role") == "Editor"]
    assert len(editor_entries) == 2, (
        f"Expected 2 editors (different surnames), got {len(editor_entries)}: "
        f"{[r['name'] for r in editor_entries]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CAST-WORD-01: Cast word-level comparison merges similar OCR variants
# ─────────────────────────────────────────────────────────────────────────────

def test_cast_word_level_merges_similar_surnames():
    """CAST-WORD-01: 'Abraham Jimenex' ve 'Abrahom Jimenez' cast'ta birleşmeli."""
    from core.export_engine import _canonicalize_cast, _HAS_RAPIDFUZZ
    if not _HAS_RAPIDFUZZ:
        import pytest
        pytest.skip("rapidfuzz not installed")

    cast = [
        {"actor_name": "Abraham Jimenex", "character_name": "", "confidence": 0.90, "seen_count": 2},
        {"actor_name": "Abrahom Jimenez", "character_name": "", "confidence": 0.85, "seen_count": 2},
    ]
    result = _canonicalize_cast(cast)
    assert len(result) == 1, (
        f"Expected 1 entry (similar OCR variants), got {len(result)}: "
        f"{[r['actor_name'] for r in result]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CAST-WORD-02: Cast word-level comparison separates different surnames
# ─────────────────────────────────────────────────────────────────────────────

def test_cast_word_level_separates_different_surnames():
    """CAST-WORD-02: 'Abraham Jimenez' ve 'Abraham Ramirez' farklı kişi — birleşmemeli."""
    from core.export_engine import _canonicalize_cast, _HAS_RAPIDFUZZ
    if not _HAS_RAPIDFUZZ:
        import pytest
        pytest.skip("rapidfuzz not installed")

    cast = [
        {"actor_name": "Abraham Jimenez", "character_name": "", "confidence": 0.90, "seen_count": 3},
        {"actor_name": "Abraham Ramirez", "character_name": "", "confidence": 0.88, "seen_count": 3},
    ]
    result = _canonicalize_cast(cast)
    assert len(result) == 2, (
        f"Expected 2 entries (different surnames), got {len(result)}: "
        f"{[r['actor_name'] for r in result]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA-PASS-01: Ollama None → entries preserved via pass-through
# ─────────────────────────────────────────────────────────────────────────────

def test_ollama_none_preserves_entries():
    """OLLAMA-PASS-01: Ollama None döndürünce girişler korunmalı (silinmemeli)."""
    from core.llm_cast_filter import LLMCastFilter

    f = LLMCastFilter(enabled=True)
    f._query_ollama = lambda prompt: None
    f._check_availability = lambda: True

    cast = [
        {"actor_name": "Nisa Serezli", "confidence": 0.7},
        {"actor_name": "Cihat Tamer", "confidence": 0.8},
    ]
    result = f.filter_cast(cast)
    assert len(result) == 2, (
        f"Expected 2 entries preserved, got {len(result)}"
    )
    assert all(e.get("is_llm_verified") for e in result), (
        "All entries should be marked as verified via pass-through"
    )
