"""
test_ocr_fuzzy_dedup.py — Fuzzy dedup ve hardcoded fix birim testleri.

Kapsanan düzeltmeler:
  FUZZY-01: _canonicalize_cast() fuzzy clustering (WRatio >= 75) ile
            benzer OCR varyantlarını tek satıra indirir.
  FUZZY-02: Post-merge fuzzy sweep kalan tekrarlı girişleri birleştirir.
  NOISE-01: seen_count <= 1 + noise_score >= 8 + confidence < 0.90
            olan girişler çıkarılır.
  HARD-01:  _HARDCODED_FIXES tablosuna NITA, SERELI, NITASERELI,
            NISASEREZLI eklendi.
"""

import sys
import os

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ─────────────────────────────────────────────────────────────────────────────
# HARD-01: Hardcoded fixes
# ─────────────────────────────────────────────────────────────────────────────

def test_hardcoded_nita_sereli():
    """HARD-01: NITA→Nisa ve SERELI→Serezli hardcoded fix'leri mevcut olmalı."""
    from core.turkish_name_db import _HARDCODED_FIXES
    assert 'NITA' in _HARDCODED_FIXES
    assert _HARDCODED_FIXES['NITA'] == 'Nisa'
    assert 'SERELI' in _HARDCODED_FIXES
    assert _HARDCODED_FIXES['SERELI'] == 'Serezli'


def test_hardcoded_nitasereli_combined():
    """HARD-01: Birleşik NITASERELI ve NISASEREZLI formları mevcut olmalı."""
    from core.turkish_name_db import _HARDCODED_FIXES
    assert 'NITASERELI' in _HARDCODED_FIXES
    assert _HARDCODED_FIXES['NITASERELI'] == 'Nisa Serezli'
    assert 'NISASEREZLI' in _HARDCODED_FIXES
    assert _HARDCODED_FIXES['NISASEREZLI'] == 'Nisa Serezli'


def test_turkish_name_db_nita_sereli_find():
    """HARD-01: TurkishNameDB.find() 'Nita Sereli' girişini 'Nisa Serezli' olarak düzeltmeli."""
    from core.turkish_name_db import TurkishNameDB
    db = TurkishNameDB()  # DB yokken sadece hardcoded tablo aktif
    result, score = db.find('Nita Sereli')
    assert result == 'Nisa Serezli', f"Expected 'Nisa Serezli', got {result!r}"
    assert score == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# FUZZY-01: Fuzzy clustering for no-char actors
# ─────────────────────────────────────────────────────────────────────────────

def test_fuzzy_clustering_ali_ozoguz_variants():
    """FUZZY-01: 'Ali Ozoqwz' / 'Ali Ozogwz' varyantları tek satırda birleşmeli.

    Kelime bazlı karşılaştırma (her kelime >= 70%) ile:
    - Ozoqwz~Ozogwz=83% ve Ozogwz~Orogwz=83% → birleşir
    - Ozoqwz~Ozoguy=50% → birleşmez (farklı kişi olarak değerlendirilir)
    """
    from core.export_engine import _canonicalize_cast, _HAS_RAPIDFUZZ
    if not _HAS_RAPIDFUZZ:
        import pytest
        pytest.skip("rapidfuzz not installed")

    cast = [
        {'actor_name': 'Ali Ozoqwz', 'character_name': '', 'confidence': 0.98, 'seen_count': 3},
        {'actor_name': 'Ali Ozogwz', 'character_name': '', 'confidence': 0.90, 'seen_count': 2},
        {'actor_name': 'Ali Orogwz', 'character_name': '', 'confidence': 0.91, 'seen_count': 2},
    ]
    result = _canonicalize_cast(cast)
    assert len(result) == 1, (
        f"Expected 1 entry, got {len(result)}: {[r['actor_name'] for r in result]}"
    )


def test_fuzzy_clustering_distinct_names_not_merged():
    """FUZZY-01: Tamamen farklı isimler ayrı satır olarak kalmalı."""
    from core.export_engine import _canonicalize_cast, _HAS_RAPIDFUZZ
    if not _HAS_RAPIDFUZZ:
        import pytest
        pytest.skip("rapidfuzz not installed")

    cast = [
        {'actor_name': 'Nisa Serezli', 'character_name': '', 'confidence': 0.99, 'seen_count': 5},
        {'actor_name': 'Kerem Yilmazer', 'character_name': '', 'confidence': 0.95, 'seen_count': 4},
        {'actor_name': 'Cihat Tamer', 'character_name': '', 'confidence': 0.99, 'seen_count': 6},
    ]
    result = _canonicalize_cast(cast)
    assert len(result) == 3, (
        f"Expected 3 entries, got {len(result)}: {[r['actor_name'] for r in result]}"
    )


def test_fuzzy_clustering_preserves_highest_confidence():
    """FUZZY-01: Fuzzy cluster'dan en yüksek confidence tutulmalı."""
    from core.export_engine import _canonicalize_cast, _HAS_RAPIDFUZZ
    if not _HAS_RAPIDFUZZ:
        import pytest
        pytest.skip("rapidfuzz not installed")

    cast = [
        {'actor_name': 'Ali Ozoqwz', 'character_name': '', 'confidence': 0.98, 'seen_count': 3},
        {'actor_name': 'Ali Ozogwz', 'character_name': '', 'confidence': 0.75, 'seen_count': 2},
    ]
    result = _canonicalize_cast(cast)
    assert len(result) == 1
    assert result[0]['confidence'] == 0.98


# ─────────────────────────────────────────────────────────────────────────────
# FUZZY-02: Post-merge fuzzy sweep
# ─────────────────────────────────────────────────────────────────────────────

def test_post_merge_sweep_merges_near_duplicates():
    """FUZZY-02: Post-merge sweep kalan benzer actor_name'leri birleştirmeli."""
    from core.export_engine import _canonicalize_cast, _HAS_RAPIDFUZZ
    if not _HAS_RAPIDFUZZ:
        import pytest
        pytest.skip("rapidfuzz not installed")

    # Karakter ismi olan girişler: benzer actor adları farklı char_key'e düşebilir
    # ama actor_name post-merge'de birleştirilmeli (aynı character olduğunda)
    cast = [
        {'actor_name': 'Ali Ozoqwz', 'character_name': 'Ahmet', 'confidence': 0.95, 'seen_count': 3},
        {'actor_name': 'Ali Ozogwz', 'character_name': 'Ahmet', 'confidence': 0.90, 'seen_count': 2},
    ]
    result = _canonicalize_cast(cast)
    # İki giriş aynı karakter ismiyle geldiğinden tek satıra düşer
    assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# NOISE-01: Seen count + noise score filtering
# ─────────────────────────────────────────────────────────────────────────────

def test_noise_filter_removes_single_frame_noisy_entry():
    """NOISE-01: seen_count=1, noise>=8, conf<0.90 olan giriş çıkarılmalı."""
    from core.export_engine import _canonicalize_cast, _noise_score

    # Boşluksuz 8+ karakter büyük harf: noise_score >= 8
    noisy_name = 'XYZABCDEFGH'  # noise_score: boşluksuz, büyük, >= 8 karakter → +6
    ns = _noise_score(noisy_name)
    assert ns >= 8, f"noise_score({noisy_name!r}) = {ns}, expected >= 8"

    cast = [
        {'actor_name': noisy_name, 'character_name': '', 'confidence': 0.85, 'seen_count': 1},
        {'actor_name': 'Nisa Serezli', 'character_name': '', 'confidence': 0.99, 'seen_count': 5},
    ]
    result = _canonicalize_cast(cast)
    names = [r['actor_name'] for r in result]
    assert noisy_name not in names, f"Noisy name should be filtered: {names}"
    assert any('Nisa' in n or 'Serezli' in n for n in names), f"Nisa Serezli should remain: {names}"


def test_noise_filter_keeps_multi_frame_noisy_entry():
    """NOISE-01: Birden fazla frame'de görülen giriş noise olsa bile tutulmalı."""
    from core.export_engine import _canonicalize_cast

    noisy_name = 'XYZABCDEFGH'
    cast = [
        {'actor_name': noisy_name, 'character_name': '', 'confidence': 0.85, 'seen_count': 5},
    ]
    result = _canonicalize_cast(cast)
    names = [r['actor_name'] for r in result]
    assert noisy_name in names, f"Multi-frame noisy entry should be kept: {names}"


def test_noise_filter_keeps_high_confidence_single_frame():
    """NOISE-01: seen_count=1 ama conf>=0.90 ise tutulmalı."""
    from core.export_engine import _canonicalize_cast

    noisy_name = 'XYZABCDEFGH'
    cast = [
        {'actor_name': noisy_name, 'character_name': '', 'confidence': 0.95, 'seen_count': 1},
    ]
    result = _canonicalize_cast(cast)
    names = [r['actor_name'] for r in result]
    assert noisy_name in names, f"High-confidence single-frame entry should be kept: {names}"
