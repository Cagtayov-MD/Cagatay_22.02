"""
Ters doğrulama (_reverse_validate) birim testleri.

REVVAL-01: Kulübe senaryosu — yanlış film (Dövüş Kulübü) REJECT edilmeli
REVVAL-02: Doğru film senaryosu (Fight Club) ACCEPT edilmeli
REVVAL-03: Yönetmen bilgisi yoksa ceza uygulanmamalı
REVVAL-04: OCR yılı bilinmiyorsa yıl puanı sıfır (ne + ne -)
REVVAL-05: Tam başlık eşleşmesi maksimum puan vermeli
REVVAL-06: verify_credits Kulübe senaryosunda rejected=True ve matched_id=0 döndürmeli
REVVAL-07: verify_credits doğru film senaryosunda rejected=False döndürmeli
REVVAL-08: OCR yılı dosya adından parse edilmeli
REVVAL-09: Başlık farklı dil → başlık kategorisi atlanır (0.0/0.0)
REVVAL-10: Başlık aynı dil ama tamamen farklı → -4.0 ceza
REVVAL-11: Cast bonus — ≥8 hits → +3.0 bonus; ≥5 → +2.0; ≥3 → +1.0
REVVAL-12: Dinamik eşik — aktif kategorilere orantılı (max × 0.40)
REVVAL-13: Cast negatif ceza — hits ≥ 3 ise ceza yok
"""
from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest

from core.tmdb_verify import TMDBVerify, TMDBVerifyResult


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def _make_verifier(log_lines=None):
    tmp = tempfile.mkdtemp()
    logs = log_lines if log_lines is not None else []
    verifier = TMDBVerify(work_dir=tmp, api_key="test-key",
                          log_cb=lambda m: logs.append(m))
    return verifier, logs


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-01: Kulübe senaryosu — REJECT bekleniyor
# ─────────────────────────────────────────────────────────────────────────────

def test_kulube_scenario_rejected():
    """
    REVVAL-01: 'KULÜBE' (OCR) vs 'Dövüş Kulübü' (1999, David Fincher)

    - Başlık fuzzy skoru düşük → negatif puan
    - Yönetmen Fritz Lang vs David Fincher → eşleşme yok → negatif puan
    - Cast oranı 3/96 = %3.1 → negatif puan
    - Yıl 1952 vs 1999 = 47 yıl → negatif puan
    → Toplam çok negatif → REJECT bekleniyor
    """
    verifier, logs = _make_verifier()

    fight_club_entry = {
        "id": 550,
        "title": "Dövüş Kulübü",
        "original_title": "Fight Club",
        "release_date": "1999-10-15",
        "media_type": "movie",
    }
    # Dövüş Kulübü credits: Edward Norton, Brad Pitt, Helena Bonham Carter...
    fight_club_credits = {
        "cast": [{"name": "Edward Norton"}, {"name": "Brad Pitt"},
                 {"name": "Helena Bonham Carter"}, {"name": "Meat Loaf"},
                 {"name": "Jared Leto"}],
        "crew": [{"name": "David Fincher", "job": "Director"}],
    }

    # OCR verisi: 96 kişi var, sadece 3 tanesi tesadüfen eşleşiyor
    ocr_cast_names = [f"Person {i}" for i in range(93)] + [
        "Edward Norton", "Brad Pitt", "Helena Bonham Carter"
    ]

    accepted, score, breakdown = verifier._reverse_validate(
        ocr_title="KULÜBE",
        ocr_cast_names=ocr_cast_names,
        ocr_director_names=["FRITZ LANG"],
        ocr_year=1952,
        tmdb_entry=fight_club_entry,
        credits_data=fight_club_credits,
        forward_hits=3,
        forward_misses=93,
    )

    assert not accepted, (
        f"Kulübe senaryosu REJECT edilmeliydi ama ACCEPT döndü. Puan: {score:.1f}"
    )
    assert score < breakdown["threshold"], f"Puan eşiğin altında olmalı: {score:.1f}"
    # Yıl farkı 47 → maksimum negatif yıl puanı
    assert breakdown["year"]["neg"] == -3.0
    # Cast hits=3 ≥ 3 → negatif cast cezası verilmez
    assert breakdown["cast"]["neg"] == 0.0
    # Yönetmen eşleşmedi → yönetmen cezası
    assert breakdown["director"]["neg"] == -2.5
    # Log'da REJECT ifadesi geçmeli
    assert any("REJECT" in line for line in logs), "Log'da REJECT ifadesi bulunamadı"


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-02: Doğru film senaryosu — ACCEPT bekleniyor
# ─────────────────────────────────────────────────────────────────────────────

def test_fight_club_correct_accepted():
    """
    REVVAL-02: 'FIGHT CLUB' (OCR) vs 'Fight Club' (1999, David Fincher)

    - Başlık exact/çok yüksek fuzzy → maksimum pozitif puan
    - Yönetmen David Fincher eşleşiyor → maksimum pozitif puan
    - Cast oranı 12/15 = %80 → maksimum pozitif puan
    - Yıl 1999 vs 1999 = 0 fark → maksimum pozitif puan
    → Toplam 10.0 → ACCEPT bekleniyor
    """
    verifier, logs = _make_verifier()

    fight_club_entry = {
        "id": 550,
        "title": "Dövüş Kulübü",
        "original_title": "Fight Club",
        "release_date": "1999-10-15",
        "media_type": "movie",
    }
    fight_club_credits = {
        "cast": [
            {"name": "Edward Norton"}, {"name": "Brad Pitt"},
            {"name": "Helena Bonham Carter"}, {"name": "Meat Loaf"},
            {"name": "Jared Leto"}, {"name": "Zach Grenier"},
            {"name": "Richmond Arquette"}, {"name": "David Andrews"},
            {"name": "George Maguire"}, {"name": "Eugenie Bondurant"},
            {"name": "Christina Cabot"}, {"name": "Sydney 'Big Dawg' Colston"},
            {"name": "Rachel Singer"}, {"name": "Christie Cronenweth"},
            {"name": "Tim De Zarn"},
        ],
        "crew": [{"name": "David Fincher", "job": "Director"}],
    }

    accepted, score, breakdown = verifier._reverse_validate(
        ocr_title="FIGHT CLUB",
        ocr_cast_names=[
            "Edward Norton", "Brad Pitt", "Helena Bonham Carter", "Meat Loaf",
            "Jared Leto", "Zach Grenier", "Richmond Arquette", "David Andrews",
            "George Maguire", "Eugenie Bondurant", "Christina Cabot", "Rachel Singer",
        ],
        ocr_director_names=["DAVID FINCHER"],
        ocr_year=1999,
        tmdb_entry=fight_club_entry,
        credits_data=fight_club_credits,
        forward_hits=12,
        forward_misses=3,
    )

    assert accepted, (
        f"Doğru film senaryosu ACCEPT edilmeliydi ama REJECT döndü. Puan: {score:.1f}"
    )
    assert score >= breakdown["threshold"], f"Puan eşiğin üzerinde olmalı: {score:.1f}"
    # Yıl farkı 0 → maksimum pozitif yıl puanı
    assert breakdown["year"]["pos"] == 2.0
    assert breakdown["year"]["neg"] == 0.0
    # Cast oranı %80 + 12 hits ≥ 8 → oran puanı 3.0 + bonus 3.0 = 6.0
    assert breakdown["cast"]["pos"] == 6.0
    # hits ≥ 3 → negatif cast cezası yok
    assert breakdown["cast"]["neg"] == 0.0
    # Log'da ACCEPT ifadesi geçmeli
    assert any("ACCEPT" in line for line in logs), "Log'da ACCEPT ifadesi bulunamadı"


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-03: Yönetmen bilgisi yoksa ceza yok
# ─────────────────────────────────────────────────────────────────────────────

def test_no_director_info_no_penalty():
    """REVVAL-03: ocr_director_names=[] ise yönetmen puanı 0 olmalı (ceza da yok)."""
    verifier, _ = _make_verifier()

    entry = {
        "id": 1, "title": "Test Film", "original_title": "Test Film",
        "release_date": "2000-01-01", "media_type": "movie",
    }
    credits = {
        "cast": [{"name": "Actor A"}, {"name": "Actor B"}, {"name": "Actor C"}],
        "crew": [{"name": "Some Director", "job": "Director"}],
    }

    accepted, score, breakdown = verifier._reverse_validate(
        ocr_title="Test Film",
        ocr_cast_names=["Actor A", "Actor B", "Actor C"],
        ocr_director_names=[],
        ocr_year=2000,
        tmdb_entry=entry,
        credits_data=credits,
        forward_hits=3,
        forward_misses=0,
    )

    assert breakdown["director"]["net"] == 0.0, (
        f"Yönetmen bilgisi yoksa puan 0.0 olmalı, şu an: {breakdown['director']}"
    )
    # Kabul edilmeli: başlık tam, cast %100, yıl tam
    assert accepted


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-04: OCR yılı bilinmiyorsa yıl puanı sıfır
# ─────────────────────────────────────────────────────────────────────────────

def test_unknown_ocr_year_no_year_score():
    """REVVAL-04: ocr_year=0 ise yıl puanı (pozitif ve negatif) sıfır olmalı."""
    verifier, _ = _make_verifier()

    entry = {
        "id": 1, "title": "Test Film", "original_title": "Test Film",
        "release_date": "2000-01-01", "media_type": "movie",
    }
    credits = {
        "cast": [{"name": "Actor A"}, {"name": "Actor B"}],
        "crew": [],
    }

    _, _, breakdown = verifier._reverse_validate(
        ocr_title="Test Film",
        ocr_cast_names=["Actor A", "Actor B"],
        ocr_director_names=[],
        ocr_year=0,   # bilinmiyor
        tmdb_entry=entry,
        credits_data=credits,
        forward_hits=2,
        forward_misses=0,
    )

    assert breakdown["year"]["pos"] == 0.0
    assert breakdown["year"]["neg"] == 0.0
    assert breakdown["year"]["net"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-05: Tam başlık eşleşmesi maksimum başlık puanı
# ─────────────────────────────────────────────────────────────────────────────

def test_exact_title_match_max_score():
    """REVVAL-05: OCR başlığı == TMDB original_title → +2.5 başlık puanı."""
    verifier, _ = _make_verifier()

    entry = {
        "id": 1, "title": "Rancho Notorious", "original_title": "Rancho Notorious",
        "release_date": "1952-01-01", "media_type": "movie",
    }
    credits = {"cast": [], "crew": []}

    _, _, breakdown = verifier._reverse_validate(
        ocr_title="Rancho Notorious",
        ocr_cast_names=[],
        ocr_director_names=[],
        ocr_year=0,
        tmdb_entry=entry,
        credits_data=credits,
        forward_hits=0,
        forward_misses=0,
    )

    assert breakdown["title"]["pos"] == 2.5, (
        f"Tam eşleşme +2.5 vermeli, aldık: {breakdown['title']}"
    )
    assert breakdown["title"]["neg"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-06: verify_credits — Kulübe senaryosunda rejected=True, matched_id=0
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_credits_kulube_rejected():
    """
    REVVAL-06: verify_credits entegrasyon testi — Kulübe senaryosunda:
    - result.rejected == True
    - result.matched_id == 0
    - result.reason == "reverse_validation_rejected"
    - cdata bozulmamış (tmdb_verified alanı eklenmemiş)
    """
    verifier, logs = _make_verifier()

    FIGHT_CLUB_ID = 550
    fight_club_entry = {
        "id": FIGHT_CLUB_ID,
        "title": "Dövüş Kulübü",
        "original_title": "Fight Club",
        "release_date": "1999-10-15",
        "media_type": "movie",
    }
    fight_club_credits = {
        "cast": [{"name": "Edward Norton"}, {"name": "Brad Pitt"},
                 {"name": "Helena Bonham Carter"}],
        "crew": [{"name": "David Fincher", "job": "Director"}],
    }

    # OCR verisi: 96 kişi, 3 tesadüf eşleşme
    ocr_cast = [{"actor_name": f"Person {i}", "confidence": 0.6} for i in range(93)]
    ocr_cast += [
        {"actor_name": "Edward Norton",       "confidence": 0.7},
        {"actor_name": "Brad Pitt",            "confidence": 0.7},
        {"actor_name": "Helena Bonham Carter", "confidence": 0.7},
    ]

    cdata = {
        "film_title": "KULÜBE",
        "cast": ocr_cast,
        "directors": [{"name": "FRITZ LANG"}],
        "filename": "1952-0052-1-0000-00-1-KULÜBE.mp4",
    }

    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(fight_club_entry, "movie", "title")), \
         patch.object(verifier, "_fetch_credits",
                      return_value=fight_club_credits), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    assert result.rejected is True, (
        f"Kulübe senaryosunda rejected=True bekleniyor, aldık: {result.rejected}"
    )
    assert result.matched_id == 0, (
        f"REJECT durumunda matched_id=0 olmalı, aldık: {result.matched_id}"
    )
    assert result.reason == "reverse_validation_rejected", (
        f"Reason 'reverse_validation_rejected' olmalı, aldık: {result.reason!r}"
    )
    assert "tmdb_verified" not in cdata, (
        "REJECT durumunda cdata'ya tmdb_verified eklenmemeli"
    )
    assert result.reverse_score < 4.0, (
        f"reverse_score eşiğin altında olmalı: {result.reverse_score}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-07: verify_credits — doğru film senaryosunda rejected=False
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_credits_correct_film_accepted():
    """
    REVVAL-07: verify_credits entegrasyon testi — Fight Club doğru eşleşmesinde:
    - result.rejected == False
    - result.matched_id == 550
    - cdata["tmdb_verified"] == True
    """
    verifier, _ = _make_verifier()

    FIGHT_CLUB_ID = 550
    fight_club_entry = {
        "id": FIGHT_CLUB_ID,
        "title": "Dövüş Kulübü",
        "original_title": "Fight Club",
        "release_date": "1999-10-15",
        "media_type": "movie",
    }
    fight_club_credits = {
        "cast": [
            {"name": "Edward Norton"}, {"name": "Brad Pitt"},
            {"name": "Helena Bonham Carter"}, {"name": "Meat Loaf"},
            {"name": "Jared Leto"}, {"name": "Zach Grenier"},
            {"name": "Richmond Arquette"}, {"name": "David Andrews"},
            {"name": "George Maguire"}, {"name": "Eugenie Bondurant"},
            {"name": "Christina Cabot"}, {"name": "Rachel Singer"},
        ],
        "crew": [{"name": "David Fincher", "job": "Director"}],
    }

    ocr_cast = [
        {"actor_name": "Edward Norton",       "confidence": 0.9},
        {"actor_name": "Brad Pitt",            "confidence": 0.9},
        {"actor_name": "Helena Bonham Carter", "confidence": 0.9},
        {"actor_name": "Meat Loaf",            "confidence": 0.8},
        {"actor_name": "Jared Leto",           "confidence": 0.8},
        {"actor_name": "Zach Grenier",         "confidence": 0.8},
        {"actor_name": "Richmond Arquette",    "confidence": 0.8},
        {"actor_name": "David Andrews",        "confidence": 0.8},
        {"actor_name": "George Maguire",       "confidence": 0.7},
        {"actor_name": "Eugenie Bondurant",    "confidence": 0.7},
        {"actor_name": "Christina Cabot",      "confidence": 0.7},
        {"actor_name": "Rachel Singer",        "confidence": 0.7},
    ]

    cdata = {
        "film_title": "FIGHT CLUB",
        "cast": ocr_cast,
        "directors": [{"name": "DAVID FINCHER"}],
        "year": 1999,
    }

    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(fight_club_entry, "movie", "title")), \
         patch.object(verifier, "_fetch_credits",
                      return_value=fight_club_credits), \
         patch.object(verifier.client, "get_movie_details",
                      return_value={"genres": [], "original_title": "Fight Club"}), \
         patch.object(verifier.client, "get_movie_keywords", return_value=[]), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    assert result.rejected is False, (
        f"Doğru film senaryosunda rejected=False bekleniyor, aldık: {result.rejected}"
    )
    assert result.matched_id == FIGHT_CLUB_ID, (
        f"matched_id={FIGHT_CLUB_ID} bekleniyor, aldık: {result.matched_id}"
    )
    assert result.reverse_score >= 4.0, (
        f"reverse_score eşiğin üzerinde olmalı: {result.reverse_score}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-08: OCR yılı dosya adından parse edilmeli
# ─────────────────────────────────────────────────────────────────────────────

def test_ocr_year_parsed_from_filename():
    """
    REVVAL-08: cdata["filename"]'de '1952' geçiyorsa _ocr_year=1952 olarak algılanmalı.
    Yıl büyük farkıyla ters doğrulama REJECT etmeli.
    """
    verifier, _ = _make_verifier()

    entry = {
        "id": 550,
        "title": "Dövüş Kulübü",
        "original_title": "Fight Club",
        "release_date": "1999-01-01",
        "media_type": "movie",
    }
    credits = {
        "cast": [{"name": "Edward Norton"}, {"name": "Brad Pitt"}],
        "crew": [{"name": "David Fincher", "job": "Director"}],
    }

    cdata = {
        "film_title": "KULÜBE",
        "cast": [{"actor_name": "Edward Norton", "confidence": 0.7},
                 {"actor_name": "Brad Pitt", "confidence": 0.7}],
        "directors": [{"name": "FRITZ LANG"}],
        # Yıl cdata'da yok ama dosya adında var
        "filename": "1952-0052-1-0000-00-1-KULÜBE.mp4",
    }

    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(entry, "movie", "title")), \
         patch.object(verifier, "_fetch_credits", return_value=credits), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    # Dosya adından 1952 parse edildi; 1952 vs 1999 = 47 yıl → REJECT bekleniyor
    assert result.rejected is True, (
        "Dosya adındaki yıl (1952) parse edilip yıl uyumsuzluğuyla REJECT edilmeliydi"
    )
    assert result.reverse_breakdown.get("year", {}).get("ocr") == 1952, (
        "Dosya adından 1952 parse edilmeliydi"
    )
    assert result.reverse_breakdown.get("year", {}).get("neg") == -3.0, (
        "47 yıl fark → -3.0 yıl cezası bekleniyor"
    )


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-09: Başlık farklı dil → başlık kategorisi atlanır
# ─────────────────────────────────────────────────────────────────────────────

def test_title_different_language_skipped():
    """
    REVVAL-09: OCR başlığı Türkçe ("Gün Batısı"), TMDB başlığı İngilizce ("The Sundowners"),
    orijinal başlığı da İngilizce → farklı dil → başlık skoru 0.0/0.0 (atlanır).
    """
    verifier, _ = _make_verifier()

    entry = {
        "id": 99,
        "title": "The Sundowners",
        "original_title": "The Sundowners",
        "release_date": "1960-01-01",
        "media_type": "movie",
    }
    credits = {"cast": [], "crew": []}

    _, _, breakdown = verifier._reverse_validate(
        ocr_title="Gün Batısı",
        ocr_cast_names=[],
        ocr_director_names=[],
        ocr_year=0,
        tmdb_entry=entry,
        credits_data=credits,
        forward_hits=0,
        forward_misses=0,
    )

    assert breakdown["title"]["pos"] == 0.0, (
        f"Farklı dil → başlık pos=0.0 olmalı: {breakdown['title']}"
    )
    assert breakdown["title"]["neg"] == 0.0, (
        f"Farklı dil → başlık neg=0.0 olmalı: {breakdown['title']}"
    )
    assert breakdown["title"]["compare"] is None, (
        "Farklı dil → compare=None olmalı (başlık atlandı)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-10: Başlık aynı dil ama tamamen farklı → -4.0 ceza
# ─────────────────────────────────────────────────────────────────────────────

def test_title_same_language_mismatch_heavy_penalty():
    """
    REVVAL-10: OCR "Yalının Kuşları" (Türkçe), TMDB title "Ölüm Yolu" (Türkçe)
    → aynı dil, tamamen farklı isim → fuzzy çok düşük → -4.0 ceza.
    """
    verifier, _ = _make_verifier()

    entry = {
        "id": 200,
        "title": "Ölüm Yolu",
        "original_title": "Road to Death",
        "release_date": "1970-01-01",
        "media_type": "movie",
    }
    credits = {"cast": [], "crew": []}

    _, _, breakdown = verifier._reverse_validate(
        ocr_title="Yalının Kuşları",
        ocr_cast_names=[],
        ocr_director_names=[],
        ocr_year=0,
        tmdb_entry=entry,
        credits_data=credits,
        forward_hits=0,
        forward_misses=0,
    )

    assert breakdown["title"]["neg"] == -4.0, (
        f"Aynı dil, tamamen farklı isim → -4.0 ceza bekleniyor: {breakdown['title']}"
    )
    assert breakdown["title"]["compare"] == "Ölüm Yolu", (
        "Türkçe OCR, Türkçe TMDB başlığıyla karşılaştırılmalı"
    )


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-11: Cast bonus — mutlak eşleşme sayısına göre
# ─────────────────────────────────────────────────────────────────────────────

def test_cast_bonus_levels():
    """
    REVVAL-11: Cast bonus seviyelerini test eder.
    - ≥8 hits → cast_pos = oran_puanı + 3.0
    - ≥5 hits → cast_pos = oran_puanı + 2.0
    - ≥3 hits → cast_pos = oran_puanı + 1.0
    """
    verifier, _ = _make_verifier()

    base_entry = {
        "id": 1, "title": "Test Film", "original_title": "Test Film",
        "release_date": "2000-01-01", "media_type": "movie",
    }
    credits = {"cast": [], "crew": []}

    # ≥8 hits → bonus +3.0; ratio=1.0 → cast_pos_ratio=3.0; total=6.0
    _, _, bd = verifier._reverse_validate(
        ocr_title="", ocr_cast_names=[], ocr_director_names=[],
        ocr_year=0, tmdb_entry=base_entry, credits_data=credits,
        forward_hits=8, forward_misses=0,
    )
    assert bd["cast"]["pos"] == 6.0, f"8 hits → 3.0+3.0=6.0 bekleniyor: {bd['cast']}"

    # ≥5 hits → bonus +2.0; ratio=1.0 → cast_pos_ratio=3.0; total=5.0
    _, _, bd = verifier._reverse_validate(
        ocr_title="", ocr_cast_names=[], ocr_director_names=[],
        ocr_year=0, tmdb_entry=base_entry, credits_data=credits,
        forward_hits=5, forward_misses=0,
    )
    assert bd["cast"]["pos"] == 5.0, f"5 hits → 3.0+2.0=5.0 bekleniyor: {bd['cast']}"

    # ≥3 hits → bonus +1.0; ratio=1.0 → cast_pos_ratio=3.0; total=4.0
    _, _, bd = verifier._reverse_validate(
        ocr_title="", ocr_cast_names=[], ocr_director_names=[],
        ocr_year=0, tmdb_entry=base_entry, credits_data=credits,
        forward_hits=3, forward_misses=0,
    )
    assert bd["cast"]["pos"] == 4.0, f"3 hits → 3.0+1.0=4.0 bekleniyor: {bd['cast']}"

    # 2 hits → bonus 0; ratio=1.0 → cast_pos_ratio=3.0; total=3.0
    _, _, bd = verifier._reverse_validate(
        ocr_title="", ocr_cast_names=[], ocr_director_names=[],
        ocr_year=0, tmdb_entry=base_entry, credits_data=credits,
        forward_hits=2, forward_misses=0,
    )
    assert bd["cast"]["pos"] == 3.0, f"2 hits → 3.0+0=3.0 bekleniyor: {bd['cast']}"


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-12: Dinamik eşik — aktif kategorilere orantılı
# ─────────────────────────────────────────────────────────────────────────────

def test_dynamic_threshold_active_categories():
    """
    REVVAL-12: Dinamik eşik = aktif kategorilerin max toplamı × 0.40.
    - 4 kategori aktif → max=13.0 → eşik=5.2
    - Yıl bilinmiyor → max=11.0 → eşik=4.4
    - Başlık atlandı (farklı dil) + yıl bilinmiyor → max=8.5 → eşik=3.4
    """
    verifier, _ = _make_verifier()

    base_credits = {"cast": [], "crew": [{"name": "Some Director", "job": "Director"}]}

    # 4 kategori aktif: başlık(TR) + yönetmen + cast + yıl → max=13.0 → eşik=5.2
    _, _, bd = verifier._reverse_validate(
        ocr_title="Dövüş Kulübü",  # Türkçe
        ocr_cast_names=[],
        ocr_director_names=["Some Director"],
        ocr_year=1999,
        tmdb_entry={
            "id": 1, "title": "Dövüş Kulübü", "original_title": "Fight Club",
            "release_date": "1999-01-01",
        },
        credits_data=base_credits,
        forward_hits=10, forward_misses=0,
    )
    assert bd["threshold"] == 5.2, (
        f"4 aktif kategori → eşik=5.2 bekleniyor: {bd['threshold']}"
    )

    # Yıl bilinmiyor → max=11.0 → eşik=4.4
    _, _, bd = verifier._reverse_validate(
        ocr_title="Dövüş Kulübü",
        ocr_cast_names=[],
        ocr_director_names=["Some Director"],
        ocr_year=0,  # bilinmiyor
        tmdb_entry={
            "id": 1, "title": "Dövüş Kulübü", "original_title": "Fight Club",
            "release_date": "1999-01-01",
        },
        credits_data=base_credits,
        forward_hits=10, forward_misses=0,
    )
    assert bd["threshold"] == 4.4, (
        f"Yıl bilinmiyor → eşik=4.4 bekleniyor: {bd['threshold']}"
    )

    # Başlık farklı dil (atlanır) + yıl bilinmiyor → max=8.5 → eşik=3.4
    _, _, bd = verifier._reverse_validate(
        ocr_title="Gün Batısı",       # Türkçe, TMDB başlığı İngilizce → atlanır
        ocr_cast_names=[],
        ocr_director_names=["Some Director"],
        ocr_year=0,
        tmdb_entry={
            "id": 1, "title": "The Sundowners", "original_title": "The Sundowners",
            "release_date": "1960-01-01",
        },
        credits_data=base_credits,
        forward_hits=10, forward_misses=0,
    )
    assert bd["threshold"] == 3.4, (
        f"Başlık atlandı + yıl bilinmiyor → eşik=3.4 bekleniyor: {bd['threshold']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# REVVAL-13: Cast negatif ceza — hits ≥ 3 ise ceza yok
# ─────────────────────────────────────────────────────────────────────────────

def test_cast_neg_suppressed_when_hits_ge_3():
    """
    REVVAL-13: forward_hits ≥ 3 ise oran düşük olsa bile cast_neg=0.0.
    100 kişilik OCR listesinden 11'i eşleşirse oran %11 ama hits=11 ≥ 3 → ceza yok.
    """
    verifier, _ = _make_verifier()

    entry = {
        "id": 1, "title": "Test Film", "original_title": "Test Film",
        "release_date": "2000-01-01", "media_type": "movie",
    }
    credits = {"cast": [], "crew": []}

    # 11 hits, 89 misses → ratio=11% → normalde -2.0 ceza gelirdi, ama hits ≥ 3 → 0
    _, _, breakdown = verifier._reverse_validate(
        ocr_title="Test Film",
        ocr_cast_names=[],
        ocr_director_names=[],
        ocr_year=0,
        tmdb_entry=entry,
        credits_data=credits,
        forward_hits=11,
        forward_misses=89,
    )

    assert breakdown["cast"]["neg"] == 0.0, (
        f"hits=11 ≥ 3 → cast_neg=0.0 bekleniyor: {breakdown['cast']}"
    )

    # 2 hits, 19 misses → ratio=2/21≈9.5% < 10% → hits < 3 → oran bazlı ceza: -2.0
    _, _, breakdown2 = verifier._reverse_validate(
        ocr_title="Test Film",
        ocr_cast_names=[],
        ocr_director_names=[],
        ocr_year=0,
        tmdb_entry=entry,
        credits_data=credits,
        forward_hits=2,
        forward_misses=19,
    )

    assert breakdown2["cast"]["neg"] == -2.0, (
        f"hits=2 < 3, ratio≈9.5% → cast_neg=-2.0 bekleniyor: {breakdown2['cast']}"
    )
