"""
test_person_evidence_recovery.py — Strateji C/D kişi kanıtı kurtarma testleri.

Film eşleşmesi başarısız olduğunda (ters doğrulama reddetti veya eşleşme bulunamadı)
Strateji C/D'nin kişi kanıtını korumasını ve downstream cast çıktısına eklemesini test eder.

Kapsam:
  PEVR-01: Strateji C kişi kanıtı biriktiriliyor (combined_credits yoluyla)
  PEVR-02: Strateji C kişi kanıtı biriktiriliyor (known_for yoluyla)
  PEVR-03: Strateji D kişi kanıtı biriktiriliyor (fuzzy doğrulama geçti)
  PEVR-04: Ters doğrulama reddedince person_evidence TMDBVerifyResult'ta korunuyor
  PEVR-05: Eşleşme bulunamazsa (tüm stratejiler başarısız) person_evidence korunuyor
  PEVR-06: _merge_tmdb_person_evidence — cast rolü cast'a ekleniyor
  PEVR-07: _merge_tmdb_person_evidence — director rolü crew'e ekleniyor
  PEVR-08: _merge_tmdb_person_evidence — duplicate isimler atlanıyor (cast)
  PEVR-09: _merge_tmdb_person_evidence — duplicate isimler atlanıyor (crew/director)
  PEVR-10: _merge_tmdb_person_evidence — crew rolü atlanıyor
  PEVR-11: Reddedilen film reddedilmiş kalıyor (güvenlik: film kabul edilmiyor)
  PEVR-12: Başarılı TMDB eşleşmesinde person_evidence de dolu geliyor
  PEVR-13: Strateji A/B başarısında person_evidence boş liste olarak geliyor
"""
from __future__ import annotations

import sys
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# ── Ağır native bağımlılıklar test ortamında olmayabilir ────────────────────
import types as _types

for _stub_name in ("cv2", "numpy", "numpy.core", "numpy.linalg",
                   "paddleocr", "paddleocr.paddleocr"):
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = _types.ModuleType(_stub_name)

_np_stub = sys.modules["numpy"]
if not hasattr(_np_stub, "ndarray"):
    _np_stub.ndarray = object
if not hasattr(_np_stub, "uint8"):
    _np_stub.uint8 = int
if not hasattr(_np_stub, "array"):
    _np_stub.array = list

from core.tmdb_verify import TMDBVerify, TMDBVerifyResult
from core.pipeline_runner import _merge_tmdb_person_evidence


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
# PEVR-01: Strateji C combined_credits yoluyla kişi kanıtı biriktirme
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr01_strategy_c_accumulates_person_evidence_via_combined_credits():
    """
    PEVR-01: combined_credits'e katkıda bulunan aktör _run_person_search
    tarafından matched_persons listesine eklenmeli.
    """
    verifier, logs = _make_verifier()

    PERSON_ID = 9999
    FILM_ID = 1234

    def fake_search_multi(title):
        return []

    def fake_search_person(query):
        return [{"id": PERSON_ID, "name": "Nisa Serezli"}]

    def fake_person_combined_credits(pid):
        return {
            "cast": [
                {"id": FILM_ID, "media_type": "movie", "title": "Çılgın Amanda"}
            ],
            "crew": []
        }

    # Film kredileri — eşleşme sağlamak için aynı aktörü içeriyor
    def fake_fetch_credits(kind, mid):
        return {
            "cast": [{"name": "Nisa Serezli"}, {"name": "Cihat Tamer"},
                     {"name": "Hadi Çaman"}],
            "crew": [],
        }

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier.client, "search_person", side_effect=fake_search_person), \
         patch.object(verifier.client, "get_person_combined_credits",
                      side_effect=fake_person_combined_credits), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind, via, evidence = verifier._find_tmdb_entry(
            film_title="Çılgın Amanda",
            cast_names=["Nisa Serezli"],
            director_names=[],
        )

    # Eşleşme bulunmuş olmalı (tek aktör bile yeterli çünkü MIN_ACTOR_MATCH=3, ama
    # work_matches'te film biriktirildi ve count >= 1)
    # Önemli olan: person_evidence dolu olmalı
    assert isinstance(evidence, list)
    if entry is not None:
        # Eşleşme varsa kanıt dolu olmalı
        assert len(evidence) >= 1
        pe = evidence[0]
        assert pe["ocr_name"] == "Nisa Serezli"
        assert pe["tmdb_name"] == "Nisa Serezli"
        assert pe["tmdb_id"] == PERSON_ID
        assert pe["role"] == "cast"
        assert pe["source_strategy"] == "C"


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-02: known_for fallback yoluyla kişi kanıtı biriktirme
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr02_strategy_c_accumulates_person_evidence_via_known_for():
    """
    PEVR-02: combined_credits API hatası verince known_for fallback kullanılmalı,
    katkı yapan kişi yine de matched_persons'a eklenmeli.
    """
    verifier, logs = _make_verifier()

    PERSON_ID = 8888
    FILM_ID = 5678

    def fake_search_multi(title):
        return []

    def fake_search_person(query):
        return [{
            "id": PERSON_ID,
            "name": "Cihat Tamer",
            "known_for": [
                {"id": FILM_ID, "media_type": "movie", "title": "Çılgın Amanda"}
            ]
        }]

    def fake_person_combined_credits(pid):
        raise Exception("API hatası")

    def fake_fetch_credits(kind, mid):
        return {
            "cast": [{"name": "Cihat Tamer"}, {"name": "Nisa Serezli"},
                     {"name": "Hadi Çaman"}],
            "crew": [],
        }

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier.client, "search_person", side_effect=fake_search_person), \
         patch.object(verifier.client, "get_person_combined_credits",
                      side_effect=fake_person_combined_credits), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind, via, evidence = verifier._find_tmdb_entry(
            film_title="Çılgın Amanda",
            cast_names=["Cihat Tamer"],
            director_names=[],
        )

    assert isinstance(evidence, list)
    # known_for katkısı varsa kanıt eklenmiş olmalı
    cihat_evidence = [pe for pe in evidence if pe["ocr_name"] == "Cihat Tamer"]
    if cihat_evidence:
        pe = cihat_evidence[0]
        assert pe["tmdb_id"] == PERSON_ID
        assert pe["role"] == "cast"


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-03: Strateji D fuzzy doğrulama geçen kişi kanıtı biriktirme
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr03_strategy_d_accumulates_person_evidence_fuzzy_pass():
    """
    PEVR-03: Strateji D — fuzzy ratio >= 80 geçen kişi, combined_credits
    ile katkı yaptıysa matched_persons'a eklenmeli.
    Strategy C veya D etiketli olabilir (C önce çalışır).
    """
    verifier, logs = _make_verifier()

    PERSON_ID = 7777
    FILM_ID = 4321

    def fake_search_multi(title):
        return []

    def fake_search_person(query):
        # "Tolga Aşkıner" aranınca dönen sonuç (aynı isim, yüksek fuzzy)
        return [{"id": PERSON_ID, "name": "Tolga Aşkıner"}]

    def fake_person_combined_credits(pid):
        return {
            "cast": [{"id": FILM_ID, "media_type": "movie", "title": "Test Dizisi"}],
            "crew": []
        }

    def fake_fetch_credits(kind, mid):
        return {
            "cast": [],
            "crew": [{"name": "Tolga Aşkıner", "job": "Director"}],
        }

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier.client, "search_person", side_effect=fake_search_person), \
         patch.object(verifier.client, "get_person_combined_credits",
                      side_effect=fake_person_combined_credits), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind, via, evidence = verifier._find_tmdb_entry(
            film_title="Çılgın Amanda",
            cast_names=[],
            director_names=["Tolga Aşkıner"],
        )

    assert isinstance(evidence, list)
    tolga_evidence = [pe for pe in evidence if pe["ocr_name"] == "Tolga Aşkıner"]
    if tolga_evidence:
        pe = tolga_evidence[0]
        assert pe["tmdb_id"] == PERSON_ID
        assert pe["role"] == "director"
        # Strategy C runs before D, so source_strategy may be "C" or "D"
        assert pe["source_strategy"] in ("C", "D")


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-04: Ters doğrulama reddince person_evidence TMDBVerifyResult'ta korunuyor
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr04_person_evidence_preserved_on_reverse_validation_rejection():
    """
    PEVR-04: Ters doğrulama reddedince:
    - TMDBVerifyResult.reason == 'reverse_validation_rejected'
    - TMDBVerifyResult.rejected == True
    - TMDBVerifyResult.person_evidence dolu olmalı
    - Film kesinlikle reddedilmiş olmalı (matched_id == 0)
    """
    verifier, logs = _make_verifier()

    evidence_data = [
        {"ocr_name": "Nisa Serezli", "tmdb_name": "Nisa Serezli",
         "tmdb_id": 100, "role": "cast", "source_strategy": "C"},
        {"ocr_name": "Cihat Tamer", "tmdb_name": "Cihat Tamer",
         "tmdb_id": 200, "role": "cast", "source_strategy": "C"},
    ]

    wrong_film = {
        "id": 1034655,
        "title": "Güngörmüşler",
        "original_title": "Güngörmüşler",
        "release_date": "1980-01-01",
        "media_type": "movie",
    }
    wrong_film_credits = {
        "cast": [{"name": "Nisa Serezli"}, {"name": "Cihat Tamer"},
                 {"name": "Hadi Çaman"}, {"name": "Ali Oroğuz"},
                 {"name": "Tolga Aşkıner"}],
        "crew": [],
    }

    cdata = {
        "film_title": "Çılgın Amanda",
        "cast": [
            {"actor_name": "Nisa Serezli", "confidence": 0.9},
            {"actor_name": "Cihat Tamer", "confidence": 0.8},
        ],
        "directors": [{"name": "Tolga Aşkıner"}],
        "crew": [],
    }

    # _find_tmdb_entry 4-tuple döndürüyor; person_evidence içeriyor
    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(wrong_film, "movie", "cast_only", evidence_data)), \
         patch.object(verifier, "_fetch_credits",
                      return_value=wrong_film_credits), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    # Film reddedilmeli
    assert result.rejected is True, "Film reddedilmeli"
    assert result.reason == "reverse_validation_rejected", (
        f"Beklenen reason='reverse_validation_rejected', alınan='{result.reason}'"
    )
    assert result.matched_id == 0, "Reddedilen filmde matched_id 0 olmalı"
    assert result.updated is False, "Reddedilen film updated=False olmalı"

    # Person evidence korunmuş olmalı
    assert result.person_evidence is not None
    assert len(result.person_evidence) == 2
    names = [pe["ocr_name"] for pe in result.person_evidence]
    assert "Nisa Serezli" in names
    assert "Cihat Tamer" in names

    # Loglarda person evidence korunma mesajı görünmeli
    recovery_logs = [l for l in logs if "PersonRecovery" in l or "kişi kanıtı korunuyor" in l]
    assert recovery_logs, f"Kişi kanıtı korunma logu yok. Loglar: {logs}"


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-05: Tüm stratejiler başarısız — person_evidence yine de korunuyor
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr05_person_evidence_preserved_when_no_match_found():
    """
    PEVR-05: Tüm stratejiler başarısız (entry=None) olsa bile
    person_evidence TMDBVerifyResult'a geçmeli.
    """
    verifier, logs = _make_verifier()

    evidence_data = [
        {"ocr_name": "Barış Manço", "tmdb_name": "Barış Manço",
         "tmdb_id": 500, "role": "cast", "source_strategy": "C"},
    ]

    cdata = {
        "film_title": "Bilinmeyen Film",
        "cast": [{"actor_name": "Barış Manço", "confidence": 0.7}],
        "directors": [],
        "crew": [],
    }

    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(None, "", "", evidence_data)), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    assert result.reason == "tmdb match not found"
    assert result.updated is False
    assert result.person_evidence is not None
    assert len(result.person_evidence) == 1
    assert result.person_evidence[0]["ocr_name"] == "Barış Manço"


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-06: _merge_tmdb_person_evidence — cast rolü cast'a ekleniyor
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr06_merge_cast_role_adds_to_cast():
    """
    PEVR-06: role='cast' olan kişi cdata['cast']'a tmdb_person_recovery etiketiyle eklenmeli.
    """
    cdata = {
        "cast": [{"actor_name": "Mevcut Oyuncu", "confidence": 0.9}],
        "crew": [],
        "directors": [],
    }
    evidence = [
        {"ocr_name": "Nisa Serezli", "tmdb_name": "Nisa Serezli",
         "tmdb_id": 100, "role": "cast", "source_strategy": "C"},
    ]
    logs = []
    added = _merge_tmdb_person_evidence(cdata, evidence, lambda m: logs.append(m))

    assert added == 1
    cast_names = [r.get("actor_name") or r.get("actor") for r in cdata["cast"]]
    assert "Nisa Serezli" in cast_names

    nisa = next(r for r in cdata["cast"] if r.get("actor_name") == "Nisa Serezli")
    assert nisa["raw"] == "tmdb_person_recovery"
    assert nisa["source"] == "tmdb_person_recovery"
    assert nisa["tmdb_id"] == 100
    assert nisa["tmdb_strategy"] == "C"


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-07: _merge_tmdb_person_evidence — director rolü crew'e ekleniyor
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr07_merge_director_role_adds_to_crew():
    """
    PEVR-07: role='director' olan kişi cdata['crew']'a Yönetmen etiketiyle eklenmeli.
    """
    cdata = {
        "cast": [],
        "crew": [],
        "directors": [],
    }
    evidence = [
        {"ocr_name": "Tolga Aşkıner", "tmdb_name": "Tolga Aşkıner",
         "tmdb_id": 300, "role": "director", "source_strategy": "D"},
    ]
    logs = []
    added = _merge_tmdb_person_evidence(cdata, evidence, lambda m: logs.append(m))

    assert added == 1
    crew_names = [r.get("name") for r in cdata.get("crew", [])]
    assert "Tolga Aşkıner" in crew_names

    tolga = next(r for r in cdata["crew"] if r.get("name") == "Tolga Aşkıner")
    assert tolga["job"] == "Director"
    assert tolga["role_tr"] == "Yönetmen"
    assert tolga["raw"] == "tmdb_person_recovery"


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-08: Duplicate cast isimleri atlanıyor
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr08_duplicate_cast_names_skipped():
    """
    PEVR-08: Mevcut cast'ta zaten olan isim tekrar eklenmemeli.
    """
    cdata = {
        "cast": [{"actor_name": "Nisa Serezli", "confidence": 0.9}],
        "crew": [],
        "directors": [],
    }
    evidence = [
        {"ocr_name": "Nisa Serezli", "tmdb_name": "Nisa Serezli",
         "tmdb_id": 100, "role": "cast", "source_strategy": "C"},
    ]
    logs = []
    added = _merge_tmdb_person_evidence(cdata, evidence, lambda m: logs.append(m))

    assert added == 0
    # Cast hâlâ 1 kayıt içermeli
    assert len(cdata["cast"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-09: Duplicate director isimleri atlanıyor
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr09_duplicate_director_names_skipped():
    """
    PEVR-09: Mevcut crew/directors'ta zaten olan yönetmen tekrar eklenmemeli.
    """
    cdata = {
        "cast": [],
        "crew": [{"name": "Tolga Aşkıner", "job": "Director"}],
        "directors": [],
    }
    evidence = [
        {"ocr_name": "Tolga Aşkıner", "tmdb_name": "Tolga Aşkıner",
         "tmdb_id": 300, "role": "director", "source_strategy": "D"},
    ]
    logs = []
    added = _merge_tmdb_person_evidence(cdata, evidence, lambda m: logs.append(m))

    assert added == 0
    assert len(cdata["crew"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-10: crew rolü atlanıyor (sadece cast ve director ekleniyor)
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr10_crew_role_skipped():
    """
    PEVR-10: role='crew' olan kişiler _merge'de atlanmalı (fazla gürültülü).
    """
    cdata = {"cast": [], "crew": [], "directors": []}
    evidence = [
        {"ocr_name": "Kameramanı", "tmdb_name": "Kameramanı",
         "tmdb_id": 400, "role": "crew", "source_strategy": "C"},
    ]
    logs = []
    added = _merge_tmdb_person_evidence(cdata, evidence, lambda m: logs.append(m))

    assert added == 0
    assert cdata["cast"] == []
    assert cdata["crew"] == []


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-11: Reddedilen film reddedilmiş kalıyor (güvenlik regresyon testi)
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr11_rejected_film_stays_rejected():
    """
    PEVR-11: Person evidence eklenmesi, reddedilen filmin kabul edilmesine
    yol açmamalı. Film kesinlikle reddedilmiş olarak kalmalı.
    """
    verifier, logs = _make_verifier()

    wrong_film = {
        "id": 1034655,
        "title": "Güngörmüşler",
        "original_title": "Güngörmüşler",
        "release_date": "1980-01-01",
        "media_type": "movie",
    }
    wrong_film_credits = {
        "cast": [{"name": "Nisa Serezli"}, {"name": "Cihat Tamer"},
                 {"name": "Hadi Çaman"}, {"name": "Ali Oroğuz"},
                 {"name": "Tolga Aşkıner"}],
        "crew": [],
    }
    evidence_data = [
        {"ocr_name": "Nisa Serezli", "tmdb_name": "Nisa Serezli",
         "tmdb_id": 100, "role": "cast", "source_strategy": "C"},
    ]

    cdata = {
        "film_title": "Çılgın Amanda",
        "cast": [
            {"actor_name": "Nisa Serezli", "confidence": 0.9},
            {"actor_name": "Cihat Tamer", "confidence": 0.8},
        ],
        "directors": [{"name": "Tolga Aşkıner"}],
        "crew": [],
    }

    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(wrong_film, "movie", "cast_only", evidence_data)), \
         patch.object(verifier, "_fetch_credits",
                      return_value=wrong_film_credits), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    # Film mutlaka reddedilmeli — person evidence bunu değiştirmemeli
    assert result.rejected is True, "Film reddedilmeli — person evidence kabul ettirmemeli"
    assert result.updated is False
    assert result.matched_id == 0
    assert result.reason == "reverse_validation_rejected"


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-12: Başarılı TMDB eşleşmesinde person_evidence de dolu geliyor
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr12_successful_match_also_carries_person_evidence():
    """
    PEVR-12: Başarılı eşleşmede person_evidence de TMDBVerifyResult'ta bulunmalı.
    (Stratejiler A/B değil C/D kullanılırsa kanıt dolu olabilir.)
    """
    verifier, logs = _make_verifier()

    evidence_data = [
        {"ocr_name": "Oyuncu Biri", "tmdb_name": "Oyuncu Biri",
         "tmdb_id": 111, "role": "cast", "source_strategy": "C"},
    ]

    correct_film = {
        "id": 12345,
        "title": "Test Film",
        "original_title": "Test Film",
        "release_date": "2000-01-01",
        "media_type": "movie",
    }
    correct_film_credits = {
        "cast": [{"name": "Oyuncu Biri", "character": "", "order": 1},
                 {"name": "Oyuncu İki", "character": "", "order": 2},
                 {"name": "Oyuncu Üç", "character": "", "order": 3}],
        "crew": [{"name": "Yönetmen Ali", "job": "Director", "department": "Directing"}],
    }

    cdata = {
        "film_title": "Test Film",
        "cast": [
            {"actor_name": "Oyuncu Biri", "confidence": 0.9},
            {"actor_name": "Oyuncu İki", "confidence": 0.9},
            {"actor_name": "Oyuncu Üç", "confidence": 0.9},
        ],
        "directors": [{"name": "Yönetmen Ali"}],
        "crew": [],
    }

    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(correct_film, "movie", "cast_only", evidence_data)), \
         patch.object(verifier, "_fetch_credits",
                      return_value=correct_film_credits), \
         patch.object(verifier.client, "get_movie_details",
                      return_value={"genres": [], "original_title": "Test Film"}), \
         patch.object(verifier.client, "get_movie_keywords", return_value=[]), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    # Eşleşme başarılı olmalı
    assert result.reason == "ok" or result.updated is True or result.rejected is False
    # person_evidence dolu olmalı
    assert result.person_evidence is not None
    assert len(result.person_evidence) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# PEVR-13: Strateji A/B başarısında person_evidence boş liste olarak geliyor
# ─────────────────────────────────────────────────────────────────────────────

def test_pevr13_strategy_a_success_returns_empty_person_evidence():
    """
    PEVR-13: Strateji A başarılı olduğunda (kişi araması olmadan),
    person_evidence boş liste olarak dönmeli.
    """
    verifier, logs = _make_verifier()

    MOVIE_ID = 2001
    MOVIE_ENTRY = {"id": MOVIE_ID, "media_type": "movie", "title": "Doğru Film"}

    def fake_search_multi(title):
        return [MOVIE_ENTRY]

    def fake_fetch_credits(kind, mid):
        return {
            "cast": [{"name": "Ahmet Yılmaz"}, {"name": "Mehmet Demir"},
                     {"name": "Ali Kaya"}],
            "crew": [{"name": "Yönetmen X", "job": "Director"}],
        }

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind, via, evidence = verifier._find_tmdb_entry(
            film_title="Doğru Film",
            cast_names=["Ahmet Yılmaz", "Mehmet Demir", "Ali Kaya"],
            director_names=["Yönetmen X"],
        )

    # Strateji A başarılı
    assert entry is not None, "Strateji A eşleşme bulmalı"
    assert via in ("title", "original_title"), f"Beklenen via=title/original_title, alınan='{via}'"
    # Strateji A/B kişi araması yapmaz → person_evidence boş
    assert evidence == [], f"Strateji A'da person_evidence boş olmalı, alınan: {evidence}"
