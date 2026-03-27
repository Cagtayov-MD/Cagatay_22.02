"""
test_tmdb_find_entry_strategies.py — A/B/C/D strateji akışının unit testleri.

Kapsam:
  STRAT-A01: Film adı + yönetmen + oyuncu → eşleşme bulundu → kabul
  STRAT-A02: Film adı + yönetmen + oyuncu → eşleşme yok → Strateji B'ye geç
  STRAT-B01: Sadece film adı + yönetmen → eşleşme bulundu + DİZİ → direkt kabul
  STRAT-B02: Sadece film adı + yönetmen → eşleşme bulundu + FİLM + yıl eşleşti → kabul
  STRAT-B03: Film + crew kıyaslaması → OCR verisi yok → kabul (kanıtsız red olmaz)
  STRAT-B04: Film + crew kıyaslaması → doğrulanamadı + OCR verisi var → Strateji C'ye geç
  STRAT-C01: Oyuncu isimleri + yönetmen → ortak film bulundu → kabul
  STRAT-D01: Kaynaği olmayan film → fuzzy doğrulama → gerçek oyuncu → ortak film bul
  STRAT-D02: Oyuncular sahte / yok → tüm stratejiler başarısız → None döner
"""

import sys
import os
import tempfile
from unittest.mock import patch

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core.tmdb_verify import TMDBVerify


def _make_verifier():
    tmp = tempfile.mkdtemp()
    return TMDBVerify(work_dir=tmp, api_key="test-key")


# ── Strateji A testleri ─────────────────────────────────────────────────────


def test_strat_a01_title_actor_director_match_accepted():
    """
    STRAT-A01: Film adı + 1 oyuncu + yönetmen eşleşmesi → kabul.
    """
    verifier = _make_verifier()

    MOVIE_ID = 1001
    MOVIE_ENTRY = {"id": MOVIE_ID, "media_type": "movie", "title": "Test Film"}

    def fake_search_multi(title):
        return [MOVIE_ENTRY]

    def fake_fetch_credits(kind, mid):
        return {
            "cast": [{"name": "Ahmet Yılmaz"}, {"name": "Bilinmeyen Kişi"}],
            "crew": [{"name": "Yönetmen Ali", "job": "Director"}],
        }

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind, via, _ = verifier._find_tmdb_entry(
            film_title="Test Film",
            cast_names=["Ahmet Yılmaz"],
            director_names=["Yönetmen Ali"],
        )

    assert entry is not None, "Strateji A: başlık + 1 oyuncu + yönetmen kabul edilmeli"
    assert entry["id"] == MOVIE_ID
    assert via in ("title", "original_title")


def test_strat_a01_title_actor_director_required():
    """
    STRAT-A01: Film adı + oyuncu eşleşmesi ama yönetmen yok → Strateji A reddeder.

    Strateji A, başlık + yönetmen zorunlu (cast-only devre dışı).
    Yönetmen olmadan Strateji B de atlanır → C/D person search boş → None.
    """
    verifier = _make_verifier()

    MOVIE_ID = 1002
    MOVIE_ENTRY = {"id": MOVIE_ID, "media_type": "movie", "title": "Test Film 2"}

    def fake_search_multi(title):
        return [MOVIE_ENTRY]

    def fake_fetch_credits(kind, mid):
        return {
            "cast": [{"name": "Oyuncu Bir"}, {"name": "Oyuncu İki"}, {"name": "Başkası"}],
            "crew": [],
        }

    def fake_search_person(name):
        return []

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier.client, "search_person", side_effect=fake_search_person), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind, via, _ = verifier._find_tmdb_entry(
            film_title="Test Film 2",
            cast_names=["Oyuncu Bir", "Oyuncu İki"],
            director_names=[],
        )

    assert entry is None, "Strateji A: yönetmen olmadan kabul edilmemeli"


def test_strat_a02_no_actor_match_falls_to_b():
    """
    STRAT-A02: Film adı + oyuncular eşleşmiyor + yönetmen yok → Strateji B/C/D'ye düşer.

    Strateji B de yönetmen olmadığından atlanır.
    Strateji C/D person search mock'u boş döndürür.
    Sonuç: None
    """
    verifier = _make_verifier()

    MOVIE_ENTRY = {"id": 999, "media_type": "movie", "title": "Bambaşka Film"}

    def fake_search_multi(title):
        return [MOVIE_ENTRY]

    def fake_fetch_credits(kind, mid):
        return {"cast": [{"name": "Alakasız Kişi"}], "crew": []}

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier.client, "search_person", return_value=[]):

        entry, kind, via, _ = verifier._find_tmdb_entry(
            film_title="Test Film",
            cast_names=["Farklı Oyuncu"],
            director_names=[],
        )

    assert entry is None, "Oyuncu eşleşmesi yoksa ve yönetmen yoksa None dönmeli"


# ── Strateji B testleri ─────────────────────────────────────────────────────


def test_strat_b01_series_director_match_accepted():
    """
    STRAT-B01: Film adı + yönetmen eşleşmesi + DİZİ (is_series=True) → direkt kabul.

    Oyuncular hiç eşleşmese bile dizi olduğundan direkt kabul edilmeli.
    """
    verifier = _make_verifier()

    TV_ID = 2001
    TV_ENTRY = {"id": TV_ID, "media_type": "tv", "name": "Test Dizi"}

    def fake_search_multi(title):
        return [TV_ENTRY]

    def fake_fetch_credits(kind, mid):
        return {
            "cast": [{"name": "Bilinmeyen Oyuncu"}],
            "crew": [{"name": "Dizi Yönetmeni", "job": "Director"}],
        }

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier.client, "search_person", return_value=[]):

        entry, kind, via, _ = verifier._find_tmdb_entry(
            film_title="Test Dizi",
            cast_names=["OCR Oyuncu Adı"],  # eşleşmiyor
            director_names=["Dizi Yönetmeni"],
            is_series=True,
        )

    assert entry is not None, "Strateji B: DİZİ + yönetmen eşleşmesi kabul edilmeli"
    assert entry["id"] == TV_ID
    assert kind == "tv"


def test_strat_b02_film_director_year_match_accepted():
    """
    STRAT-B02: Film adı + yönetmen + FİLM + yıl eşleşmesi → kabul.

    Oyuncular eşleşmez ama yönetmen + yıl eşleşmesi Strateji B'de yeterli.
    """
    verifier = _make_verifier()

    MOVIE_ID = 2002
    MOVIE_ENTRY = {
        "id": MOVIE_ID, "media_type": "movie",
        "title": "Yıl Filmi", "release_date": "2015-06-20",
    }

    def fake_search_multi(title):
        return [MOVIE_ENTRY]

    def fake_fetch_credits(kind, mid):
        return {
            "cast": [{"name": "Bilinmeyen Oyuncu"}],
            "crew": [{"name": "Film Yönetmeni", "job": "Director"}],
        }

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier.client, "search_person", return_value=[]):

        entry, kind, via, _ = verifier._find_tmdb_entry(
            film_title="Yıl Filmi",
            cast_names=["OCR Oyuncu Adı"],  # eşleşmiyor
            director_names=["Film Yönetmeni"],
            is_series=False,
            ocr_year=2015,  # TMDB yılıyla eşleşiyor
        )

    assert entry is not None, "Strateji B: FİLM + yönetmen + yıl eşleşmesi kabul edilmeli"
    assert entry["id"] == MOVIE_ID


def test_strat_b03_film_no_ocr_data_accepted():
    """
    STRAT-B03: Film adı + yönetmen + FİLM + OCR verisi yok → kabul.

    Karşılaştırılacak OCR yılı/crew yoksa reddetme — kanıtsız red olmaz.
    """
    verifier = _make_verifier()

    MOVIE_ID = 2003
    MOVIE_ENTRY = {"id": MOVIE_ID, "media_type": "movie", "title": "Boş OCR Filmi"}

    def fake_search_multi(title):
        return [MOVIE_ENTRY]

    def fake_fetch_credits(kind, mid):
        return {
            "cast": [],
            "crew": [{"name": "Yönetmen X", "job": "Director"}],
        }

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind, via, _ = verifier._find_tmdb_entry(
            film_title="Boş OCR Filmi",
            cast_names=[],  # cast yok
            director_names=["Yönetmen X"],
            is_series=False,
            ocr_year=0,          # yıl yok
            ocr_crew_dicts=[],   # crew yok
        )

    assert entry is not None, (
        "Strateji B: OCR karşılaştırma verisi yoksa başlık + yönetmen eşleşmesi yeterli olmalı"
    )
    assert entry["id"] == MOVIE_ID


def test_strat_b04_film_crew_mismatch_falls_to_c():
    """
    STRAT-B04: Film adı + yönetmen + FİLM + yıl eşleşmiyor → Strateji C/D'ye geç.

    OCR yılı 2010, TMDB yılı 2015 → eşleşme yok. Strateji C/D'ye düşmeli.
    Strateji C/D mock'u boş döndürür → None
    """
    verifier = _make_verifier()

    MOVIE_ENTRY = {
        "id": 2004, "media_type": "movie",
        "title": "Yanlış Yıl Filmi", "release_date": "2015-01-01",
    }

    def fake_search_multi(title):
        return [MOVIE_ENTRY]

    def fake_fetch_credits(kind, mid):
        return {
            "cast": [{"name": "Başka Oyuncu"}],
            "crew": [{"name": "Yönetmen Y", "job": "Director"}],
        }

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier.client, "search_person", return_value=[]):

        entry, kind, via, _ = verifier._find_tmdb_entry(
            film_title="Yanlış Yıl Filmi",
            cast_names=["OCR Oyuncu"],         # eşleşmiyor
            director_names=["Yönetmen Y"],
            is_series=False,
            ocr_year=2010,                     # TMDB=2015, fark 5 > 1 → eşleşmiyor
            ocr_crew_dicts=[
                {"name": "OCR DoP Adı", "job": "Director of Photography"},
            ],
        )

    assert entry is None, (
        "Strateji B: Yıl uyuşmuyor ve OCR crew eşleşmesi yoksa None dönmeli"
    )


# ── Strateji C testleri ─────────────────────────────────────────────────────


def test_strat_c01_person_search_common_film():
    """
    STRAT-C01: Oyuncu isimleri + yönetmen → ortak film bulundu → kabul.

    film_title boş, oyuncularla search_person → combined_credits → ortak film.
    """
    verifier = _make_verifier()

    MOVIE_ID = 3001
    WORK_ENTRY = {"id": MOVIE_ID, "media_type": "movie", "title": "Ortak Film"}

    def fake_search_person(name):
        return [{"id": 101, "name": name}]

    def fake_get_person_combined_credits(person_id):
        return {
            "cast": [{"id": MOVIE_ID, "media_type": "movie", "title": "Ortak Film"}],
            "crew": [],
        }

    def fake_fetch_credits(kind, mid):
        if mid == MOVIE_ID:
            return {
                "cast": [{"name": "Oyuncu Alfa"}, {"name": "Oyuncu Beta"}],
                "crew": [{"name": "Yönetmen Gamma", "job": "Director"}],
            }
        return None

    with patch.object(verifier.client, "search_multi", return_value=[]), \
         patch.object(verifier.client, "search_person", side_effect=fake_search_person), \
         patch.object(verifier.client, "get_person_combined_credits",
                      side_effect=fake_get_person_combined_credits), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind, via, _ = verifier._find_tmdb_entry(
            film_title="",  # başlık yok
            cast_names=["Oyuncu Alfa", "Oyuncu Beta"],
            director_names=["Yönetmen Gamma"],
        )

    assert entry is not None, "Strateji C: kişi aramasıyla ortak film bulunmalı"
    assert entry["id"] == MOVIE_ID
    assert via == "cast_only"


# ── Strateji D testleri ─────────────────────────────────────────────────────


def test_strat_d01_fuzzy_name_validation_passes():
    """
    STRAT-D01: Fuzzy isim doğrulaması → OCR adı TMDB adıyla yeterince eşleşiyor → kabul.

    Strateji C: 3 oyuncu da TMDB kişileriyle eşleşiyor → ortak film → kabul.
    """
    verifier = _make_verifier()

    MOVIE_ID = 4001

    def fake_search_person(name):
        return [{"id": hash(name) & 0xFFFF, "name": name}]

    def fake_get_person_combined_credits(person_id):
        return {
            "cast": [{"id": MOVIE_ID, "media_type": "movie", "title": "Gerçek Film"}],
            "crew": [],
        }

    def fake_fetch_credits(kind, mid):
        if mid == MOVIE_ID:
            return {
                "cast": [
                    {"name": "Ahmet Yilmaz"},
                    {"name": "Mehmet Kaya"},
                    {"name": "Fatma Demir"},
                ],
                "crew": [],
            }
        return None

    with patch.object(verifier.client, "search_multi", return_value=[]), \
         patch.object(verifier.client, "search_person", side_effect=fake_search_person), \
         patch.object(verifier.client, "get_person_combined_credits",
                      side_effect=fake_get_person_combined_credits), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind, via, _ = verifier._find_tmdb_entry(
            film_title="",
            cast_names=["Ahmet Yilmaz", "Mehmet Kaya", "Fatma Demir"],
            director_names=[],
        )

    # Strateji C: 3 kişi ortak filme işaret ediyor + credit doğrulama geçiyor → kabul
    assert entry is not None, "Strateji C: 3 kişi ortak film → kabul edilmeli"
    assert via == "cast_only"


def test_strat_d02_fake_actors_no_match():
    """
    STRAT-D02: Sahte/OCR-gürültüsü oyuncu isimleri → tüm stratejiler başarısız → None.

    Hem Strateji C hem D person search boş döndürüyor → None.
    """
    verifier = _make_verifier()

    with patch.object(verifier.client, "search_multi", return_value=[]), \
         patch.object(verifier.client, "search_person", return_value=[]), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind, via, _ = verifier._find_tmdb_entry(
            film_title="",
            cast_names=["XzXzXz Gürültü", "QqQqQq Sahte", "1234 Geçersiz"],
            director_names=[],
        )

    assert entry is None, "Tüm stratejiler başarısız → None dönmeli"
    assert kind == ""


def test_strat_d02_fuzzy_threshold_rejects_low_match():
    """
    STRAT-D02 (fuzzy eşik): TMDB kişisi OCR adıyla çok az benziyor → Strateji D reddeder.

    Strateji C: "Ahmet" için "Xxxxxxxxx Zzzz" döndürüyor → ortak film bul denenir ama TMDB
    credits eşleşmiyor. Strateji D: fuzzy ratio < 80 → kişi tamamen atlanıyor → None.
    """
    verifier = _make_verifier()

    MOVIE_ID = 4002

    def fake_search_person(name):
        # Tamamen alakasız isim döndür
        return [{"id": 301, "name": "Xxxxxxxxx Zzzz"}]

    def fake_get_person_combined_credits(person_id):
        return {
            "cast": [{"id": MOVIE_ID, "media_type": "movie", "title": "Alakasız Film"}],
            "crew": [],
        }

    def fake_fetch_credits(kind, mid):
        if mid == MOVIE_ID:
            return {
                "cast": [{"name": "Xxxxxxxxx Zzzz"}],
                "crew": [],
            }
        return None

    with patch.object(verifier.client, "search_multi", return_value=[]), \
         patch.object(verifier.client, "search_person", side_effect=fake_search_person), \
         patch.object(verifier.client, "get_person_combined_credits",
                      side_effect=fake_get_person_combined_credits), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind, via, _ = verifier._find_tmdb_entry(
            film_title="",
            cast_names=["Ahmet Yılmaz"],
            director_names=[],
        )

    # Strateji C: "Xxxxxxxxx Zzzz" ile ortak film bulunabilir (C fuzzy validate etmez),
    # ama credits karşılaştırmasında "Ahmet Yılmaz" ≠ "Xxxxxxxxx Zzzz" → threshold altında
    # Strateji D: fuzzy validate açık, "Ahmet Yılmaz" vs "Xxxxxxxxx Zzzz" ratio < 80 → atla
    assert entry is None, (
        "Strateji D: tamamen alakasız TMDB kişisi fuzzy threshold'u geçmemeli → None"
    )


# ── is_series bayrağı pipeline entegrasyon testleri ────────────────────────


def test_is_series_flag_propagated_to_find_entry():
    """
    is_series=True bayrağı verify_credits → _find_tmdb_entry zincirinde iletilmeli.

    _find_tmdb_entry mock'lanarak is_series'in doğru iletildiği doğrulanır.
    """
    tmp = tempfile.mkdtemp()
    verifier = TMDBVerify(work_dir=tmp, api_key="test-key")

    captured = {}

    original_find = verifier._find_tmdb_entry

    def capturing_find(*args, **kwargs):
        captured["is_series"] = kwargs.get("is_series", None)
        return None, "", "", []

    cdata = {
        "film_title": "Test",
        "cast": [{"actor_name": "Oyuncu", "confidence": 0.9}],
        "directors": [{"name": "Yönetmen"}],
        "crew": [],
    }

    with patch.object(verifier, "_find_tmdb_entry", side_effect=capturing_find):
        verifier.verify_credits(cdata, is_series=True)

    assert captured.get("is_series") is True, (
        "is_series=True, _find_tmdb_entry'ye iletilmeli"
    )


def test_is_series_false_default():
    """
    is_series varsayılan False olmalı — pipeline'da açıkça belirtilmezse film kabul edilir.
    """
    tmp = tempfile.mkdtemp()
    verifier = TMDBVerify(work_dir=tmp, api_key="test-key")

    captured = {}

    def capturing_find(*args, **kwargs):
        captured["is_series"] = kwargs.get("is_series", "NOT_SET")
        return None, "", "", []

    cdata = {
        "film_title": "Test",
        "cast": [{"actor_name": "Oyuncu", "confidence": 0.9}],
        "directors": [],
        "crew": [],
    }

    with patch.object(verifier, "_find_tmdb_entry", side_effect=capturing_find):
        verifier.verify_credits(cdata)  # is_series belirtilmiyor

    assert captured.get("is_series") is False, (
        "is_series varsayılan False olmalı"
    )
