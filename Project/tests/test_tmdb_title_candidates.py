"""TMDB başlık varyantı üretimi testleri."""


def test_madam_to_madame():
    """Madam → Madame varyantı üretilmeli."""
    from core.tmdb_verify import _title_candidates
    result = _title_candidates("Madam Bovary")
    assert "Madame Bovary" in result, f"Madame Bovary varyantı bulunamadı: {result}"
    assert "Madam Bovary" in result
    assert result[0] == "Madam Bovary"


def test_no_duplicate():
    """Tekrarlı varyant olmamalı."""
    from core.tmdb_verify import _title_candidates
    result = _title_candidates("Madame Bovary")
    assert result.count("Madame Bovary") == 1


def test_normal_title_unchanged():
    """Değiştirilecek kelime yoksa yalnızca orijinal döner."""
    from core.tmdb_verify import _title_candidates
    result = _title_candidates("Casablanca")
    assert result == ["Casablanca"]


def test_company_filter_editions_musicales():
    """'editions musicales' cast_names'ten elenmeli."""
    from core.tmdb_verify import _looks_like_company
    assert _looks_like_company("editions musicales") is True
    assert _looks_like_company("MK2PRODUCTIONSSA") is True
    assert _looks_like_company("ISABELLE HUPPERT") is False
    assert _looks_like_company("JEAN-FRANCOIS BALMER") is False


def test_company_filter_mk2():
    """MK2 yapım şirketi cast'tan elenmeli."""
    from core.tmdb_verify import _looks_like_company
    assert _looks_like_company("MK2 PRODUCTIONS SA") is True


def test_plausible_actor_not_filtered():
    """Gerçek oyuncu isimleri elenmemeli."""
    from core.tmdb_verify import _looks_like_company
    assert _looks_like_company("CHRISTOPHE MALAVOY") is False
    assert _looks_like_company("LUCAS BELVAUX") is False


def test_strategy2_filters_company_names():
    """Strateji 2 persons_to_search listesi şirket isimi içermemeli."""
    from core.tmdb_verify import _looks_like_company
    company_names = ['editions musicales', 'MK2PRODUCTIONS SA', 'editionsmusicales']
    for name in company_names:
        assert _looks_like_company(name) is True, f"Şirket ismi olarak tanınmadı: {name}"


def test_looks_like_company_detects_company_names():
    """editions musicales, MK2 Productions gibi şirket isimleri reddedilmeli."""
    from core.tmdb_verify import _looks_like_company
    assert _looks_like_company("editions musicales") is True
    assert _looks_like_company("editionsmusicales editions musicales") is True
    assert _looks_like_company("MK2PRODUCTIONS SA MK2PRODUCTIONSSA") is True
    assert _looks_like_company("editions-musicales") is True


def test_title_candidates_all_caps():
    """Tüm büyük harfli başlık → title-case varyantı üretilmeli."""
    from core.tmdb_verify import _title_candidates
    result = _title_candidates("MADAM BOVARY")
    assert "Madam Bovary" in result, f"title-case varyantı bulunamadı: {result}"
    assert "MADAM BOVARY" in result
    assert result[0] == "MADAM BOVARY"


def test_title_candidates_all_caps_madame():
    """Tüm büyük harfli Madam → title-case üzerinden Madame varyantı da üretilmeli."""
    from core.tmdb_verify import _title_candidates
    result = _title_candidates("MADAM BOVARY")
    assert "Madame Bovary" in result, f"Madame Bovary varyantı bulunamadı: {result}"


def test_gemini_extractor_line_limit():
    """GeminiCastExtractor.extract() max_lines parametresi 200 varsayılanıyla kabul edilmeli."""
    import inspect
    from core.gemini_cast_extractor import GeminiCastExtractor

    extractor = GeminiCastExtractor(api_key="")
    sig = inspect.signature(extractor.extract)
    assert "max_lines" in sig.parameters, "extract() fonksiyonu max_lines parametresi içermiyor"
    assert sig.parameters["max_lines"].default == 200, "max_lines varsayılan değeri 200 olmalı"


# ── Strategy 2: combined_credits crew bölümü testleri ─────────────────────────


def test_strategy2_director_contributes_via_crew():
    """
    STRAT2-CREW-01: Yönetmen combined_credits'teki crew bölümüyle work_matches'e katkı yapmalı.

    Film adı yanlış (Strategy 1 başarısız), cast yeterli değil, ama 1 oyuncu + 1 yönetmen +
    1 oyuncu daha (3 kişi) aynı yapıta işaret edince eşleşme bulunabilmeli.

    Senaryo: Madame Bovary (1991, id=999)
      - Isabelle Huppert → cast['Madame Bovary'] → count 1
      - Jean-Francois Balmer → cast['Madame Bovary'] → count 2
      - Claude Chabrol (yönetmen) → crew['Madame Bovary'] → count 3  ← ÖNCEKİ KOD BUNU SAYMAZDI
    """
    from unittest.mock import patch
    from core.tmdb_verify import TMDBVerify
    import tempfile

    tmp = tempfile.mkdtemp()
    verifier = TMDBVerify(work_dir=tmp, api_key="test-key")

    MOVIE_ID = 999
    MOVIE_ENTRY = {"id": MOVIE_ID, "media_type": "movie", "title": "Madame Bovary",
                   "release_date": "1991-01-01"}

    # Kişi arama: her isim için 1 TMDB person kaydı döner
    def fake_search_person(name):
        return [{"id": hash(name) & 0xFFFF, "name": name, "known_for": []}]

    # combined_credits: oyuncular cast'ta, yönetmen crew'da görünür
    def fake_combined_credits(person_id):
        huppert_id = hash("Isabelle Huppert") & 0xFFFF
        balmer_id = hash("Jean-Francois Balmer") & 0xFFFF
        chabrol_id = hash("Claude Chabrol") & 0xFFFF

        if person_id == huppert_id:
            return {"cast": [MOVIE_ENTRY], "crew": []}
        if person_id == balmer_id:
            return {"cast": [MOVIE_ENTRY], "crew": []}
        if person_id == chabrol_id:
            # Yönetmen: sadece crew'da görünür, cast'ta YOK
            return {"cast": [], "crew": [dict(MOVIE_ENTRY, job="Director")]}
        return {"cast": [], "crew": []}

    # _fetch_credits: Madame Bovary için Isabelle + Jean-Francois cast'ta, Chabrol crew'da
    def fake_fetch_credits(kind, mid):
        if mid == MOVIE_ID:
            return {
                "cast": [
                    {"name": "Isabelle Huppert"},
                    {"name": "Jean-Francois Balmer"},
                ],
                "crew": [
                    {"name": "Claude Chabrol", "job": "Director"},
                ],
            }
        return None

    with patch.object(verifier.client, "search_person", side_effect=fake_search_person), \
         patch.object(verifier.client, "get_person_combined_credits",
                      side_effect=fake_combined_credits), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind = verifier._find_tmdb_entry(
            film_title="",  # Film adı yok — Strategy 2 tetiklenmeli
            cast_names=["Isabelle Huppert", "Jean-Francois Balmer"],
            director_names=["Claude Chabrol"],
        )

    assert entry is not None, (
        "Strategy 2 yönetmen crew eşleşmesiyle sonuç bulmalıydı, None döndü. "
        "Muhtemelen combined_credits crew bölümü okunmuyor."
    )
    assert entry["id"] == MOVIE_ID
    assert kind == "movie"


def test_strategy2_director_no_double_count_for_actor_director():
    """
    STRAT2-CREW-02: Hem oyuncu hem yönetmen olan kişi aynı yapıt için iki kez sayılmamalı.

    Senaryo: Bir yapıt için combined_credits cast + crew bölümlerinde aynı wid varsa,
    kişi başına sadece 1 kez sayılmalı.
    """
    from unittest.mock import patch
    from core.tmdb_verify import TMDBVerify
    import tempfile

    tmp = tempfile.mkdtemp()
    verifier = TMDBVerify(work_dir=tmp, api_key="test-key")

    MOVIE_ID = 42
    MOVIE_ENTRY = {"id": MOVIE_ID, "media_type": "movie", "title": "Test Film"}

    def fake_search_person(name):
        return [{"id": 1, "name": name, "known_for": []}]

    # Bu kişi aynı filmde hem cast (oyuncu) hem crew (yönetmen)
    def fake_combined_credits(person_id):
        return {
            "cast": [MOVIE_ENTRY],
            "crew": [dict(MOVIE_ENTRY, job="Director")],  # aynı wid=42
        }

    def fake_fetch_credits(kind, mid):
        return {"cast": [{"name": "Actor One"}], "crew": [{"name": "Actor One", "job": "Director"}]}

    with patch.object(verifier.client, "search_person", side_effect=fake_search_person), \
         patch.object(verifier.client, "get_person_combined_credits",
                      side_effect=fake_combined_credits), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        # Sadece 1 kişi arıyoruz; count en fazla 1 olmalı → MIN_ACTOR_MATCH=3 geçilemez
        entry, kind = verifier._find_tmdb_entry(
            film_title="",
            cast_names=["Actor One"],
            director_names=[],
        )

    # 1 kişiyle MIN_ACTOR_MATCH=3 karşılanamaz → None beklenir
    assert entry is None, (
        "Tek kişiden çift sayım yapılmış olabilir; count 1 kalmalı ve eşleşme olmamalı."
    )


# ── Strateji 1 + 2: cast boş, sadece yönetmen mevcut ─────────────────────────


def test_strategy1_director_only_no_cast():
    """
    STRAT1-DIR-01: film_title + cast=[] + director → Strateji 1'de yönetmen eşleşmesiyle kabul edilmeli.

    Senaryo: film_title="Teşkilat", cast=[], directors=["Ozan Bodur"]
      - search_multi "Teşkilat" → id:119806 döner
      - credits'te "Ozan Bodur" crew'da bulunuyor
      → kabul edilmeli
    """
    from unittest.mock import patch
    from core.tmdb_verify import TMDBVerify
    import tempfile

    tmp = tempfile.mkdtemp()
    verifier = TMDBVerify(work_dir=tmp, api_key="test-key")

    TV_ID = 119806
    TV_ENTRY = {"id": TV_ID, "media_type": "tv", "name": "Teşkilat"}

    def fake_search_multi(title):
        return [TV_ENTRY]

    def fake_fetch_credits(kind, mid):
        if mid == TV_ID:
            return {
                "cast": [],
                "crew": [{"name": "Ozan Bodur", "job": "Director"}],
            }
        return None

    with patch.object(verifier.client, "search_multi", side_effect=fake_search_multi), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind = verifier._find_tmdb_entry(
            film_title="Teşkilat",
            cast_names=[],
            director_names=["Ozan Bodur"],
        )

    assert entry is not None, (
        "Strateji 1: film adı + yönetmen eşleşmesi (cast yok) kabul edilmeliydi, None döndü."
    )
    assert entry["id"] == TV_ID
    assert kind == "tv"


def test_strategy2_director_only_no_cast():
    """
    STRAT2-DIR-01: film_title yok, cast=[], sadece yönetmen → Strateji 2'de min_match_threshold=1 ile eşleşmeli.

    Senaryo: film_title="", cast=[], directors=["Ozan Bodur"]
      - search_person "Ozan Bodur" → id:5001 döner
      - combined_credits crew'da id:119806 bulunuyor
      - _fetch_credits credits'te "Ozan Bodur" crew'da var
      → kabul edilmeli
    """
    from unittest.mock import patch
    from core.tmdb_verify import TMDBVerify
    import tempfile

    tmp = tempfile.mkdtemp()
    verifier = TMDBVerify(work_dir=tmp, api_key="test-key")

    TV_ID = 119806
    TV_ENTRY = {"id": TV_ID, "media_type": "tv", "name": "Teşkilat"}

    def fake_search_person(name):
        return [{"id": 5001, "name": name, "known_for": []}]

    def fake_combined_credits(person_id):
        if person_id == 5001:
            return {"cast": [], "crew": [dict(TV_ENTRY, job="Director")]}
        return {"cast": [], "crew": []}

    def fake_fetch_credits(kind, mid):
        if mid == TV_ID:
            return {
                "cast": [],
                "crew": [{"name": "Ozan Bodur", "job": "Director"}],
            }
        return None

    with patch.object(verifier.client, "search_person", side_effect=fake_search_person), \
         patch.object(verifier.client, "get_person_combined_credits",
                      side_effect=fake_combined_credits), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind = verifier._find_tmdb_entry(
            film_title="",
            cast_names=[],
            director_names=["Ozan Bodur"],
        )

    assert entry is not None, (
        "Strateji 2: cast boş + tek yönetmen eşleşmesi (min_match_threshold=1) kabul edilmeliydi, None döndü."
    )
    assert entry["id"] == TV_ID
    assert kind == "tv"


def test_strategy2_cast_present_still_uses_min_actor_match():
    """
    STRAT2-THRESHOLD-01: cast doluyken MIN_ACTOR_MATCH=3 eşiği korunmalı.

    cast=[A, B] + director=[C] — sadece 2 eşleşme → min_match_threshold=3 aşılamaz → None beklenir.
    """
    from unittest.mock import patch
    from core.tmdb_verify import TMDBVerify
    import tempfile

    tmp = tempfile.mkdtemp()
    verifier = TMDBVerify(work_dir=tmp, api_key="test-key")

    MOVIE_ID = 77
    MOVIE_ENTRY = {"id": MOVIE_ID, "media_type": "movie", "title": "Some Film"}

    def fake_search_person(name):
        return [{"id": hash(name) & 0xFFFF, "name": name, "known_for": []}]

    def fake_combined_credits(person_id):
        actor_a_id = hash("Actor A") & 0xFFFF
        actor_b_id = hash("Actor B") & 0xFFFF
        if person_id in (actor_a_id, actor_b_id):
            return {"cast": [MOVIE_ENTRY], "crew": []}
        return {"cast": [], "crew": []}

    def fake_fetch_credits(kind, mid):
        if mid == MOVIE_ID:
            return {
                "cast": [{"name": "Actor A"}, {"name": "Actor B"}],
                "crew": [],
            }
        return None

    with patch.object(verifier.client, "search_person", side_effect=fake_search_person), \
         patch.object(verifier.client, "get_person_combined_credits",
                      side_effect=fake_combined_credits), \
         patch.object(verifier, "_fetch_credits", side_effect=fake_fetch_credits), \
         patch.object(verifier, "_load_cache", return_value=None), \
         patch.object(verifier, "_save_cache"):

        entry, kind = verifier._find_tmdb_entry(
            film_title="",
            cast_names=["Actor A", "Actor B"],
            director_names=[],
        )

    # cast doluyken 2 eşleşme MIN_ACTOR_MATCH=3'ü karşılamaz → None beklenir
    assert entry is None, (
        "cast doluyken MIN_ACTOR_MATCH=3 eşiği korunmalı; 2 eşleşmeyle None dönmeliydi."
    )