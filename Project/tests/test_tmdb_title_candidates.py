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