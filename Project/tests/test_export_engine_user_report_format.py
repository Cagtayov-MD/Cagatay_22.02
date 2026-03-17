import os
import sys

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def test_to_upper_tr_converts_ascii_turkish_words():
    from core.export_engine import _to_upper_tr

    text = "kariyerinin kabullenmeyip intihar etmesi"
    assert _to_upper_tr(text) == "KARİYERİNİN KABULLENMEYİP İNTİHAR ETMESİ"


def test_to_upper_tr_keeps_foreign_names_ascii_with_protection():
    from core.export_engine import _to_upper_tr

    text = "chris washington'a gider"
    protected = {"chris", "washington"}
    assert _to_upper_tr(text, protected_words=protected) == "CHRIS WASHINGTON'A GİDER"


def test_collect_summary_name_candidates_detects_foreign_proper_names():
    from core.export_engine import _collect_summary_name_candidates

    summary = "Karen Chris'in teklifini reddedip Washington'a gider."
    result = _collect_summary_name_candidates(summary)
    assert "karen" in result
    assert "chris" in result
    assert "washington" in result


def test_turkish_original_title_words_not_added_to_protected_words():
    """Türkçe karakter içeren kelimeler protected_words'e eklenmemeli (Sorun 1 fix)."""
    from core.export_engine import _is_turkish_word, _is_known_name

    turkish_title = "KADIN VE DENİZCİ"
    protected_words: set = set()
    for token in turkish_title.split():
        t = token.strip("''`\".,;:!?()[]{}")
        if t and not _is_turkish_word(t) and not _is_known_name(t):
            protected_words.add(t.casefold())

    # Türkçe karakterli kelimeler (DENİZCİ → İ içeriyor) protected olmamalı
    assert "denizci" not in protected_words


def test_turkish_title_i_uppercased_correctly_with_dot():
    """Türkçe başlıktaki 'i' harfi 'İ' (noktalı) olarak büyütülmeli."""
    from core.export_engine import _to_upper_tr

    # Türkçe kelimeleri içeren metin — protected_words boş (title Turkish)
    text = "denizci filmin başrolünde"
    result = _to_upper_tr(text, protected_words=set())
    assert result == "DENİZCİ FİLMİN BAŞROLÜNDE"


def test_upper_word_turkish_no_dotless_i():
    """_upper_word_turkish: 'i' harfi 'İ' (noktalı) olarak büyütülmeli, 'I' (noktasız) değil."""
    from core.export_engine import _upper_word_turkish

    assert _upper_word_turkish("filmi") == "FİLMİ"
    assert _upper_word_turkish("intihar") == "İNTİHAR"
    assert _upper_word_turkish("denizci") == "DENİZCİ"


def test_upper_word_turkish_dotted_i_preserved():
    """_upper_word_turkish: büyük 'İ' zaten büyük olduğu için değişmemeli."""
    from core.export_engine import _upper_word_turkish

    assert _upper_word_turkish("İstanbul") == "İSTANBUL"
