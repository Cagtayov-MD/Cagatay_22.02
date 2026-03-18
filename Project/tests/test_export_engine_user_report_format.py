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


def _make_minimal_report(summary_value=None):
    """_write_report için gerekli minimum rapor dict'i döndür."""
    import tempfile
    r = {
        "generated_at": "2026-01-01T00:00:00",
        "profile": "test",
        "file_info": {
            "filename": "test.mp4",
            "filepath": "/tmp/test.mp4",
            "filesize_bytes": 1024 * 1024,
            "duration_seconds": 60,
            "duration_human": "00:01:00",
            "resolution": "1920x1080",
            "fps": 25,
        },
        "processing": {
            "scope": "full",
            "content_type": "FilmDizi",
            "ocr_engine": "PaddleOCR",
            "first_segment_min": 0,
            "last_segment_min": 5,
            "stages": [],
            "total_duration_sec": 10.0,
            "speed_ratio": 6.0,
        },
        "credits": {"cast": [], "directors": []},
        "film_title": "Test Filmi",
        "keywords": [],
        "logos_detected": [],
        "ocr_results": [],
        "errors": [],
    }
    audio_result = None
    if summary_value is not None:
        audio_result = {
            "status": "ok",
            "summary": summary_value,
            "detected_language": "tr",
        }
    return r, audio_result


def test_write_report_summary_dict_shows_turkish_text(tmp_path):
    """RAPOR-01: summary dict ise Türkçe metin gösterilmeli, Python dict repr değil."""
    from core.export_engine import ExportEngine

    turkish_text = "Ahmet köyünden ayrılıp İstanbul'a gider ve yeni bir hayata başlar."
    summary_dict = {"en": turkish_text, "model_used": "gemini-2.5-pro"}

    r, audio_result = _make_minimal_report(summary_value=summary_dict)
    out_path = str(tmp_path / "teknik.txt")

    engine = ExportEngine(str(tmp_path))
    engine._write_report(r, out_path, audio_result=audio_result)

    content = open(out_path, encoding="utf-8-sig").read()
    assert turkish_text in content, "Türkçe özet metni raporda görünmeli"
    assert "model_used" not in content, "Python dict repr raporda görünmemeli"
    assert "'en'" not in content, "Python dict key 'en' raporda görünmemeli"


def test_write_report_summary_plain_string_works(tmp_path):
    """RAPOR-02: summary düz string ise doğrudan gösterilmeli."""
    from core.export_engine import ExportEngine

    turkish_text = "Mehmet emekli olup memlekete döner."
    r, audio_result = _make_minimal_report(summary_value=turkish_text)
    out_path = str(tmp_path / "teknik2.txt")

    engine = ExportEngine(str(tmp_path))
    engine._write_report(r, out_path, audio_result=audio_result)

    content = open(out_path, encoding="utf-8-sig").read()
    assert turkish_text in content


def test_write_report_no_summary_shows_default_message(tmp_path):
    """RAPOR-03: summary yoksa 'Özet oluşturma aktif değil' mesajı gösterilmeli."""
    from core.export_engine import ExportEngine

    r, audio_result = _make_minimal_report(summary_value=None)
    out_path = str(tmp_path / "teknik3.txt")

    engine = ExportEngine(str(tmp_path))
    engine._write_report(r, out_path, audio_result=audio_result)

    content = open(out_path, encoding="utf-8-sig").read()
    assert "Özet oluşturma aktif değil" in content


def test_write_report_summary_dict_tr_key_preferred(tmp_path):
    """RAPOR-04: summary dict'te 'tr' anahtarı varsa önce kullanılmalı."""
    from core.export_engine import ExportEngine

    tr_text = "Bu Türkçe özet metnidir."
    summary_dict = {"tr": tr_text, "en": "This is an English fallback.", "model_used": "gemini-2.5-flash"}

    r, audio_result = _make_minimal_report(summary_value=summary_dict)
    out_path = str(tmp_path / "teknik4.txt")

    engine = ExportEngine(str(tmp_path))
    engine._write_report(r, out_path, audio_result=audio_result)

    content = open(out_path, encoding="utf-8-sig").read()
    assert tr_text in content
