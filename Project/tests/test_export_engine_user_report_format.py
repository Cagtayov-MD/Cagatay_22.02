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


# ─────────────────────────────────────────────────────────────────────────────
# USER-REPORT-FORMAT-01..06 — Sabit format (doğrulama logu + özet yer tutucusu)
# ─────────────────────────────────────────────────────────────────────────────

import tempfile
import os


def _make_minimal_report(content_type="FilmDizi-Paddle", vlog_text="", verification_status=""):
    """Minimal geçerli rapor dict'i döndürür."""
    from datetime import datetime

    cr = {
        "cast": [],
        "technical_crew": [],
        "directors": [],
        "film_title": "Test Film",
        "year": "2024",
    }
    if vlog_text:
        cr["_verification_log_text"] = vlog_text
    if verification_status:
        cr["verification_status"] = verification_status
    return {
        "generated_at": datetime.now().isoformat(),
        "profile": "test",
        "file_info": {
            "filename": "TEST_123_Test.mp4",
            "resolution": "1920x1080",
            "fps": 25,
            "duration_human": "01:30:00",
        },
        "processing": {"content_type": content_type},
        "credits": cr,
    }


def test_user_report_always_has_dogrulama_logu_section_for_filmdizi_paddle():
    """USER-REPORT-FORMAT-01: FilmDizi-Paddle profilinde de DOĞRULAMA LOGU bölümü çıkmalı."""
    from core.export_engine import ExportEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ExportEngine(tmpdir)
        r = _make_minimal_report(content_type="FilmDizi-Paddle", vlog_text="")
        path = os.path.join(tmpdir, "out.txt")
        engine._write_user_report(r, path)
        content = open(path, encoding="utf-8-sig").read()
        assert "DOĞRULAMA LOGU" in content, (
            "FilmDizi-Paddle profilinde DOĞRULAMA LOGU bölümü eksik"
        )


def test_user_report_dogrulama_logu_placeholder_when_no_log():
    """USER-REPORT-FORMAT-02: Doğrulama logu yoksa yer tutucu satır yazılmalı."""
    from core.export_engine import ExportEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ExportEngine(tmpdir)
        r = _make_minimal_report(vlog_text="")
        path = os.path.join(tmpdir, "out.txt")
        engine._write_user_report(r, path)
        content = open(path, encoding="utf-8-sig").read()
        assert "DOĞRULAMA LOGU MEVCUT DEĞİL" in content, (
            "Doğrulama logu yokken yer tutucu satır bekleniyor"
        )


def test_user_report_dogrulama_logu_shows_vlog_text_when_present():
    """USER-REPORT-FORMAT-03: _verification_log_text varsa içeriği raporta gösterilmeli."""
    from core.export_engine import ExportEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ExportEngine(tmpdir)
        r = _make_minimal_report(vlog_text="YONETMEN ONAYLANDI")
        path = os.path.join(tmpdir, "out.txt")
        engine._write_user_report(r, path)
        content = open(path, encoding="utf-8-sig").read()
        assert "YONETMEN ONAYLANDI" in content, (
            "Doğrulama log metni kullanıcı raporunda görünmüyor"
        )


def test_user_report_ozet_section_always_present():
    """USER-REPORT-FORMAT-04: ÖZET bölümü her zaman raporta yer almalı."""
    from core.export_engine import ExportEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ExportEngine(tmpdir)
        r = _make_minimal_report()
        path = os.path.join(tmpdir, "out.txt")
        # audio_result=None → özet yok
        engine._write_user_report(r, path, audio_result=None)
        content = open(path, encoding="utf-8-sig").read()
        assert "ÖZET" in content, "ÖZET bölümü audio_result=None durumunda eksik"


def test_user_report_ozet_placeholder_when_no_audio():
    """USER-REPORT-FORMAT-05: Ses analizi çalışmadıysa açıklayıcı yer tutucu olmalı."""
    from core.export_engine import ExportEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ExportEngine(tmpdir)
        r = _make_minimal_report()
        path = os.path.join(tmpdir, "out.txt")
        engine._write_user_report(r, path, audio_result=None)
        content = open(path, encoding="utf-8-sig").read()
        assert "SES ANALİZİ ÇALIŞMADI" in content, (
            "audio_result=None durumunda açıklayıcı özet yer tutucusu eksik"
        )


def test_user_report_yapim_ekibi_always_present():
    """USER-REPORT-FORMAT-06: YAPIM EKİBİ bölümü her zaman tüm 7 rolü göstermeli."""
    from core.export_engine import ExportEngine, _OUTPUT_ROLES

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = ExportEngine(tmpdir)
        r = _make_minimal_report()
        path = os.path.join(tmpdir, "out.txt")
        engine._write_user_report(r, path)
        content = open(path, encoding="utf-8-sig").read()
        assert "YAPIM EKİBİ" in content, "YAPIM EKİBİ bölümü eksik"
        for role in _OUTPUT_ROLES:
            assert role in content, f"Rol eksik: {role}"
