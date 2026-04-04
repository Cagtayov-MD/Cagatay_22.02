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


def test_to_upper_tr_keeps_turkish_proper_names_without_foreign_guess():
    from core.export_engine import _to_upper_tr

    text = "İlker'in kararı değişir"
    protected = {"i̇lker", "ilker"}
    assert _to_upper_tr(text, protected_words=protected) == "İLKER'İN KARARI DEĞİŞİR"


def test_collect_summary_name_candidates_detects_foreign_proper_names():
    from core.export_engine import _collect_summary_name_candidates

    summary = "Karen Chris'in teklifini reddedip Washington'a gider."
    result = _collect_summary_name_candidates(summary)
    assert "karen" in result
    assert "chris" in result
    assert "washington" in result


def test_collect_summary_name_candidates_skips_common_ascii_sentence_words():
    from core.export_engine import _collect_summary_name_candidates

    summary = "Bir gün Chris Washington'a gider."
    result = _collect_summary_name_candidates(summary)
    assert "bir" not in result
    assert "chris" in result
    assert "washington" in result


def test_upper_word_proper_name_handles_turkish_and_foreign_without_language_db():
    from core.export_engine import _upper_word_proper_name

    assert _upper_word_proper_name("Chris'in") == "CHRIS'İN"
    assert _upper_word_proper_name("İlker'in") == "İLKER'İN"
    assert _upper_word_proper_name("Washington'a") == "WASHINGTON'A"


def test_to_upper_tr_ozet_distinguishes_turkish_and_foreign_names():
    from core.export_engine import _to_upper_tr_ozet

    text = "İrfan Ivan ve Jérôme ile konuşur."
    result = _to_upper_tr_ozet(
        text,
        foreign_nouns={"IVAN", "JEROME"},
        proper_names={"irfan", "ivan", "jérôme", "jerome"},
    )

    assert "İRFAN" in result
    assert "IVAN" in result
    assert "JEROME" in result


def test_user_report_summary_fallback_preserves_unwrapped_foreign_names(tmp_path):
    from core.export_engine import ExportEngine

    engine = ExportEngine(output_dir=str(tmp_path), name_db=None)
    report = {
        "generated_at": "2026-03-30T12:00:00",
        "file_info": {
            "filename": "2023-0001-1-0000-00-1-TEST_FILM.mp4",
            "resolution": "1920x1080",
            "fps": 25,
            "duration_human": "01:00:00",
        },
        "credits": {
            "film_title": "Test Film",
            "original_title": "Test Film",
            "verification_status": "ocr_parsed",
            "cast": [],
            "crew": [],
            "technical_crew": [],
            "directors": [],
        },
    }
    out_path = tmp_path / "report.txt"

    engine._write_user_report(
        report,
        out_path,
        audio_result={
            "summary": "Karen Chris'in teklifini reddedip Washington'a gider.",
        },
        film_name_tr="TEST FILM",
        film_id="2023-0001-1-0000-00-1",
        ocr_lines=[],
    )

    text = out_path.read_text(encoding="utf-8-sig")
    assert "CHRIS'İN" in text
    assert "WASHINGTON'A" in text
    assert "CHRİS'İN" not in text
    assert "WASHİNGTON'A" not in text


def test_extract_final_summary_text_rejects_legacy_english_dict():
    from core.export_engine import _extract_final_summary_text

    summary = _extract_final_summary_text({
        "summary": {"en": "A teacher keeps a soldier hostage."},
    })

    assert summary == ""


def test_extract_final_summary_text_accepts_new_payload():
    from core.export_engine import _extract_final_summary_text

    summary = _extract_final_summary_text({
        "summary": {
            "text": "İrfan eve döner ve Ivan ile vedalaşır.",
            "language": "tr",
        },
    })

    assert summary == "İrfan eve döner ve Ivan ile vedalaşır."


def test_extract_final_summary_text_preserves_two_paragraphs():
    from core.export_engine import _extract_final_summary_text

    summary = _extract_final_summary_text({
        "summary": {
            "text": "Bir kadın eve döner.\n\nIvan ile vedalaşıp Washington'a gider.",
            "language": "tr",
        },
    })

    assert summary == "Bir kadın eve döner.\n\nIvan ile vedalaşıp Washington'a gider."


def test_user_report_rejects_english_summary_payload(tmp_path):
    from core.export_engine import ExportEngine

    engine = ExportEngine(output_dir=str(tmp_path), name_db=None)
    report = {
        "generated_at": "2026-03-30T12:00:00",
        "file_info": {
            "filename": "2023-0001-1-0000-00-1-TEST_FILM.mp4",
            "resolution": "1920x1080",
            "fps": 25,
            "duration_human": "01:00:00",
        },
        "credits": {
            "film_title": "Test Film",
            "original_title": "Test Film",
            "verification_status": "ocr_parsed",
            "cast": [],
            "crew": [],
            "technical_crew": [],
            "directors": [],
        },
    }
    out_path = tmp_path / "report.txt"

    engine._write_user_report(
        report,
        out_path,
        audio_result={
            "summary": {"en": "A teacher keeps a soldier hostage."},
        },
        film_name_tr="TEST FILM",
        film_id="2023-0001-1-0000-00-1",
        ocr_lines=[],
    )

    text = out_path.read_text(encoding="utf-8-sig")
    assert "ÖZET OLUŞTURULAMADI." in text
    assert "A TEACHER KEEPS A SOLDIER HOSTAGE." not in text


def test_user_report_is_mirrored_to_secondary_directory(tmp_path):
    from core.export_engine import ExportEngine

    work_dir = tmp_path / "work"
    mirror_dir = tmp_path / "export"
    engine = ExportEngine(
        output_dir=str(work_dir),
        name_db=None,
        user_report_mirror_dir=str(mirror_dir),
    )
    report = {
        "generated_at": "2026-03-30T12:00:00",
        "file_info": {
            "filename": "2023-0001-1-0000-00-1-TEST_FILM.mp4",
            "resolution": "1920x1080",
            "fps": 25,
            "duration_human": "01:00:00",
        },
        "credits": {
            "film_title": "Test Film",
            "original_title": "Test Film",
            "verification_status": "ocr_parsed",
            "cast": [],
            "crew": [],
            "technical_crew": [],
            "directors": [],
        },
    }
    out_path = work_dir / "2023-0001-1-0000-00-1 TEST FILM.txt"

    engine._write_user_report(
        report,
        out_path,
        audio_result={
            "summary": "Karen Chris'in teklifini reddedip Washington'a gider.",
        },
        film_name_tr="TEST FILM",
        film_id="2023-0001-1-0000-00-1",
        ocr_lines=[],
    )

    mirror_path = mirror_dir / out_path.name
    assert mirror_path.is_file()
    assert mirror_path.read_text(encoding="utf-8-sig") == out_path.read_text(encoding="utf-8-sig")


def test_user_report_preserves_summary_paragraph_breaks(tmp_path):
    from core.export_engine import ExportEngine

    engine = ExportEngine(output_dir=str(tmp_path), name_db=None)
    report = {
        "generated_at": "2026-03-30T12:00:00",
        "file_info": {
            "filename": "2023-0001-1-0000-00-1-TEST_FILM.mp4",
            "resolution": "1920x1080",
            "fps": 25,
            "duration_human": "01:00:00",
        },
        "credits": {
            "film_title": "Test Film",
            "original_title": "Test Film",
            "verification_status": "ocr_parsed",
            "cast": [],
            "crew": [],
            "technical_crew": [],
            "directors": [],
        },
    }
    out_path = tmp_path / "report.txt"

    engine._write_user_report(
        report,
        out_path,
        audio_result={
            "summary": "Bir kadın eve döner.\n\nChris ile vedalaşıp Washington'a gider.",
        },
        film_name_tr="TEST FILM",
        film_id="2023-0001-1-0000-00-1",
        ocr_lines=[],
    )

    text = out_path.read_text(encoding="utf-8-sig")
    assert "BİR KADIN EVE DÖNER.\n\n  CHRIS İLE VEDALAŞIP WASHINGTON'A GİDER." in text


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


def test_format_language_block_emits_all_languages():
    from core.export_engine import _format_language_block

    lines = _format_language_block({
        "transcript_language": "en",
        "summary_language": "tr",
        "report_language": "tr",
    })

    assert any("Transcript" in line and "EN (İNG)" in line for line in lines)
    assert any("Özet" in line and "TR (TR)" in line for line in lines)
    assert any("Rapor" in line and "TR (TR)" in line for line in lines)


def test_format_language_block_ignores_missing_values():
    from core.export_engine import _format_language_block

    lines = _format_language_block({
        "transcript_language": "ar",
        # summary_language intentionally missing
        "report_language": "tr",
    })

    assert any("AR (ARA)" in line for line in lines)
    assert not any("Özet" in line for line in lines)
