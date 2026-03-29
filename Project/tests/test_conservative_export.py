"""
test_conservative_export.py — Değişiklik 6: ocr_parsed muhafazakar export modu.
Ham crew yerine güvenilir kaynak seçilir.
"""
import os
import sys
from pathlib import Path

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def _make_engine(tmp_path):
    from core.export_engine import ExportEngine
    return ExportEngine(output_dir=str(tmp_path), name_db=None)


def _base_file_info():
    return {
        "filename": "evoArcadmin_TEST2_2023-0001-1-0000-00-1-TEST_FILM.mp4",
        "resolution": "1920x1080",
        "fps": 25,
        "duration_human": "01:00:00",
    }


def test_ocr_only_verified_crew_roles_preserved(tmp_path):
    """ocr_parsed + _verified_crew_roles'da YAPIMCI var → korunur."""
    engine = _make_engine(tmp_path)
    report = {
        "generated_at": "2026-03-28T12:00:00",
        "file_info": _base_file_info(),
        "credits": {
            "film_title": "Test Film",
            "original_title": "Test Film",
            "verification_status": "ocr_parsed",
            "cast": [],
            "directors": [{"name": "Dir A"}],
            "crew": [{"name": "Raw OCR Name", "role": "YAPIMCI"}],
            "_verified_crew_roles": {"YAPIMCI": ["Verified Producer"]},
        },
    }
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(report, out_path,
                              film_name_tr="TEST FILM",
                              film_id="2023-0001-1-0000-00-1",
                              ocr_lines=[])
    text = out_path.read_text(encoding="utf-8-sig")
    # _verified_crew_roles öncelikli — korunur
    assert "VERIFIED PRODUCER" in text


def test_ocr_only_uses_technical_crew_not_raw_crew(tmp_path):
    """ocr_parsed + _verified_crew_roles yok + technical_crew var → technical_crew kullanılır."""
    engine = _make_engine(tmp_path)
    report = {
        "generated_at": "2026-03-28T12:00:00",
        "file_info": _base_file_info(),
        "credits": {
            "film_title": "Test Film",
            "original_title": "Test Film",
            "verification_status": "ocr_parsed",
            "cast": [],
            "directors": [{"name": "Dir A"}],
            "crew": [{"name": "Raw Crew Producer", "role": "YAPIMCI"}],
            "technical_crew": [{"name": "Technical Producer", "role": "YAPIMCI"}],
        },
    }
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(report, out_path,
                              film_name_tr="TEST FILM",
                              film_id="2023-0001-1-0000-00-1",
                              ocr_lines=[])
    text = out_path.read_text(encoding="utf-8-sig")
    # technical_crew öncelikli
    assert "TECHNICAL PRODUCER" in text
    assert "RAW CREW PRODUCER" not in text


def test_tmdb_verified_conservative_mode_not_applied(tmp_path):
    """tmdb_verified → conservative mode uygulanmaz (external_locked aktif)."""
    engine = _make_engine(tmp_path)
    report = {
        "generated_at": "2026-03-28T12:00:00",
        "file_info": _base_file_info(),
        "credits": {
            "film_title": "Test Film",
            "original_title": "Test Film Original",
            "verification_status": "tmdb_verified",
            "cast": [],
            "directors": [{"name": "Dir A"}],
            "crew": [
                {"name": "Dir A", "job": "Director", "raw": "tmdb"},
                {"name": "TMDB Producer", "job": "Producer", "raw": "tmdb"},
            ],
        },
    }
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(report, out_path,
                              film_name_tr="TEST FILM",
                              film_id="2023-0001-1-0000-00-1",
                              ocr_lines=[])
    text = out_path.read_text(encoding="utf-8-sig")
    assert "TMDB PRODUCER" in text


def test_gemini_verified_conservative_mode_not_applied(tmp_path):
    """_gemini_crew_roles varsa → ocr_only_mode devreye girmez."""
    engine = _make_engine(tmp_path)
    report = {
        "generated_at": "2026-03-28T12:00:00",
        "file_info": _base_file_info(),
        "credits": {
            "film_title": "Test Film",
            "original_title": "Test Film",
            "verification_status": None,
            "cast": [],
            "directors": [{"name": "Dir A"}],
            "crew": [],
            "_gemini_crew_roles": {"YAPIMCI": ["Gemini Producer"], "YÖNETMEN": ["Dir A"]},
        },
    }
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(report, out_path,
                              film_name_tr="TEST FILM",
                              film_id="2023-0001-1-0000-00-1",
                              ocr_lines=[])
    text = out_path.read_text(encoding="utf-8-sig")
    assert "GEMINI PRODUCER" in text
