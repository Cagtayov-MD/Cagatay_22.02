"""
test_external_lock_backfill.py — Değişiklik 2+3: external_locked crew kaynağı ve backfill davranışı.

Mevcut test_external_lock_cleanup.py güvenlik bariyerini korur (OCR producer sızmaz).
Bu dosya yeni backfill davranışını doğrular.
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


def _base_report():
    return {
        "generated_at": "2026-03-28T12:00:00",
        "file_info": {
            "filename": "evoArcadmin_TEST2_2023-0001-1-0000-00-1-TEST_FILM.mp4",
            "resolution": "1920x1080",
            "fps": 25,
            "duration_human": "01:00:00",
        },
    }


def test_external_lock_backfill_senaryo_from_technical_crew(tmp_path):
    """SENARYO boşsa ve technical_crew'da Writer varsa → backfill olur."""
    engine = _make_engine(tmp_path)
    report = _base_report()
    report["credits"] = {
        "film_title": "Test Film",
        "original_title": "Test Film Original",
        "verification_status": "imdb_verified",
        "cast": [{"actor_name": "Lead Actor", "raw": "imdb"}],
        "directors": [{"name": "Alice Director"}],
        "crew": [
            {"name": "Alice Director", "job": "Director", "raw": "imdb"},
        ],
        "technical_crew": [
            {"name": "Bob Writer", "role": "Screenplay", "raw": "imdb"},
        ],
    }
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(
        report, out_path,
        film_name_tr="TEST FILM",
        film_id="2023-0001-1-0000-00-1",
        ocr_lines=[],
    )
    text = out_path.read_text(encoding="utf-8-sig")
    assert "BOB WRITER" in text, "technical_crew'dan SENARYO backfill çalışmalı"
    assert "ALICE DIRECTOR" in text


def test_external_lock_verified_crew_roles_does_not_leak(tmp_path):
    """external_locked + _verified_crew_roles'da YAPIMCI var → external crew YAPIMCI ile doluysa sızmaz."""
    engine = _make_engine(tmp_path)
    report = _base_report()
    report["credits"] = {
        "film_title": "Test Film",
        "original_title": "Test Film Original",
        "verification_status": "imdb_verified",
        "cast": [{"actor_name": "Lead Actor", "raw": "imdb"}],
        "directors": [{"name": "Alice Director"}],
        "crew": [
            {"name": "Alice Director", "job": "Director", "raw": "imdb"},
            {"name": "Real Producer", "job": "Producer", "raw": "imdb"},
        ],
        "_verified_crew_roles": {"YAPIMCI": ["OCR Producer"]},
    }
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(
        report, out_path,
        film_name_tr="TEST FILM",
        film_id="2023-0001-1-0000-00-1",
        ocr_lines=[],
    )
    text = out_path.read_text(encoding="utf-8-sig")
    # YAPIMCI dolu olduğu için OCR Producer sızmamalı
    assert "OCR PRODUCER" not in text
    assert "REAL PRODUCER" in text


def test_external_lock_technical_crew_priority_over_crew(tmp_path):
    """external_locked: technical_crew TMDB-tagged girişi varsa crew'dan önce gelir."""
    engine = _make_engine(tmp_path)
    report = _base_report()
    report["credits"] = {
        "film_title": "Test Film",
        "original_title": "Test Film Original",
        "verification_status": "imdb_verified",
        "cast": [{"actor_name": "Lead Actor", "raw": "imdb"}],
        "directors": [{"name": "Alice Director"}],
        "crew": [
            {"name": "Alice Director", "job": "Director", "raw": "imdb"},
            {"name": "Crew Producer", "job": "Producer", "raw": "imdb"},
        ],
        "technical_crew": [
            {"name": "Alice Director", "job": "Director", "raw": "imdb"},
            {"name": "Tech Producer", "job": "Producer", "raw": "imdb"},
        ],
    }
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(
        report, out_path,
        film_name_tr="TEST FILM",
        film_id="2023-0001-1-0000-00-1",
        ocr_lines=[],
    )
    text = out_path.read_text(encoding="utf-8-sig")
    # technical_crew öncelikli olduğu için Tech Producer görünmeli
    assert "TECH PRODUCER" in text
    # crew'daki Crew Producer görünmemeli (technical_crew override etti)
    assert "CREW PRODUCER" not in text
