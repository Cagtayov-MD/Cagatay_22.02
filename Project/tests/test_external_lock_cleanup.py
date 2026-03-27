import os
import sys
from pathlib import Path

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def test_write_user_report_external_lock_skips_ocr_fallback_and_non_allowed_roles(tmp_path):
    from core.export_engine import ExportEngine

    engine = ExportEngine(output_dir=str(tmp_path), name_db=None)
    report = {
        "generated_at": "2026-03-24T12:00:00",
        "file_info": {
            "filename": "evoArcadmin_TEST2_2023-0001-1-0000-00-1-TEST_FILM.mp4",
            "resolution": "1920x1080",
            "fps": 25,
            "duration_human": "01:00:00",
        },
        "credits": {
            "film_title": "Test Film",
            "original_title": "Original Film",
            "verification_status": "imdb_verified",
            "cast": [{"actor_name": "Lead Actor", "raw": "imdb"}],
            "directors": [{"name": "Alice Director"}],
            "crew": [
                {"name": "Alice Director", "job": "Director", "raw": "imdb"},
                {"name": "Craig Campobasso", "job": "Casting Director", "raw": "imdb"},
                {"name": "Sylvie Chesneau", "job": "Script Supervisor", "raw": "imdb"},
                {"name": "Bin Li", "job": "Assistant Editor", "raw": "imdb"},
                {"name": "Gene Rudolf", "job": "Art Direction", "raw": "imdb"},
                {"name": "Monett Holderer", "job": "Foley Artist", "raw": "imdb"},
            ],
            "_verified_crew_roles": {"YAPIMCI": ["OCR Producer"]},
        },
    }
    out_path = Path(tmp_path) / "user_report.txt"

    engine._write_user_report(
        report,
        out_path,
        film_name_tr="TEST FILM",
        film_id="2023-0001-1-0000-00-1",
        ocr_lines=[],
    )

    text = out_path.read_text(encoding="utf-8-sig")

    assert "ALICE DIRECTOR" in text
    assert "OCR PRODUCER" not in text
    assert "CRAIG CAMPOBASSO" not in text
    assert "SYLVIE CHESNEAU" not in text
    assert "BIN LI" not in text
    assert "GENE RUDOLF" not in text
    assert "MONETT HOLDERER" not in text
    assert "SINIFLANDIRILMAMIŞ" not in text
    assert "YAPIMCI" in text and "VERİ YOK" in text
