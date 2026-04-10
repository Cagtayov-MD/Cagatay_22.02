"""
test_txt_role_limits.py — Değişiklik 4: TXT çıktısında rol başına gösterim limiti.
report.json tam veri taşır; bu limit yalnızca TXT görüntülemede uygulanır.
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


def _base_report(credits_override):
    report = {
        "generated_at": "2026-03-28T12:00:00",
        "file_info": {
            "filename": "evoArcadmin_TEST2_2023-0001-1-0000-00-1-TEST_FILM.mp4",
            "resolution": "1920x1080",
            "fps": 25,
            "duration_human": "01:00:00",
        },
        "credits": credits_override,
    }
    return report


def test_yapimci_limit_four(tmp_path):
    """YAPIMCI limit 4: 6 isim girilirse TXT'de ilk 4 görünür."""
    engine = _make_engine(tmp_path)
    producers = [f"Producer {i}" for i in range(1, 7)]
    report = _base_report({
        "film_title": "Test Film",
        "original_title": "Test Film Original",
        "verification_status": "imdb_verified",
        "cast": [],
        "directors": [{"name": "Dir A"}],
        "crew": [{"name": f"Producer {i}", "job": "Producer", "raw": "imdb"} for i in range(1, 7)]
              + [{"name": "Dir A", "job": "Director", "raw": "imdb"}],
    })
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(report, out_path,
                              film_name_tr="TEST FILM",
                              film_id="2023-0001-1-0000-00-1",
                              ocr_lines=[])
    text = out_path.read_text(encoding="utf-8-sig")
    shown = [f"PRODUCER {i}" for i in range(1, 5) if f"PRODUCER {i}" in text]
    hidden = [f"PRODUCER {i}" for i in range(5, 7) if f"PRODUCER {i}" in text]
    assert len(shown) == 4, f"4 yapımcı görünmeli, görünen: {shown}"
    assert len(hidden) == 0, f"5. ve 6. yapımcı gizlenmeli, ama: {hidden}"


def test_non_export_roles_are_not_listed(tmp_path):
    """KAMERA gibi desteklenmeyen roller TXT'de hiç görünmemeli."""
    engine = _make_engine(tmp_path)
    report = _base_report({
        "film_title": "Test Film",
        "original_title": "Test Film Original",
        "verification_status": "imdb_verified",
        "cast": [],
        "directors": [{"name": "Dir A"}],
        "crew": [{"name": f"Camera {i}", "job": "Camera Operator", "raw": "imdb"} for i in range(1, 5)]
              + [{"name": "Dir A", "job": "Director", "raw": "imdb"}],
    })
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(report, out_path,
                              film_name_tr="TEST FILM",
                              film_id="2023-0001-1-0000-00-1",
                              ocr_lines=[])
    text = out_path.read_text(encoding="utf-8-sig")
    assert "DIR A" in text
    assert "CAMERA 1" not in text
    assert "CAMERA 2" not in text
    assert "CAMERA 3" not in text
    assert "CAMERA 4" not in text


def test_yonetmen_no_limit_in_txt_role_limits(tmp_path):
    """YÖNETMEN _TXT_ROLE_LIMITS'te yok (MAX_DIRECTORS upstream tarafından yönetilir)."""
    from core.export_engine import _TXT_ROLE_LIMITS
    assert "YÖNETMEN" not in _TXT_ROLE_LIMITS


def test_yapimci_exact_limit_four_shows_all(tmp_path):
    """Tam limit kadar yapımcı varsa hepsi görünür."""
    engine = _make_engine(tmp_path)
    report = _base_report({
        "film_title": "Test Film",
        "original_title": "Test Film Original",
        "verification_status": "imdb_verified",
        "cast": [],
        "directors": [{"name": "Dir A"}],
        "crew": [{"name": f"Producer {i}", "job": "Producer", "raw": "imdb"} for i in range(1, 5)]
              + [{"name": "Dir A", "job": "Director", "raw": "imdb"}],
    })
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(report, out_path,
                              film_name_tr="TEST FILM",
                              film_id="2023-0001-1-0000-00-1",
                              ocr_lines=[])
    text = out_path.read_text(encoding="utf-8-sig")
    assert "PRODUCER 1" in text
    assert "PRODUCER 4" in text


def test_cast_limit_twenty(tmp_path):
    """Cast limiti 20: 25 oyuncu girilirse OYUNCULAR bölümünde ilk 20 görünür + '... ve 5 oyuncu daha'."""
    engine = _make_engine(tmp_path)
    cast_entries = [{"actor_name": f"Actor {i}"} for i in range(1, 26)]
    report = _base_report({
        "film_title": "Test Film",
        "original_title": "Test Film Original",
        "verification_status": "imdb_verified",
        "cast": cast_entries,
        "directors": [{"name": "Dir A"}],
        "crew": [{"name": "Dir A", "job": "Director", "raw": "imdb"}],
    })
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(report, out_path,
                              film_name_tr="TEST FILM",
                              film_id="2023-0001-1-0000-00-1",
                              ocr_lines=[])
    text = out_path.read_text(encoding="utf-8-sig")
    # OYUNCULAR başlığından sonra YAPIM EKİBİ'ne kadar olan bölümü izole et
    # (ANAHTAR SÖZCÜKLER tüm isimleri listeler, bu yüzden tam bölümü alıyoruz)
    parts = text.split("OYUNCULAR")
    assert len(parts) >= 2, "OYUNCULAR bölümü bulunamadı"
    cast_section = parts[-1].split("YAPIM EK")[0]
    assert "ACTOR 1" in cast_section
    assert "ACTOR 20" in cast_section
    assert "ACTOR 21" not in cast_section
    assert "5 OYUNCU DAHA" in cast_section


def test_cast_limit_not_triggered_below_limit(tmp_path):
    """15 oyuncu → hepsi görünür, '... ve' satırı yok."""
    engine = _make_engine(tmp_path)
    cast_entries = [{"actor_name": f"Actor {i}"} for i in range(1, 16)]
    report = _base_report({
        "film_title": "Test Film",
        "original_title": "Test Film Original",
        "verification_status": "imdb_verified",
        "cast": cast_entries,
        "directors": [{"name": "Dir A"}],
        "crew": [{"name": "Dir A", "job": "Director", "raw": "imdb"}],
    })
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(report, out_path,
                              film_name_tr="TEST FILM",
                              film_id="2023-0001-1-0000-00-1",
                              ocr_lines=[])
    text = out_path.read_text(encoding="utf-8-sig")
    assert "ACTOR 15" in text
    assert "oyuncu daha" not in text
