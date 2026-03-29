"""
test_episode_label.py — Değişiklik 1: Dizi bölüm etiketi FİLMİN ORJİNAL ADI'na sızmaz.
"""
import os
import sys
from pathlib import Path

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ── Unit: _is_episode_label ──────────────────────────────────────────────────

def test_is_episode_label_avsnitt():
    from core.export_engine import _is_episode_label
    assert _is_episode_label("Avsnitt 8")
    assert _is_episode_label("avsnitt 1")
    assert _is_episode_label("AVSNITT 12")


def test_is_episode_label_episode():
    from core.export_engine import _is_episode_label
    assert _is_episode_label("Episode 12")
    assert _is_episode_label("episode 3")
    assert _is_episode_label("Ep. 5")
    assert _is_episode_label("Ep 7")


def test_is_episode_label_bolum():
    from core.export_engine import _is_episode_label
    assert _is_episode_label("Bölüm 5")
    assert _is_episode_label("bolum 9")


def test_is_episode_label_sxxexx():
    from core.export_engine import _is_episode_label
    assert _is_episode_label("S01E03")
    assert _is_episode_label("S1E12")
    assert _is_episode_label("S12E100")


def test_is_episode_label_false_for_titles():
    from core.export_engine import _is_episode_label
    assert not _is_episode_label("The Great Escape")
    assert not _is_episode_label("Avsnitt")  # numara yok
    assert not _is_episode_label("Episode")  # numara yok
    assert not _is_episode_label("Breaking Bad")
    assert not _is_episode_label("")
    assert not _is_episode_label("Moby Dick")


def test_is_episode_label_hash_format():
    """TMDB 'Episode #season.episode' formatı yakalanmalı."""
    from core.export_engine import _is_episode_label
    assert _is_episode_label("Episode #1.4")
    assert _is_episode_label("Episode #5")
    assert _is_episode_label("episode #12.1")
    assert _is_episode_label("Episode #1.12")


# ── Integration: TXT çıktısında episode label sızmamalı ──────────────────────

def _make_engine(tmp_path):
    from core.export_engine import ExportEngine
    return ExportEngine(output_dir=str(tmp_path), name_db=None)


def test_series_episode_title_does_not_leak_to_original_title(tmp_path):
    """Dizi kaydı + film_title='Avsnitt 8' + original_title yok → FİLMİN ORJİNAL ADI: VERİ YOK"""
    engine = _make_engine(tmp_path)
    report = {
        "generated_at": "2026-03-28T12:00:00",
        "file_info": {
            "filename": "evoArcadmin_TEST2_2023-0001-0-0008-00-1-ZENGIN_KIZ.mp4",
            "resolution": "1920x1080",
            "fps": 25,
            "duration_human": "00:45:00",
        },
        "credits": {
            "film_title": "Avsnitt 8",
            # original_title yok
            "verification_status": "tmdb_verified",
            "cast": [],
            "directors": [{"name": "Dir A"}],
            "crew": [{"name": "Dir A", "job": "Director", "raw": "tmdb"}],
        },
    }
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(
        report, out_path,
        film_name_tr="ZENGIN KIZ",
        film_id="2023-0001-0-0008-00-1",  # block[2]=0 → dizi
        ocr_lines=[],
    )
    text = out_path.read_text(encoding="utf-8-sig")
    assert "AVSNITT 8" not in text, "Episode label TXT'ye sızmamalı"
    assert "VERİ YOK" in text


def test_film_title_preserved_for_non_episode(tmp_path):
    """Film kaydı + film_title gerçek yabancı başlık → FİLMİN ORJİNAL ADI'nda görünür"""
    engine = _make_engine(tmp_path)
    report = {
        "generated_at": "2026-03-28T12:00:00",
        "file_info": {
            "filename": "evoArcadmin_TEST2_2023-0001-1-0000-00-1-BUYUK_KACIS.mp4",
            "resolution": "1920x1080",
            "fps": 25,
            "duration_human": "02:00:00",
        },
        "credits": {
            "film_title": "The Great Escape",
            "verification_status": "tmdb_verified",
            "cast": [],
            "directors": [{"name": "Dir A"}],
            "crew": [{"name": "Dir A", "job": "Director", "raw": "tmdb"}],
        },
    }
    out_path = Path(tmp_path) / "report.txt"
    engine._write_user_report(
        report, out_path,
        film_name_tr="BUYUK KACIS",
        film_id="2023-0001-1-0000-00-1",  # block[2]=1 → film
        ocr_lines=[],
    )
    text = out_path.read_text(encoding="utf-8-sig")
    assert "THE GREAT ESCAPE" in text, "Gerçek film başlığı gösterilmeli"
