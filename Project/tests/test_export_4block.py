"""test_export_4block.py — Validates that _write_report() produces exactly 4 blocks.

Covers:
  BLOCK-01: Output contains all 4 block headers in correct order
  BLOCK-02: BLOK 4 shows Gemini summary when audio_result has summary
  BLOCK-02b: BLOK 4 shows "Özet oluşturma aktif değil" when audio_result has no summary
  BLOCK-03: BLOK 4 shows "Özet oluşturma aktif değil" when no audio_result
  BLOCK-04: BLOK 1 shows ASR Motor line when audio_result is provided
  BLOCK-05: Spor profile with match_data shows match info in BLOK 4 (not a separate section)
"""

import os
import sys
import tempfile

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def _make_report(content_type="FilmDizi", match_data=None):
    return {
        "file_info": {
            "filename": "test_film.mp4",
            "duration_human": "1:46:17",
            "resolution": "1920x1080",
            "fps": "25.0",
            "filesize_bytes": 2321.0 * 1024 * 1024,
        },
        "processing": {
            "content_type": content_type,
            "ocr_engine": "PaddleOCR (GPU)",
            "total_duration_sec": 598.3,
            "speed_ratio": "0.09x",
            "stages": [
                {"name": "INGEST", "status": "ok", "duration_sec": 1.2},
                {"name": "FRAME_EXTRACT", "status": "ok", "duration_sec": 13.3},
            ],
        },
        "profile": "WORKSTATION",
        "credits": {
            "cast": [],
            "directors": [],
            "technical_crew": [],
        },
        "generated_at": "2026-03-04T01:03:40",
        "match_data": match_data,
    }


def _run_write_report(report, audio_result=None):
    from core.export_engine import ExportEngine
    engine = ExportEngine.__new__(ExportEngine)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        tmp_path = f.name
    try:
        engine._write_report(report, tmp_path, audio_result=audio_result)
        with open(tmp_path, encoding="utf-8-sig") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def test_four_blocks_present_in_correct_order():
    """BLOCK-01: All 4 block headers must appear in order."""
    content = _run_write_report(_make_report())
    b1 = content.find("BLOK 1")
    b2 = content.find("BLOK 2")
    b3 = content.find("BLOK 3")
    b4 = content.find("BLOK 4")
    assert b1 != -1, "BLOK 1 bulunamadı"
    assert b2 != -1, "BLOK 2 bulunamadı"
    assert b3 != -1, "BLOK 3 bulunamadı"
    assert b4 != -1, "BLOK 4 bulunamadı"
    assert b1 < b2 < b3 < b4, "Bloklar yanlış sırada"


def test_blok4_shows_summary_when_audio_result_has_summary():
    """BLOCK-02: When audio_result has summary, BLOK 4 shows the summary text."""
    audio_result = {
        "status": "ok",
        "asr_engine": "faster-whisper",
        "whisper_model": "large-v3",
        "transcript": [
            {"start": 12.0, "text": "Karanlık bir odada..."},
            {"start": 105.0, "text": "Kapı çalınır..."},
        ],
        "summary": "Bu film karanlık bir odada geçen olayları konu almaktadır.",
    }
    content = _run_write_report(_make_report(), audio_result=audio_result)
    assert "Bu film karanlık bir odada geçen olayları konu almaktadır." in content, "Özet BLOK 4'te yok"
    # Transcript lines should NOT appear directly in BLOK 4 anymore
    assert "Karanlık bir odada..." not in content
    assert "Kapı çalınır..." not in content
    # Should not show "Özet oluşturma aktif değil" when summary is provided
    assert "Özet oluşturma aktif değil" not in content


def test_blok4_shows_no_summary_message_when_audio_but_no_summary():
    """BLOCK-02b: When audio_result exists but has no summary, shows 'Özet oluşturma aktif değil'."""
    audio_result = {
        "status": "ok",
        "asr_engine": "faster-whisper",
        "whisper_model": "large-v3",
        "transcript": [
            {"start": 12.0, "text": "Karanlık bir odada..."},
        ],
    }
    content = _run_write_report(_make_report(), audio_result=audio_result)
    assert "Özet oluşturma aktif değil" in content


def test_blok4_shows_no_summary_message_when_no_audio():
    """BLOCK-03: When no audio_result, BLOK 4 shows 'Özet oluşturma aktif değil'."""
    content = _run_write_report(_make_report(), audio_result=None)
    assert "Özet oluşturma aktif değil" in content


def test_blok1_shows_asr_motor_when_audio_result_provided():
    """BLOCK-04: When audio_result is provided, BLOK 1 includes ASR Motor line."""
    audio_result = {
        "status": "ok",
        "asr_engine": "faster-whisper",
        "whisper_model": "large-v3",
        "transcript": [],
    }
    content = _run_write_report(_make_report(), audio_result=audio_result)
    assert "ASR Motor  : faster-whisper (large-v3)" in content


def test_blok4_shows_match_data_for_spor_profile():
    """BLOCK-05: For Spor profile with match_data, BLOK 4 shows match info."""
    match_data = {
        "spor_turu": "Futbol",
        "lig": "Süper Lig",
        "takimlar": [
            {"isim": "Galatasaray", "skor": "2"},
            {"isim": "Fenerbahçe", "skor": "1"},
        ],
    }
    content = _run_write_report(_make_report(content_type="Spor", match_data=match_data))
    b4_pos = content.find("BLOK 4")
    assert b4_pos != -1
    # Match info should appear after BLOK 4 header
    after_b4 = content[b4_pos:]
    assert "Galatasaray" in after_b4
    assert "Süper Lig" in after_b4
    # Old separate "MAC BILGILERI" section should not appear outside BLOK 4
    assert "MAC BILGILERI" not in content
