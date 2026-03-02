"""
test_database_writer.py — Unit tests for PipelineRunner DATABASE write functionality.

Covers:
  DB-01: _build_ocr_scores maps OCR lines to cast entries (KEEP verdict)
  DB-02: _build_ocr_scores produces REJECTED verdict for unmatched lines
  DB-03: _build_ocr_scores handles both attribute-style and dict-style OCR lines
  DB-04: _write_database skips when database_enabled=False
  DB-05: _write_database creates correct directory structure and files
  DB-06: _write_database uses env var fallback for database_root
  DB-07: pipeline run() wraps _write_database in try/except
  DB-08: asr_engine key present in audio_pipeline result dict
"""

import json
import os
import sys
import types
from pathlib import Path

# Ensure Project directory is on sys.path
_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# Stub heavy native modules so pipeline_runner can be imported without cv2/numpy
for _stub_name in ("cv2", "numpy", "numpy.core", "numpy.linalg",
                   "paddleocr", "paddleocr.paddleocr"):
    if _stub_name not in sys.modules:
        _m = types.ModuleType(_stub_name)
        sys.modules[_stub_name] = _m
# numpy needs a minimal array stub
sys.modules["numpy"].array = lambda *a, **k: None

from core.pipeline_runner import PipelineRunner  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _FakeOCRLine:
    """Attribute-style OCR line (as returned by OCREngine)."""
    def __init__(self, text, avg_confidence=0.9, seen_count=3):
        self.text = text
        self.avg_confidence = avg_confidence
        self.seen_count = seen_count


def _make_credits(cast):
    return {
        "film_title": "Test Film",
        "cast": cast,
        "crew": [],
        "total_actors": len(cast),
        "total_crew": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DB-01: KEEP verdict for matched actor
# ─────────────────────────────────────────────────────────────────────────────

def test_build_ocr_scores_keep_for_matched_actor():
    """DB-01: OCR line matching cast actor_name -> KEEP verdict with pipeline info."""
    cast = [{
        "actor_name": "Nisa Serezli",
        "character_name": "Aysel",
        "confidence": 0.99,
        "is_verified_name": True,
        "is_llm_verified": True,
    }]
    ocr_lines = [_FakeOCRLine("Nisa Serezli", avg_confidence=0.92, seen_count=5)]
    result = PipelineRunner._build_ocr_scores(ocr_lines, _make_credits(cast))

    scores = result["scores"]
    assert len(scores) == 1
    s = scores[0]
    assert s["text"] == "Nisa Serezli"
    assert s["ocr_confidence"] == 0.92
    assert s["pipeline_confidence"] == 0.99
    assert s["seen_count"] == 5
    assert s["name_db_match"] is True
    assert s["llm_verified"] is True
    assert s["verdict"] == "KEEP"


# ─────────────────────────────────────────────────────────────────────────────
# DB-02: REJECTED verdict for unmatched OCR line
# ─────────────────────────────────────────────────────────────────────────────

def test_build_ocr_scores_rejected_for_unmatched():
    """DB-02: OCR line not in cast -> REJECTED verdict with null pipeline info."""
    ocr_lines = [_FakeOCRLine("COCO", avg_confidence=0.98, seen_count=12)]
    result = PipelineRunner._build_ocr_scores(ocr_lines, _make_credits([]))

    scores = result["scores"]
    assert len(scores) == 1
    s = scores[0]
    assert s["pipeline_confidence"] is None
    assert s["name_db_match"] is False
    assert s["llm_verified"] is False
    assert s["verdict"] == "REJECTED"


# ─────────────────────────────────────────────────────────────────────────────
# DB-03: dict-style and edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_build_ocr_scores_dict_style_ocr_lines():
    """DB-03: dict-style OCR lines (serialized) are handled correctly."""
    cast = [{
        "actor_name": "Volkan Girgin",
        "character_name": "",
        "confidence": 0.85,
        "is_verified_name": False,
        "is_llm_verified": True,
    }]
    ocr_lines = [{"text": "Volkan Girgin", "avg_confidence": 0.75, "seen_count": 2}]
    result = PipelineRunner._build_ocr_scores(ocr_lines, _make_credits(cast))

    s = result["scores"][0]
    assert s["verdict"] == "KEEP"
    assert s["ocr_confidence"] == 0.75
    assert s["pipeline_confidence"] == 0.85
    assert s["llm_verified"] is True


def test_build_ocr_scores_matched_by_character_name():
    """DB-01 variant: matching by character_name also counts as KEEP."""
    cast = [{
        "actor_name": "Ahmet Yilmaz",
        "character_name": "Kaptan",
        "confidence": 0.77,
        "is_verified_name": True,
        "is_llm_verified": False,
    }]
    ocr_lines = [_FakeOCRLine("Kaptan", avg_confidence=0.88, seen_count=1)]
    result = PipelineRunner._build_ocr_scores(ocr_lines, _make_credits(cast))

    assert result["scores"][0]["verdict"] == "KEEP"
    assert result["scores"][0]["pipeline_confidence"] == 0.77


def test_build_ocr_scores_empty_inputs():
    """DB-03 edge: empty ocr_lines returns empty scores list."""
    assert PipelineRunner._build_ocr_scores([], _make_credits([])) == {"scores": []}


# ─────────────────────────────────────────────────────────────────────────────
# DB-04: database_enabled=False skips writing
# ─────────────────────────────────────────────────────────────────────────────

def test_write_database_skips_when_disabled(tmp_path):
    """DB-04: _write_database does nothing when database_enabled=False."""
    runner = PipelineRunner.__new__(PipelineRunner)
    runner.config = {"database_enabled": False, "database_root": str(tmp_path / "db")}
    runner._log_messages = []

    runner._write_database(
        video_info={"filename": "test.mp4"},
        credits_data=_make_credits([]),
        credits_raw=None,
        ocr_lines=[],
        stage_stats={},
        audio_result=None,
        work_dir=str(tmp_path),
        content_profile_name="FilmDizi",
        ts="010125-1200",
    )

    assert not (tmp_path / "db").exists()


# ─────────────────────────────────────────────────────────────────────────────
# DB-05: _write_database creates correct files
# ─────────────────────────────────────────────────────────────────────────────

def test_write_database_creates_all_files(tmp_path):
    """DB-05: _write_database creates all expected output files."""
    ts = "010125-1200"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / f"myvideo_{ts}.txt").write_text("report text", encoding="utf-8")
    (work_dir / f"myvideo-tscr_{ts}.txt").write_text("transcript", encoding="utf-8")
    (work_dir / "myvideo_report.json").write_text('{"test": 1}', encoding="utf-8")

    db_root = tmp_path / "DATABASE"
    runner = PipelineRunner.__new__(PipelineRunner)
    runner.config = {"database_enabled": True, "database_root": str(db_root)}
    runner._log_messages = ["log line 1", "log line 2"]
    runner._log = lambda msg: None

    cast = [{
        "actor_name": "Test Actor",
        "character_name": "Hero",
        "confidence": 0.95,
        "is_verified_name": True,
        "is_llm_verified": True,
    }]
    ocr_lines = [_FakeOCRLine("Test Actor", avg_confidence=0.88, seen_count=4)]
    audio_result = {
        "status": "ok",
        "transcript": [{"start": 0.0, "end": 2.5, "text": "Merhaba dunya"}],
    }

    runner._write_database(
        video_info={"filename": "myvideo.mp4"},
        credits_data=_make_credits(cast),
        credits_raw={"raw": True},
        ocr_lines=ocr_lines,
        stage_stats={},
        audio_result=audio_result,
        work_dir=str(work_dir),
        content_profile_name="FilmDizi",
        ts=ts,
    )

    db_dir = db_root / "myvideo"
    assert db_dir.is_dir()

    files = list(db_dir.iterdir())
    names = [f.name for f in files]

    assert any(n.endswith(".txt") for n in names)
    assert any(n.endswith("_report.json") for n in names)
    assert any(n.endswith("_ocr_scores.json") for n in names)
    assert any(n.endswith("_credits_raw.json") for n in names)
    assert any(n.endswith("_transcript.json") for n in names)
    assert any(n.endswith("_debug.log") for n in names)

    # Verify work_dir files were copied to db_dir
    assert f"myvideo_{ts}.txt" in names, "Report TXT not copied from work_dir"
    assert f"myvideo-tscr_{ts}.txt" in names, "Transcript TXT not copied from work_dir"
    assert "myvideo_report.json" in names, "Report JSON not copied from work_dir"

    # ocr_scores content
    ocr_file = next(f for f in files if f.name.endswith("_ocr_scores.json"))
    data = json.loads(ocr_file.read_text(encoding="utf-8"))
    assert data["scores"][0]["verdict"] == "KEEP"
    assert data["scores"][0]["ocr_confidence"] == 0.88

    # transcript content
    tr_file = next(f for f in files if f.name.endswith("_transcript.json"))
    tr_data = json.loads(tr_file.read_text(encoding="utf-8"))
    assert tr_data[0]["text"] == "Merhaba dunya"

    # debug log content
    log_file = next(f for f in files if f.name.endswith("_debug.log"))
    assert "log line 1" in log_file.read_text(encoding="utf-8")

    # credits_raw content
    raw_file = next(f for f in files if f.name.endswith("_credits_raw.json"))
    assert json.loads(raw_file.read_text(encoding="utf-8")).get("raw") is True


def test_write_database_filename_contains_timestamp(tmp_path):
    """DB-05 variant: output files include DDMMYY-HHmm timestamp in name."""
    import re
    ts = "010125-1200"
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    runner = PipelineRunner.__new__(PipelineRunner)
    runner.config = {"database_enabled": True, "database_root": str(tmp_path / "DB")}
    runner._log_messages = []
    runner._log = lambda msg: None

    runner._write_database(
        video_info={"filename": "clip.mp4"},
        credits_data=_make_credits([]),
        credits_raw=None,
        ocr_lines=[],
        stage_stats={},
        audio_result=None,
        work_dir=str(work_dir),
        content_profile_name="FilmDizi",
        ts=ts,
    )

    db_dir = tmp_path / "DB" / "clip"
    for f in db_dir.iterdir():
        assert re.search(r"clip_\d{6}-\d{4}", f.name), \
            f"Timestamp not found in filename: {f.name}"


# ─────────────────────────────────────────────────────────────────────────────
# DB-06: env var fallback for database_root
# ─────────────────────────────────────────────────────────────────────────────

def test_write_database_uses_env_var_fallback(tmp_path, monkeypatch):
    """DB-06: VITOS_DATABASE_ROOT env var is used when config has no database_root."""
    db_root = tmp_path / "ENV_DATABASE"
    monkeypatch.setenv("VITOS_DATABASE_ROOT", str(db_root))

    runner = PipelineRunner.__new__(PipelineRunner)
    runner.config = {"database_enabled": True}
    runner._log_messages = []
    runner._log = lambda msg: None

    runner._write_database(
        video_info={"filename": "clip.mp4"},
        credits_data=_make_credits([]),
        credits_raw=None,
        ocr_lines=[],
        stage_stats={},
        audio_result=None,
        work_dir=str(tmp_path),
        content_profile_name="Spor",
        ts="010125-1200",
    )

    assert (db_root / "clip").is_dir()


# ─────────────────────────────────────────────────────────────────────────────
# DB-06b: _safe_path collision protection
# ─────────────────────────────────────────────────────────────────────────────

def test_write_database_collision_protection(tmp_path):
    """DB-06b: _write_database uses _safe_path so existing files get _2 suffix."""
    ts = "010125-1200"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    report_txt = work_dir / f"clip_{ts}.txt"
    report_txt.write_text("first run", encoding="utf-8")

    db_root = tmp_path / "DB"
    runner = PipelineRunner.__new__(PipelineRunner)
    runner.config = {"database_enabled": True, "database_root": str(db_root)}
    runner._log_messages = []
    runner._log = lambda msg: None

    _call = dict(
        video_info={"filename": "clip.mp4"},
        credits_data=_make_credits([]),
        credits_raw=None,
        ocr_lines=[],
        stage_stats={},
        audio_result=None,
        work_dir=str(work_dir),
        content_profile_name="FilmDizi",
        ts=ts,
    )

    runner._write_database(**_call)
    runner._write_database(**_call)

    db_dir = db_root / "clip"
    names = [f.name for f in db_dir.iterdir()]
    # Both copies of report TXT should exist (original + _2 variant)
    assert f"clip_{ts}.txt" in names
    assert f"clip_{ts}_2.txt" in names


# ─────────────────────────────────────────────────────────────────────────────
# DB-07: pipeline run() wraps _write_database in try/except
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_runner_run_wraps_database_in_try_except():
    """DB-07: pipeline_runner.run() wraps _write_database in try/except."""
    import inspect
    source = inspect.getsource(PipelineRunner.run)
    assert "_write_database" in source
    assert "DATABASE" in source and "Yazma" in source


# ─────────────────────────────────────────────────────────────────────────────
# DB-08: asr_engine key in audio_pipeline result
# ─────────────────────────────────────────────────────────────────────────────

def test_audio_pipeline_result_has_asr_engine_key():
    """DB-08: AudioPipeline result dict always contains asr_engine key."""
    import inspect
    from core.audio_pipeline import AudioPipeline

    source = inspect.getsource(AudioPipeline.run)
    assert '"asr_engine"' in source
    assert "faster-whisper" in source
