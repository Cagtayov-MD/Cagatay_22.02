import json
from pathlib import Path

from core import audio_worker


def test_write_json_atomic_and_schema_fields(tmp_path):
    payload = {
        "version": "1.0",
        "status": "ok",
        "processing_time_sec": 1.2,
        "transcript": [],
        "speakers": {},
        "stages": {},
    }
    out = tmp_path / "audio_result.json"
    audio_worker._write_json_atomic(str(out), payload)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["version"] == "1.0"


def test_error_result_writes_json_and_txt(tmp_path):
    json_path = tmp_path / "audio_result.json"
    txt_path = tmp_path / "audio_transcript.txt"

    err = audio_worker._write_error_result(
        str(json_path),
        "boom",
        0.7,
        error_code="PIPELINE_ERROR",
        trace_id="abc123",
    )
    audio_worker._write_transcript_txt_atomic(str(txt_path), err)

    body = json.loads(json_path.read_text(encoding="utf-8"))
    txt = txt_path.read_text(encoding="utf-8")

    assert body["error_code"] == "PIPELINE_ERROR"
    assert body["trace_id"] == "abc123"
    assert "Transcript üretilemedi." in txt
    assert "status=error" in txt


def test_transcript_txt_handles_bad_timestamps(tmp_path):
    txt_path = tmp_path / "audio_transcript.txt"
    audio_worker._write_transcript_txt_atomic(
        str(txt_path),
        {
            "status": "ok",
            "transcript": [
                {"start": "1.25", "end": "oops", "speaker": "SPEAKER_00", "text": "Merhaba"}
            ],
        },
    )
    txt = txt_path.read_text(encoding="utf-8")
    assert "[00:00:01.250 - 00:00:00.000] SPEAKER_00: Merhaba" in txt


def test_lock_acquire_release(tmp_path):
    lock_path = tmp_path / "audio_worker.lock"
    fd1 = audio_worker._acquire_lock(str(lock_path))
    assert fd1 is not None

    fd2 = audio_worker._acquire_lock(str(lock_path))
    assert fd2 is None

    audio_worker._release_lock(fd1, str(lock_path))
    assert not lock_path.exists()
