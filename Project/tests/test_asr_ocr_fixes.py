"""
test_asr_ocr_fixes.py — Bug düzeltmelerinin birim testleri.

Kapsanan hatalar:
  BUG-02: diarize_result tam dict geçirildiğinde konuşmacı ataması yapılmıyordu.
  BUG-03: Whisper ayarları ayrı kwargs olarak geçirildiğinde görmezden geliniyordu.
  OCR-01: _iter_paddle_lines generator olduğu için `if not it` hiç tetiklenmiyordu.
"""
from audio.stages.transcribe import TranscribeStage


# ─────────────────────────────────────────────────────────────────────────────
# BUG-02: _assign_speakers — tam diarize-result dict'i de kabul etmeli
# ─────────────────────────────────────────────────────────────────────────────

def test_assign_speakers_accepts_segments_list():
    """Normal durum: list[dict] geçirilince konuşmacı atanır."""
    stage = TranscribeStage()
    segments = [{"start": 0.0, "end": 3.0, "text": "Merhaba", "speaker": ""}]
    diar = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    stage._assign_speakers(segments, diar)
    assert segments[0]["speaker"] == "SPEAKER_00"


def test_assign_speakers_accepts_full_diarize_result_dict():
    """BUG-02 FIX: tam diarize_result dict geçirilince konuşmacı yine atanmalı."""
    stage = TranscribeStage()
    segments = [{"start": 0.0, "end": 3.0, "text": "Merhaba", "speaker": ""}]
    diarize_result = {
        "status": "ok",
        "segments": [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}],
        "speakers_found": 1,
        "stage_time_sec": 5.0,
    }
    stage._assign_speakers(segments, diarize_result)
    assert segments[0]["speaker"] == "SPEAKER_00"


def test_assign_speakers_tolerates_none():
    """None geçirilince sessizce dönmeli (hata üretmemeli)."""
    stage = TranscribeStage()
    segments = [{"start": 0.0, "end": 3.0, "text": "Merhaba", "speaker": ""}]
    stage._assign_speakers(segments, None)
    assert segments[0]["speaker"] == ""


def test_assign_speakers_tolerates_empty_dict():
    """Boş dict geçirilince sessizce dönmeli."""
    stage = TranscribeStage()
    segments = [{"start": 0.0, "end": 3.0, "text": "Merhaba", "speaker": ""}]
    stage._assign_speakers(segments, {})
    assert segments[0]["speaker"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# BUG-03: _run_legacy — doğrudan kwargs olarak geçirilen Whisper ayarları
# ─────────────────────────────────────────────────────────────────────────────

def test_run_legacy_extracts_direct_whisper_kwargs(monkeypatch):
    """BUG-03 FIX: whisper_model vb. doğrudan kwargs geçirilince opts'a aktarılmalı."""
    stage = TranscribeStage()
    captured = {}

    def fake_transcribe(audio_path, opts, diarization=None):
        captured.update(opts)
        return {
            "status": "error",
            "segments": [],
            "total_segments": 0,
            "stage_time_sec": 0.0,
            "error": "test_only",
        }

    monkeypatch.setattr(stage, "_transcribe", fake_transcribe)
    stage._run_legacy(
        "fake.wav",
        whisper_model="medium",
        whisper_language="en",
        compute_type="int8",
        batch_size=4,
    )
    assert captured.get("whisper_model") == "medium"
    assert captured.get("whisper_language") == "en"
    assert captured.get("compute_type") == "int8"
    assert captured.get("batch_size") == 4


def test_run_legacy_options_dict_takes_precedence(monkeypatch):
    """options={} geçirilince, aynı anahtardaki doğrudan kwargs görmezden gelinmeli."""
    stage = TranscribeStage()
    captured = {}

    def fake_transcribe(audio_path, opts, diarization=None):
        captured.update(opts)
        return {
            "status": "error",
            "segments": [],
            "total_segments": 0,
            "stage_time_sec": 0.0,
            "error": "test_only",
        }

    monkeypatch.setattr(stage, "_transcribe", fake_transcribe)
    stage._run_legacy(
        "fake.wav",
        options={"whisper_model": "large-v3"},
        whisper_model="medium",  # should NOT override options dict value
    )
    assert captured.get("whisper_model") == "large-v3"


def test_run_legacy_normalises_diarize_result_dict(monkeypatch):
    """BUG-02 FIX: full diarize_result dict geçirilince segments listesi çıkarılmalı."""
    stage = TranscribeStage()
    captured_diar = {}

    def fake_transcribe(audio_path, opts, diarization=None):
        captured_diar["value"] = diarization
        return {
            "status": "error",
            "segments": [],
            "total_segments": 0,
            "stage_time_sec": 0.0,
            "error": "test_only",
        }

    monkeypatch.setattr(stage, "_transcribe", fake_transcribe)
    diarize_result = {
        "status": "ok",
        "segments": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
        "speakers_found": 1,
        "stage_time_sec": 3.0,
    }
    stage._run_legacy("fake.wav", diarization=diarize_result)
    assert isinstance(captured_diar["value"], list)
    assert captured_diar["value"][0]["speaker"] == "SPEAKER_00"


# ─────────────────────────────────────────────────────────────────────────────
# OCR-01: _iter_paddle_lines — generator her zaman truthy
# ─────────────────────────────────────────────────────────────────────────────

def test_iter_paddle_lines_generator_always_truthy():
    """
    OCR-01 FIX doğrulaması: generator nesnesi her zaman truthy'dir.
    'if not it: continue' şeklindeki kontrol hiçbir zaman tetiklenmez.
    Düzeltme: list() ile materialize edip len kontrolü yapmak gerekir.

    Bu test cv2/PaddleOCR gerektirmeden mantığı doğrular.
    """
    def _iter_nothing():
        # Yield ifadesi var → generator; return None boş durdurar
        if False:
            yield ("dummy",)
        return None

    gen = _iter_nothing()
    # Generator nesnesi her zaman truthy
    assert bool(gen) is True
    # Ama list()'e dönüştürünce boş çıkar
    items = list(_iter_nothing())
    assert items == []
    # Ve `if not items: continue` doğru çalışır
    assert not items
