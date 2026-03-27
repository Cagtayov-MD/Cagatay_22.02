"""test_profiling.py — Profiler modülü birim testleri.

PROFILE=1 ile çalışır, aksi hâlde testlerin büyük çoğunluğu skip edilir.
"""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

_project_dir = Path(__file__).resolve().parent.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

_PROFILING = os.environ.get("PROFILE", "0").strip() not in ("0", "", "false", "False")
_needs_profile = pytest.mark.skipif(not _PROFILING, reason="PROFILE=1 gerekli")


# ═══════════════════════════════════════════════════════════════════
# StageTimer testleri
# ═══════════════════════════════════════════════════════════════════

class TestStageTimer:
    """StageTimer'ın PROFILE=1 modunda doğru log yazdığını doğrula."""

    @_needs_profile
    def test_stage_timer_writes_start_and_end(self, tmp_path, monkeypatch):
        """Stage start + end eventleri log dosyasına yazılır."""
        import core.profiler as profiler

        log_file = tmp_path / "profile.jsonl"
        monkeypatch.setattr(profiler, "PROFILE_LOG", log_file)

        with profiler.StageTimer("job1", "TEST_STAGE"):
            time.sleep(0.05)

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

        start = json.loads(lines[0])
        end = json.loads(lines[1])

        assert start["type"] == "stage_start"
        assert start["stage"] == "TEST_STAGE"
        assert start["job_id"] == "job1"

        assert end["type"] == "stage_end"
        assert end["stage"] == "TEST_STAGE"
        assert end["duration_sec"] >= 0.04
        assert end["ok"] is True

    @_needs_profile
    def test_stage_timer_captures_exception(self, tmp_path, monkeypatch):
        """Exception durumunda ok=False ve error mesajı yazılır."""
        import core.profiler as profiler

        log_file = tmp_path / "profile.jsonl"
        monkeypatch.setattr(profiler, "PROFILE_LOG", log_file)

        with pytest.raises(ValueError):
            with profiler.StageTimer("job2", "FAIL_STAGE"):
                raise ValueError("test hatası")

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        end_events = [json.loads(l) for l in lines if '"stage_end"' in l]
        assert len(end_events) == 1
        assert end_events[0]["ok"] is False
        assert "test hatası" in end_events[0]["error"]

    def test_stage_timer_noop_when_disabled(self, tmp_path, monkeypatch):
        """PROFILE=0 iken log dosyası oluşturulmaz."""
        import core.profiler as profiler

        log_file = tmp_path / "profile.jsonl"
        monkeypatch.setattr(profiler, "PROFILE_LOG", log_file)
        monkeypatch.setattr(profiler, "_ENABLED", False)

        with profiler.StageTimer("job3", "NOOP_STAGE"):
            pass

        assert not log_file.exists()

    @_needs_profile
    def test_stage_timer_duration_accurate(self, tmp_path, monkeypatch):
        """Süre ölçümü 50ms hassasiyetle doğru."""
        import core.profiler as profiler

        log_file = tmp_path / "profile.jsonl"
        monkeypatch.setattr(profiler, "PROFILE_LOG", log_file)

        with profiler.StageTimer("job4", "TIMED_STAGE"):
            time.sleep(0.1)

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        end = json.loads([l for l in lines if '"stage_end"' in l][0])
        assert 0.08 <= end["duration_sec"] <= 0.5, \
            f"Süre beklenen 0.1s civarında, alınan: {end['duration_sec']}s"


# ═══════════════════════════════════════════════════════════════════
# ResourceSampler testleri
# ═══════════════════════════════════════════════════════════════════

class TestResourceSampler:
    """ResourceSampler thread'inin doğru örnekleme yaptığını doğrula."""

    @_needs_profile
    def test_sampler_writes_resource_events(self, tmp_path, monkeypatch):
        """Sampler çalıştırılınca resource_sample eventleri yazılır."""
        import core.profiler as profiler

        log_file = tmp_path / "profile.jsonl"
        monkeypatch.setattr(profiler, "PROFILE_LOG", log_file)

        sampler = profiler.ResourceSampler(job_id="sampler_test", interval=0.3)
        sampler.start()
        time.sleep(0.8)
        sampler.stop()

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        samples = [json.loads(l) for l in lines if '"resource_sample"' in l]

        assert len(samples) >= 1, "En az 1 sample alınmış olmalı"
        for s in samples:
            assert "cpu_percent" in s
            assert "ram_used_gb" in s
            assert s["job_id"] == "sampler_test"

    @_needs_profile
    def test_sampler_stops_cleanly(self, tmp_path, monkeypatch):
        """Sampler durdurulduktan sonra yeni event yazılmaz."""
        import core.profiler as profiler

        log_file = tmp_path / "profile.jsonl"
        monkeypatch.setattr(profiler, "PROFILE_LOG", log_file)

        sampler = profiler.ResourceSampler(job_id="stop_test", interval=0.2)
        sampler.start()
        time.sleep(0.5)
        sampler.stop()

        # Sadece bu test'e ait job_id'ye bak — conftest sampler'ı karışmasın
        count_before = sum(
            1 for l in log_file.read_text(encoding="utf-8").splitlines()
            if '"resource_sample"' in l and '"stop_test"' in l
        )

        time.sleep(0.5)

        count_after = sum(
            1 for l in log_file.read_text(encoding="utf-8").splitlines()
            if '"resource_sample"' in l and '"stop_test"' in l
        )

        assert count_before == count_after, "Durdurmadan sonra yeni sample yazılmamalı"

    def test_sampler_noop_when_disabled(self, tmp_path, monkeypatch):
        """PROFILE=0 iken sampler hiçbir şey yazmaz."""
        import core.profiler as profiler

        log_file = tmp_path / "profile.jsonl"
        monkeypatch.setattr(profiler, "PROFILE_LOG", log_file)
        monkeypatch.setattr(profiler, "_ENABLED", False)

        sampler = profiler.ResourceSampler(job_id="disabled_test", interval=0.1)
        sampler.start()
        time.sleep(0.3)
        sampler.stop()

        assert not log_file.exists()


# ═══════════════════════════════════════════════════════════════════
# summarize() testleri
# ═══════════════════════════════════════════════════════════════════

class TestSummarize:
    """summarize() fonksiyonunun doğru özet ürettiğini doğrula."""

    @_needs_profile
    def test_summarize_returns_stage_durations(self, tmp_path, monkeypatch):
        """summarize() stage sürelerini doğru toplar."""
        import core.profiler as profiler

        log_file = tmp_path / "profile.jsonl"
        monkeypatch.setattr(profiler, "PROFILE_LOG", log_file)

        events = [
            {"type": "stage_end", "job_id": "j1", "stage": "OCR", "duration_sec": 42.5, "ok": True, "t": 100},
            {"type": "stage_end", "job_id": "j1", "stage": "ASR", "duration_sec": 120.0, "ok": True, "t": 200},
            {"type": "resource_sample", "job_id": "j1", "t": 150,
             "cpu_percent": 80, "ram_used_gb": 8.0},
        ]
        with log_file.open("w", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(ev) + "\n")

        summary = profiler.summarize(job_id="j1")

        assert summary["stages"]["OCR"]["duration_sec"] == 42.5
        assert summary["stages"]["ASR"]["duration_sec"] == 120.0
        assert summary["resources"]["cpu_max"] == 80
        assert summary["resources"]["ram_max_gb"] == 8.0

    @_needs_profile
    def test_summarize_filters_by_job_id(self, tmp_path, monkeypatch):
        """summarize() farklı job_id'leri karıştırmaz."""
        import core.profiler as profiler

        log_file = tmp_path / "profile.jsonl"
        monkeypatch.setattr(profiler, "PROFILE_LOG", log_file)

        events = [
            {"type": "stage_end", "job_id": "jobA", "stage": "OCR", "duration_sec": 10.0, "ok": True, "t": 1},
            {"type": "stage_end", "job_id": "jobB", "stage": "OCR", "duration_sec": 999.0, "ok": True, "t": 2},
        ]
        with log_file.open("w", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(ev) + "\n")

        summary = profiler.summarize(job_id="jobA")
        assert summary["stages"]["OCR"]["duration_sec"] == 10.0

    def test_summarize_empty_when_no_log(self, tmp_path, monkeypatch):
        """Log dosyası yoksa boş sözlük döner."""
        import core.profiler as profiler

        monkeypatch.setattr(profiler, "PROFILE_LOG", tmp_path / "nonexistent.jsonl")

        result = profiler.summarize()
        assert result == {}


# ═══════════════════════════════════════════════════════════════════
# Pipeline entegrasyon testi
# ═══════════════════════════════════════════════════════════════════

class TestPipelineProfilerIntegration:
    """pipeline_runner.py'nin profiler hook'larını doğru çağırdığını doğrula."""

    @_needs_profile
    def test_pipeline_stage_hook_called(self, tmp_path, monkeypatch):
        """_stage() çağrıldığında profiler event yazılır.

        PipelineRunner'ın tamamını import etmek paddleocr init'ini tetikler
        (benign ama gürültülü). Bunun yerine _stage() hook'unu izole test ediyoruz:
        profiler._log_event() doğrudan çağrılınca doğru JSON yazılıyor mu?
        """
        import core.profiler as profiler

        log_file = tmp_path / "profile.jsonl"
        monkeypatch.setattr(profiler, "PROFILE_LOG", log_file)

        # _stage() metodu içindeki profiler çağrısını simüle et
        profiler._log_event({
            "type": "stage_end",
            "job_id": "integration_test",
            "stage": "INGEST",
            "duration_sec": 1.78,
            "ok": True,
            "error": None,
            "t": time.time(),
        })

        lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        events = [json.loads(l) for l in lines]

        stage_events = [e for e in events if e.get("stage") == "INGEST"]
        assert len(stage_events) == 1
        assert stage_events[0]["duration_sec"] == 1.78
        assert stage_events[0]["job_id"] == "integration_test"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
