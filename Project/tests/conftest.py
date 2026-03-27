"""conftest.py — Pytest fixture'ları.

PROFILE=1 ortam değişkeni ile:
  - Her test session'ı için ayrı profil log dosyası oluşturulur.
  - ResourceSampler session boyunca çalışır.
  - Session sonunda `profile_summary` fixture ile özet yazdırılır.
"""

import os
import sys
import time
from pathlib import Path

# Project dizinini sys.path'e ekle
_project_dir = Path(__file__).resolve().parent.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

import pytest


# ── PROFILE=1 aktif mi? ───────────────────────────────────────────────────────
_PROFILING = os.environ.get("PROFILE", "0").strip() not in ("0", "", "false", "False")


@pytest.fixture(scope="session", autouse=True)
def profiling_session():
    """Test session'ı boyunca ResourceSampler çalıştır (PROFILE=1 gerekir)."""
    if not _PROFILING:
        yield
        return

    # Her session için ayrı log dosyası
    from core.profiler import ResourceSampler, PROFILE_LOG, _log_event

    session_id = f"pytest_{int(time.time())}"
    _log_event({
        "type": "session_start",
        "job_id": session_id,
        "t": time.perf_counter(),
    })

    sampler = ResourceSampler(job_id=session_id, interval=2.0)
    sampler.start()

    yield session_id

    sampler.stop()
    _log_event({
        "type": "session_end",
        "job_id": session_id,
        "t": time.perf_counter(),
    })

    # Özet yazdır
    try:
        from core.profiler import summarize
        summary = summarize(job_id=session_id)
        if summary:
            print(f"\n{'='*60}")
            print(f"  PROFIL ÖZETİ — {session_id}")
            print(f"{'='*60}")
            if summary.get("stages"):
                print("  Stage süreleri:")
                for sname, sdata in summary["stages"].items():
                    dur = sdata.get("duration_sec", "?")
                    ok = "OK" if sdata.get("ok") else "HATA"
                    print(f"    {sname:<20} {dur:>8.2f}s  [{ok}]")
            if summary.get("resources"):
                res = summary["resources"]
                print("  Kaynak kullanımı:")
                if "cpu_avg" in res:
                    print(f"    CPU ort/maks: {res['cpu_avg']}% / {res['cpu_max']}%")
                if "ram_max_gb" in res:
                    print(f"    RAM ort/maks: {res['ram_avg_gb']:.2f} / {res['ram_max_gb']:.2f} GB")
                if "gpu_max" in res:
                    print(f"    GPU ort/maks: {res['gpu_avg']}% / {res['gpu_max']}%")
                if "vram_max_gb" in res:
                    print(f"    VRAM ort/maks: {res['vram_avg_gb']:.2f} / {res['vram_max_gb']:.2f} GB")
            print(f"{'='*60}")
            print(f"  Log: {PROFILE_LOG}")
    except Exception as exc:
        print(f"\n  [Profiler] Özet hesaplanamadı: {exc}")
