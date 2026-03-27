"""
run_profile_batch.py — 7 videoyu PROFILE=1 ile sırayla pipeline'dan geçir.
Her video için ayrı profile_<stem>.jsonl üretir, sonunda özet tablo basar.

Kullanım:
    PROFILE=1 python run_profile_batch.py
"""
import json
import os
import sys
import time
from pathlib import Path

# ── Ortam ayarları ────────────────────────────────────────────────────────────
os.environ["PROFILE"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["HEADLESS"] = "1"
os.environ["VENV_AUDIO_PYTHON"] = r"F:\Root\venv_audio\Scripts\python.exe"

# UTF-8 çıktı
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Proje dizini sys.path'e ekle ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ── Video listesi ─────────────────────────────────────────────────────────────
VIDEOS = [
    r"V:\web_client_CAG_1939-0003-1-000-00-1-KÜÇÜK_KIZIN_RÜYASI.mp4",
    r"V:\web_client_CAG_1965-0057-1-0000-00-1-NEW_YORK'TAKİ_JANDARMA.mp4",
    r"V:\web_client_CAG_1973-0177-1-0000-90-1-AĞLIYORUM.mp4",
    r"V:\web_client_CAG_1989-0058-1-0000-00-1-ESAS_HEDEF.mp4",
    r"V:\web_client_CAG_1995-0288-1-0000-00-1-KARA_RAHİP.mp4",
    r"V:\web_client_CAG_2011-9252-0-0001-88-1-DİLEKLER_ZAMANI.mp4",
    r"V:\web_client_CAG_2021-2164-1-0000-56-1-BEYAZ_BALON.mp4",
]


def _sep(char="═", width=68):
    return char * width


def _print_summary(stem: str, log_file: Path):
    """Tek video için profil özetini tablo olarak yazdır."""
    if not log_file.exists():
        print(f"  [UYARI] Log dosyası bulunamadı: {log_file}")
        return

    from core.profiler import summarize  # noqa: PLC0415

    # geçici olarak PROFILE_LOG'u bu dosyaya yönlendir
    import core.profiler as _profiler  # noqa: PLC0415
    _orig = _profiler.PROFILE_LOG
    _profiler.PROFILE_LOG = log_file
    summary = summarize(job_id=stem)
    _profiler.PROFILE_LOG = _orig

    stages = summary.get("stages", {})
    resources = summary.get("resources", {})

    print(f"\n  {'Aşama':<28} {'Süre (s)':>10}  {'Durum'}")
    print(f"  {'-'*28} {'-'*10}  {'-'*8}")
    for stage, info in stages.items():
        dur = info.get("duration_sec")
        ok_flag = "OK" if info.get("ok") else "HATA"
        dur_str = f"{dur:.2f}" if dur is not None else "?"
        print(f"  {stage:<28} {dur_str:>10}  {ok_flag}")

    if resources:
        print(f"\n  Kaynak (max değerler):")
        for k, v in resources.items():
            print(f"    {k}: {v}")


def main():
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    logging.getLogger("paddleocr").setLevel(logging.ERROR)
    logging.getLogger("paddle").setLevel(logging.ERROR)

    from config.runtime_paths import FFMPEG_BIN_DIR  # noqa: PLC0415
    from core.pipeline_runner import PipelineRunner  # noqa: PLC0415
    import core.profiler as _profiler  # noqa: PLC0415

    ffmpeg = str(Path(FFMPEG_BIN_DIR) / "ffmpeg.exe")
    ffprobe = str(Path(FFMPEG_BIN_DIR) / "ffprobe.exe")

    total = len(VIDEOS)
    results = []

    for idx, video_path_str in enumerate(VIDEOS, start=1):
        video_path = Path(video_path_str)
        stem = video_path.stem
        log_file = PROJECT_DIR / f"profile_{stem}.jsonl"

        # Bu video için log dosyasını ayarla
        _profiler.PROFILE_LOG = log_file
        if log_file.exists():
            log_file.unlink()  # önceki çalıştırmadan temizle

        print()
        print(_sep())
        print(f"  [{idx}/{total}] {stem}")
        print(f"  Log  → {log_file.name}")
        print(_sep())

        t_start = time.perf_counter()
        try:
            runner = PipelineRunner(ffmpeg, ffprobe)
            runner.run(video_path=video_path_str, scope="video+audio")
            elapsed = time.perf_counter() - t_start
            status = "OK"
            print(f"\n  ✓ Tamamlandı — {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            status = f"HATA: {exc}"
            print(f"\n  ✗ Hata ({elapsed:.1f}s): {exc}")

        results.append((stem, log_file, elapsed, status))
        _print_summary(stem, log_file)

    # ── Genel özet tablosu ────────────────────────────────────────────────────
    print()
    print(_sep("═"))
    print("  GENEL ÖZET")
    print(_sep("═"))
    print(f"  {'Film':<52} {'Süre':>7}  Durum")
    print(f"  {'-'*52} {'-'*7}  {'-'*8}")
    for stem, _lf, elapsed, status in results:
        short = stem[stem.rfind("1-")+2:] if "1-" in stem else stem[-40:]
        print(f"  {short:<52} {elapsed:>6.0f}s  {status}")
    print(_sep("═"))


if __name__ == "__main__":
    main()
