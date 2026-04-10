"""
parallel_runner.py — Paralel video işleme çalıştırıcısı.

Kullanım:
    python parallel_runner.py                        # varsayılan: V:\ÇAĞATAY, FilmDizi-Hybrid
    python parallel_runner.py --source "D:\Videolar"
    python parallel_runner.py --profile SpeechToText
    python parallel_runner.py --reset-state           # state JSON'u temizle, baştan başla

GUI başlar → worker sayısı seç (1/2/3/4) → pipeline'lar paralel koşar.
Her subprocess: python main.py --headless <video>  (ayrı CUDA context, tam izolasyon)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# VIDEO_EXTENSIONS'ı queue_manager'dan al
try:
    from core.queue_manager import VIDEO_EXTENSIONS
except ImportError:
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".wmv"}

DEFAULT_SOURCE = r"V:\ÇAĞATAY"
DEFAULT_PROFILE = "FilmDizi-Hybrid"
STATE_FILENAME = "_parallel_state.json"


# ---------------------------------------------------------------------------
# State dosyası
# ---------------------------------------------------------------------------

def _state_path(source_dir: str) -> Path:
    return Path(source_dir) / STATE_FILENAME


def _load_state(source_dir: str) -> dict:
    p = _state_path(source_dir)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed": {}, "errors": {}}


def _save_state(source_dir: str, state: dict) -> None:
    """Atomic write: önce .tmp, sonra os.replace."""
    p = _state_path(source_dir)
    tmp = p.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except Exception as e:
        print(f"[UYARI] State yazılamadı: {e}", flush=True)


# ---------------------------------------------------------------------------
# Video tarama
# ---------------------------------------------------------------------------

def _scan_videos(source_dir: str) -> list[str]:
    """source_dir altındaki tüm videoları sıralı döndür (recursive)."""
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Kaynak klasör bulunamadı: {source_dir}")
    videos = sorted(
        str(p)
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        and p.name != STATE_FILENAME
    )
    return videos


# ---------------------------------------------------------------------------
# PySide6 Worker Seçim Dialogu
# ---------------------------------------------------------------------------

def _ask_worker_count_gui() -> int | None:
    """
    PySide6 ile küçük bir dialog açar.
    Seçilen worker sayısını (1-4) döndürür; iptal edilirse None.
    """
    try:
        from PySide6.QtWidgets import (
            QApplication, QDialog, QVBoxLayout, QHBoxLayout,
            QPushButton, QLabel
        )
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont
    except ImportError:
        print("PySide6 bulunamadı — terminal moduna geçiliyor.", flush=True)
        return _ask_worker_count_terminal()

    app = QApplication.instance() or QApplication(sys.argv)

    dialog = QDialog()
    dialog.setWindowTitle("Paralel Worker Sayısı")
    dialog.setFixedSize(340, 160)
    dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)

    layout = QVBoxLayout(dialog)
    layout.setSpacing(12)
    layout.setContentsMargins(20, 16, 20, 16)

    label = QLabel("Aynı anda kaç video işlensin?")
    label.setFont(QFont("Segoe UI", 11))
    label.setAlignment(Qt.AlignCenter)
    layout.addWidget(label)

    btn_layout = QHBoxLayout()
    btn_layout.setSpacing(8)

    chosen = [None]

    for n in (1, 2, 3, 4):
        btn = QPushButton(str(n))
        btn.setFixedSize(60, 44)
        btn.setFont(QFont("Segoe UI", 13, QFont.Bold))
        if n == 4:
            btn.setToolTip("24 GB VRAM sınırına dayanır — OOM riski")
            btn.setStyleSheet("QPushButton { color: #b05000; }")
        def _clicked(_, val=n):
            chosen[0] = val
            dialog.accept()
        btn.clicked.connect(_clicked)
        btn_layout.addWidget(btn)

    layout.addLayout(btn_layout)

    cancel_btn = QPushButton("İptal")
    cancel_btn.setFixedHeight(28)
    cancel_btn.setFont(QFont("Segoe UI", 9))
    cancel_btn.clicked.connect(dialog.reject)
    layout.addWidget(cancel_btn, alignment=Qt.AlignRight)

    dialog.exec()
    return chosen[0]


def _ask_worker_count_terminal() -> int | None:
    """Fallback: terminal prompt."""
    while True:
        try:
            raw = input("Kaç worker? [1/2/3/4] (q=çıkış): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw in {"q", "quit", "exit", ""}:
            return None
        if raw in {"1", "2", "3", "4"}:
            return int(raw)
        print("Geçersiz seçim. 1, 2, 3 veya 4 girin.")


# ---------------------------------------------------------------------------
# Ana çalıştırıcı
# ---------------------------------------------------------------------------

def run(source_dir: str, profile: str, worker_count: int, reset_state: bool) -> None:
    state = _load_state(source_dir)
    if reset_state:
        state = {"completed": {}, "errors": {}}
        _save_state(source_dir, state)
        print("[STATE] Sıfırlandı.", flush=True)

    all_videos = _scan_videos(source_dir)
    skip_set = set(state.get("completed", {}).keys())
    pending = [v for v in all_videos if v not in skip_set]

    print(
        f"[INFO] Toplam: {len(all_videos)} video  |  "
        f"Tamamlanmış: {len(skip_set)}  |  "
        f"Bekleyen: {len(pending)}  |  "
        f"Worker: {worker_count}",
        flush=True,
    )

    if not pending:
        print("[INFO] İşlenecek video kalmadı.", flush=True)
        return

    python_exe = sys.executable
    main_py = str(PROJECT_ROOT / "main.py")

    active: list[tuple[subprocess.Popen, str, float]] = []  # (proc, video_path, start_time)
    queue = list(pending)
    interrupted = False

    def _terminate_all():
        for proc, vpath, _ in active:
            if proc.poll() is None:
                proc.terminate()
                print(f"  [DURDURULDU] {Path(vpath).name}", flush=True)

    def _sigint_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[Ctrl+C] Durduruluyor...", flush=True)
        _terminate_all()

    signal.signal(signal.SIGINT, _sigint_handler)

    def _launch_next() -> bool:
        """Kuyruktan bir video alıp subprocess başlatır. Boş ise False döner."""
        if not queue:
            return False
        vpath = queue.pop(0)
        env = os.environ.copy()
        env["CONTENT_PROFILE"] = profile
        env["SCOPE"] = "video+audio"
        env["HEADLESS"] = "1"
        proc = subprocess.Popen(
            [python_exe, main_py, vpath],
            env=env,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        active.append((proc, vpath, time.monotonic()))
        print(f"[BAŞLADI ] #{len(skip_set) + len(pending) - len(queue)}/"
              f"{len(pending)}  {Path(vpath).name}  (PID={proc.pid})", flush=True)
        return True

    # İlk dalgayı doldur
    for _ in range(min(worker_count, len(queue))):
        _launch_next()

    # Ana döngü
    while active and not interrupted:
        time.sleep(0.5)
        still_running = []
        for proc, vpath, t0 in active:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, vpath, t0))
                continue
            # Bitti
            elapsed = time.monotonic() - t0
            if rc == 0:
                state["completed"][vpath] = {
                    "returncode": 0,
                    "duration_sec": round(elapsed, 1),
                    "finished_at": datetime.now().isoformat(),
                }
                print(
                    f"[TAMAM   ] {Path(vpath).name}  "
                    f"({elapsed / 60:.1f} dk)",
                    flush=True,
                )
            else:
                state["errors"][vpath] = {
                    "returncode": rc,
                    "finished_at": datetime.now().isoformat(),
                }
                print(
                    f"[HATA    ] {Path(vpath).name}  rc={rc}",
                    flush=True,
                )
            _save_state(source_dir, state)
            # Slot boş: yenisini başlat
            _launch_next()

        active[:] = still_running

    if not interrupted:
        print(
            f"\n[BİTTİ] Tamamlanan: {len(state['completed'])}  "
            f"Hatalı: {len(state['errors'])}",
            flush=True,
        )
        print(f"[STATE ] {_state_path(source_dir)}", flush=True)


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paralel video pipeline çalıştırıcısı",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        metavar="KLASOR",
        help=f"Video kaynak klasörü (varsayılan: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        metavar="PROFİL",
        help=f"İçerik profili (varsayılan: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="State JSON'u sıfırla, tüm videoları tekrar işle",
    )
    args = parser.parse_args()

    # Worker sayısını sor
    worker_count = _ask_worker_count_gui()
    if worker_count is None:
        print("İptal edildi.", flush=True)
        return 0

    run(
        source_dir=args.source,
        profile=args.profile,
        worker_count=worker_count,
        reset_state=args.reset_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
