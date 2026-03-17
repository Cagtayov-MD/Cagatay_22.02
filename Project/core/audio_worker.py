"""
audio_worker.py — Ses pipeline CLI giriş noktası.

Kullanım:
    python Project/core/audio_worker.py <config.json>

Çıkış:
  - work_dir/audio_result.json
  - work_dir/audio_transcript.txt

Exit kodları:
  0 = başarılı
  1 = config hatası
  2 = pipeline veya import hatası

Not:
- Bu modülde proje içi modüllere (`utils`, `audio`, vb.) erişim sağlamak için
  PROJE kökü `sys.path`'e importlardan önce eklenir.
"""

from __future__ import annotations

# ── PROJE KÖKÜNÜ sys.path'E EKLE (IMPORT ÖNCESİ) ──
# Bu blok, proje içi importların çalışması için dosyanın en başında olmalıdır.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Project/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ------------------------------------------------

# Windows console encoding fix
import io
import os
import json
import tempfile
import time
import traceback
import uuid
from typing import Optional, Dict, Any

# Windows konsolunda UTF-8 kullan
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

# Şimdi güvenle proje-içi modülleri import edebiliriz
from utils.time_utils import fmt_hms as _fmt_hms_shared


def main() -> None:
    # Argüman kontrolü
    if len(sys.argv) < 2:
        print("Kullanım: python audio_worker.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    config_file = Path(config_path)
    if not config_file.is_file():
        print(f"Config dosyası bulunamadı: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Config yükle
    try:
        with config_file.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Config parse hatası: {e}", file=sys.stderr)
        sys.exit(1)

    # Zorunlu alan kontrolü
    video_path = config.get("video_path", "")
    work_dir = config.get("work_dir", "")

    if not video_path or not Path(video_path).is_file():
        print(f"Video bulunamadı: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not work_dir:
        print("work_dir belirtilmemiş", file=sys.stderr)
        sys.exit(1)

    os.makedirs(work_dir, exist_ok=True)

    # Log yolu ve yardımcı log fonksiyonu
    log_path = str(Path(work_dir) / "audio_pipeline.log")

    def log_cb(msg: str) -> None:
        print(msg, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8", newline="\n") as lf:
                lf.write(msg + "\n")
        except Exception:
            # Log yazılamaması kritik değil; pipeline devam edebilir
            pass

    # Başlangıç logları
    log_cb(f"[AudioWorker] Başlıyor — {Path(video_path).name}")
    log_cb(f"[AudioWorker] Config: {config_path}")
    log_cb(f"[AudioWorker] Python: {sys.executable}")
    log_cb(f"[AudioWorker] CWD: {os.getcwd()}")

    t0 = time.time()
    trace_id = uuid.uuid4().hex[:12]
    result_path = str(Path(work_dir) / "audio_result.json")
    transcript_txt_path = str(Path(work_dir) / "audio_transcript.txt")
    lock_path = str(Path(work_dir) / "audio_worker.lock")

    lock_fd = _acquire_lock(lock_path)
    if lock_fd is None:
        error_result = _build_error_result(
            error="work_dir is locked by another audio worker",
            elapsed=time.time() - t0,
            error_code="WORKDIR_LOCKED",
            trace_id=trace_id,
        )
        _write_json_atomic(result_path, error_result)
        _write_transcript_txt_atomic(transcript_txt_path, error_result)
        log_cb("[AudioWorker] Çalışma klasörü kilitli — başka worker aktif olabilir")
        sys.exit(2)

    try:
        # audio paketi import — bu noktada venv_audio aktif olmalı
        # ImportError ayrıntılı şekilde yakalanır ve kullanıcıya loglanır.
        from audio.audio_pipeline import AudioPipeline  # type: ignore

        pipeline = AudioPipeline(config=config, log_cb=log_cb)
        result = pipeline.run()

        # Sonucu JSON'a yaz
        _write_json_atomic(result_path, result)
        _write_transcript_txt_atomic(transcript_txt_path, result)

        elapsed = time.time() - t0
        log_cb(f"\n[AudioWorker] Tamamlandı ({elapsed:.1f}s) → {result_path} | {transcript_txt_path}")

        # status=error ise exit 2 dön (bridge daha doğru anlasın)
        if isinstance(result, dict) and result.get("status") == "error":
            sys.exit(2)

        sys.exit(0)

    except ImportError as e:
        # Tipik nedenler: venv aktif değil veya PATH/sys.path eksik
        log_cb(f"\n[AudioWorker] Import hatası: {e}")
        log_cb("Olası neden: venv_audio aktif değil veya paket eksik")
        log_cb(traceback.format_exc())

        error_result = _write_error_result(
            result_path,
            str(e),
            time.time() - t0,
            error_code="IMPORT_ERROR",
            trace_id=trace_id,
        )
        _write_transcript_txt_atomic(transcript_txt_path, error_result)
        sys.exit(2)

    except Exception as e:
        log_cb(f"\n[AudioWorker] Pipeline hatası: {e}")
        log_cb(traceback.format_exc())

        error_result = _write_error_result(
            result_path,
            str(e),
            time.time() - t0,
            error_code="PIPELINE_ERROR",
            trace_id=trace_id,
        )
        _write_transcript_txt_atomic(transcript_txt_path, error_result)
        sys.exit(2)

    finally:
        _release_lock(lock_fd, lock_path)


# ----------------- Yardımcı fonksiyonlar -----------------


def _write_error_result(path: str, error: str, elapsed: float, error_code: str = "PIPELINE_ERROR", trace_id: str = "") -> dict:
    """Hata durumunda minimal sonuç JSON'ı yaz ve döndür."""
    result = _build_error_result(error, elapsed, error_code=error_code, trace_id=trace_id)
    try:
        _write_json_atomic(path, result)
    except Exception:
        pass
    return result


def _write_transcript_txt_atomic(path: str, result: dict) -> None:
    """Transcript segmentlerini okunabilir TXT formatında yaz."""
    segments = result.get("transcript") or []
    lines = []

    for seg in segments:
        start = _safe_float(seg.get("start", 0.0))
        end = _safe_float(seg.get("end", 0.0))
        speaker = (seg.get("speaker") or "UNK").strip() or "UNK"
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{_fmt_hms(start)} - {_fmt_hms(end)}] {speaker}: {text}")

    if not lines:
        status = result.get("status", "unknown")
        err = result.get("error", "")
        # Stage-level transcribe hatasını da göster
        if not err:
            err = (result.get("stages", {}) .get("transcribe", {}) .get("error", ""))
        lines = ["Transcript üretilemedi.", f"status={status}"]
        if err:
            lines.append(f"error={err}")

    _write_text_atomic(path, "\n".join(lines) + "\n")


def _fmt_hms(seconds: float) -> str:
    return _fmt_hms_shared(seconds, with_ms=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_error_result(error: str, elapsed: float, error_code: str = "PIPELINE_ERROR", trace_id: str = "") -> dict:
    return {
        "version": "1.0",
        "status": "error",
        "error": error,
        "error_code": error_code,
        "trace_id": trace_id,
        "processing_time_sec": round(elapsed, 2),
        "transcript": [],
        "speakers": {},
        "stages": {},
    }


def _write_json_atomic(path: str, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    _write_text_atomic(path, body)


def _write_text_atomic(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # some environments (Windows on network FS) may not support fsync on temp files
                pass
        os.replace(tmp_path, target)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _is_pid_alive(pid: int) -> bool:
    """PID hâlâ çalışıyor mu kontrol et."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # sinyal göndermez, sadece varlık kontrolü
        return True
    except ProcessLookupError:
        return False  # process yok
    except PermissionError:
        return True  # process var ama erişim yok — canlı say
    except OSError:
        return False


def _try_clean_stale_lock(lock_path: str) -> bool:
    """
    Stale lock tespiti: lock dosyasındaki PID ölmüşse kilidi sil.
    Returns True if stale lock was cleaned.
    """
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            content = f.read()
        # "pid=12345 time=..." formatını parse et
        pid = 0
        for part in content.split():
            if part.startswith("pid="):
                pid = int(part.split("=", 1)[1])
                break

        if pid > 0 and not _is_pid_alive(pid):
            os.remove(lock_path)
            return True
    except (OSError, ValueError):
        # Dosya okunamıyorsa veya parse edilemiyorsa temizlemeyi dene
        try:
            os.remove(lock_path)
            return True
        except OSError:
            pass
    return False


def _acquire_lock(lock_path: str) -> Optional[int]:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(lock_path, flags)
        os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
        return fd
    except FileExistsError:
        # Stale lock kontrolü: eski process ölmüşse kilidi temizle ve tekrar dene
        if _try_clean_stale_lock(lock_path):
            try:
                fd = os.open(lock_path, flags)
                os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
                return fd
            except OSError:
                return None
        return None
    except OSError:
        return None


def _release_lock(lock_fd: Optional[int], lock_path: str) -> None:
    try:
        if lock_fd is not None:
            os.close(lock_fd)
    except OSError:
        pass
    try:
        if lock_fd is not None and os.path.exists(lock_path):
            os.remove(lock_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()