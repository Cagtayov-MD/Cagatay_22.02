"""
audio_worker.py — Ses pipeline CLI giriş noktası.

venv_audio içinden çağrılır:
  venv_audio\\Scripts\\python.exe audio/audio_worker.py config.json

config.json formatı:
  {
    "video_path": "F:/input/film.mp4",
    "work_dir": "F:/output/arsiv_film_20260222/",
    "ffmpeg": "F:/Source/ffmpeg/bin/ffmpeg.exe",
    "hf_token": "hf_xxx",
    "ollama_url": "http://localhost:11434",
    "tmdb_cast": [],
    "stages": ["extract", "denoise", "diarize", "transcribe", "post_process"],
    "options": {
      "whisper_model": "large-v3",
      "whisper_language": "tr",
      "compute_type": "float16",
      "max_speakers": 10,
      "denoise_enabled": true,
      "ollama_model": "llama3.1:8b"
    }
  }

Çıktı:
  - work_dir/audio_result.json
  - work_dir/audio_transcript.txt
Exit codes:
  0 = başarılı
  1 = config hatası
  2 = pipeline hatası
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path


def main():
    # ── Argüman kontrolü ──
    if len(sys.argv) < 2:
        print("Kullanım: python audio_worker.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    if not Path(config_path).is_file():
        print(f"Config dosyası bulunamadı: {config_path}", file=sys.stderr)
        sys.exit(1)

    # ── Config yükle ──
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Config parse hatası: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Zorunlu alan kontrolü ──
    video_path = config.get("video_path", "")
    work_dir = config.get("work_dir", "")

    if not video_path or not Path(video_path).is_file():
        print(f"Video bulunamadı: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not work_dir:
        print("work_dir belirtilmemiş", file=sys.stderr)
        sys.exit(1)

    os.makedirs(work_dir, exist_ok=True)

    # ── Log fonksiyonu ──
    log_path = str(Path(work_dir) / "audio_pipeline.log")

    def log_cb(msg: str):
        """Hem stdout'a hem log dosyasına yaz."""
        print(msg, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        except Exception:
            pass

    # ── Pipeline çalıştır ──
    log_cb(f"[AudioWorker] Başlıyor — {Path(video_path).name}")
    log_cb(f"[AudioWorker] Config: {config_path}")
    log_cb(f"[AudioWorker] Python: {sys.executable}")
    log_cb(f"[AudioWorker] CWD: {os.getcwd()}")

    t0 = time.time()
    result_path = str(Path(work_dir) / "audio_result.json")
    transcript_txt_path = str(Path(work_dir) / "audio_transcript.txt")

    try:
        # audio paketi import — bu noktada venv_audio aktif olmalı
        from audio.audio_pipeline import AudioPipeline

        pipeline = AudioPipeline(config=config, log_cb=log_cb)
        result = pipeline.run()

        # Sonucu JSON'a yaz
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        _write_transcript_txt(transcript_txt_path, result)

        elapsed = time.time() - t0
        log_cb(
            f"\n[AudioWorker] Tamamlandı ({elapsed:.1f}s) → {result_path} | {transcript_txt_path}"
        )
        sys.exit(0)

    except ImportError as e:
        log_cb(f"\n[AudioWorker] Import hatası: {e}")
        log_cb("Olası neden: venv_audio aktif değil veya paket eksik")
        log_cb(traceback.format_exc())

        # Hata sonucunu da JSON'a yaz — bridge okuyabilsin
        error_result = _write_error_result(result_path, str(e), time.time() - t0)
        _write_transcript_txt(transcript_txt_path, error_result)
        sys.exit(2)

    except Exception as e:
        log_cb(f"\n[AudioWorker] Pipeline hatası: {e}")
        log_cb(traceback.format_exc())

        error_result = _write_error_result(result_path, str(e), time.time() - t0)
        _write_transcript_txt(transcript_txt_path, error_result)
        sys.exit(2)


def _write_error_result(path: str, error: str, elapsed: float) -> dict:
    """Hata durumunda minimal sonuç JSON'ı yaz."""
    result = {
        "version": "1.0",
        "status": "error",
        "error": error,
        "processing_time_sec": round(elapsed, 2),
        "transcript": [],
        "speakers": {},
        "stages": {},
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return result


def _write_transcript_txt(path: str, result: dict):
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
        lines = ["Transcript üretilemedi.", f"status={status}"]
        if err:
            lines.append(f"error={err}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _fmt_hms(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
