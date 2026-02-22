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

Çıktı: work_dir/audio_result.json
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

    try:
        # audio paketi import — bu noktada venv_audio aktif olmalı
        from audio.audio_pipeline import AudioPipeline

        pipeline = AudioPipeline(config=config, log_cb=log_cb)
        result = pipeline.run()

        # Sonucu JSON'a yaz
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t0
        log_cb(f"\n[AudioWorker] Tamamlandı ({elapsed:.1f}s) → {result_path}")
        sys.exit(0)

    except ImportError as e:
        log_cb(f"\n[AudioWorker] Import hatası: {e}")
        log_cb("Olası neden: venv_audio aktif değil veya paket eksik")
        log_cb(traceback.format_exc())

        # Hata sonucunu da JSON'a yaz — bridge okuyabilsin
        _write_error_result(result_path, str(e), time.time() - t0)
        sys.exit(2)

    except Exception as e:
        log_cb(f"\n[AudioWorker] Pipeline hatası: {e}")
        log_cb(traceback.format_exc())

        _write_error_result(result_path, str(e), time.time() - t0)
        sys.exit(2)


def _write_error_result(path: str, error: str, elapsed: float):
    """Hata durumunda minimal sonuç JSON'ı yaz."""
    try:
        result = {
            "version": "1.0",
            "status": "error",
            "error": error,
            "processing_time_sec": round(elapsed, 2),
            "transcript": [],
            "speakers": {},
            "stages": {},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()
