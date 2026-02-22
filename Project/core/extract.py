"""
extract.py — [A] FFmpeg ile video → WAV ses çıkarma.

İki çıktı üretir:
  - audio_raw_16k.wav  → 16kHz mono (PyAnnote, WhisperX için)
  - audio_raw_48k.wav  → 48kHz mono (DeepFilterNet3 için — DF3 48kHz istiyor)

Güvenilirlik: returncode + stderr her zaman kontrol edilir.
"""

import os
import subprocess
import time
from pathlib import Path


class ExtractStage:
    """FFmpeg ile video dosyasından ses çıkarma."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", log_cb=None):
        self._ffmpeg = ffmpeg_path
        self._log = log_cb or print

    def run(self, video_path: str, work_dir: str, **opts) -> dict:
        """
        Video → WAV çıkarma.

        Args:
            video_path: Kaynak video dosyası
            work_dir: Çıktı klasörü

        Returns:
            {
                "status": "ok",
                "wav_16k": "path/audio_raw_16k.wav",
                "wav_48k": "path/audio_raw_48k.wav",
                "duration_sec": 5400.0,
                "stage_time_sec": 5.2
            }
        """
        t0 = time.time()
        os.makedirs(work_dir, exist_ok=True)

        wav_16k = str(Path(work_dir) / "audio_raw_16k.wav")
        wav_48k = str(Path(work_dir) / "audio_raw_48k.wav")

        # ── 48kHz mono WAV (DF3 için) — ana çıkarma ──
        self._log("  [Extract] 48kHz WAV çıkarılıyor...")
        self._ffmpeg_extract(video_path, wav_48k, sample_rate=48000)

        # ── 16kHz mono WAV (PyAnnote + WhisperX) — 48kHz'den resample ──
        # PERF-4 FIX: Video tekrar decode etmek yerine WAV→WAV resample (~1s)
        self._log("  [Extract] 48kHz → 16kHz resample...")
        self._ffmpeg_extract(wav_48k, wav_16k, sample_rate=16000)

        # Süre hesabı
        duration = self._get_duration(wav_16k)

        elapsed = round(time.time() - t0, 2)
        self._log(f"  [Extract] Tamamlandı: {duration:.0f}s audio ({elapsed:.1f}s)")

        return {
            "status": "ok",
            "wav_16k": wav_16k,
            "wav_48k": wav_48k,
            "duration_sec": duration,
            "stage_time_sec": elapsed,
        }

    def _ffmpeg_extract(self, video_path: str, out_path: str,
                        sample_rate: int = 16000):
        """FFmpeg subprocess çalıştır + hata kontrolü."""
        cmd = [
            self._ffmpeg, '-y',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            out_path
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding='utf-8', errors='replace',
                timeout=600,
            )
            if result.returncode != 0:
                err = result.stderr.strip()[-500:] if result.stderr else "bilinmeyen hata"
                raise RuntimeError(
                    f"FFmpeg hatası (rc={result.returncode}): {err}"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"FFmpeg timeout: 600s aşıldı")

        if not Path(out_path).is_file():
            raise FileNotFoundError(f"FFmpeg çıktı dosyası oluşmadı: {out_path}")

        size_mb = Path(out_path).stat().st_size / (1024 * 1024)
        if size_mb < 0.001:
            raise RuntimeError(f"FFmpeg çıktısı boş: {out_path} ({size_mb:.3f} MB)")

    def _get_duration(self, wav_path: str) -> float:
        """WAV süresini saniye cinsinden döndür."""
        try:
            import wave
            with wave.open(wav_path, 'rb') as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0
