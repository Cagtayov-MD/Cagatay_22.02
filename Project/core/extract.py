"""
extract.py — [A] FFmpeg ile video → WAV ses çıkarma.

İki çıktı üretir:
  - audio_raw_16k.wav  → 16kHz mono (PyAnnote, faster-whisper için)
  - audio_raw_48k.wav  → 48kHz mono (DeepFilterNet3 için — DF3 48kHz istiyor)

v2: selected_channel desteği — detect_language stage'inden gelen
    kanal bilgisine göre belirli kanalı çıkarabilir.

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

    def run(self, video_path: str, work_dir: str,
            selected_channel: int | None = None,
            max_duration_sec: float | None = None,
            start_offset_sec: float = 0.0,
            output_prefix: str = "audio_raw",
            **opts) -> dict:
        """
        Video → WAV çıkarma.

        Args:
            video_path:       Kaynak video dosyası
            work_dir:         Çıktı klasörü
            selected_channel: None → karışık mono (-ac 1, eski davranış)
                              0    → sol kanal  (-af "pan=mono|c0=c0")
                              1    → sağ kanal  (-af "pan=mono|c0=c1")
            start_offset_sec: Ses çıkarma başlangıcı (saniye)
            output_prefix:    Çıktı WAV dosyaları için önek

        Returns:
            {
                "status": "ok",
                "wav_16k": "path/audio_raw_16k.wav",
                "wav_48k": "path/audio_raw_48k.wav",
                "duration_sec": 5400.0,
                "stage_time_sec": 5.2,
                "selected_channel": None | 0 | 1,
                "start_offset_sec": 0.0,
            }
        """
        t0 = time.time()

        # Input validation
        if not video_path or not Path(video_path).is_file():
            self._log(f"  [Extract] HATA: Video dosyası bulunamadı: {video_path}")
            return {
                "status": "error",
                "wav_16k": "",
                "wav_48k": "",
                "duration_sec": 0.0,
                "stage_time_sec": round(time.time() - t0, 2),
                "selected_channel": selected_channel,
                "start_offset_sec": start_offset_sec,
                "error": f"Video dosyası bulunamadı: {video_path}",
            }

        os.makedirs(work_dir, exist_ok=True)

        prefix = (output_prefix or "audio_raw").strip() or "audio_raw"
        wav_16k = str(Path(work_dir) / f"{prefix}_16k.wav")
        wav_48k = str(Path(work_dir) / f"{prefix}_48k.wav")

        if selected_channel is not None:
            self._log(f"  [Extract] Kanal seçimi aktif: kanal {selected_channel}")
        else:
            self._log(f"  [Extract] Karışık mono (tüm kanallar)")

        if start_offset_sec:
            self._log(f"  [Extract] Başlangıç ofseti: {start_offset_sec:.1f}s")
        if max_duration_sec:
            self._log(
                f"  [Extract] Süre sınırı: {max_duration_sec:.1f}s "
                f"({int(max_duration_sec) // 60}dk)"
            )

        try:
            # ── 48kHz mono WAV (DF3 için) — ana çıkarma ──
            self._log("  [Extract] 48kHz WAV çıkarılıyor...")
            self._ffmpeg_extract(
                video_path, wav_48k,
                sample_rate=48000,
                selected_channel=selected_channel,
                start_offset_sec=start_offset_sec,
                max_duration_sec=max_duration_sec,
            )

            # ── 16kHz mono WAV (PyAnnote + faster-whisper) — 48kHz'den resample ──
            # PERF-4 FIX: Video tekrar decode etmek yerine WAV→WAV resample (~1s)
            # NOT: 48k zaten mono olarak çıktı, resample'da kanal seçimine gerek yok
            self._log("  [Extract] 48kHz → 16kHz resample...")
            self._ffmpeg_extract(wav_48k, wav_16k, sample_rate=16000)
        except Exception as e:
            self._log(f"  [Extract] HATA: {e}")
            return {
                "status": "error",
                "wav_16k": "",
                "wav_48k": "",
                "duration_sec": 0.0,
                "stage_time_sec": round(time.time() - t0, 2),
                "selected_channel": selected_channel,
                "start_offset_sec": start_offset_sec,
                "error": str(e),
            }

        # Süre hesabı - validate file exists first
        if not Path(wav_16k).is_file():
            self._log(f"  [Extract] UYARI: 16kHz WAV oluşmadı: {wav_16k}")
            duration = 0.0
        else:
            duration = self._get_duration(wav_16k)

        elapsed = round(time.time() - t0, 2)
        self._log(f"  [Extract] Tamamlandı: {duration:.0f}s audio ({elapsed:.1f}s)")

        return {
            "status": "ok",
            "wav_16k": wav_16k,
            "wav_48k": wav_48k,
            "duration_sec": duration,
            "stage_time_sec": elapsed,
            "selected_channel": selected_channel,
            "start_offset_sec": start_offset_sec,
        }

    def _ffmpeg_extract(self, video_path: str, out_path: str,
                        sample_rate: int = 16000,
                        selected_channel: int | None = None,
                        start_offset_sec: float = 0.0,
                        max_duration_sec: int | None = None):
        """FFmpeg subprocess çalıştır + hata kontrolü."""
        cmd = [
            self._ffmpeg, '-y',
        ]
        if start_offset_sec:
            cmd += ['-ss', str(start_offset_sec)]
        cmd += [
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
        ]
        if max_duration_sec:
            cmd += ['-t', str(max_duration_sec)]

        if selected_channel is not None:
            # Belirli kanalı al → mono
            cmd += ['-af', f'pan=mono|c0=c{selected_channel}']
        else:
            # Karışık mono (eski davranış)
            cmd += ['-ac', '1']

        cmd.append(out_path)

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
        except Exception as e:
            import logging
            logging.warning(f"Failed to get WAV duration for {wav_path}: {e}")
            return 0.0
