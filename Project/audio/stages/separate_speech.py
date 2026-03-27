"""
separate_speech.py — Demucs ile spiker sesini efekten ayır.

Spor yayınlarında spiker sesi tribün gürültüsüyle karışık gelir.
Whisper bu karışık sesi zayıf tanır. Bu stage Demucs'u kullanarak
yalnızca "vocals" (spiker) kanalını ayırır; ASR buna uygulanır.

Model: htdemucs_ft  (fine-tuned, vocals/no-vocals ayrımı)
Çalışma: venv_audio subprocess (demucs torch gerektirir)

Entegrasyon:
    stage = SpeechSeparationStage(venv_python=..., ffmpeg=..., log_cb=...)
    vocals_path = stage.run(audio_path, work_dir)
    # vocals_path → None ise orijinal yol kullanılır (fallback)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path


class SpeechSeparationStage:
    """
    Demucs htdemucs_ft ile spiker sesini tribün gürültüsünden ayırır.

    Başarısız olursa (demucs yok / model indi değil / hata) None döner;
    çağıran katman orijinal sesi kullanmaya devam eder.
    """

    MODEL = "htdemucs_ft"   # vocals/no-vocals için fine-tuned model

    def __init__(
        self,
        venv_python: str = r"F:\Root\venv_audio\Scripts\python.exe",
        ffmpeg: str = "ffmpeg",
        log_cb=None,
        timeout_sec: int = 600,
    ):
        self.venv_python = venv_python
        self.ffmpeg = ffmpeg
        self._log = log_cb or print
        self.timeout_sec = timeout_sec

    def run(self, audio_path: str, work_dir: str) -> str | None:
        """
        Args:
            audio_path: Kaynak WAV dosyası (segment_first_ch1.wav vb.)
            work_dir:   Geçici dosyalar için dizin

        Returns:
            vocals WAV dosyasının yolu, başarısız olursa None.
        """
        t0 = time.time()
        audio_path = str(audio_path)

        if not os.path.isfile(audio_path):
            self._log(f"  [AYRIM] Kaynak dosya bulunamadı: {audio_path}")
            return None

        sep_dir = os.path.join(work_dir, "demucs_out")
        os.makedirs(sep_dir, exist_ok=True)

        # ── Demucs'u venv_audio subprocess'te çalıştır ──
        # Çıktı: sep_dir/htdemucs_ft/<dosyaadı>/{vocals,no_vocals}.wav
        script = f"""
import sys, subprocess, os
try:
    from demucs.separate import main as _sep
    sys.argv = [
        'demucs',
        '--two-stems', 'vocals',
        '-n', {self.MODEL!r},
        '--out', {sep_dir!r},
        '--mp3',          # hız için (daha sonra WAV'a çevrilecek)
        '--mp3-bitrate', '192',
        {audio_path!r},
    ]
    _sep()
    print('OK')
except Exception as e:
    print(f'ERR: {{e}}', file=sys.stderr)
    sys.exit(1)
"""
        try:
            proc = subprocess.run(
                [self.venv_python, "-c", script],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            if proc.returncode != 0:
                self._log(
                    f"  [AYRIM] Demucs hatası: {proc.stderr.strip()[:200]}"
                )
                return None
        except subprocess.TimeoutExpired:
            self._log(f"  [AYRIM] Demucs zaman aşımı ({self.timeout_sec}s)")
            return None
        except Exception as e:
            self._log(f"  [AYRIM] Demucs başlatılamadı: {e}")
            return None

        # ── Çıktı dosyasını bul ──
        stem = Path(audio_path).stem
        # Demucs çıktısı: sep_dir/htdemucs_ft/<stem>/vocals.mp3
        vocals_mp3 = Path(sep_dir) / self.MODEL / stem / "vocals.mp3"
        vocals_wav = Path(sep_dir) / self.MODEL / stem / "vocals.wav"

        # mp3 → wav (16kHz mono, whisper için)
        if vocals_mp3.is_file():
            try:
                subprocess.run(
                    [
                        self.ffmpeg, "-y", "-i", str(vocals_mp3),
                        "-ar", "16000", "-ac", "1",
                        "-acodec", "pcm_s16le",
                        str(vocals_wav),
                    ],
                    capture_output=True,
                    timeout=120,
                    check=True,
                )
            except Exception as e:
                self._log(f"  [AYRIM] MP3→WAV dönüşüm hatası: {e}")
                return None
        elif not vocals_wav.is_file():
            self._log(f"  [AYRIM] Vocals çıktısı bulunamadı: {vocals_mp3}")
            return None

        elapsed = round(time.time() - t0, 1)
        size_mb = round(vocals_wav.stat().st_size / 1_048_576, 1)
        self._log(
            f"  [AYRIM] ✓ Spiker sesi ayrıldı: {vocals_wav.name} "
            f"({size_mb} MB, {elapsed}s)"
        )
        return str(vocals_wav)
