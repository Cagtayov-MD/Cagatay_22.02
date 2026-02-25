"""
audio_bridge.py — Ana venv'den ses pipeline'ı çağıran subprocess köprü.

pipeline_runner.py → AudioBridge.run() → subprocess → venv_audio/audio_worker.py

İki venv arası iletişim:
  Giriş:  config.json (disk üzerinden)
  Çıkış:  audio_result.json (disk üzerinden)
  Köprü:  subprocess.run([venv_audio_python, audio_worker.py, config.json])

Prensip: Ses pipeline patlarsa video pipeline sonuçları kaybolmaz.
"""

import json
import os
import subprocess
import time
from pathlib import Path

from config.runtime_paths import AUDIO_WORKER_SCRIPT, PROJECT_ROOT, VENV_AUDIO_PYTHON


class AudioBridge:
    """
    Ana venv (PaddleOCR) → venv_audio (WhisperX/PyAnnote) subprocess köprüsü.

    Neden subprocess?
      - PaddleOCR ve WhisperX farklı PyTorch versiyonları gerektirebilir
      - Aynı venv'de çakışma riski yüksek
      - VRAM izolasyonu: subprocess bitince tüm GPU bellek otomatik serbest kalır
    """

    # ── Varsayılan yollar — env var öncelikli ──
    DEFAULT_VENV_PYTHON = str(VENV_AUDIO_PYTHON)
    DEFAULT_WORKER_SCRIPT = str(AUDIO_WORKER_SCRIPT)

    # Timeout: 1 saat (90 dakikalık film WhisperX ~20-30dk sürer)
    DEFAULT_TIMEOUT = 3600

    def __init__(self, log_cb=None, config: dict = None):
        cfg = config or {}
        self._log = log_cb or print
        self._venv_python = cfg.get("venv_audio_python", self.DEFAULT_VENV_PYTHON)
        self._worker_script = cfg.get("audio_worker_script", self.DEFAULT_WORKER_SCRIPT)
        self._timeout = cfg.get("audio_timeout", self.DEFAULT_TIMEOUT)

    def run(self, video_path: str, work_dir: str, config: dict) -> dict:
        """
        Ses pipeline'ı subprocess ile çalıştır.

        Args:
            video_path: Kaynak video dosyası
            work_dir: Çalışma klasörü (config + result JSON burada)
            config: Pipeline yapılandırması (hf_token, options, stages vs.)

        Returns:
            audio_result.json içeriği (dict)
            Hata durumunda: {"status": "error", "error": "..."}
        """
        t0 = time.time()
        os.makedirs(work_dir, exist_ok=True)

        # ── 1. Ön kontroller ──
        if not self._check_prerequisites(video_path):
            return self._error_result("Ön kontrol başarısız")

        # ── 2. Config JSON oluştur ──
        full_config = {
            "video_path": str(video_path),
            "work_dir": str(work_dir),
            **config,
        }
        config_path = str(Path(work_dir) / "audio_config.json")
        result_path = str(Path(work_dir) / "audio_result.json")

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(full_config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"  [AudioBridge] Config yazma hatası: {e}")
            return self._error_result(f"Config yazma hatası: {e}")

        # ── 3. Subprocess çalıştır ──
        cmd = [self._venv_python, "-m", "core.audio_worker", config_path]
        self._log(f"  [AudioBridge] Başlatılıyor: {Path(self._venv_python).name}")
        self._log(f"  [AudioBridge] Komut: {' '.join(cmd)}")

        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
                cwd=str(PROJECT_ROOT),
                env=env,
            )

            # stdout'u logla (pipeline progress mesajları)
            if proc.stdout:
                for line in proc.stdout.strip().split("\n"):
                    self._log(f"  [audio] {line}")

            if proc.returncode != 0:
                stderr_tail = proc.stderr.strip()[-500:] if proc.stderr else ""
                self._log(f"  [AudioBridge] Hata (rc={proc.returncode}): {stderr_tail}")

                # Subprocess hatalı bitti ama audio_result.json yazılmış olabilir
                if Path(result_path).is_file():
                    return self._read_result(result_path)

                return self._error_result(
                    f"Subprocess hatası (rc={proc.returncode}): {stderr_tail}"
                )

        except subprocess.TimeoutExpired:
            self._log(f"  [AudioBridge] Timeout! ({self._timeout}s aşıldı)")
            return self._error_result(f"Ses pipeline timeout ({self._timeout}s)")

        except FileNotFoundError:
            self._log(f"  [AudioBridge] Python bulunamadı: {self._venv_python}")
            return self._error_result(
                f"venv_audio python bulunamadı: {self._venv_python}"
            )

        except Exception as e:
            self._log(f"  [AudioBridge] Beklenmeyen hata: {e}")
            return self._error_result(str(e))

        # ── 4. Sonuç JSON oku ──
        if not Path(result_path).is_file():
            self._log(f"  [AudioBridge] Sonuç dosyası bulunamadı: {result_path}")
            return self._error_result("audio_result.json oluşmadı")

        result = self._read_result(result_path)
        elapsed = round(time.time() - t0, 2)
        self._log(
            f"  [AudioBridge] Tamamlandı ({elapsed:.1f}s) — "
            f"{len(result.get('transcript', []))} segment"
        )
        return result

    def is_available(self) -> bool:
        """
        venv_audio erişilebilir mi?
        Pipeline başlamadan önce kontrol et — UI'da gösterilebilir.
        """
        python_ok = Path(self._venv_python).is_file()
        worker_ok = Path(self._worker_script).is_file()

        if not python_ok:
            self._log(f"  [AudioBridge] venv_audio python yok: {self._venv_python}")
        if not worker_ok:
            self._log(f"  [AudioBridge] worker script yok: {self._worker_script}")

        return python_ok and worker_ok

    def _check_prerequisites(self, video_path: str) -> bool:
        """Ön kontroller — hızlı fail."""
        if not Path(video_path).is_file():
            self._log(f"  [AudioBridge] Video bulunamadı: {video_path}")
            return False

        if not Path(self._venv_python).is_file():
            self._log(f"  [AudioBridge] venv_audio python bulunamadı: {self._venv_python}")
            self._log(f"  [AudioBridge] Beklenen: {self._venv_python}")
            self._log(f"  [AudioBridge] Çözüm: python -m venv F:\\Root\\venv_audio")
            return False

        if not Path(self._worker_script).is_file():
            self._log(f"  [AudioBridge] Worker script bulunamadı: {self._worker_script}")
            return False

        return True

    def _read_result(self, path: str) -> dict:
        """audio_result.json oku."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._log(f"  [AudioBridge] JSON parse hatası: {e}")
            return self._error_result(f"JSON parse hatası: {e}")
        except Exception as e:
            return self._error_result(str(e))

    @staticmethod
    def _error_result(error: str) -> dict:
        """Standart hata sonucu."""
        return {
            "status": "error",
            "error": error,
            "transcript": [],
            "speakers": {},
            "stages": {},
        }
