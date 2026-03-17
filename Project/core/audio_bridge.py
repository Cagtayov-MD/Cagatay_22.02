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
import signal
import subprocess
import threading
import time
from pathlib import Path


class AudioBridge:
    """
    Ana venv (PaddleOCR) → venv_audio (faster-whisper/PyAnnote) subprocess köprüsü.

    Neden subprocess?
      - PaddleOCR ve faster-whisper farklı PyTorch versiyonları gerektirebilir
      - Aynı venv'de çakışma riski yüksek
      - VRAM izolasyonu: subprocess bitince tüm GPU bellek otomatik serbest kalır
    """

    # ── Varsayılan yollar — env var öncelikli ──
    # Cross-platform default paths
    _default_venv = Path.home() / "venv_audio" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    DEFAULT_VENV_PYTHON = os.environ.get(
        "VENV_AUDIO_PYTHON", str(_default_venv)
    )
    # Worker script defaults to project core directory
    _default_worker = Path(__file__).parent / "audio_worker.py"
    DEFAULT_WORKER_SCRIPT = os.environ.get(
        "AUDIO_WORKER_SCRIPT", str(_default_worker)
    )

    # Timeout: 1 saat (90 dakikalık film faster-whisper ~20-30dk sürer)
    DEFAULT_TIMEOUT = 3600

    def __init__(self, log_cb=None, config: dict = None):
        cfg = config or {}
        self._log = log_cb or print
        self._venv_python = cfg.get("venv_audio_python", self.DEFAULT_VENV_PYTHON)
        self._worker_script = cfg.get("audio_worker_script") or self._resolve_worker_script()
        self._timeout = cfg.get("audio_timeout", self.DEFAULT_TIMEOUT)

    def _resolve_worker_script(self) -> str:
        """
        Worker script yolunu güvenli şekilde çöz.

        Öncelik:
          1) AUDIO_WORKER_SCRIPT env (DEFAULT_WORKER_SCRIPT içinde)
          2) repo içi core/audio_worker.py
          3) legacy audio/audio_worker.py
          4) DEFAULT_WORKER_SCRIPT (son fallback)
        """
        default_path = Path(self.DEFAULT_WORKER_SCRIPT)
        if default_path.is_file():
            return str(default_path)

        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            project_root / "core" / "audio_worker.py",
            project_root / "audio" / "audio_worker.py",
        ]
        for candidate in candidates:
            if candidate.is_file():
                self._log(f"  [AudioBridge] Worker otomatik bulundu: {candidate}")
                return str(candidate)

        return str(default_path)

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

        # ── 3. Subprocess çalıştır (Popen — gerçek zamanlı log) ──
        cmd = [self._venv_python, self._worker_script, config_path]
        self._log(f"  [AudioBridge] Başlatılıyor: {Path(self._venv_python).name}")
        self._log(f"  [AudioBridge] Komut: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            # stderr'ı arka planda topla (deadlock önleme)
            stderr_lines: list[str] = []

            def _drain_stderr():
                assert proc.stderr is not None
                for line in proc.stderr:
                    stderr_lines.append(line)

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            # stdout'u gerçek zamanlı logla
            assert proc.stdout is not None
            for line in proc.stdout:
                stripped = line.rstrip("\n\r")
                if stripped:
                    self._log(f"  [audio] {stripped}")

            # Process'in bitmesini bekle (timeout ile)
            try:
                proc.wait(timeout=self._timeout)
            except subprocess.TimeoutExpired:
                self._log(f"  [AudioBridge] Timeout! ({self._timeout}s aşıldı) — process kill")
                proc.kill()
                proc.wait(timeout=10)
                return self._error_result(f"Ses pipeline timeout ({self._timeout}s)")

            stderr_thread.join(timeout=5)

            if proc.returncode != 0:
                stderr_tail = "".join(stderr_lines).strip()[-500:]
                self._log(f"  [AudioBridge] Hata (rc={proc.returncode}): {stderr_tail}")

                # Subprocess hatalı bitti ama audio_result.json yazılmış olabilir
                if Path(result_path).is_file():
                    return self._read_result(result_path)

                return self._error_result(
                    f"Subprocess hatası (rc={proc.returncode}): {stderr_tail}"
                )

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
            venv_path = Path.home() / "venv_audio"
            self._log(f"  [AudioBridge] Çözüm: python -m venv {venv_path}")
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
