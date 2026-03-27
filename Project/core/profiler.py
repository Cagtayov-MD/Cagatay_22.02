"""profiler.py — Stage timer + sistem kaynak örnekleyici.

Kullanım:
    PROFILE=1 ile aktif hale gelir, aksi hâlde tüm çağrılar no-op'tur.

    with StageTimer(job_id, "OCR_CREDITS"):
        run_ocr(...)

    sampler = ResourceSampler(job_id)
    sampler.start()
    run_pipeline(...)
    sampler.stop()

Çıktı: profile_log.jsonl (satır başına bir JSON event)
"""

import json
import os
import threading
import time
from pathlib import Path

# ── Aktivasyon kontrolü ──────────────────────────────────────────────────────
_ENABLED = os.environ.get("PROFILE", "0").strip() not in ("0", "", "false", "False")

# ── Log dosyası ──────────────────────────────────────────────────────────────
PROFILE_LOG = Path(os.environ.get("PROFILE_LOG", "profile_log.jsonl"))

# ── GPU backend (opsiyonel) ──────────────────────────────────────────────────
try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo,
    )
    nvmlInit()
    _GPU_HANDLE = nvmlDeviceGetHandleByIndex(0)
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False
    _GPU_HANDLE = None


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _gpu_snapshot() -> dict:
    """GPU util + VRAM anlık değer (sözlük). Erişilemezse boş sözlük."""
    if _HAS_NVML and _GPU_HANDLE is not None:
        try:
            util = nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
            mem = nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
            return {
                "gpu_util_percent": util.gpu,
                "vram_used_gb": round(mem.used / (1024 ** 3), 2),
                "vram_total_gb": round(mem.total / (1024 ** 3), 2),
            }
        except Exception:
            pass
    return {}


def cuda_sync():
    """GPU kuyruğunu boşalt (PaddlePaddle veya PyTorch). Yoksa no-op.

    Import lazy yapılır — module yüklenirken CUDA init tetiklenmez.
    """
    try:
        import paddle  # noqa: PLC0415
        paddle.device.cuda.synchronize()
        return
    except Exception:
        pass
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _log_event(event: dict):
    """Profiling eventini JSONL dosyasına yaz."""
    if not _ENABLED:
        return
    with PROFILE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


# ── StageTimer ────────────────────────────────────────────────────────────────

class StageTimer:
    """Context manager: stage başlangıç/bitiş eventlerini logla.

    PROFILE=0 (varsayılan) iken overhead sıfırdır.

    Args:
        job_id:     İş tanımlayıcı (örn. video dosya adı).
        stage:      Aşama adı (örn. "OCR_CREDITS").
        gpu_sync:   True ise girişte/çıkışta CUDA/Paddle sync yapılır.
    """

    __slots__ = ("job_id", "stage", "gpu_sync", "t0")

    def __init__(self, job_id: str, stage: str, *, gpu_sync: bool = False):
        self.job_id = job_id
        self.stage = stage
        self.gpu_sync = gpu_sync
        self.t0 = 0.0

    def __enter__(self):
        if not _ENABLED:
            return self
        if self.gpu_sync:
            cuda_sync()
        self.t0 = time.perf_counter()
        _log_event({
            "type": "stage_start",
            "job_id": self.job_id,
            "stage": self.stage,
            "t": self.t0,
        })
        return self

    def __exit__(self, exc_type, exc, tb):
        if not _ENABLED:
            return
        if self.gpu_sync:
            cuda_sync()
        t1 = time.perf_counter()
        _log_event({
            "type": "stage_end",
            "job_id": self.job_id,
            "stage": self.stage,
            "t": t1,
            "duration_sec": round(t1 - self.t0, 3),
            "ok": exc is None,
            "error": str(exc) if exc else None,
            **_gpu_snapshot(),
        })


# ── ResourceSampler ───────────────────────────────────────────────────────────

class ResourceSampler:
    """Ayrı thread ile sistem kaynaklarını örnekler (PROFILE=1 gerekir).

    CPU %, RAM, disk I/O, GPU util, VRAM değerlerini ``interval`` saniyede bir
    JSONL dosyasına yazar.

    Kullanım::

        sampler = ResourceSampler(job_id, interval=2.0)
        sampler.start()
        run_pipeline(...)
        sampler.stop()
    """

    def __init__(self, job_id: str, interval: float = 2.0):
        self.job_id = job_id
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if not _ENABLED:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"profiler-sampler-{self.job_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        if not _ENABLED:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self):
        try:
            import psutil
        except ImportError:
            return  # psutil yoksa örnekleme yapılamaz

        disk_base = psutil.disk_io_counters()
        base_read = disk_base.read_bytes if disk_base else 0
        base_write = disk_base.write_bytes if disk_base else 0

        while not self._stop_event.wait(timeout=self.interval):
            try:
                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=None)
                disk = psutil.disk_io_counters()
                disk_read = (disk.read_bytes - base_read) if disk else 0
                disk_write = (disk.write_bytes - base_write) if disk else 0

                event: dict = {
                    "type": "resource_sample",
                    "job_id": self.job_id,
                    "t": time.perf_counter(),
                    "cpu_percent": cpu,
                    "ram_used_gb": round(mem.used / (1024 ** 3), 2),
                    "ram_total_gb": round(mem.total / (1024 ** 3), 2),
                    "disk_read_mb": round(disk_read / (1024 ** 2), 2),
                    "disk_write_mb": round(disk_write / (1024 ** 2), 2),
                }
                event.update(_gpu_snapshot())
                _log_event(event)
            except Exception:
                pass  # örnekleme hatası pipeline'ı durdurmamalı


# ── Özet raporu ───────────────────────────────────────────────────────────────

def summarize(job_id: str | None = None) -> dict:
    """profile_log.jsonl dosyasını okuyup özet istatistik döndür.

    Args:
        job_id: Belirtilirse yalnızca o iş için özet hesaplar.

    Returns::

        {
          "stages": {"STAGE_NAME": {"duration_sec": ..., "ok": ...}},
          "resources": {"cpu_max": ..., "ram_max_gb": ...,
                        "gpu_max": ..., "vram_max_gb": ...}
        }
    """
    if not PROFILE_LOG.exists():
        return {}

    starts: dict[str, float] = {}
    stages: dict[str, dict] = {}
    cpu_vals, ram_vals, gpu_vals, vram_vals = [], [], [], []

    with PROFILE_LOG.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            if job_id and ev.get("job_id") != job_id:
                continue

            ev_type = ev.get("type")
            if ev_type == "stage_start":
                starts[ev["stage"]] = ev["t"]
            elif ev_type == "stage_end":
                stages[ev["stage"]] = {
                    "duration_sec": ev.get("duration_sec"),
                    "ok": ev.get("ok", True),
                    "error": ev.get("error"),
                }
            elif ev_type == "resource_sample":
                if "cpu_percent" in ev:
                    cpu_vals.append(ev["cpu_percent"])
                if "ram_used_gb" in ev:
                    ram_vals.append(ev["ram_used_gb"])
                if "gpu_util_percent" in ev:
                    gpu_vals.append(ev["gpu_util_percent"])
                if "vram_used_gb" in ev:
                    vram_vals.append(ev["vram_used_gb"])

    resources: dict = {}
    if cpu_vals:
        resources["cpu_avg"] = round(sum(cpu_vals) / len(cpu_vals), 1)
        resources["cpu_max"] = max(cpu_vals)
    if ram_vals:
        resources["ram_avg_gb"] = round(sum(ram_vals) / len(ram_vals), 2)
        resources["ram_max_gb"] = max(ram_vals)
    if gpu_vals:
        resources["gpu_avg"] = round(sum(gpu_vals) / len(gpu_vals), 1)
        resources["gpu_max"] = max(gpu_vals)
    if vram_vals:
        resources["vram_avg_gb"] = round(sum(vram_vals) / len(vram_vals), 2)
        resources["vram_max_gb"] = max(vram_vals)

    return {"stages": stages, "resources": resources}
