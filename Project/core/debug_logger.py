"""debug_logger.py — Pipeline debug log writer.

Adım adım detaylı pipeline log'u oluşturur.
Her stage için başlangıç, detay, başarı/başarısızlık bilgisi yazar.
"""

from datetime import datetime
from pathlib import Path


class DebugLogger:
    """Pipeline debug log writer — adım adım detaylı log."""

    SEP_THICK = "=" * 80
    SEP_THIN  = "─" * 80

    def __init__(self, log_path: str):
        self.path   = Path(log_path)
        self._lines: list[str] = []

    # ──────────────────────────────────────────────────────────────────────────
    def header(self, video_path: str, profile_name: str, system_info: dict = None):
        """Log başlığını yaz."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        si  = system_info or {}
        cpu = si.get("cpu", "?")
        gpu = si.get("gpu", "?")
        ram = si.get("ram", "?")

        self._lines += [
            self.SEP_THICK,
            "  VİTOS DEBUG LOG",
            f"  Dosya    : {Path(video_path).name}",
            f"  Profil   : {profile_name}",
            f"  Tarih    : {now}",
            f"  Sistem   : CPU: {cpu} | GPU: {gpu} | RAM: {ram}",
            self.SEP_THICK,
            "",
            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] ▶ PIPELINE START",
            f"  Video    : {video_path}",
            f"  Profil   : {profile_name}",
            "",
        ]

    # ──────────────────────────────────────────────────────────────────────────
    def stage_start(self, stage_num: int, total_stages: int, stage_name: str):
        """Stage başlangıcını logla."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._lines += [
            self.SEP_THIN,
            f"[{ts}] [{stage_num}/{total_stages}] {stage_name}",
            self.SEP_THIN,
        ]

    # ──────────────────────────────────────────────────────────────────────────
    def stage_detail(self, key: str, value):
        """Stage içi detay satırı ekle."""
        self._lines.append(f"  {key:<18}: {value}")

    # ──────────────────────────────────────────────────────────────────────────
    def stage_ok(self, stage_name: str, elapsed_sec: float, **details):
        """Stage başarılı tamamlandı."""
        for k, v in details.items():
            self._lines.append(f"  {k:<18}: {v}")
        self._lines.append(f"  ✅ OK ({elapsed_sec:.1f}s)")
        self._lines.append("")

    # ──────────────────────────────────────────────────────────────────────────
    def stage_fail(self, stage_name: str, elapsed_sec: float,
                   error: str, traceback_str: str = None, **details):
        """Stage başarısız — sebep ve traceback dahil."""
        for k, v in details.items():
            self._lines.append(f"  {k:<18}: {v}")
        self._lines.append(f"  Hata detayı : {error}")
        if traceback_str:
            # Son 5 satırı al
            tb_lines = [l for l in traceback_str.strip().splitlines() if l.strip()]
            for tl in tb_lines[-5:]:
                self._lines.append(f"  | {tl}")
        self._lines.append(f"  ❌ FAIL ({elapsed_sec:.1f}s) — {error}")
        self._lines.append("")

    # ──────────────────────────────────────────────────────────────────────────
    def stage_skip(self, stage_name: str, reason: str = ""):
        """Stage atlandı."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        note = f" — {reason}" if reason else ""
        self._lines.append(f"  -- SKIPPED{note}")
        self._lines.append("")

    # ──────────────────────────────────────────────────────────────────────────
    def footer(self, total_sec: float, stages_ok: int, stages_fail: int,
               output_paths: dict = None):
        """Log sonunu yaz."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        total_stages = stages_ok + stages_fail
        speed = ""
        self._lines += [
            self.SEP_THIN,
            f"[{ts}] ■ PIPELINE END",
            self.SEP_THIN,
            f"  Toplam süre : {total_sec:.1f}s ({_fmt_duration(total_sec)})",
            f"  Başarılı    : {stages_ok}/{total_stages} stage",
        ]
        if stages_fail:
            self._lines.append(f"  Başarısız   : {stages_fail}/{total_stages} stage")
        if output_paths:
            self._lines.append("")
            self._lines.append("  Çıktılar:")
            for label, path in (output_paths or {}).items():
                self._lines.append(f"    {label:<12}: {path}")
        self._lines.append(self.SEP_THICK)

    # ──────────────────────────────────────────────────────────────────────────
    def flush(self):
        """Tüm log satırlarını dosyaya yaz."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("\n".join(self._lines))
                f.write("\n")
        except Exception as exc:
            print(f"[DebugLogger] flush hatası: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
def _fmt_duration(sec: float) -> str:
    """Saniyeyi M:SS formatına çevir."""
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
