"""path_resolver.py — Sabit araç yolları.

Kullanıcı seçimi kaldırıldı. Tüm araç yolları sabitlendi:
  FFmpeg/FFprobe : F:\\Source\\ffmpeg\\bin
  Google API JSON: F:\\Project\\config\\video-analiz-sistemi-a59996f04788.json
  LOGOLAR        : F:\\Source\\Logo
"""
import shutil
from pathlib import Path

from config.runtime_paths import FFMPEG_BIN_DIR, GOOGLE_KEYS_JSON, LOGOLAR_DIR

class PathResolver:
    """
    Tüm araç yollarını sabit konumlardan çözer.
    Hiçbir seçim penceresi açmaz — yollar kodda tanımlı.
    """

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

        # Çözülen yollar
        self.ffmpeg:    str = ""
        self.ffprobe:   str = ""
        self.tesseract: str = ""
        self.tessdata:  str = ""
        self.logolar:   str = ""
        self.google_json: str = ""

    # ------------------------------------------------------------------
    def resolve_all(self) -> bool:
        self._resolve_ffmpeg()
        self._resolve_ffprobe()
        self._resolve_tesseract()
        self._resolve_logolar()
        self._resolve_google_json()
        return len(self.errors) == 0

    # ------------------------------------------------------------------
    def _resolve_ffmpeg(self):
        candidates = [
            Path(FFMPEG_BIN_DIR) / "ffmpeg.exe",
            Path(FFMPEG_BIN_DIR) / "ffmpeg",        # Linux/Mac uyumu
        ]
        for c in candidates:
            if c.exists():
                self.ffmpeg = str(c)
                return
        # PATH fallback
        found = shutil.which("ffmpeg")
        if found:
            self.ffmpeg = found
            self.warnings.append(f"ffmpeg PATH'ten bulundu: {found}")
        else:
            self.errors.append(
                f"ffmpeg bulunamadı! Beklenen: {FFMPEG_BIN_DIR}"
            )

    def _resolve_ffprobe(self):
        if self.ffmpeg:
            p = Path(self.ffmpeg).parent / "ffprobe.exe"
            if p.exists():
                self.ffprobe = str(p)
                return
            p2 = Path(self.ffmpeg).parent / "ffprobe"
            if p2.exists():
                self.ffprobe = str(p2)
                return
        found = shutil.which("ffprobe")
        if found:
            self.ffprobe = found
        else:
            self.warnings.append("ffprobe bulunamadı (ffmpeg ile birlikte gelmeli).")

    def _resolve_tesseract(self):
        """Tesseract opsiyonel — PaddleOCR birincil."""
        common = [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
        for c in common:
            if c.exists():
                self.tesseract = str(c)
                td = c.parent / "tessdata"
                self.tessdata = str(td) if td.is_dir() else ""
                return
        found = shutil.which("tesseract")
        if found:
            self.tesseract = found
        # Tesseract yoksa uyarı yok — PaddleOCR kullanılıyor

    def _resolve_logolar(self):
        p = Path(LOGOLAR_DIR)
        if p.is_dir():
            self.logolar = str(p)
        else:
            self.warnings.append(f"LOGOLAR dizini yok: {LOGOLAR_DIR}")

    def _resolve_google_json(self):
        p = Path(GOOGLE_KEYS_JSON)
        if p.is_file():
            self.google_json = str(p)
        else:
            self.warnings.append(f"Google API JSON yok: {GOOGLE_KEYS_JSON}")

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            "═══ Path Resolver ═══",
            f"  FFmpeg     : {self.ffmpeg or '❌'}",
            f"  FFprobe    : {self.ffprobe or '⚠️ yok'}",
            f"  Tesseract  : {self.tesseract or '⚠️ yok (PaddleOCR birincil)'}",
            f"  LOGOLAR    : {self.logolar or '⚠️ yok'}",
            f"  Google JSON: {self.google_json or '⚠️ yok'}",
        ]
        for e in self.errors:
            lines.append(f"  ❌ {e}")
        for w in self.warnings:
            lines.append(f"  ⚠️  {w}")
        return "\n".join(lines)
