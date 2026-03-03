"""
time_utils.py — Ortak zaman formatlama yardimci fonksiyonlari.
"""


def fmt_hms(seconds: float, with_ms: bool = True) -> str:
    """Saniyeyi HH:MM:SS.mmm formatina cevir."""
    total_ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    if with_ms:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"
