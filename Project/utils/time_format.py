"""Ortak zaman formatlama yardımcıları."""


def fmt_hms(seconds: float) -> str:
    """Saniyeyi HH:MM:SS.mmm formatına çevir."""
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
