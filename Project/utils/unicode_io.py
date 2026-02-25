"""unicode_io.py — OpenCV Türkçe yol desteği.

OpenCV'nin bazı ortamlarda (özellikle headless CI/sunucu) import sırasında
dinamik kütüphane hatası vermesi mümkündür. Bu nedenle import işlemini ihtiyaç
anına taşırız; böylece modülün kendisi güvenle import edilebilir.
"""

from pathlib import Path


def _load_cv2_np():
    """OpenCV ve NumPy'ı gerektiğinde yükler; yoksa (None, None) döner."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        return cv2, np
    except Exception:
        return None, None


def imread_unicode(filepath, flags=None):
    cv2, np = _load_cv2_np()
    if cv2 is None or np is None:
        return None

    if flags is None:
        flags = cv2.IMREAD_COLOR

    try:
        return cv2.imdecode(np.fromfile(str(filepath), dtype=np.uint8), flags)
    except Exception:
        return None


def imwrite_unicode(filepath, img, params=None):
    cv2, _ = _load_cv2_np()
    if cv2 is None:
        return False

    try:
        ext = Path(str(filepath)).suffix
        if not ext:
            ext = ".png"

        ok, buf = cv2.imencode(ext, img, params) if params else cv2.imencode(ext, img)
        if ok:
            buf.tofile(str(filepath))
            return True
    except Exception:
        pass

    return False
