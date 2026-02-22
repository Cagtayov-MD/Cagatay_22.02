"""unicode_io.py — OpenCV Türkçe yol desteği."""
import cv2, numpy as np
from pathlib import Path

def imread_unicode(filepath, flags=cv2.IMREAD_COLOR):
    try:
        return cv2.imdecode(np.fromfile(str(filepath), dtype=np.uint8), flags)
    except Exception:
        return None

def imwrite_unicode(filepath, img, params=None):
    try:
        ext = Path(str(filepath)).suffix
        ok, buf = cv2.imencode(ext, img, params) if params else cv2.imencode(ext, img)
        if ok: buf.tofile(str(filepath)); return True
    except Exception: pass
    return False
