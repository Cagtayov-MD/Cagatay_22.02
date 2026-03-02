"""
frame_preprocessor.py — Çoklu ROI crop + upscale ön işleme.

Amaç:
    Frame'leri OCR öncesinde ön işlemden geçirerek okunabilirliği artırmak.
    Birden fazla ROI (Region of Interest) bölgesi deneyerek yazının
    nerede olursa olsun yakalanmasını sağlamak.

Stratejiler:
    1. Çoklu ROI crop: Alt bölge (credits), orta bölge, üst bölge (başlık)
    2. Upscale: Küçük yazıyı büyüterek okunabilirliği artırma (3x Lanczos)
    3. %20 güven payı (padding): Crop sınırlarını genişleterek kaymayı tolere etme
    4. Fallback: Crop sonucu şüpheliyse raw frame'e düşme

ROI tanımları (frame yüksekliği oranları):
    - CREDITS_ROI: y=0.30..1.00 (alt %70 — genişletilmiş B crop)
    - SUBTITLE_ROI: y=0.80..1.00 (en alt %20 — altyazı şeridi)
    - HEADER_ROI: y=0.00..0.30 (üst %30 — başlık/logo)
"""

import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# ROI tanımları: (y_start_ratio, y_end_ratio, label)
# %20 güven payı (padding) zaten sınırlara dahil edilmiş
DEFAULT_ROIS = [
    {"name": "credits",  "y_start": 0.30, "y_end": 1.00, "priority": 1},
    {"name": "subtitle", "y_start": 0.80, "y_end": 1.00, "priority": 3},
    {"name": "header",   "y_start": 0.00, "y_end": 0.30, "priority": 2},
]

# Crop3x = Upscale 3 katına, interpolation=Lanczos
DEFAULT_UPSCALE_FACTOR = 3
# cv2.INTER_LANCZOS4 evaluated lazily to avoid ImportError when cv2 is absent
_DEFAULT_INTERPOLATION_ID = 4  # cv2.INTER_LANCZOS4 == 4


@dataclass
class ROICrop:
    """Tek bir ROI crop sonucu."""
    name: str               # ROI adı: "credits", "subtitle", "header"
    image: object           # Crop + upscale edilmiş görüntü (np.ndarray)
    y_start_px: int         # Orijinal frame'deki başlangıç y (pixel)
    y_end_px: int           # Orijinal frame'deki bitiş y (pixel)
    upscale_factor: int     # Uygulanan upscale faktörü
    temp_path: str = ""     # Geçici dosya yolu (OCR için)
    priority: int = 1       # Düşük = daha öncelikli


@dataclass
class PreprocessResult:
    """Frame ön işleme sonucu."""
    frame_path: str
    crops: list = field(default_factory=list)
    raw_path: str = ""      # Ham frame'in yolu (fallback için)


class FramePreprocessor:
    """
    Çoklu ROI crop + upscale ön işleyici.

    Config parametreleri:
        preprocess_rois: list[dict]  — ROI tanımları
        preprocess_upscale: int      — Upscale faktörü (default: 3)
        preprocess_enabled: bool     — Ön işleme aktif mi (default: True)
        preprocess_padding: float    — Crop güven payı oranı (default: 0.20)
    """

    def __init__(self, cfg: dict = None, log_cb=None):
        self.cfg = cfg or {}
        self._log = log_cb or (lambda m: None)

        self.enabled = bool(self.cfg.get("preprocess_enabled", True))
        self.upscale = int(self.cfg.get("preprocess_upscale", DEFAULT_UPSCALE_FACTOR))
        self.padding = float(self.cfg.get("preprocess_padding", 0.20))
        self.rois = self.cfg.get("preprocess_rois", DEFAULT_ROIS)
        self._temp_dir = None

    def _get_temp_dir(self, work_dir: str = "") -> str:
        """Geçici dosyalar için dizin."""
        if work_dir:
            d = os.path.join(work_dir, "crops_vlm")
            os.makedirs(d, exist_ok=True)
            return d
        if not self._temp_dir:
            self._temp_dir = tempfile.mkdtemp(prefix="vlm_crops_")
        return self._temp_dir

    def preprocess_frame(self, frame_path: str, work_dir: str = "") -> PreprocessResult:
        """
        Tek frame'i çoklu ROI ile crop + upscale et.

        Returns:
            PreprocessResult: crop listesi + raw frame yolu
        """
        result = PreprocessResult(frame_path=frame_path, raw_path=frame_path)

        if not self.enabled:
            return result

        if not HAS_CV2:
            return result

        # Deferred import: avoids circular/missing dependency when cv2 is absent
        from utils.unicode_io import imread_unicode
        img = imread_unicode(frame_path)
        if img is None:
            return result

        h, w = img.shape[:2]
        temp_dir = self._get_temp_dir(work_dir)
        stem = Path(frame_path).stem

        for roi_def in self.rois:
            name = roi_def["name"]
            y_start_ratio = roi_def["y_start"]
            y_end_ratio = roi_def["y_end"]
            priority = roi_def.get("priority", 1)

            # Padding uygula (crop yüksekliğinin %20'si)
            crop_height_ratio = y_end_ratio - y_start_ratio
            pad = crop_height_ratio * self.padding

            y_start = max(0.0, y_start_ratio - pad)
            y_end = min(1.0, y_end_ratio + pad)

            y1 = int(h * y_start)
            y2 = int(h * y_end)

            # Minimum crop boyutu kontrolü
            if (y2 - y1) < 20:
                continue

            crop = img[y1:y2, 0:w].copy()

            # Upscale (Lanczos)
            if self.upscale > 1:
                new_w = crop.shape[1] * self.upscale
                new_h = crop.shape[0] * self.upscale
                interp = (cv2.INTER_LANCZOS4 if HAS_CV2
                          else _DEFAULT_INTERPOLATION_ID)
                crop = cv2.resize(crop, (new_w, new_h),
                                  interpolation=interp)

            # Geçici dosyaya yaz
            crop_path = os.path.join(temp_dir, f"{stem}_{name}_crop{self.upscale}x.jpg")
            ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok or buf is None:
                continue
            buf.tofile(crop_path)

            result.crops.append(ROICrop(
                name=name,
                image=crop,
                y_start_px=y1,
                y_end_px=y2,
                upscale_factor=self.upscale,
                temp_path=crop_path,
                priority=priority,
            ))

        return result

    def preprocess_frames(self, frame_list: list,
                          work_dir: str = "",
                          log_cb=None) -> list:
        """Birden fazla frame'i toplu ön işle."""
        cb = log_cb or self._log
        results = []
        total = len(frame_list)

        for i, frame_info in enumerate(frame_list):
            if i % 20 == 0:
                cb(f"  🖼️ Preprocess: {i+1}/{total}")
            path = frame_info.get("path", "")
            if not path or not Path(path).exists():
                continue
            result = self.preprocess_frame(path, work_dir)
            results.append(result)

        cb(f"  🖼️ Preprocess tamamlandı: {len(results)} frame, "
           f"{sum(len(r.crops) for r in results)} crop")
        return results
