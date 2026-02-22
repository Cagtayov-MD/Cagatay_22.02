"""text_filter.py — yazı adayı filtresi.

Amaç: OCR'a girmeden önce “yazı olma ihtimali düşük” frame'leri ele.

v2.2:
- Mod profilleri (light/medium/heavy) ile eşikleri tek yerden yönet.
- Adaptif eşik: içerik çok "zor" ise (düşük textness) eşiği otomatik yumuşat.
- Fallback'te entry/exit ayrı quota korunur.
"""

import cv2
import numpy as np
from typing import Optional

from utils.unicode_io import imread_unicode


PROFILE_PRESETS = {
    # Daha az frame → daha hızlı
    "light":  {"threshold": 0.30, "mser_min_boxes": 4, "max_per_segment": 70},
    # Varsayılan
    "medium": {"threshold": 0.25, "mser_min_boxes": 3, "max_per_segment": 110},
    # Daha çok frame → daha iyi yakalama
    "heavy":  {"threshold": 0.20, "mser_min_boxes": 2, "max_per_segment": 160},
}


class TextFilter:
    def __init__(self, threshold=0.25, mser_min_boxes: int = 3,
                 adaptive: bool = True, max_per_segment: int = 110):
        self.threshold = float(threshold)
        self.mser_min_boxes = int(mser_min_boxes)
        self.adaptive = bool(adaptive)
        self.max_per_segment = int(max_per_segment)

    @classmethod
    def from_config(cls, cfg: dict):
        cfg = cfg or {}
        profile = (cfg.get("difficulty") or cfg.get("profile") or cfg.get("level") or "medium")
        profile = str(profile).strip().lower()
        base = PROFILE_PRESETS.get(profile, PROFILE_PRESETS["medium"]).copy()
        # override ile ince ayar
        base["threshold"] = float(cfg.get("text_filter_threshold", base["threshold"]))
        base["mser_min_boxes"] = int(cfg.get("text_filter_mser_min_boxes", base["mser_min_boxes"]))
        base["max_per_segment"] = int(cfg.get("text_filter_max_per_segment", base["max_per_segment"]))
        adaptive = bool(cfg.get("text_filter_adaptive", True))
        return cls(threshold=base["threshold"], mser_min_boxes=base["mser_min_boxes"],
                   adaptive=adaptive, max_per_segment=base["max_per_segment"])

    def filter_frames(self, frames: list[dict]) -> list[dict]:
        """İki aşamalı filtre: textness + (gerekirse) MSER teyidi."""
        scored = []
        for f in frames:
            img = imread_unicode(f["path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            score = self._textness(img)
            scored.append((score, f, img))

        if not scored:
            return []

        # Adaptif eşik: eğer tüm skorlar düşükse eşiği bir tık yumuşat.
        thr = self.threshold
        if self.adaptive:
            scores = np.array([s for s, _, _ in scored], dtype=np.float32)
            p80 = float(np.percentile(scores, 80))
            # Çok zor içerik: yüksek percentil bile düşükse, eşiği düşür.
            if p80 < thr:
                thr = max(0.12, p80 * 0.85)

        candidates = []
        for score, f, img in scored:
            if score < thr:
                continue
            has_text, bcount = self._mser_check(img, min_boxes=self.mser_min_boxes)
            if not has_text and score < 0.50:
                continue
            f["textness"] = round(float(score), 3)
            f["bbox_count"] = int(bcount)
            f["difficulty"] = self._difficulty(img)
            candidates.append(f)
        return candidates

    def fallback_filter(self, entry_frames: list[dict], exit_frames: list[dict],
                        max_per_segment: Optional[int] = None) -> list[dict]:
        """
        TextFilter boş dönünce: entry ve exit'ten eşit sayıda frame al.
        Rastgele 200 yerine her segmentten max_per_segment.
        """
        fallback = []
        limit = int(max_per_segment or self.max_per_segment)
        for frames in (entry_frames, exit_frames):
            # Eşit aralıkla örnekle (uniform sampling)
            step = max(1, len(frames) // max(limit, 1))
            selected = frames[::step][:limit]
            for f in selected:
                f["textness"] = 0.3
                f["difficulty"] = "hard"
                f["bbox_count"] = 0
            fallback.extend(selected)
        return fallback

    def _textness(self, gray):
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        edge_d = np.count_nonzero(edges) / (h * w)
        sob = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        horiz = np.mean(np.abs(sob)) / 255.0
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        m = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = 0
        for c in cnts:
            _, _, bw_, bh_ = cv2.boundingRect(c)
            if bh_ > 0 and 2 < bw_ / bh_ < 30 and 100 < bw_ * bh_ < h * w * 0.3:
                blobs += 1
        return 0.35*min(edge_d/0.15, 1) + 0.30*min(horiz/0.05, 1) + 0.35*min(blobs/15, 1)

    def _mser_check(self, gray, min_boxes: int = 3):
        try:
            mser = cv2.MSER_create()
            mser.setMinArea(60)
            mser.setMaxArea(14400)
            sc = 640 / max(gray.shape[1], 1)
            sm = cv2.resize(gray, None, fx=min(sc, 1), fy=min(sc, 1)) if sc < 1 else gray
            _, bboxes = mser.detectRegions(sm)
            if bboxes is None or len(bboxes) == 0:
                return False, 0
            tb = sum(1 for x, y, w, h in bboxes if 0.1 < w / max(h, 1) < 15 and h > 5)
            return tb >= int(min_boxes), tb
        except Exception:
            return True, 0  # Güvenli taraf: frame'i atma

    def _difficulty(self, gray):
        mn, sd = np.mean(gray), np.std(gray)
        contrast = sd / max(mn, 1)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if contrast > 0.6 and blur > 500:
            return "easy"
        elif contrast > 0.3 or blur > 200:
            return "medium"
        return "hard"
