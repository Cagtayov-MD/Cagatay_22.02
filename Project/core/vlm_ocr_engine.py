"""
vlm_ocr_engine.py — VLM tabanlı çoklu ROI OCR motoru.

Amaç:
    Çoklu ROI crop'larını VLM ile OCR edip en iyi sonucu seçmek.
    PaddleOCR'a alternatif veya tamamlayıcı olarak kullanılır.

Akış:
    1. FramePreprocessor ile frame'den çoklu ROI crop üret
    2. Her crop'u VLM'e gönder (Qwen2.5-VL tercihli)
    3. Sonuçları skorla
    4. En iyi sonucu seç
    5. Sonuç şüpheliyse raw frame ile fallback dene

Skorlama kriterleri:
    - Satır sayısı (newline count)
    - Harf sayısı (alpha char count)
    - "........" / "____" gibi credits ayraçları (+puan)
    - "oynayanlar" gibi anahtar kelimeler (+puan)
    - Boş veya çok kısa ise (-puan)
"""

import base64
import json
import re
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass, field

from core.frame_preprocessor import FramePreprocessor, PreprocessResult, ROICrop

# Sabitler
OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_VLM_OCR_MODEL = "qwen2.5vl:7b"
REQUEST_TIMEOUT = 60

# OCR prompt — test konuşmalarında en iyi sonuç veren prompt
VLM_OCR_SYSTEM_PROMPT = (
    "You are an OCR engine. Language is Turkish. "
    "Output ONLY the exact text you can read. "
    "Preserve line breaks. No explanations. "
    "Do not invent missing words. "
    "Pay attention to similar letters: N/W/M, i/l, d/t."
)

# Türkçe kredi yazısı anahtar kelimeleri (skorlama için)
CREDITS_KEYWORDS = {
    "oynayanlar", "oyuncular", "cast", "yönetmen", "yonetmen",
    "senaryo", "müzik", "muzik", "yapımcı", "yapimci",
    "görüntü", "goruntu", "montaj", "kurgu",
}

# Ayraç desenleri (credits formatı)
SEPARATOR_PATTERN = re.compile(r'[.…_\-]{3,}')

# Yanıt temizleme
_STRIP_PATTERNS = re.compile(
    r"<think>.*?</think>"
    r"|<\|.*?\|>"
    r"|\[INST\].*?\[/INST\]"
    r"|<<SYS>>.*?<</SYS>>",
    re.DOTALL
)


@dataclass
class VLMOCRResult:
    """Tek ROI'nin VLM OCR sonucu."""
    roi_name: str
    text: str
    score: float
    elapsed_sec: float
    frame_path: str
    crop_path: str = ""
    is_fallback: bool = False  # raw frame ile mi elde edildi?


@dataclass
class VLMFrameResult:
    """Bir frame'in tüm ROI sonuçları + seçilen en iyi sonuç."""
    frame_path: str
    all_results: list = field(default_factory=list)
    best_result: VLMOCRResult = None

    @property
    def best_text(self) -> str:
        return self.best_result.text if self.best_result else ""


class VLMOCREngine:
    """
    Çoklu ROI + VLM OCR motoru.

    Config parametreleri:
        vlm_ocr_enabled: bool       — VLM OCR aktif mi (default: False)
        vlm_ocr_model: str          — VLM model adı (default: qwen2.5vl:7b)
        vlm_ocr_fallback: bool      — Crop başarısızsa raw frame dene (default: True)
        vlm_ocr_min_score: float    — Minimum kabul skoru (default: 5.0)
        preprocess_enabled: bool    — Crop ön işleme aktif mi (default: True)
        preprocess_upscale: int     — Upscale faktörü (default: 3)
        preprocess_rois: list       — ROI tanımları
    """

    def __init__(self, cfg: dict = None, log_cb=None):
        self.cfg = cfg or {}
        self._log = log_cb or (lambda m: None)

        self.enabled = bool(self.cfg.get("vlm_ocr_enabled", False))
        self.model = (
            self.cfg.get("vlm_ocr_model") or
            self.cfg.get("vlm_model") or
            DEFAULT_VLM_OCR_MODEL
        )
        self.fallback_enabled = bool(self.cfg.get("vlm_ocr_fallback", True))
        self.min_score = float(self.cfg.get("vlm_ocr_min_score", 5.0))
        self.ollama_url = self.cfg.get("ollama_url", OLLAMA_API_URL)

        self._preprocessor = FramePreprocessor(cfg=self.cfg, log_cb=log_cb)
        self._available = None

    def is_available(self) -> bool:
        """Ollama çalışıyor mu ve model var mı?"""
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                base_name = self.model.split(":")[0]
                self._available = any(base_name in m for m in models)
        except Exception:
            self._available = False
        return self._available

    def ocr_single_image(self, image_path: str) -> str:
        """Tek görüntüyü VLM ile OCR et."""
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": VLM_OCR_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": "OCR",
                    "images": [b64],
                },
            ],
            "options": {
                "temperature": 0,
                "top_p": 0.2,
                "num_predict": 220,  # Credits metni için yeterli token sayısı
            },
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.ollama_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                response = json.loads(resp.read())

            raw = response.get("message", {}).get("content", "").strip()
            # Kontrol token'larını temizle
            raw = _STRIP_PATTERNS.sub("", raw).strip()
            return raw
        except Exception:
            return ""

    def score_text(self, text: str) -> float:
        """
        OCR çıktısını skorla.
        Yüksek skor = daha güvenilir credits metni.
        """
        if not text or not text.strip():
            return 0.0

        score = 0.0
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

        # Satır sayısı (1 satır = 2 puan, max 20 puan)
        score += min(len(lines) * 2.0, 20.0)

        # Toplam harf sayısı (her 10 harf = 1 puan, max 15 puan)
        alpha_count = sum(1 for c in text if c.isalpha())
        score += min(alpha_count / 10.0, 15.0)

        # Credits ayraçları (........, ____, ----)
        separator_count = len(SEPARATOR_PATTERN.findall(text))
        score += min(separator_count * 3.0, 15.0)

        # Anahtar kelimeler
        text_lower = text.lower()
        for kw in CREDITS_KEYWORDS:
            if kw in text_lower:
                score += 5.0
                break  # max 1 kez

        # 2+ kelimeli satırlar (isim soyisim) (+2 puan her biri, max 10)
        name_like_lines = sum(
            1 for l in lines
            if len(l.split()) >= 2 and all(
                len(w) > 0 and w[0].isupper() for w in l.split()
            )
        )
        score += min(name_like_lines * 2.0, 10.0)

        # Çok kısa metin cezası
        if len(text.strip()) < 10:
            score *= 0.3

        return round(score, 2)

    def process_frame(self, frame_path: str,
                      work_dir: str = "") -> VLMFrameResult:
        """
        Tek frame'i çoklu ROI + fallback ile OCR et.
        """
        import time

        frame_result = VLMFrameResult(frame_path=frame_path)

        # 1. Çoklu ROI crop + upscale
        preprocess = self._preprocessor.preprocess_frame(frame_path, work_dir)

        # 2. Her crop'u OCR et
        for crop in sorted(preprocess.crops, key=lambda c: c.priority):
            if not crop.temp_path or not Path(crop.temp_path).exists():
                continue

            t0 = time.time()
            text = self.ocr_single_image(crop.temp_path)
            elapsed = time.time() - t0

            result = VLMOCRResult(
                roi_name=crop.name,
                text=text,
                score=self.score_text(text),
                elapsed_sec=round(elapsed, 3),
                frame_path=frame_path,
                crop_path=crop.temp_path,
            )
            frame_result.all_results.append(result)

        # 3. En iyi sonucu seç
        if frame_result.all_results:
            frame_result.best_result = max(
                frame_result.all_results, key=lambda r: r.score
            )

        # 4. Fallback: en iyi skor çok düşükse raw frame dene
        best_score = frame_result.best_result.score if frame_result.best_result else 0
        if self.fallback_enabled and best_score < self.min_score:
            t0 = time.time()
            raw_text = self.ocr_single_image(frame_path)
            elapsed = time.time() - t0

            raw_result = VLMOCRResult(
                roi_name="raw_fallback",
                text=raw_text,
                score=self.score_text(raw_text),
                elapsed_sec=round(elapsed, 3),
                frame_path=frame_path,
                is_fallback=True,
            )
            frame_result.all_results.append(raw_result)

            # Raw daha iyiyse onu seç
            if raw_result.score > best_score:
                frame_result.best_result = raw_result

        return frame_result

    def process_frames(self, candidate_frames: list,
                       work_dir: str = "",
                       log_cb=None) -> list:
        """
        Birden fazla frame'i VLM OCR ile işle.

        Returns:
            list of OCRLine-compatible dicts
        """
        import time

        cb = log_cb or self._log
        if not self.enabled:
            cb("  [VLM-OCR] Devre dışı")
            return []

        if not self.is_available():
            cb(f"  [VLM-OCR] Model '{self.model}' mevcut değil — atlanıyor")
            return []

        cb(f"  [VLM-OCR] {len(candidate_frames)} frame, model={self.model}")

        all_lines = []
        total = len(candidate_frames)
        total_crops_ocrd = 0
        fallback_count = 0

        for i, frame_info in enumerate(candidate_frames):
            if i % 5 == 0:
                cb(f"  🔍 VLM-OCR: {i+1}/{total}")

            path = frame_info.get("path", "")
            timecode = frame_info.get("timecode_sec", 0.0)

            if not path or not Path(path).exists():
                continue

            result = self.process_frame(path, work_dir)
            total_crops_ocrd += len(result.all_results)

            if result.best_result and result.best_result.text:
                if result.best_result.is_fallback:
                    fallback_count += 1

                # Çok satırlı metni satırlara böl
                for line_text in result.best_result.text.splitlines():
                    line_text = line_text.strip()
                    if line_text and len(line_text) >= 2:
                        all_lines.append({
                            "text": line_text,
                            "avg_confidence": 0.85,
                            "first_seen": timecode,
                            "last_seen": timecode,
                            "seen_count": 1,
                            "bbox": [],
                            "frame_path": path,
                            "source": f"vlm_{result.best_result.roi_name}",
                        })

        cb(f"  [VLM-OCR] Tamamlandı: {total_crops_ocrd} crop OCR, "
           f"{len(all_lines)} satır, {fallback_count} fallback")

        return all_lines
