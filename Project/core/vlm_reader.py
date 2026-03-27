"""
vlm_reader.py — VLM tabanlı metin okuma (OCR alternatifi).

Amaç:
    PaddleOCR yerine veya OCR başarısız/düşük güvenli olduğunda
    VLM (GLM-4.6V-Flash vb.) ile frame/crop'tan metin okuma.

Varsayılan olarak KAPALI — config'de ``vlm_ocr_enabled: true`` veya
``use_vlm_for_ocr: true`` ile etkinleştirilir.

Kullanım:
    reader = VLMReader(model="glm4.6v-flash:q4_K_M", enabled=True)
    result = reader.read_text_from_frame("frame.png", bbox=[x1,y1,x2,y2], lang="tr")
    # {"text": "...", "avg_confidence": 0.85, "bbox": [...], "frame_path": "...", "source": "vlm"}
"""

import base64
import json
import re
import urllib.request
import urllib.error
from pathlib import Path

from core._ollama_url import normalize_ollama_url
from utils.unicode_io import imread_unicode

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

# ── Sabitler ──────────────────────────────────────────────────────
OLLAMA_API_URL  = "http://localhost:11434/api/chat"
DEFAULT_MODEL   = "glm4.6v-flash:q4_K_M"
REQUEST_TIMEOUT = 30

PROMPT_READ_TR = (
    "Bu görüntüdeki tüm Türkçe metni satır satır oku ve yaz. "
    "Yalnızca gördüğün metni yaz — başka açıklama, etiket veya önek ekleme. "
    "Türkçe karakterleri (Ş,ş,Ğ,ğ,Ü,ü,Ö,ö,Ç,ç,İ,ı) doğru kullan. "
    "Görüntüde metin yoksa boş satır döndür."
)

PROMPT_READ_EN = (
    "Read all text visible in this image, line by line. "
    "Output only the text — no explanations, labels, or prefixes."
)

# Yanıt temizleme: thinking blokları ve kontrol token'ları
_STRIP_PATTERNS = re.compile(
    r"<think>.*?</think>"
    r"|<\|.*?\|>"
    r"|\[INST\].*?\[/INST\]"
    r"|<<SYS>>.*?<</SYS>>",
    re.DOTALL
)

# Heuristic confidence — VLM çıktısı için sabit değer (OCR gibi gerçek conf yok)
VLM_HEURISTIC_CONFIDENCE = 0.85


class VLMReader:
    """
    VLM tabanlı metin okuma motoru.
    OCR alternatifi veya düşük güvenli OCR satırları için fallback.
    Varsayılan olarak kapalıdır (enabled=False).
    """

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 ollama_url: str = OLLAMA_API_URL,
                 enabled: bool = False):
        self.model   = model
        _base        = normalize_ollama_url(ollama_url)
        self.url     = f"{_base}/api/chat"
        self._base_url = _base
        self.enabled = enabled
        self._available = None  # lazy check

    def is_available(self) -> bool:
        """Ollama çalışıyor mu ve model var mı? (ilk çağrıda kontrol eder)"""
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(
                f"{self._base_url}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                self._available = any(
                    self.model.split(":")[0] in m for m in models
                )
        except Exception:
            self._available = False
        return self._available

    def read_text_from_frame(self,
                             frame_path: str,
                             bbox: list = None,
                             lang: str = "tr") -> dict | None:
        """
        Frame (veya bbox crop) içindeki metni VLM ile oku.

        Returns:
            dict with keys: text, avg_confidence, bbox, frame_path, source
            None on failure.
        """
        if not self.enabled:
            return None

        try:
            img_b64 = self._encode_image(frame_path, bbox)
            if not img_b64:
                return None

            prompt = PROMPT_READ_TR if lang == "tr" else PROMPT_READ_EN
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 200,
                }
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                response = json.loads(resp.read())

            raw = (
                response.get("message", {})
                        .get("content", "")
                        .strip()
            )

            # Kontrol token'larını temizle
            raw = _STRIP_PATTERNS.sub("", raw).strip()

            if not raw:
                return None

            return {
                "text": raw,
                "avg_confidence": VLM_HEURISTIC_CONFIDENCE,
                "bbox": bbox or [],
                "frame_path": str(frame_path),
                "source": "vlm",
            }

        except urllib.error.URLError:
            return None
        except Exception:
            return None

    def augment_ocr_lines(self,
                          ocr_lines: list,
                          frame_paths: list,
                          log_cb=None) -> list:
        """
        OCR sonuçlarını VLM ile tamamla / fallback sağla.

        Boş OCR çıktısı olan frame'ler için VLM'den metin oku ve
        sonuç listesine ekle. Mevcut OCR satırlarına dokunmaz.

        Args:
            ocr_lines:   Mevcut OCR sonuçları (dict listesi)
            frame_paths: Tüm aday frame yolları
            log_cb:      Opsiyonel log callback

        Returns:
            Güncellenmiş ocr_lines (yeni VLM satırları eklenmiş)
        """
        log = log_cb or (lambda m: None)
        if not self.enabled or not frame_paths:
            return ocr_lines

        # Halihazırda OCR sonucu olan frame'leri bul
        covered = {
            (line.get("frame_path") if isinstance(line, dict)
             else getattr(line, "frame_path", ""))
            for line in ocr_lines
        }

        added = 0
        for fp in frame_paths:
            fp_str = str(fp)
            if fp_str in covered:
                continue  # OCR zaten okudu
            if not Path(fp_str).exists():
                continue

            result = self.read_text_from_frame(fp_str, lang="tr")
            if result and result.get("text"):
                # Çok satırlı ise satırlara böl
                for line_text in result["text"].splitlines():
                    line_text = line_text.strip()
                    if line_text:
                        ocr_lines.append({
                            "text": line_text,
                            "avg_confidence": VLM_HEURISTIC_CONFIDENCE,
                            "bbox": [],
                            "frame_path": fp_str,
                            "source": "vlm",
                        })
                        added += 1

        if added:
            log(f"  [VLM-OCR] {added} satır eklendi (VLM fallback)")
        return ocr_lines

    # ── Yardımcı: görüntü kodlama ─────────────────────────────────
    def _encode_image(self, frame_path: str, bbox: list = None) -> str | None:
        """Frame'i (veya bbox crop'unu) base64'e çevir."""
        try:
            if bbox and len(bbox) == 4 and HAS_CV2 and HAS_NUMPY:
                img = imread_unicode(str(frame_path))
                if img is not None:
                    h, w = img.shape[:2]
                    x1, y1, x2, y2 = bbox
                    pad_x = max(int((x2 - x1) * 0.20), 10)
                    pad_y = max(int((y2 - y1) * 0.20), 8)
                    cx1 = max(0, int(x1 - pad_x))
                    cy1 = max(0, int(y1 - pad_y))
                    cx2 = min(w, int(x2 + pad_x))
                    cy2 = min(h, int(y2 + pad_y))
                    if (cx2 - cx1) > 5 and (cy2 - cy1) > 5:
                        crop = img[cy1:cy2, cx1:cx2]
                        _, buf = cv2.imencode(".png", crop)
                        return base64.b64encode(buf.tobytes()).decode("utf-8")

            with open(frame_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None


def strip_vlm_tokens(text: str) -> str:
    """
    VLM çıktısındaki kontrol token'larını ve thinking bloklarını temizle.
    Test/utility fonksiyonu — bağımsız kullanım için.
    """
    return _STRIP_PATTERNS.sub("", text).strip()
