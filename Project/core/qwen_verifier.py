"""
qwen_verifier.py — Qwen3-VL ile OCR sonrası doğrulama.

Amaç:
    PaddleOCR'ın düşük confidence'lı satırlarını Qwen3-VL'ye göndererek
    doğrulat veya düzelt. Özellikle Türkçe özel karakter bozulmalarını
    (Ş→S, ğ→g, ü→u vb.) ve el yazısı font hatalarını yakalar.

Kullanım:
    verifier = QwenVerifier(model="qwen3-vl:8b", confidence_threshold=0.40)
    ocr_lines = verifier.verify(ocr_lines, log_cb=self._log)

Gereksinim:
    - Ollama kurulu ve çalışıyor olmalı
    - qwen3-vl:8b modeli indirilmiş olmalı (ollama pull qwen3-vl:8b)

Değişiklikler (Qwen3_ocr_test branch):
    - DEFAULT_THRESHOLD: 0.80 → 0.40
      El yazısı/kaligrafi fontlu eski film jeneriği frameleri için
      PaddleOCR düşük confidence verse bile Qwen devreye girer.
    - num_predict: 50 → 100 (el yazısı okuma için daha fazla token)
    - REQUEST_TIMEOUT: 30 → 60 saniye
"""

import base64
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

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
DEFAULT_MODEL   = "qwen3-vl:8b"
DEFAULT_THRESHOLD = 0.40   # DEĞİŞTİRİLDİ: 0.80 → 0.40 (el yazısı font desteği)
REQUEST_TIMEOUT = 60        # DEĞİŞTİRİLDİ: 30 → 60 saniye

PROMPT_TEMPLATE = (
    "Bu görüntüde gördüğün Türkçe metni, özellikle kişi adlarını oku. "
    "Şu metin OCR tarafından okundu: '{ocr_text}' "
    "Eğer bu metin doğruysa sadece aynı metni yaz. "
    "Eğer yanlışsa düzeltilmiş halini yaz. "
    "Sadece metin yaz, açıklama ekleme. Türkçe karakterleri (Ş,ş,Ğ,ğ,Ü,ü,Ö,ö,Ç,ç,İ,ı) doğru kullan."
)

PROMPT_TEMPLATE_CROP = (
    "Bu kırpılmış görüntüde tam olarak ne yazıyor? "
    "El yazısı veya kaligrafi font olabilir, dikkatlice oku. "
    "OCR şunu okudu: '{ocr_text}' "
    "Doğruysa aynısını yaz. Yanlışsa düzeltilmiş halini yaz. "
    "Sadece metin yaz, açıklama ekleme. Türkçe karakterleri (Ş,ş,Ğ,ğ,Ü,ü,Ö,ö,Ç,ç,İ,ı) doğru kullan."
)


@dataclass
class VerifyResult:
    original: str
    corrected: str
    was_fixed: bool
    confidence_before: float


class QwenVerifier:
    """
    Qwen3-VL ile OCR sonrası doğrulama motoru.
    Sadece düşük confidence'lı satırları işler → performans kaybı minimal.
    """

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 confidence_threshold: float = DEFAULT_THRESHOLD,
                 ollama_url: str = OLLAMA_API_URL,
                 enabled: bool = True,
                 name_checker=None):
        self.model = model
        self.threshold = confidence_threshold
        self.url = ollama_url
        self.enabled = enabled
        self.name_checker = name_checker
        self._available = None  # lazy check

    def is_available(self) -> bool:
        """Ollama çalışıyor mu ve model var mı? (ilk çağrıda kontrol eder)"""
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
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

    def verify(self, ocr_lines: list, log_cb=None, resolution: str = "") -> list:
        """
        OCRLine listesini al, şüphelileri Qwen ile doğrula.
        Döndürür: güncellenmiş ocr_lines (dict listesi veya OCRLine listesi)
        """
        log = log_cb or (lambda m: None)

        if not self.enabled:
            return ocr_lines

        if not self.is_available():
            log(f"  [Qwen] Ollama bağlantısı yok veya '{self.model}' yüklü değil — atlanıyor")
            return ocr_lines

        # Çözünürlüğe göre gürültü threshold'u
        noise_threshold = 0.25  # DEĞİŞTİRİLDİ: 0.60 → 0.25 (el yazısı için daha toleranslı)
        if resolution:
            try:
                # "384x288" veya "1920x1080" formatında
                height = int(resolution.split("x")[-1])
                if height <= 360:
                    noise_threshold = 0.15
                    log(f"  [Qwen] Düşük çözünürlük ({resolution}) — gürültü eşiği: {noise_threshold}")
                elif height <= 480:
                    noise_threshold = 0.20
            except (ValueError, IndexError):
                pass

        # Şüpheli satırları belirle — conf + metin kalitesi bazlı
        suspicious = []
        skipped_noise = 0
        for i, line in enumerate(ocr_lines):
            conf = self._get_conf(line)
            text = self._get_text(line)

            # conf < noise_threshold: zaten gürültü, Qwen de düzeltemez → atla
            if conf < noise_threshold:
                skipped_noise += 1
                continue

            # Metin kaliteli mi? (Türkçe karakter, düzgün büyük/küçük harf)
            if conf >= self.threshold and self._is_quality_text(text):
                if not self._should_force_verify_name_like_text(text):
                    continue  # güvenilir, dokunma

            # conf >= threshold ama kalitesiz metin → Qwen'e gönder (DYAYUCE gibi durumlar)
            # veya conf noise_threshold-threshold arası → Qwen'e gönder
            suspicious.append(i)

        if skipped_noise:
            log(f"  [Qwen] {skipped_noise} gürültü satırı atlandı (conf<{noise_threshold})")

        if not suspicious:
            log(f"  [Qwen] Tüm satırlar kaliteli — doğrulama gerekmedi")
            return ocr_lines

        log(f"  [Qwen] {len(suspicious)}/{len(ocr_lines)} satır doğrulanıyor...")

        fixed_count = 0
        for i in suspicious:
            line = ocr_lines[i]
            text = self._get_text(line)
            frame_path = self._get_frame_path(line)
            bbox = self._get_bbox(line)

            if not text or not frame_path or not Path(frame_path).exists():
                continue

            result = self._verify_single(text, frame_path, bbox=bbox)
            if result and result.was_fixed:
                self._set_text(ocr_lines[i], result.corrected)
                # Orijinal metni sakla
                self._set_original(ocr_lines[i], result.original)
                fixed_count += 1
                log(f"  [Qwen] ✓ '{result.original}' → '{result.corrected}'")

        log(f"  [Qwen] {fixed_count} satır düzeltildi")
        return ocr_lines

    def _should_force_verify_name_like_text(self, text: str) -> bool:
        """
        Yüksek confidence satırlar için ek güvenlik:
        Kişi adı gibi görünen satır NameDB'de yoksa Qwen doğrulamasına zorla gönder.
        """
        if not self.name_checker:
            return False
        if not self._looks_like_person_name(text):
            return False
        try:
            return not bool(self.name_checker(text))
        except Exception:
            return False

    @staticmethod
    def _looks_like_person_name(text: str) -> bool:
        words = [w for w in text.strip().split() if w]
        if len(words) < 2:
            return False
        for w in words:
            if not w[0].isalpha() or not w[0].isupper():
                return False
            tail = w[1:]
            if tail and not all(ch.isalpha() or ch in "'-" for ch in tail):
                return False
        return True

    def _is_quality_text(self, text: str) -> bool:
        """
        Metin kaliteli mi? Kaliteliyse Qwen'e gönderme.
        Kalitesizse (DYAYUCE gibi) Qwen'e gönder.
        """
        if not text or len(text) < 2:
            return False

        words = text.split()

        # Boşluksuz 6+ karakter tamamen büyük harf → birleşik/bozuk → kalitesiz
        if ' ' not in text and text.isupper() and len(text) >= 6:
            return False

        # Türkçe özel karakter + Title Case veya çok kelimeli → kaliteli
        _turk = set("çğıöşüÇĞİÖŞÜ")
        has_turk = any(c in _turk for c in text)

        if has_turk:
            # Türkçe karakter VAR ama tek kelime ve kısa (< 4 harf) → şüpheli
            if len(words) == 1 and len(text) < 4:
                return False
            # Türkçe karakter + düzgün kelime yapısı → kaliteli
            return True

        # Tüm kelimeler Title Case → düzgün yazım → kaliteli
        if len(words) >= 2 and all(len(w) >= 2 and w[0].isupper() and w[1:].islower() for w in words):
            return True

        # Tek kelime Title Case ve 4+ harf → muhtemelen isim → kaliteli
        if len(words) == 1 and len(text) >= 4 and text[0].isupper() and text[1:].islower():
            return True

        # Tamamen büyük harf çok kelimeli → şüpheli
        if len(words) >= 2 and all(w.isupper() for w in words):
            return False

        # Karışık karakter (rakam, nokta vb.) → kalitesiz
        alpha_ratio = sum(1 for c in text if c.isalpha() or c == ' ') / max(len(text), 1)
        if alpha_ratio < 0.80:
            return False

        return True

    def _verify_single(self, ocr_text: str, frame_path: str,
                       bbox: list = None) -> VerifyResult | None:
        """
        Tek satırı Qwen'e gönder, cevabı al.
        bbox varsa → frame'den ilgili bölgeyi kırp, sadece crop gönder.
        bbox yoksa → tüm frame gönderilir (eski davranış).
        """
        try:
            img_b64 = self._encode_image_with_crop(frame_path, bbox)
            if not img_b64:
                return None

            # bbox crop varsa daha odaklı prompt kullan
            if bbox and len(bbox) == 4 and HAS_CV2 and HAS_NUMPY:
                prompt = PROMPT_TEMPLATE_CROP.format(ocr_text=ocr_text)
            else:
                prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)

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
                    "temperature": 0.0,   # deterministik
                    "num_predict": 100,   # DEĞİŞTİRİLDİ: 50 → 100 (el yazısı için)
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

            corrected = (
                response.get("message", {})
                        .get("content", "")
                        .strip()
            )

            # Thinking modunda <think>...</think> bloğunu temizle
            import re
            corrected = re.sub(r"<think>.*?</think>", "", corrected, flags=re.DOTALL).strip()

            if not corrected:
                return None

            # Çok uzun cevap → güvenilmez, atla
            if len(corrected) > len(ocr_text) * 3:
                return None

            was_fixed = corrected.lower().strip() != ocr_text.lower().strip()
            return VerifyResult(
                original=ocr_text,
                corrected=corrected,
                was_fixed=was_fixed,
                confidence_before=0.0
            )

        except urllib.error.URLError:
            return None
        except Exception:
            return None

    def _encode_image_with_crop(self, frame_path: str,
                               bbox: list = None) -> str | None:
        """
        Görüntüyü base64'e çevir.
        bbox varsa → frame'den ilgili bölgeyi kırp + %30 padding ekle.
        bbox yoksa → tüm frame'i gönder (eski davranış).

        Crop avantajı: 1920x1080 frame'de 15px yüksekliğindeki isim satırını
        Qwen aramak zorunda kalmıyor — direkt ilgili bölgeyi görüyor.
        Padding (%30): bağlam korunsun (üst/alt satırdaki ipuçları).
        """
        try:
            if bbox and len(bbox) == 4 and HAS_CV2 and HAS_NUMPY:
                img = cv2.imread(frame_path)
                if img is not None:
                    h, w = img.shape[:2]
                    x1, y1, x2, y2 = bbox

                    # Padding: bbox boyutunun %30'u kadar her yöne genişlet
                    bw = x2 - x1
                    bh = y2 - y1
                    pad_x = int(bw * 0.30)
                    pad_y = int(bh * 0.30)

                    # Minimum padding — çok küçük bbox'larda bağlam kaybolmasın
                    pad_x = max(pad_x, 20)
                    pad_y = max(pad_y, 15)

                    cx1 = max(0, int(x1 - pad_x))
                    cy1 = max(0, int(y1 - pad_y))
                    cx2 = min(w, int(x2 + pad_x))
                    cy2 = min(h, int(y2 + pad_y))

                    # Crop'un geçerli boyutta olduğundan emin ol
                    if (cx2 - cx1) > 10 and (cy2 - cy1) > 5:
                        crop = img[cy1:cy2, cx1:cx2]

                        # Crop çok küçükse 2x upscale (Qwen daha iyi okur)
                        crop_h, crop_w = crop.shape[:2]
                        if crop_h < 40:
                            scale = max(2, 80 // crop_h)
                            crop = cv2.resize(
                                crop, (crop_w * scale, crop_h * scale),
                                interpolation=cv2.INTER_CUBIC
                            )

                        _, buf = cv2.imencode('.png', crop)
                        return base64.b64encode(buf.tobytes()).decode("utf-8")

            # Fallback: tüm frame
            with open(frame_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

    def _encode_image(self, frame_path: str) -> str | None:
        """Geriye uyumluluk: tüm frame'i base64'e çevir."""
        return self._encode_image_with_crop(frame_path, bbox=None)

    # ── OCRLine / dict uyumlu getter/setter'lar ────────────────────
    def _get_conf(self, line) -> float:
        if isinstance(line, dict):
            return float(line.get("avg_confidence", line.get("confidence", 1.0)))
        return float(getattr(line, "avg_confidence", getattr(line, "confidence", 1.0)))

    def _get_text(self, line) -> str:
        if isinstance(line, dict):
            return line.get("text", "")
        return getattr(line, "text", "")

    def _get_frame_path(self, line) -> str:
        if isinstance(line, dict):
            return line.get("frame_path", "")
        return getattr(line, "frame_path", "")

    def _get_bbox(self, line) -> list:
        """OCRLine / dict'ten bbox [x1, y1, x2, y2] al."""
        if isinstance(line, dict):
            return line.get("bbox", [])
        return getattr(line, "bbox", [])

    def _set_text(self, line, text: str):
        if isinstance(line, dict):
            line["text"] = text
        else:
            line.text = text

    def _set_original(self, line, text: str):
        if isinstance(line, dict):
            line["text_qwen_original"] = text
        else:
            line.text_qwen_original = text  # dynamic attr