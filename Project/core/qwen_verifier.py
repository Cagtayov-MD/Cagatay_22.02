"""
qwen_verifier.py — Model-agnostic VLM ile OCR sonrası doğrulama.

Amaç:
    PaddleOCR'ın düşük confidence'lı satırlarını bir VLM'ye göndererek
    doğrulat veya düzelt. Özellikle Türkçe özel karakter bozulmalarını
    (Ş→S, ğ→g, ü→u vb.) ve el yazısı font hatalarını yakalar.

Desteklenen modeller:
    - glm4.6v-flash:q4_K_M  (varsayılan; ollama pull glm4.6v-flash:q4_K_M)
    - qwen3-vl:8b            (geriye dönük uyumluluk)

Kullanım:
    verifier = QwenVerifier(model="glm4.6v-flash:q4_K_M", confidence_threshold=0.80)
    ocr_lines = verifier.verify(ocr_lines, log_cb=self._log)

Gereksinim:
    - Ollama kurulu ve çalışıyor olmalı
    - Model indirilmiş olmalı (ör. ollama pull glm4.6v-flash:q4_K_M)
"""

import base64
import concurrent.futures
import json
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
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
DEFAULT_THRESHOLD = 0.80   # Bu değerin altı şüpheli
REQUEST_TIMEOUT = 30        # saniye
MAX_VLM_WORKERS = 4         # Paralel batch HTTP iş parçacığı sayısı
MAX_SINGLE_FALLBACKS = 10   # Batch parse başarısız olunca max tek-tek fallback
VLM_MAX_TOTAL_SEC = int(os.environ.get("VLM_MAX_TOTAL_SEC", "180"))  # Toplam zaman bütçesi

PROMPT_TEMPLATE = (
    "Görüntüdeki Türkçe metni — özellikle kişi adlarını — oku. "
    "OCR şunu okudu: '{ocr_text}'. "
    "Eğer doğruysa yalnızca aynı metni yaz. "
    "Yanlışsa yalnızca düzeltilmiş halini yaz. "
    "Başka hiçbir açıklama, etiket veya önek ekleme. "
    "Türkçe karakterleri (Ş,ş,Ğ,ğ,Ü,ü,Ö,ö,Ç,ç,İ,ı) doğru kullan."
)

PROMPT_TEMPLATE_CROP = (
    "Bu kırpılmış görüntüde tam olarak ne yazıyor? "
    "OCR şunu okudu: '{ocr_text}'. "
    "Doğruysa yalnızca aynı metni yaz. Yanlışsa yalnızca düzeltilmiş halini yaz. "
    "Başka hiçbir açıklama, etiket veya önek ekleme. "
    "Türkçe karakterleri (Ş,ş,Ğ,ğ,Ü,ü,Ö,ö,Ç,ç,İ,ı) doğru kullan."
)

BATCH_PROMPT_TEMPLATE = (
    "Görüntüdeki Türkçe metinleri oku ve OCR hatalarını düzelt.\n"
    "OCR şu satırları okudu:\n"
    "{numbered_list}\n"
    "Her satır için: doğruysa aynısını, yanlışsa düzeltilmişini yaz.\n"
    "Sadece '1. metin', '2. metin', ... formatında cevap ver. "
    "Başka açıklama ekleme.\n"
    "Türkçe karakterleri (Ş,ş,Ğ,ğ,Ü,ü,Ö,ö,Ç,ç,İ,ı) doğru kullan."
)

# Yanıt temizleme: <think> blokları ve diğer kontrol token'ları
_STRIP_PATTERNS = re.compile(
    r"<think>.*?</think>"          # Qwen3 / GLM thinking blocks
    r"|<\|.*?\|>"                  # <|assistant|> benzeri kontrol token'ları
    r"|\[INST\].*?\[/INST\]"       # Llama instruction tags
    r"|<<SYS>>.*?<</SYS>>",        # System tags
    re.DOTALL
)


@dataclass
class VerifyResult:
    original: str
    corrected: str
    was_fixed: bool
    confidence_before: float


class QwenVerifier:
    """
    VLM (GLM-4.6V-Flash / Qwen3-VL) ile OCR sonrası doğrulama motoru.
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
        _base = normalize_ollama_url(ollama_url)
        self.url = f"{_base}/api/chat"
        self._base_url = _base
        self.enabled = enabled
        self.name_checker = name_checker
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

    def verify(self, ocr_lines: list, log_cb=None, resolution: str = "") -> list:
        """
        OCRLine listesini al, şüphelileri Qwen ile doğrula.
        Döndürür: güncellenmiş ocr_lines (dict listesi veya OCRLine listesi)
        """
        log = log_cb or (lambda m: None)

        if not self.enabled:
            return ocr_lines

        if not self.is_available():
            log(f"  [VLM] Ollama bağlantısı yok veya '{self.model}' yüklü değil — atlanıyor")
            return ocr_lines

        # Çözünürlüğe göre gürültü threshold'u
        noise_threshold = 0.60  # varsayılan
        if resolution:
            try:
                # "384x288" veya "1920x1080" formatında
                height = int(resolution.split("x")[-1])
                if height <= 360:
                    noise_threshold = 0.50
                    log(f"  [Qwen] Düşük çözünürlük ({resolution}) — gürültü eşiği: {noise_threshold}")
                elif height <= 480:
                    noise_threshold = 0.50
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
            log(f"  [VLM] {skipped_noise} gürültü satırı atlandı (conf<{noise_threshold})")

        if not suspicious:
            log(f"  [VLM] Tüm satırlar kaliteli — doğrulama gerekmedi")
            return ocr_lines

        log(f"  [VLM] {len(suspicious)}/{len(ocr_lines)} satır doğrulanıyor...")

        # Frame bazlı gruplama: aynı frame'den gelen satırlar → tek HTTP çağrısı
        groups = {}
        for i in suspicious:
            line = ocr_lines[i]
            text = self._get_text(line)
            conf = self._get_conf(line)
            frame_path = self._get_frame_path(line)
            bbox = self._get_bbox(line)

            if not text or not frame_path or not Path(frame_path).exists():
                continue

            if frame_path not in groups:
                groups[frame_path] = []
            groups[frame_path].append((i, text, conf, bbox))

        log(f"  [VLM] {len(groups)} frame grubunda batch doğrulama")

        fixed_count = 0
        fallback_count = 0
        fallback_limit_logged = False
        groups_list = list(groups.items())

        _max_total_sec = int(os.environ.get("VLM_MAX_TOTAL_SEC", str(VLM_MAX_TOTAL_SEC)))
        _t_verify_start = time.time()

        def _process_group(idx_frame_group):
            idx, (frame_path, group) = idx_frame_group
            t0 = time.time()
            log(f"  [VLM] Batch {idx+1}/{len(groups_list)} ({len(group)} satır)...")
            result = self._verify_batch(group, frame_path)
            elapsed = time.time() - t0
            if elapsed > 2:
                log(f"  [VLM] Batch {idx+1} tamamlandı ({elapsed:.1f}s)")
            return idx, frame_path, group, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_VLM_WORKERS) as executor:
            futures = {
                executor.submit(_process_group, (idx, item)): idx
                for idx, item in enumerate(groups_list)
            }
            batch_outputs = [None] * len(groups_list)
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, frame_path, group, batch_results = future.result()
                    batch_outputs[idx] = (frame_path, group, batch_results)
                except Exception:
                    pass
                if time.time() - _t_verify_start > _max_total_sec:
                    log(
                        f"  [VLM] ⏱ Zaman bütçesi aşıldı ({_max_total_sec}s) — "
                        "kalan batch'ler iptal ediliyor, kısmi sonuçlar döndürülüyor"
                    )
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break

        for entry in batch_outputs:
            if entry is None:
                continue
            frame_path, group, batch_results = entry
            for grp_idx, (i, text, conf, bbox) in enumerate(group):
                result = batch_results.get(grp_idx)
                if result is None:
                    # Batch parse başarısız → tek tek doğrula (limitli)
                    if fallback_count < MAX_SINGLE_FALLBACKS:
                        result = self._verify_single(text, frame_path, bbox=bbox,
                                                     confidence_before=conf)
                        fallback_count += 1
                    elif not fallback_limit_logged:
                        log(f"  [VLM] Fallback limiti aşıldı (max {MAX_SINGLE_FALLBACKS}), kalan satırlar atlanıyor")
                        fallback_limit_logged = True

                if result and result.was_fixed:
                    self._set_text(ocr_lines[i], result.corrected)
                    # Orijinal metni sakla
                    self._set_original(ocr_lines[i], result.original)
                    fixed_count += 1
                    log(f"  [VLM] ✓ '{result.original}' → '{result.corrected}'")

        log(f"  [VLM] {fixed_count} satır düzeltildi")
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

    def _verify_batch(self, group: list, frame_path: str) -> dict:
        """
        Aynı frame'deki satırları tek HTTP çağrısıyla doğrula.

        group: [(i, text, conf, bbox), ...]  — grup öğeleri (grp_idx = enumerate index'i)
        frame_path: frame dosya yolu (tüm grup için ortak)

        Döndürür: {grp_idx: VerifyResult} — parse edilemeyen satırlar eksik kalır
        (eksik satırlar için caller _verify_single fallback yapar).
        Herhangi bir hata veya HTTP başarısızlığında boş dict döner.
        """
        try:
            img_b64 = self._encode_image(frame_path)
            if not img_b64:
                return {}

            numbered_list = "\n".join(
                f'{idx + 1}. "{item[1]}"'
                for idx, item in enumerate(group)
            )
            prompt = BATCH_PROMPT_TEMPLATE.format(numbered_list=numbered_list)

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": max(200, len(group) * 30),
                },
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                response = json.loads(resp.read())

            content = (
                response.get("message", {}).get("content", "").strip()
            )
            content = _STRIP_PATTERNS.sub("", content).strip()

            if not content:
                return {}

            results = {}
            for line in content.splitlines():
                m = re.match(r"^\s*(\d+)[.)]\s*(.+)$", line.strip())
                if not m:
                    continue
                grp_idx = int(m.group(1)) - 1
                corrected = m.group(2).strip().strip("\"'")
                corrected = _STRIP_PATTERNS.sub("", corrected).strip()
                if not (0 <= grp_idx < len(group)):
                    continue
                _, original, conf, _ = group[grp_idx]
                if not corrected or len(corrected) > len(original) * 3:
                    continue
                was_fixed = corrected.lower().strip() != original.lower().strip()
                results[grp_idx] = VerifyResult(
                    original=original,
                    corrected=corrected,
                    was_fixed=was_fixed,
                    confidence_before=conf,
                )

            return results

        except (urllib.error.URLError, Exception):
            return {}

    def _verify_single(self, ocr_text: str, frame_path: str,
                       bbox: list = None,
                       confidence_before: float = 0.0) -> VerifyResult | None:
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
                    "num_predict": 50,    # kısa cevap yeterli
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

            # Kontrol token'larını ve thinking bloklarını temizle
            corrected = _STRIP_PATTERNS.sub("", corrected).strip()

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
                confidence_before=confidence_before,
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
                img = imread_unicode(frame_path)
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

                        # Crop çok küçükse 2x-8x upscale (Qwen daha iyi okur)
                        crop_h, crop_w = crop.shape[:2]
                        if crop_h < 40:
                            scale = min(max(2, 80 // crop_h), 8)
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
