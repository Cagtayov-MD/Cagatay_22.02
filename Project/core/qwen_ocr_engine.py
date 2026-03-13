"""
qwen_ocr_engine.py — Qwen2.5-VL tabanlı OCR motoru.

OCREngine ile birebir aynı arayüz:
    engine = QwenOCREngine(cfg=config, log_cb=log, name_db=name_db)
    ocr_lines, layout_pairs = engine.process_frames(candidate_frames)

Fark:
    PaddleOCR yerine Qwen2.5-VL (Ollama) ile okur.
    Her frame için "bu jenerik karesinde ne yazıyor, rolleri neler?" diye sorar.
    Confidence yoktur — VLM çıktısı seen_count ile ağırlıklandırılır.
    layout_pairs boş döner (bbox bilgisi yok).

Kullanım:
    content_profiles.json içinde "ocr_engine": "qwen" olan profillerde devreye girer.
    FilmDizi profili etkilenmez.
"""

import base64
import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

from core._ollama_url import normalize_ollama_url
from core.ocr_engine import OCRLine  # Aynı veri yapısını kullan

# ── Sabitler ──────────────────────────────────────────────────────
OLLAMA_API_URL   = "http://localhost:11434/api/chat"
DEFAULT_MODEL    = "qwen2.5vl:7b"
REQUEST_TIMEOUT  = 60   # VLM yavaş olabilir
VLM_CONFIDENCE   = 0.92  # Heuristic — VLM gerçek conf üretmez

# Qwen'e gönderilecek prompt — jenerik okuma için optimize edildi
PROMPT_JENERIK = (
    "Bu görüntü bir Türkçe film veya dizi jeneriğinden alınmış bir karedir. "
    "Görüntüdeki tüm metni satır satır oku. "
    "Her satırı olduğu gibi yaz — fazladan açıklama, etiket veya yorum ekleme. "
    "Türkçe karakterleri doğru kullan: Ş ş Ğ ğ Ü ü Ö ö Ç ç İ ı. "
    "Görüntüde metin yoksa hiçbir şey yazma."
)

# Yanıt temizleme: thinking blokları ve kontrol token'ları
_STRIP = re.compile(
    r"<think>.*?</think>"
    r"|<\|.*?\|>"
    r"|\[INST\].*?\[/INST\]"
    r"|<<SYS>>.*?<</SYS>>",
    re.DOTALL
)

# Anlamsız satırları filtrele (OCREngine blacklist'iyle uyumlu)
_BLACKLIST = [
    r"^hd$", r"^sd$", r"^4k$",
    r"^www\.", r"^http",
    r"^\d{1,2}:\d{2}$",
    r"^altyazı", r"^subtitle",
]
_BL_RE = [re.compile(p, re.IGNORECASE) for p in _BLACKLIST]


def _is_blacklisted(text: str) -> bool:
    t = text.strip()
    return any(r.match(t) for r in _BL_RE)


class QwenOCREngine:
    """
    Qwen2.5-VL tabanlı OCR motoru.
    OCREngine ile aynı process_frames() arayüzü.
    """

    def __init__(self,
                 cfg: dict = None,
                 log_cb=None,
                 name_db=None,
                 ollama_url: str = OLLAMA_API_URL):
        self.cfg      = cfg or {}
        self._log     = log_cb or (lambda m: None)
        self._name_db = name_db
        self.model    = (
            self.cfg.get("vlm_model") or
            self.cfg.get("qwen_model") or
            DEFAULT_MODEL
        )
        _base      = normalize_ollama_url(ollama_url)
        self.url   = f"{_base}/api/chat"
        self._base = _base
        self._available = None

    # ── Ollama bağlantı kontrolü ───────────────────────────────────
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(f"{self._base}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data   = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                base   = self.model.split(":")[0]
                self._available = any(base in m for m in models)
        except Exception:
            self._available = False
        return self._available

    # ── Ana arayüz — OCREngine ile aynı imza ──────────────────────
    def process_frames(self,
                       candidate_frames: list,
                       log_callback=None) -> tuple:
        """
        Döndürür: (ocr_lines: list[OCRLine], layout_pairs: list)
        layout_pairs her zaman boştur — VLM bbox üretmez.
        """
        cb    = log_callback or self._log
        total = len(candidate_frames)

        if not self.is_available():
            cb(f"  [QwenOCR] !! Ollama bağlantısı yok veya model yüklü değil: {self.model}")
            cb(f"  [QwenOCR] Ollama çalışıyor mu? Model kurulu mu?  →  ollama pull {self.model}")
            return [], []

        cb(f"  [QwenOCR] Model: {self.model} | {total} frame işlenecek")

        # Frame başına ham metin topla — dedup için tüm instance'ları sakla
        # key: normalize edilmiş metin → list[dict]  (text, timecode, frame_path)
        raw_groups: dict[str, list] = {}

        t0 = time.time()
        for i, frame_info in enumerate(candidate_frames):
            if i % 5 == 0 or i == total - 1:
                elapsed = time.time() - t0
                eta     = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
                cb(f"  [QwenOCR] {i+1}/{total} | geçen:{elapsed:.0f}s | kalan:~{eta:.0f}s")

            frame_path = frame_info.get("path") or frame_info.get("frame_path", "")
            timecode   = float(frame_info.get("timecode_sec", 0.0))

            if not frame_path or not Path(frame_path).exists():
                continue

            lines = self._read_frame(frame_path)
            for line_text in lines:
                line_text = line_text.strip()
                if not line_text:
                    continue
                if len(line_text) < 2:
                    continue
                if _is_blacklisted(line_text):
                    continue

                norm = self._normalize(line_text)
                if norm not in raw_groups:
                    raw_groups[norm] = []
                raw_groups[norm].append({
                    "text":       line_text,
                    "timecode":   timecode,
                    "frame_path": frame_path,
                })

        # ── Dedup + çoğunluk oyu ──────────────────────────────────
        # Aynı normalize metni için en çok oy alan yazım biçimini seç
        ocr_lines = []
        for norm, instances in raw_groups.items():
            if not instances:
                continue

            # Watermark guard: çok fazla frame'de geçiyorsa logo/watermark olabilir
            if len(instances) >= 15:
                continue

            # Çoğunluk oyu: hangi yazım biçimi daha çok çıktı?
            vote: dict[str, int] = {}
            for inst in instances:
                t = inst["text"]
                vote[t] = vote.get(t, 0) + 1

            best_text  = max(vote, key=vote.__getitem__)
            seen_count = len(instances)

            # Tek görüntüde görüldüyse daha düşük güven
            confidence = VLM_CONFIDENCE if seen_count >= 2 else round(VLM_CONFIDENCE * 0.75, 3)

            # NameDB onarımı (isteğe bağlı)
            if self._name_db:
                fixed = self._name_db.correct_line(best_text)
                if fixed != best_text:
                    cb(f"    [NameDB] '{best_text}' → '{fixed}'")
                    best_text = fixed

            first_inst = min(instances, key=lambda x: x["timecode"])

            ocr_lines.append(OCRLine(
                text          = best_text,
                first_seen    = first_inst["timecode"],
                last_seen     = max(x["timecode"] for x in instances),
                seen_count    = seen_count,
                avg_confidence= confidence,
                bbox          = [],
                frame_path    = first_inst["frame_path"],
                source        = "qwen_vlm",
            ))

        ocr_lines.sort(key=lambda l: l.first_seen)

        elapsed_total = time.time() - t0
        cb(f"  [QwenOCR] Tamamlandı: {len(ocr_lines)} satır ({elapsed_total:.1f}s)")

        return ocr_lines, []   # layout_pairs boş

    # ── Tek frame okuma ───────────────────────────────────────────
    def _read_frame(self, frame_path: str) -> list[str]:
        """Frame'i Qwen'e gönder, satır listesi olarak al."""
        try:
            img_b64 = self._encode_image(frame_path)
            if not img_b64:
                return []

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role":    "user",
                        "content": PROMPT_JENERIK,
                        "images":  [img_b64],
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 300,
                },
            }

            data = json.dumps(payload).encode("utf-8")
            req  = urllib.request.Request(
                self.url,
                data    = data,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                response = json.loads(resp.read())

            raw = (
                response.get("message", {})
                        .get("content", "")
                        .strip()
            )

            # Kontrol token'larını temizle
            raw = _STRIP.sub("", raw).strip()

            if not raw:
                return []

            return [line.strip() for line in raw.splitlines() if line.strip()]

        except urllib.error.URLError:
            return []
        except Exception:
            return []

    # ── Yardımcı ──────────────────────────────────────────────────
    @staticmethod
    def _encode_image(frame_path: str) -> str | None:
        try:
            with open(frame_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

    @staticmethod
    def _normalize(text: str) -> str:
        """Dedup için normalize et — büyük/küçük harf + Türkçe karakter agnostik."""
        _TR = str.maketrans(
            "çğışöüÇĞİŞÖÜ",
            "cgisouCGISOu"   # basit ASCII eşleme
        )
        return re.sub(r"\s+", " ", text.lower().translate(_TR)).strip()
