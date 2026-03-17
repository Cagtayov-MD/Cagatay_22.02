"""
oneocr_engine.py — Windows 11 Snipping Tool OCR motoru.

OCREngine / QwenOCREngine ile birebir aynı arayüz:
    engine = OneOCREngine(cfg=config, log_cb=log, name_db=name_db)
    ocr_lines, layout_pairs = engine.process_frames(candidate_frames)

Motor: Windows 11 Snipping Tool AI OCR (oneocr paketi)
Avantajlar:
    - PaddleOCR'dan çok daha iyi Latin karakter tanıma
    - Word düzeyinde confidence skoru (PaddleOCR'da yok)
    - Word düzeyinde 4 noktalı polygon bbox
    - Birleşik kelime sorunu yok (JACKKOSSLYN → JACK KOSSLYN)
    - Türkçe karakter tanıma (ğ, ü, ö, ş, ç, ı) daha iyi
    - CPU tabanlı — GPU gerektirmez
    - Preprocessing varyantı gerektirmez — tek okuma yeterli

Kısıtlar:
    - Sadece Windows 11 (Snipping Tool DLL gerekli)
    - Cursive/el yazısı fontlarda hala hatalı (ama PaddleOCR'dan iyi)
    - Cursive için Qwen VLM fallback önerilir

Kullanım:
    content_profiles.json'da "ocr_engine": "oneocr" ile etkinleşir.
    FilmDiziONEOCR profili bu motoru kullanır.

Gereksinimler:
    pip install oneocr
    Windows 11 + güncel Snipping Tool (Microsoft Store)
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import oneocr
    HAS_ONEOCR = True
except ImportError:
    HAS_ONEOCR = False

from core.ocr_engine import OCRLine  # Aynı veri yapısını kullan

# ── Sabitler ──────────────────────────────────────────────────
WATERMARK_THRESHOLD = 15    # Bu kadar frame'de geçen metin → watermark
MIN_TEXT_LEN = 2            # 2 karakterden kısa → atla
MIN_CONFIDENCE = 0.30       # Çok düşük confidence → atla
BLACKLIST_PATTERNS = [
    r"^hd$", r"^sd$", r"^4k$", r"^uhd$",
    r"^yeni$", r"^canli$", r"^canlı$", r"^live$",
    r"^bolum$", r"^\d+\s*\.?\s*bolum$", r"^episode\s*\d*$",
    r"^www\.", r"^http", r"\.com$", r"\.tr$",
    r"^\d{1,2}:\d{2}$",
    r"^\d{1,2}\.\d{2}\.\d{4}$",
    r"^altyazı", r"^subtitle",
]
_BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS]

# ── Türkçe normalizasyon ─────────────────────────────────────
from utils.turkish import normalize_for_dedup as _normalize


def _is_blacklisted(text: str) -> bool:
    t = text.strip()
    return any(r.search(t) for r in _BLACKLIST_RE)


class OneOCREngine:
    """
    Windows 11 Snipping Tool OCR motoru.
    OCREngine / QwenOCREngine ile aynı process_frames() arayüzü.
    """

    def __init__(self, cfg: dict = None, log_cb=None, name_db=None):
        if not HAS_ONEOCR:
            raise ImportError(
                "oneocr kurulu değil! pip install oneocr\n"
                "Windows 11 + güncel Snipping Tool gereklidir."
            )

        self.cfg = cfg or {}
        self._log = log_cb or (lambda m: None)
        self._name_db = name_db
        self._model = oneocr.OcrEngine()
        self._log(f"  [OneOCR] Windows Snipping Tool OCR motoru başlatıldı")

    # ── Ana arayüz — OCREngine ile aynı imza ──────────────────
    def process_frames(self,
                       candidate_frames: list,
                       log_callback=None) -> tuple:
        """
        Döndürür: (ocr_lines: list[OCRLine], layout_pairs: list)

        oneocr word düzeyinde bbox veriyor — layout analizi
        pipeline_runner'da mevcut LayoutAnalyzer ile yapılabilir.
        """
        cb = log_callback or self._log
        total = len(candidate_frames)

        cb(f"  [OneOCR] {total} frame işlenecek")

        # Frame başına ham sonuçları topla
        raw_groups: dict[str, list] = {}
        all_layout_data: list = []  # Layout analizi için bbox verileri

        t0 = time.time()
        error_count = 0

        for i, frame_info in enumerate(candidate_frames):
            if i % 10 == 0 or i == total - 1:
                elapsed = time.time() - t0
                eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
                cb(f"  [OneOCR] {i+1}/{total} | geçen:{elapsed:.0f}s | kalan:~{eta:.0f}s")

            frame_path = frame_info.get("path") or frame_info.get("frame_path", "")
            timecode = float(frame_info.get("timecode_sec", 0.0))

            if not frame_path or not Path(frame_path).exists():
                continue

            try:
                lines_data = self._read_frame(frame_path)
            except Exception as e:
                error_count += 1
                if error_count <= 3:
                    cb(f"  [OneOCR] Frame {i} hatası: {e}")
                continue

            for line_info in lines_data:
                text = line_info["text"].strip()
                if not text or len(text) < MIN_TEXT_LEN:
                    continue
                if _is_blacklisted(text):
                    continue

                norm = _normalize(text)
                if norm not in raw_groups:
                    raw_groups[norm] = []
                raw_groups[norm].append({
                    "text": text,
                    "confidence": line_info["confidence"],
                    "bbox": line_info["bbox"],
                    "words": line_info.get("words", []),
                    "timecode": timecode,
                    "frame_path": frame_path,
                })

                # Layout analizi için bbox verisi topla
                if line_info.get("words"):
                    all_layout_data.append({
                        "frame_path": frame_path,
                        "timecode": timecode,
                        "words": line_info["words"],
                        "line_text": text,
                    })

        if error_count > 0:
            cb(f"  [OneOCR] {error_count}/{total} frame'de hata oluştu")

        # ── Dedup + çoğunluk oyu + confidence birleştirme ─────
        ocr_lines = []
        for norm, instances in raw_groups.items():
            if not instances:
                continue

            # Watermark guard
            if len(instances) >= WATERMARK_THRESHOLD:
                continue

            # Çoğunluk oyu: hangi yazım biçimi daha çok çıktı?
            vote: dict[str, int] = {}
            conf_sum: dict[str, float] = {}
            for inst in instances:
                t = inst["text"]
                vote[t] = vote.get(t, 0) + 1
                conf_sum[t] = conf_sum.get(t, 0.0) + inst["confidence"]

            best_text = max(vote, key=vote.__getitem__)
            seen_count = len(instances)

            # Ortalama confidence (gerçek oneocr skoru)
            avg_conf = round(conf_sum[best_text] / vote[best_text], 3)

            # Tek seferlik görüntü → confidence penalty
            if seen_count == 1:
                avg_conf = round(avg_conf * 0.80, 3)

            # Düşük confidence filtresi
            if avg_conf < MIN_CONFIDENCE:
                continue

            # En iyi bbox'u al (en yüksek confidence'lı instance'dan)
            best_inst = max(
                [inst for inst in instances if inst["text"] == best_text],
                key=lambda x: x["confidence"]
            )

            # NameDB onarımı
            if self._name_db:
                fixed = self._name_db.correct_line(best_text)
                if fixed != best_text:
                    cb(f"    [NameDB] '{best_text}' → '{fixed}'")
                    best_text = fixed

            first_inst = min(instances, key=lambda x: x["timecode"])

            ocr_lines.append(OCRLine(
                text=best_text,
                first_seen=first_inst["timecode"],
                last_seen=max(x["timecode"] for x in instances),
                seen_count=seen_count,
                avg_confidence=avg_conf,
                bbox=best_inst["bbox"],
                frame_path=first_inst["frame_path"],
                source="oneocr",
            ))

        ocr_lines.sort(key=lambda l: l.first_seen)

        # ── Layout pairs (oneocr bbox'lardan) ─────────────────
        layout_pairs = self._extract_layout_pairs(all_layout_data, cb)

        elapsed_total = time.time() - t0
        cb(f"  [OneOCR] Tamamlandı: {len(ocr_lines)} satır, "
           f"{len(layout_pairs)} layout pair ({elapsed_total:.1f}s)")

        return ocr_lines, layout_pairs

    # ── Tek frame okuma ────────────────────────────────────────
    def _read_frame(self, frame_path: str) -> list[dict]:
        """
        Frame'i oneocr ile oku.

        Döndürür: [
            {
                "text": "KATHLEEN HERBERT",
                "confidence": 0.987,
                "bbox": [x1, y1, x2, y2],
                "words": [{"text": "KATHLEEN", "confidence": 0.99, "bbox": [...]}, ...]
            }, ...
        ]
        """
        # cv2 ile oku (unicode yol desteği)
        if HAS_CV2 and HAS_NUMPY:
            img = cv2.imread(str(frame_path))
            if img is None:
                # Unicode path fallback
                img = cv2.imdecode(
                    np.fromfile(str(frame_path), dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
            if img is None:
                return []
            result = self._model.recognize_cv2(img)
        else:
            # PIL fallback
            try:
                from PIL import Image
                img = Image.open(str(frame_path))
                result = self._model.recognize_pil(img)
            except Exception:
                return []

        if not result or not result.get("lines"):
            return []

        lines_out = []
        for line in result["lines"]:
            line_text = line.get("text", "").strip()
            if not line_text:
                continue

            # Line bbox (4 noktalı polygon → [x1,y1,x2,y2])
            lbbox = line.get("bounding_rect", {})
            line_bbox = self._polygon_to_rect(lbbox)

            # Word düzeyinde veri
            words = []
            word_confs = []
            for word in line.get("words", []):
                w_text = word.get("text", "").strip()
                w_conf = float(word.get("confidence", 0.0))
                w_bbox_raw = word.get("bounding_rect", {})
                w_bbox = self._polygon_to_rect(w_bbox_raw)

                if w_text:
                    words.append({
                        "text": w_text,
                        "confidence": round(w_conf, 4),
                        "bbox": w_bbox,
                    })
                    word_confs.append(w_conf)

            # Line confidence = word confidence'ların ortalaması
            line_conf = (
                round(sum(word_confs) / len(word_confs), 4)
                if word_confs else 0.0
            )

            lines_out.append({
                "text": line_text,
                "confidence": line_conf,
                "bbox": line_bbox,
                "words": words,
            })

        return lines_out

    # ── Layout pair çıkarımı (bbox'lardan) ─────────────────────
    def _extract_layout_pairs(self, layout_data: list, cb) -> list:
        """
        oneocr word bbox'larından iki sütunlu layout tespiti.

        Jeneriklerde yaygın format:
            Sol sütun: karakter adı (veya rol)
            Sağ sütun: oyuncu adı

        oneocr line düzeyinde zaten birleştirme yapıyor,
        ama iki sütunlu yapıda sol ve sağ ayrı line olabilir.
        Burada aynı y-seviyesindeki sol/sağ satırları eşleştiriyoruz.
        """
        try:
            from core.layout_analyzer import LayoutAnalyzer, CastPair
        except ImportError:
            return []

        if not layout_data:
            return []

        # Frame bazlı gruplama
        frame_groups: dict[str, list] = {}
        for item in layout_data:
            fp = item["frame_path"]
            if fp not in frame_groups:
                frame_groups[fp] = []
            frame_groups[fp].append(item)

        all_pairs = []
        seen_pairs = set()

        for frame_path, items in frame_groups.items():
            # Bu frame'deki tüm word'leri topla
            frame_words = []
            for item in items:
                tc = item["timecode"]
                for w in item.get("words", []):
                    if w["bbox"]:
                        frame_words.append({
                            "text": w["text"],
                            "bbox": w["bbox"],
                            "confidence": w["confidence"],
                        })

            if len(frame_words) < 2:
                continue

            # LayoutAnalyzer'a gönder
            analyzer = LayoutAnalyzer()
            layout = analyzer.analyze_frame_results(frame_words)
            for pair in layout.get("pairs", []):
                key = (
                    pair.character_name.lower().strip(),
                    pair.actor_name.lower().strip()
                )
                if key not in seen_pairs and key[0] and key[1]:
                    seen_pairs.add(key)
                    # Timecode ekle
                    pair.timecode_sec = items[0]["timecode"] if items else 0.0
                    all_pairs.append(pair)

        if all_pairs:
            cb(f"  [OneOCR] Layout: {len(all_pairs)} karakter↔oyuncu eşleşmesi")

        return all_pairs

    # ── Font tipi tahmini (bbox geometrisinden) ────────────────
    def estimate_font_type(self, frame_path: str) -> str:
        """
        Tek frame'in font tipini tahmin et.
        oneocr bbox geometrisi + edge density analizi.

        Returns: "standard" | "handwriting" | "decorative" | "unknown"
        """
        if not HAS_CV2 or not HAS_NUMPY:
            return "unknown"

        try:
            img = cv2.imread(str(frame_path))
            if img is None:
                return "unknown"

            lines_data = self._read_frame(frame_path)
            if not lines_data:
                return "unknown"

            heights = []
            edge_densities = []

            for line in lines_data:
                bbox = line.get("bbox", [])
                if not bbox or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox
                h_val = y2 - y1
                if h_val < 4:
                    continue
                heights.append(float(h_val))

                # Edge density
                region = img[int(y1):int(y2), int(x1):int(x2)]
                if region.size == 0:
                    continue
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                density = float(np.count_nonzero(edges)) / float(edges.size)
                edge_densities.append(density)

            if len(heights) < 2:
                return "unknown"

            h_arr = np.array(heights, dtype=float)
            h_mean = float(np.mean(h_arr))
            h_cv = float(np.std(h_arr) / h_mean) if h_mean > 0 else 0.0
            edge_density = float(np.mean(edge_densities)) if edge_densities else 0.0

            if h_cv < 0.15 and edge_density < 0.25:
                return "standard"
            if h_cv > 0.35 or edge_density > 0.40:
                return "handwriting"
            return "decorative"

        except Exception:
            return "unknown"

    # ── Yardımcı: polygon bbox → rect ──────────────────────────
    @staticmethod
    def _polygon_to_rect(bbox_data) -> list:
        """
        oneocr bbox formatı → [x1, y1, x2, y2]

        oneocr formatı (4 noktalı polygon):
            {'x1': 503.0, 'y1': 296.0, 'x2': 978.0, 'y2': 296.0,
             'x3': 978.0, 'y3': 331.0, 'x4': 503.0, 'y4': 331.0}

        veya winocr formatı (rect):
            {'x': 503, 'y': 296, 'width': 475, 'height': 35}
        """
        if not bbox_data:
            return []

        try:
            # oneocr 4 noktalı polygon formatı
            if 'x1' in bbox_data:
                x_vals = [bbox_data.get('x1', 0), bbox_data.get('x2', 0),
                          bbox_data.get('x3', 0), bbox_data.get('x4', 0)]
                y_vals = [bbox_data.get('y1', 0), bbox_data.get('y2', 0),
                          bbox_data.get('y3', 0), bbox_data.get('y4', 0)]
                return [
                    round(min(x_vals)),
                    round(min(y_vals)),
                    round(max(x_vals)),
                    round(max(y_vals)),
                ]

            # winocr rect formatı (fallback)
            if 'x' in bbox_data and 'width' in bbox_data:
                x = bbox_data['x']
                y = bbox_data['y']
                return [
                    round(x),
                    round(y),
                    round(x + bbox_data['width']),
                    round(y + bbox_data['height']),
                ]
        except Exception:
            pass

        return []
