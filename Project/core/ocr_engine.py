"""
ocr_engine.py — PaddleOCR GPU-powered OCR engine.

v2.1 GÜNCELLEMELER:
- Birleşik isim bölme (Turkish name splitter)
- Sayısal noise filtresi (%55+ digit → drop)
- Kısa anlamsız string filtresi
- Daha iyi watermark tespiti
"""

import cv2
import numpy as np
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

try:
    from rapidfuzz import fuzz
    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False

from utils.unicode_io import imread_unicode
from utils.turkish import split_concatenated_name


# ═══════════════════════════════════════════════════════════════════
# VERİ YAPILARI
# ═══════════════════════════════════════════════════════════════════
@dataclass
class OCRResult:
    text: str
    confidence: float
    timecode_sec: float
    frame_path: str
    bbox: list = field(default_factory=list)
    source: str = "paddleocr"

@dataclass
class OCRLine:
    """Deduplicate edilmiş, temiz bir metin satırı."""
    text: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    seen_count: int = 1
    avg_confidence: float = 0.0
    bbox: list = field(default_factory=list)
    frame_path: str = ""
    source: str = "paddleocr"


# ═══════════════════════════════════════════════════════════════════
# KARA LİSTE (jenerik dışı noise)
# ═══════════════════════════════════════════════════════════════════
BLACKLIST_PATTERNS = [
    r"^hd$", r"^sd$", r"^4k$", r"^uhd$",
    r"^yeni$", r"^canli$", r"^canlı$", r"^live$",
    r"^bolum$", r"^\d+\s*\.?\s*bolum$", r"^episode\s*\d*$",
    r"^previously\s+on$", r"^daha\s+önce$",
    r"^www\.", r"^http", r"\.com$", r"\.tr$",
    r"^\d{1,2}:\d{2}$",
    r"^\d{1,2}\.\d{2}\.\d{4}$",
    r"^altyazı", r"^subtitle",
    r"^yaptirilmistir$",
]
BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS]


# ═══════════════════════════════════════════════════════════════════
# ANA MOTOR
# ═══════════════════════════════════════════════════════════════════
class OCREngine:
    """PaddleOCR GPU engine + agresif dedup + isim bölme."""

    def __init__(self, use_gpu=True, lang="en", cfg=None, log_cb=None, name_db=None):
        if not HAS_PADDLE:
            raise ImportError("PaddleOCR kurulu değil! pip install paddlepaddle paddleocr")

        self.cfg = cfg or {}
        self._log = log_cb or (lambda m: None)
        self._gpu = use_gpu
        self._name_db = name_db   # TurkishNameDB instance — DP-tabanlı 356k isim
        self.ocr = self._init_paddle(use_gpu, lang)

        self.noise_chars = set("|_~^`{}[]<>\\©®™•§¶†‡░▒▓█▄▀")
        self.min_text_len = 2
        self.min_confidence = 0.50
        self.fuzzy_threshold = 82
        self.watermark_threshold = 15
        self.max_digit_ratio = 0.55

    def _init_paddle(self, use_gpu, lang):
        """
        PaddleOCR 2.x / 3.x her ikisinde de çalışan dayanıklı init.

        Strateji:
        1. 3.x arg seti ile başla (text_det_thresh, device, vb.)
        2. Her ValueError/TypeError'da hata mesajından kötü arg'ı çıkar, tekrar dene.
        3. "mutually exclusive" çiftleri akıllıca çöz (öncelik: text_* > det_db_*)
        4. Tüm denemeler başarısız → minimal fallback (sadece lang + device/use_gpu)
        5. set_optimization_level gibi Paddle iç hataları → CPU'ya geç, işe yararsa döndür
        """
        import os

        env_lang = (os.environ.get("OCR_LANG") or "").strip()
        if env_lang:
            lang = env_lang

        raw_ver = (self.cfg.get("ocr_version") or os.environ.get("OCR_VERSION") or "").strip()
        ocr_version = raw_ver if raw_ver.lower() not in ("none", "null", "nil", "") else ""

        l = str(lang or "").strip().lower()
        mapped_lang = "tr" if l in (
        "latin", "tr", "tur", "turkish", "türkçe", "turkce", "tr-tr", "tr_tr"
        ) else (l or "tr")

        # Mutually exclusive çiftler: (tercih edilen, alternatif)
        EXCL_PAIRS = [
            ("text_det_thresh",          "det_db_thresh"),
            ("text_det_box_thresh",      "det_db_box_thresh"),
            ("text_recognition_batch_size", "rec_batch_num"),
            ("use_textline_orientation", "use_angle_cls"),
        ]

        def _parse_bad_arg(msg: str):
            m = re.search(r"Unknown argument[:\s]+([A-Za-z_]\w*)", msg)
            if m: return m.group(1)
            m = re.search(r"unexpected keyword argument ['\"]([A-Za-z_]\w*)['\"]", msg)
            if m: return m.group(1)
            return None

        def _fix_exclusive(msg: str, w: dict) -> bool:
            """Hata mesajındaki mutually exclusive çifti çöz. True → düzeltme yapıldı."""
            # Regex: `param_a` and `param_b` are mutually exclusive
            m = re.search(r"`([^`]+)`\s+and\s+`([^`]+)`\s+are mutually exclusive", msg)
            if m:
                a, b = m.group(1), m.group(2)
            elif "mutually exclusive" in msg:
                # fallback: çiftleri elle tara
                a, b = None, None
                for pa, pb in EXCL_PAIRS:
                    if pa in msg and pb in msg:
                        a, b = pa, pb
                        break
                if not a:
                    return False
            else:
                return False

            # Öncelik: text_* > det_db_*, use_textline_orientation > use_angle_cls
            def _score(k):
                if k == "use_textline_orientation": return 10
                if k.startswith("text_"):            return 9
                if k == "use_angle_cls":             return 1
                return 0  # det_db_*, rec_batch_num, use_gpu vb.

            if a in w and b in w:
                drop = a if _score(a) < _score(b) else b
                w.pop(drop)
                return True
            elif a in w:
                w.pop(a)
                return True
            elif b in w:
                w.pop(b)
                return True
            return False

        def _attempt(w):
            return PaddleOCR(**w)

        def _try_with(w: dict):
            """Verilen kwargs ile en fazla 30 iterasyon dene, argüman budayarak."""
            working = dict(w)
            for _ in range(30):
                try:
                    return _attempt(working)
                except ValueError as e:
                    msg = str(e)
                    bad = _parse_bad_arg(msg)
                    if bad and bad in working:
                        working.pop(bad); continue
                    if _fix_exclusive(msg, working):
                        continue
                    raise
                except TypeError as e:
                    msg = str(e)
                    bad = _parse_bad_arg(msg)
                    if bad and bad in working:
                        working.pop(bad); continue
                    if _fix_exclusive(msg, working):
                        continue
                    raise
            raise RuntimeError("Argüman budama limiti aşıldı")

        # ── 1. Deneme: 3.x arg seti + GPU ──────────────────────────────
        kwargs_3x = {
            "lang":                    mapped_lang,
            "device":                  "gpu:0" if use_gpu else "cpu",
            "show_log":                False,
            "use_textline_orientation": True,
            "text_det_thresh":         0.3,
            "text_det_box_thresh":     0.5,
            "text_recognition_batch_size": 6,
        }
        if ocr_version:
            kwargs_3x["ocr_version"] = ocr_version

        # ── 2. Deneme: 2.x arg seti + GPU ──────────────────────────────
        kwargs_2x = {
            "lang":             mapped_lang,
            "use_gpu":          use_gpu,
            "use_angle_cls":    True,
            "show_log":         False,
            "det_db_thresh":    0.3,
            "det_db_box_thresh": 0.5,
            "rec_batch_num":    6,
        }

        attempts = [
            ("3.x GPU",  kwargs_3x),
            ("2.x GPU",  kwargs_2x),
            ("3.x CPU",  {**kwargs_3x, "device": "cpu"}),
            ("2.x CPU",  {**kwargs_2x, "use_gpu": False}),
            ("minimal",  {"lang": mapped_lang}),
        ]

        gpu_failed = False
        last_err = None
        for label, kw in attempts:
            # GPU denemelerini use_gpu=False ise atla
            if not use_gpu and "GPU" in label:
                continue
            # GPU denemesi başarısız olduysa CPU denemelerine geçildi — uyar
            if gpu_failed and "CPU" in label and label != "minimal":
                self._log(
                    f"\n  {'!'*60}\n"
                    f"  !! DİKKAT: GPU başlatma başarısız — CPU'ya geçildi!\n"
                    f"  !! Olası sebepler:\n"
                    f"  !!   1. 'paddlepaddle' kurulu, 'paddlepaddle-gpu' DEĞİL\n"
                    f"  !!      → pip uninstall paddlepaddle && pip install paddlepaddle-gpu\n"
                    f"  !!   2. CUDA/cuDNN sürüm uyumsuzluğu\n"
                    f"  !!      → python -c \"import paddle; paddle.utils.run_check()\"\n"
                    f"  !!   3. GPU belleği yetersiz\n"
                    f"  {'!'*60}\n"
                )
            try:
                ocr = _try_with(kw)
                is_gpu_attempt = "GPU" in label
                # Gerçek GPU doğrulaması: paddle.device.get_device() ile kontrol
                actual_device = "?"
                try:
                    import paddle
                    actual_device = paddle.device.get_device()
                except Exception:
                    actual_device = "gpu" if is_gpu_attempt else "cpu"

                using_gpu = "gpu" in str(actual_device).lower()
                dev_str = f"GPU ({actual_device})" if using_gpu else f"CPU ({actual_device})"

                if is_gpu_attempt and not using_gpu:
                    # GPU istedik ama CPU ile başladı — bu sessiz düşüş
                    self._log(
                        f"\n  {'!'*60}\n"
                        f"  !! UYARI: GPU istendi ama PaddleOCR CPU ile başlatıldı!\n"
                        f"  !! Gerçek device: {actual_device}\n"
                        f"  !! 'paddlepaddle-gpu' kurulu mu kontrol et:\n"
                        f"  !!   pip show paddlepaddle paddlepaddle-gpu\n"
                        f"  {'!'*60}\n"
                    )
                    self._gpu = False
                    gpu_failed = True
                    continue  # CPU alternatifiyle devam et

                self._log(f"  🔧 PaddleOCR başlatıldı ({label} / {dev_str}) lang:{mapped_lang}")
                if not using_gpu:
                    self._gpu = False
                return ocr
            except Exception as e:
                last_err = e
                if "GPU" in label:
                    gpu_failed = True
                self._log(f"  ⚠️  {label} başarısız: {type(e).__name__}: {str(e)[:120]}")
                continue

        raise RuntimeError(f"PaddleOCR başlatılamadı (tüm denemeler): {last_err}") from last_err

    # ═══════════════════════════════════════════════════════════════
    # ANA İŞLEM
    # ═══════════════════════════════════════════════════════════════
    def process_frames(self, candidate_frames: list[dict],
                       log_callback=None) -> tuple:
        """
        Ana OCR işlemi. Döndürür: (ocr_lines, layout_pairs)
        - ocr_lines: Sıralı, deduplicate edilmiş metin satırları
        - layout_pairs: Layout analizinden gelen karakter↔oyuncu çiftleri
        """
        from core.layout_analyzer import LayoutAnalyzer

        cb = log_callback or self._log
        raw_results: list[OCRResult] = []
        all_layout_pairs = []
        layout_analyzer = LayoutAnalyzer()
        total = len(candidate_frames)
        error_count = 0
        first_errors = []

        for i, frame_info in enumerate(candidate_frames):
            if i % 10 == 0 or i == total - 1:
                cb(f"  🔍 OCR: {i+1}/{total}")
            try:
                results = self._process_single(frame_info)
                raw_results.extend(results)

                # Per-frame layout analizi (bbox'lardan satır/sütun)
                if results:
                    frame_ocr = [
                        {"text": r.text, "bbox": r.bbox, "confidence": r.confidence}
                        for r in results if r.bbox
                    ]
                    if frame_ocr:
                        layout = layout_analyzer.analyze_frame_results(frame_ocr)
                        for pair in layout.get("pairs", []):
                            pair.timecode_sec = frame_info.get("timecode_sec", 0.0)
                            all_layout_pairs.append(pair)

            except Exception as e:
                error_count += 1
                if len(first_errors) < 3:
                    first_errors.append(f"Frame {i}: {e}")

        if error_count > 0:
            cb(f"  ⚠️ OCR hataları: {error_count}/{total} frame")
            for err in first_errors:
                cb(f"    → {err}")
        if total > 0 and error_count / total > 0.5:
            cb(f"  ❌ OCR hata oranı çok yüksek (%{error_count*100//total})!")

        # Layout çiftlerini dedup et
        unique_pairs = self._dedup_layout_pairs(all_layout_pairs)
        if unique_pairs:
            cb(f"  🔗 Layout: {len(unique_pairs)} karakter↔oyuncu eşleşmesi")

        # ═══ 8-AŞAMALI FİLTRELEME ═══
        step1 = self._noise_filter(raw_results)
        step2 = self._length_filter(step1)
        step3 = self._confidence_filter(step2)
        step4 = self._digit_noise_filter(step3)
        step5 = self._blacklist_filter(step4)
        step6 = self._name_split_pass(step5)
        step7 = self._fuzzy_dedup(step6)
        step8 = self._persistence_and_watermark(step7)

        cb(f"  📝 OCR: {len(raw_results)} ham → {len(step8)} temiz satır")
        return step8, unique_pairs

    def _dedup_layout_pairs(self, pairs: list) -> list:
        """Aynı karakter↔oyuncu çiftini tekrar eden frame'lerden temizle."""
        if not pairs:
            return []
        seen = set()
        unique = []
        for p in pairs:
            key = (p.character_name.lower().strip(), p.actor_name.lower().strip())
            if key not in seen and key[0] and key[1]:
                seen.add(key)
                unique.append(p)
        return unique

    # ═══════════════════════════════════════════════════════════════
    # TEK FRAME
    # ═══════════════════════════════════════════════════════════════

    def _run_paddle(self, img):
        """
        PaddleOCR 2.x / 3.x outputlarını tek noktada çalıştır.
        3.x pipeline'da `cls` arg'ı yok (hata: PaddleOCR.predict() got an unexpected keyword argument 'cls').
        """
        o = self.ocr
        if hasattr(o, "ocr"):
            try:
                return o.ocr(img, cls=True)
            except TypeError:
                return o.ocr(img)
            except ValueError as e:
                if "cls" in str(e).lower():
                    return o.ocr(img)
                raise
        if hasattr(o, "predict"):
            return o.predict(img)
        if callable(o):
            return o(img)
        raise RuntimeError("PaddleOCR nesnesinde çalıştırılabilir OCR metodu yok (ocr/predict/callable).")

    def _iter_paddle_lines(self, ocr_out):
        """
        2.x list formatı / 3.x dict veya list-dict formatlarından
        (bbox_points, text, conf) üretir.
        """
        if ocr_out is None:
            return None

        # 2.x: [ [ [bbox, (text, conf)], ... ] ]
        if isinstance(ocr_out, list) and ocr_out:
            if isinstance(ocr_out[0], list):
                for line_data in ocr_out[0]:
                    if not line_data or len(line_data) < 2:
                        continue
                    bbox_points = line_data[0]
                    text_conf = line_data[1]
                    if not text_conf or len(text_conf) < 2:
                        continue
                    yield bbox_points, str(text_conf[0]), float(text_conf[1])
                return

            # 3.x: list[dict]
            if isinstance(ocr_out[0], dict):
                for d in ocr_out:
                    txt = d.get("rec_text") or d.get("text") or d.get("label") or ""
                    conf = d.get("rec_score") or d.get("score") or d.get("confidence") or 0.0
                    bbox = d.get("dt_poly") or d.get("poly") or d.get("bbox") or d.get("points") or None
                    if not txt:
                        continue
                    yield bbox, str(txt), float(conf) if conf is not None else 0.0
                return

        # 3.x: dict
        if isinstance(ocr_out, dict):
            texts = (ocr_out.get("rec_texts") or ocr_out.get("texts") or ocr_out.get("text") or [])
            scores = (ocr_out.get("rec_scores") or ocr_out.get("scores") or ocr_out.get("score") or [])
            polys = (ocr_out.get("dt_polys") or ocr_out.get("polys") or ocr_out.get("boxes") or ocr_out.get("bboxes") or [])

            if isinstance(texts, str):
                texts = [texts]
            if isinstance(scores, (int, float)):
                scores = [scores]
            if not isinstance(polys, list):
                polys = [polys]

            n = max(len(texts), len(scores), len(polys))
            for i in range(n):
                txt = texts[i] if i < len(texts) else ""
                conf = scores[i] if i < len(scores) else 0.0
                bbox = polys[i] if i < len(polys) else None
                if not txt:
                    continue
                yield bbox, str(txt), float(conf) if conf is not None else 0.0
            return

        return None


    def _process_single(self, frame_info: dict) -> list[OCRResult]:
        img = imread_unicode(frame_info["path"])
        if img is None:
            return []

        difficulty = frame_info.get("difficulty", "medium")
        timecode = frame_info.get("timecode_sec", 0.0)
        variants = self._prepare_variants(img, difficulty)
        all_results: dict[str, OCRResult] = {}

        for variant in variants:
            ocr_out = self._run_paddle(variant)
            # _iter_paddle_lines is a generator -- always truthy even when empty.
            # Materialise into a list so the empty-check actually works.
            lines = list(self._iter_paddle_lines(ocr_out))
            if not lines:
                continue

            for bbox_points, raw_text, conf in lines:
                text = str(raw_text).strip()
                conf = float(conf) if conf is not None else 0.0
                text = self._clean_text(text)
                if not text:
                    continue

                norm = self._normalize(text)
                if norm in all_results:
                    if conf > all_results[norm].confidence:
                        all_results[norm].text = text
                        all_results[norm].confidence = conf
                    continue

                bbox = []
                if bbox_points:
                    try:
                        # 2.x: 4 nokta listesi [[x,y],...]
                        if isinstance(bbox_points, list) and bbox_points and isinstance(bbox_points[0], (list, tuple)):
                            x1 = min(p[0] for p in bbox_points)
                            y1 = min(p[1] for p in bbox_points)
                            x2 = max(p[0] for p in bbox_points)
                            y2 = max(p[1] for p in bbox_points)
                            bbox = [float(x1), float(y1), float(x2), float(y2)]
                        # [x1,y1,x2,y2]
                        elif isinstance(bbox_points, (list, tuple)) and len(bbox_points) == 4:
                            bbox = [float(bbox_points[0]), float(bbox_points[1]), float(bbox_points[2]), float(bbox_points[3])]
                    except Exception:
                        bbox = []

                all_results[norm] = OCRResult(
                    text=text,
                    confidence=conf,
                    timecode_sec=timecode,
                    frame_path=frame_info["path"],
                    bbox=bbox,
                )

        return list(all_results.values())
    def _prepare_variants(self, img, difficulty) -> list:
        """
        Preprocessing v2.0 — Kullanıcı eleştirisi doğrultusunda yeniden yazıldı.

        Eleştiri: "Sadece 3 şey yapıyor (resize, invert, CLAHE). Binarization
        stratejileri yok, ROI crop yok, gradient temizleme yok, renk kanalı
        ayrımı yok."

        Düzeltmeler:
          - Renk kanalı ayrımı (B,G,R ayrı ayrı → en yüksek kontrast)
          - Çoklu binarization (Otsu + adaptive + Sauvola benzeri)
          - Arka plan gradient temizleme (morphological top-hat)
          - Alt bölge (ROI) crop — jenerik yazıları frame alt %30'unda yoğun
          - Hard frame için özel strateji (daha agresif preprocessing)
        """
        h, w = img.shape[:2]

        # ── 1. Upscale (düşük çözünürlük) ──────────────────────────
        if h < 400:
            scale = max(2, 600 // h)
            img = cv2.resize(img, (w * scale, h * scale),
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variants = []

        # ── 2. Orijinal BGR (her zaman) ─────────────────────────────
        variants.append(img)

        # ── 3. İnvert (koyu arka plan tespiti) ──────────────────────
        mean_lum = float(np.mean(gray))
        if mean_lum < 110:
            variants.append(cv2.bitwise_not(img))

        # ── 4. Renk kanalı ayrımı — en yüksek kontrast kanalı ──────
        b, g, r = cv2.split(img)
        channel_stds = [(np.std(b), b), (np.std(g), g), (np.std(r), r)]
        best_channel = max(channel_stds, key=lambda x: x[0])[1]
        channel_bgr = cv2.cvtColor(best_channel, cv2.COLOR_GRAY2BGR)
        variants.append(channel_bgr)

        # ── 5. CLAHE (medium ve hard) ────────────────────────────────
        if difficulty in ("medium", "hard"):
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            variants.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

        # ── 6. Binarization stratejileri (hard için) ─────────────────
        if difficulty == "hard":
            # Otsu binarization
            _, otsu = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_bgr = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
            variants.append(otsu_bgr)

            # Adaptive threshold (küçük metin için)
            adapt = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            adapt_bgr = cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)
            variants.append(adapt_bgr)

            # Gradient arka plan temizleme (morphological top-hat)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 10))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            _, tophat_bin = cv2.threshold(tophat, 30, 255, cv2.THRESH_BINARY)
            tophat_bgr = cv2.cvtColor(tophat_bin, cv2.COLOR_GRAY2BGR)
            variants.append(tophat_bgr)

        # ── 7. Alt bölge (ROI) crop — jenerik yazılar alt %30 da ────
        # Jenerik yazılar genellikle frame'in alt 1/3'ünde yer alır.
        # Tüm frame'e ek olarak sadece alt bölgeyi de kuyruğa ekle.
        roi_y = int(h * 0.65)
        if roi_y > 20 and (h - roi_y) > 20:
            roi = img[roi_y:, :]
            # ROI'yi upscale et (küçük metin daha iyi okunur)
            roi_scaled = cv2.resize(
                roi, (w, int((h - roi_y) * 2.0)),
                interpolation=cv2.INTER_CUBIC
            )
            variants.append(roi_scaled)

        # Hard için üst bölge de ekle (bazı içeriklerde üstte yazı var)
        if difficulty == "hard":
            roi_top = img[:int(h * 0.35), :]
            if roi_top.shape[0] > 20:
                roi_top_scaled = cv2.resize(
                    roi_top, (w, int(h * 0.35 * 2.0)),
                    interpolation=cv2.INTER_CUBIC
                )
                variants.append(roi_top_scaled)

        # Maksimum variant sayısı: easy=3, medium=5, hard=8
        max_variants = {"easy": 3, "medium": 5, "hard": 8}.get(difficulty, 5)
        return variants[:max_variants]

    # ═══════════════════════════════════════════════════════════════
    # TEMİZLEME + FİLTRELEME
    # ═══════════════════════════════════════════════════════════════
    def _clean_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip('.-_~=+|/\\')
        text = re.sub(r'^[\W_]+$', '', text)
        return text.strip()

    def _normalize(self, text: str) -> str:
        t = text.lower().strip()
        t = re.sub(r'[^\w\s]', '', t)
        t = re.sub(r'\s+', ' ', t)
        return t

    def _noise_filter(self, results: list[OCRResult]) -> list[OCRResult]:
        return [r for r in results
                if sum(1 for c in r.text if c in self.noise_chars) / max(len(r.text), 1) < 0.25]

    def _length_filter(self, results: list[OCRResult]) -> list[OCRResult]:
        return [r for r in results if len(r.text) >= self.min_text_len]

    def _confidence_filter(self, results: list[OCRResult]) -> list[OCRResult]:
        return [r for r in results if r.confidence >= self.min_confidence]

    def _digit_noise_filter(self, results: list[OCRResult]) -> list[OCRResult]:
        """
        %55+ digit → drop (plaka, timecode, numara).
        Yıl sayıları (1900-2100) korunur.
        """
        clean = []
        for r in results:
            text = r.text.strip()
            if not text:
                continue
            # 4 haneli yıl → koru
            if text.isdigit() and len(text) == 4:
                val = int(text)
                if 1900 <= val <= 2100:
                    clean.append(r)
                    continue
            digits = sum(1 for c in text if c.isdigit())
            ratio = digits / len(text)
            if len(text) <= 8 and ratio > self.max_digit_ratio:
                continue
            if ratio > 0.7:
                continue
            clean.append(r)
        return clean

    def _blacklist_filter(self, results: list[OCRResult]) -> list[OCRResult]:
        clean = []
        for r in results:
            text_lower = r.text.lower().strip()
            if any(pat.search(text_lower) for pat in BLACKLIST_RE):
                continue
            clean.append(r)
        return clean

    def _name_split_pass(self, results: list[OCRResult]) -> list[OCRResult]:
        """
        Birleşik isimleri böl: "SEBNEMSONMEZ" → "SEBNEM SONMEZ"

        v2.2: TurkishNameDB varsa DP-tabanlı 356k kayıt destekli split kullan.
              Yoksa eski ALL_NAMES (~300) ile devam et.
              Koşullar gevşetildi: sadece ALL-CAPS değil, mixed case de denenir.
        """
        for r in results:
            text = r.text.strip()

            # Zaten boşluklu ve 2+ kelime → bölmeye gerek yok
            if " " in text and len(text.split()) >= 2:
                continue

            # Minimum uzunluk ve alfanumerik kontrol
            if len(text) < 5 or not text.replace(" ", "").isalpha():
                continue

            # ── TurkishNameDB (356k kayıt, DP split) ──
            if self._name_db is not None:
                parts = self._name_db.split_concatenated(text)
                if len(parts) >= 2:
                    r.text = " ".join(parts)
                    continue

            # ── Fallback: eski ALL_NAMES (~300 isim) ──
            # Sadece ALL-CAPS + boşluksuz + 6+ harf (orijinal koşul)
            if text.isupper() and " " not in text and len(text) >= 6:
                split = split_concatenated_name(text)
                if split != text:
                    r.text = split

        return results

    def _fuzzy_dedup(self, results: list[OCRResult]) -> list[OCRResult]:
        if not results:
            return []
        sorted_r = sorted(results, key=lambda r: r.timecode_sec)
        unique: list[OCRResult] = []
        used = set()

        for i, r in enumerate(sorted_r):
            if i in used:
                continue
            best = r
            cluster = [r]
            for j in range(i + 1, len(sorted_r)):
                if j in used:
                    continue
                sim = self._similarity(r.text, sorted_r[j].text)
                if sim >= self.fuzzy_threshold:
                    used.add(j)
                    cluster.append(sorted_r[j])
                    if sorted_r[j].confidence > best.confidence:
                        best = sorted_r[j]

            unique.append(OCRResult(
                text=best.text,
                confidence=max(c.confidence for c in cluster),
                timecode_sec=min(c.timecode_sec for c in cluster),
                frame_path=best.frame_path,
                bbox=best.bbox,
            ))
        return unique

    def _persistence_and_watermark(self, results: list[OCRResult]) -> list[OCRLine]:
        groups: dict[str, list[OCRResult]] = {}
        for r in results:
            norm = self._normalize(r.text)
            if not norm:
                continue
            groups.setdefault(norm, []).append(r)

        lines = []
        for norm, group in groups.items():
            if len(group) >= self.watermark_threshold:
                continue
            best = max(group, key=lambda r: r.confidence)
            lines.append(OCRLine(
                text=best.text,
                first_seen=min(r.timecode_sec for r in group),
                last_seen=max(r.timecode_sec for r in group),
                seen_count=len(group),
                avg_confidence=round(sum(r.confidence for r in group) / len(group), 3),
                bbox=best.bbox,
                frame_path=best.frame_path,
                source="paddleocr",
            ))
        lines.sort(key=lambda l: l.first_seen)
        return lines

    def _similarity(self, a: str, b: str) -> float:
        if HAS_FUZZ:
            return fuzz.ratio(a.lower(), b.lower())
        a_n, b_n = self._normalize(a), self._normalize(b)
        if a_n == b_n:
            return 100.0
        common = sum(1 for ca, cb in zip(a_n, b_n) if ca == cb)
        return (common / max(len(a_n), len(b_n), 1)) * 100
