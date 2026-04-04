"""
hybrid_ocr_router.py — oneocr birincil + Qwen VLM handwriting fallback hibrit OCR motoru.

OCREngine / QwenOCREngine ile birebir aynı arayüz:
    router = HybridOCRRouter(cfg=config, log_cb=log, name_db=name_db)
    ocr_lines, layout_pairs = router.process_frames(candidate_frames)

Çalışma mantığı:
    Phase 1: oneocr ile TÜM frame'leri oku (hızlı, CPU, ~0.3s/frame)
    Phase 2: 3 katmanlı karar mekanizması — hangi frame'ler Qwen'e gidecek?
      Katman 1 — oneocr confidence: ≥ 0.90 → yeterli, < 0.70 → Qwen gerek
      Katman 2 — Font geometrisi: standard → atla, handwriting → Qwen gerek
      Katman 3 — NameDB eşleşme oranı: ≥ 0.60 → atla, < 0.60 → Qwen gerek
    Phase 3: Sadece handwriting frame'leri Qwen'e gönder (tam frame, asla crop değil!)
      Qwen sonucu birincil, oneocr eksik satırları tamamlar

Fallback zinciri:
    oneocr kurulu → Hybrid (oneocr + Qwen)
    oneocr kurulu değil → QwenOCREngine (tek başına)
    qwen_fallback_on_handwriting: false → oneocr tek başına

Mimari ilkeler:
    - Qwen'e ASLA bbox crop gönderilmez — tam frame modu
    - Ham okuma asla silinmez (OCR sonuçları ayrı kayıt)
    - GPU sadece Qwen için, oneocr CPU'da
"""

import time
from pathlib import Path

from core.ocr_engine import OCRLine

# ── Opsiyonel import'lar ─────────────────────────────────────────────────────

try:
    from core.oneocr_engine import OneOCREngine
    _HAS_ONEOCR = True
except ImportError:
    OneOCREngine = None
    _HAS_ONEOCR = False

try:
    from core.qwen_ocr_engine import QwenOCREngine
    _HAS_QWEN = True
except ImportError:
    QwenOCREngine = None
    _HAS_QWEN = False

# ── Eşikler ──────────────────────────────────────────────────────────────────

# Confidence eşikleri (Katman 1)
CONF_HIGH   = 0.30   # ≥ bu değer → oneocr yeterli
CONF_LOW    = 0.20   # < bu değer → Qwen gerek

# NameDB eşleşme oranı (Katman 3)
NAMEDB_MATCH_THRESHOLD = 0.60  # < bu oran → Qwen gerek

# Handwriting frame oranı — tüm film handwriting sayılır
HW_RATIO_FULL = 0.50   # > bu oran → tüm frame'leri Qwen'e gönder


class HybridOCRRouter:
    """
    oneocr birincil motor + Qwen VLM handwriting fallback.
    OCREngine / QwenOCREngine ile aynı process_frames() arayüzü.
    """

    def __init__(self, cfg: dict = None, log_cb=None, name_db=None):
        self.cfg = cfg or {}
        self._log = log_cb or (lambda m: None)
        self._name_db = name_db
        self._font_cache: dict[str, str] = {}

        self._qwen_fallback = self.cfg.get("qwen_fallback_on_handwriting", True)

        # ── oneocr motoru ─────────────────────────────────────────────────
        if _HAS_ONEOCR:
            self._oneocr = OneOCREngine(
                cfg=self.cfg,
                log_cb=self._log,
                name_db=self._name_db,
            )
            self._log("  [Hybrid] oneocr motoru hazır (birincil)")
        else:
            self._oneocr = None
            self._log("  [Hybrid] oneocr kurulu değil — sadece Qwen kullanılacak")

        # ── Qwen motoru ───────────────────────────────────────────────────
        if _HAS_QWEN and self._qwen_fallback:
            ollama_url = self.cfg.get(
                "ollama_url",
                self.cfg.get("vlm_url", "http://localhost:11434")
            )
            self._qwen = QwenOCREngine(
                cfg=self.cfg,
                log_cb=self._log,
                name_db=self._name_db,
                ollama_url=ollama_url,
            )
            self._log("  [Hybrid] Qwen motoru hazır (handwriting fallback)")
        else:
            self._qwen = None
            if not self._qwen_fallback:
                self._log("  [Hybrid] Qwen fallback devre dışı (qwen_fallback_on_handwriting: false)")

    # ── Ana arayüz — OCREngine ile aynı imza ─────────────────────────────────
    def process_frames(self,
                       candidate_frames: list,
                       log_callback=None) -> tuple:
        """
        Döndürür: (ocr_lines: list[OCRLine], layout_pairs: list)

        oneocr birincil, Qwen handwriting fallback olarak çalışır.
        """
        cb = log_callback or self._log
        total = len(candidate_frames)

        cb(f"\n  [Hybrid] {total} frame işlenecek | oneocr={'✓' if self._oneocr else '✗'}"
           f" | qwen={'✓' if self._qwen else '✗'}")

        # ── Sadece Qwen varsa (oneocr kurulu değil) ───────────────────────
        if self._oneocr is None:
            if self._qwen is not None:
                cb("  [Hybrid] oneocr yok → Qwen tek başına çalışıyor")
                return self._qwen.process_frames(candidate_frames, log_callback=cb)
            cb("  [Hybrid] !! Ne oneocr ne de Qwen kullanılabilir — boş döndürülüyor")
            return [], []

        t0 = time.time()

        # ══ Phase 1: oneocr ile TÜM frame'leri oku ════════════════════════
        cb("  [Hybrid] Phase 1: oneocr tüm frame'leri okuyor...")
        oneocr_lines, layout_pairs = self._oneocr.process_frames(
            candidate_frames, log_callback=cb)
        frame_meta = dict(getattr(self._oneocr, "_last_frame_meta", {}) or {})

        phase1_time = time.time() - t0
        cb(f"  [Hybrid] Phase 1 tamamlandı: {len(oneocr_lines)} satır ({phase1_time:.1f}s)")

        # Qwen devre dışıysa veya kurulu değilse — doğrudan dön
        if self._qwen is None or not self._qwen_fallback:
            return oneocr_lines, layout_pairs

        # ══ Phase 2: Karar mekanizması ════════════════════════════════════
        cb("  [Hybrid] Phase 2: handwriting frame analizi...")
        handwriting_frames = self._decide_qwen_frames(
            candidate_frames, oneocr_lines, frame_meta, cb)

        if not handwriting_frames:
            cb("  [Hybrid] Tüm frame'ler standart font — Qwen atlanıyor")
            return oneocr_lines, layout_pairs

        hw_ratio = len(handwriting_frames) / max(total, 1)
        cb(f"  [Hybrid] Handwriting frame oranı: {hw_ratio:.0%} "
           f"({len(handwriting_frames)}/{total})")

        # ══ Phase 3: Handwriting frame'leri Qwen'e gönder ═════════════════
        cb(f"  [Hybrid] Phase 3: {len(handwriting_frames)} frame Qwen'e gönderiliyor...")

        # hw_ratio > 0.50 ise tüm frame'ler Qwen'e gitsin
        qwen_frames = candidate_frames if hw_ratio > HW_RATIO_FULL else handwriting_frames

        if not self._qwen.is_available():
            cb("  [Hybrid] !! Qwen/Ollama bağlantısı yok — oneocr sonucu kullanılıyor")
            cb("  [Hybrid]    Ollama çalışıyor mu? `ollama serve` ile başlatabilirsiniz.")
            return oneocr_lines, layout_pairs

        t2 = time.time()
        qwen_lines, _ = self._qwen.process_frames(qwen_frames, log_callback=cb)
        phase3_time = time.time() - t2
        cb(f"  [Hybrid] Phase 3 tamamlandı: {len(qwen_lines)} satır ({phase3_time:.1f}s)")

        # ── Merge: Qwen birincil, oneocr eksik satırları tamamlar ─────────
        merged_lines = self._merge_results(
            oneocr_lines=oneocr_lines,
            qwen_lines=qwen_lines,
            cb=cb,
        )

        total_time = time.time() - t0
        cb(f"  [Hybrid] Tamamlandı: {len(merged_lines)} satır "
           f"(oneocr:{len(oneocr_lines)} + qwen:{len(qwen_lines)} → merged:{len(merged_lines)}) "
           f"[{total_time:.1f}s]")

        return merged_lines, layout_pairs

    # ══ Phase 2 — Karar mekanizması ══════════════════════════════════════════

    def _decide_qwen_frames(self,
                            candidate_frames: list,
                            oneocr_lines: list,
                            frame_meta: dict,
                            cb) -> list:
        """
        3 katmanlı karar: hangi frame'ler Qwen'e gidecek?

        Katman 1 — oneocr confidence (frame bazlı)
        Katman 2 — Font geometrisi (frame bazlı)
        Katman 3 — NameDB eşleşme oranı (toplu)
        """
        if not candidate_frames:
            return []

        handwriting_frames = []
        total = len(candidate_frames)

        for idx, frame_info in enumerate(candidate_frames, start=1):
            frame_path = frame_info.get("path") or frame_info.get("frame_path", "")
            if not frame_path or not Path(frame_path).exists():
                continue

            meta = frame_meta.get(frame_path)
            if meta is not None:
                avg_conf = self._meta_value(meta, "avg_confidence", 0.0)
                has_text = bool(self._meta_value(meta, "has_text", False))
                font_type = self._meta_value(meta, "font_type", "unknown")

                # Frame'de usable OCR sonucu yok → sadece handwriting ise aday yap
                if not has_text:
                    if font_type == "handwriting":
                        handwriting_frames.append(frame_info)
                else:
                    # Katman 1: Yüksek confidence → Qwen gerek yok
                    if avg_conf >= CONF_HIGH:
                        pass
                    # Katman 1: Çok düşük confidence → Qwen gerek
                    elif avg_conf < CONF_LOW:
                        handwriting_frames.append(frame_info)
                    else:
                        # Katman 2: Orta confidence → font tipini kontrol et
                        if font_type == "standard":
                            pass
                        elif font_type == "handwriting":
                            handwriting_frames.append(frame_info)
                        else:
                            # decorative/unknown → Katman 3'e bak (toplu karar)
                            handwriting_frames.append(frame_info)
            else:
                # Güvenlik fallback'i: metadata yoksa eski font tahmin yolunu kullan.
                font_type = self._estimate_font_type(frame_path)
                if font_type == "handwriting":
                    handwriting_frames.append(frame_info)

            if idx % 50 == 0 or idx == total:
                cb(f"  [Hybrid] Phase 2 ilerleme: {idx}/{total} | aday:{len(handwriting_frames)}")

        if not handwriting_frames:
            return []

        # ── Katman 3: NameDB eşleşme oranı (toplu) ───────────────────────
        if self._name_db is not None:
            match_ratio = self._compute_namedb_ratio(oneocr_lines)
            if match_ratio >= NAMEDB_MATCH_THRESHOLD:
                cb(f"  [Hybrid] Katman 3: NameDB oranı {match_ratio:.0%} ≥ "
                   f"{NAMEDB_MATCH_THRESHOLD:.0%} — Qwen atlanıyor")
                return []

        return handwriting_frames

    @staticmethod
    def _meta_value(meta, field_name: str, default=None):
        """FrameMeta dataclass veya dict üzerinde güvenli alan okuma."""
        if meta is None:
            return default
        if isinstance(meta, dict):
            return meta.get(field_name, default)
        return getattr(meta, field_name, default)

    def _estimate_font_type(self, frame_path: str) -> str:
        """
        Frame'in font tipini tahmin et (cache'li).
        OneOCREngine.estimate_font_type() metodunu kullanır.
        Returns: "standard" | "handwriting" | "decorative" | "unknown"
        """
        if frame_path in self._font_cache:
            return self._font_cache[frame_path]
        if self._oneocr is None:
            return "unknown"
        try:
            result = self._oneocr.estimate_font_type(frame_path)
        except Exception:
            result = "unknown"
        self._font_cache[frame_path] = result
        return result

    def _compute_namedb_ratio(self, ocr_lines: list) -> float:
        """
        OCR satırlarının NameDB eşleşme oranını hesapla.
        Döndürür: 0.0 – 1.0
        """
        if self._name_db is None or not ocr_lines:
            return 0.0

        matched = 0
        total = 0

        for line in ocr_lines:
            text = getattr(line, "text", "") or ""
            # Çok kısa satırları atla (teknik terimler, sayılar)
            if len(text.strip()) < 4:
                continue
            total += 1
            try:
                if self._name_db.is_name(text):
                    matched += 1
            except Exception:
                pass

        return matched / total if total > 0 else 0.0

    # ══ Phase 3 — Merge ═══════════════════════════════════════════════════════

    def _merge_results(self,
                       oneocr_lines: list,
                       qwen_lines: list,
                       cb) -> list:
        """
        Qwen sonucu birincil, oneocr eksik satırları tamamlar.

        - Qwen'deki her satır alınır
        - oneocr'da Qwen'de olmayan satırlar eklenir (flag: source="oneocr+hybrid")
        - Sonuç zaman sırasına göre sıralanır
        """
        if not qwen_lines:
            return oneocr_lines
        if not oneocr_lines:
            return qwen_lines

        try:
            from rapidfuzz.fuzz import WRatio
            _has_rapidfuzz = True
        except ImportError:
            _has_rapidfuzz = False

        # Qwen satırlarının normalize metinlerini set olarak tut
        _TR_MAP = str.maketrans("çğışöüÇĞİŞÖÜ", "cgisouCGISOu")

        def _norm(t: str) -> str:
            import re
            return re.sub(r"\s+", " ", t.lower().translate(_TR_MAP)).strip()

        qwen_norms = {_norm(getattr(l, "text", "") or "") for l in qwen_lines}

        # oneocr'dan sadece Qwen'de olmayan satırları ekle
        extra_lines = []
        for line in oneocr_lines:
            text = getattr(line, "text", "") or ""
            norm = _norm(text)
            if not norm:
                continue

            # Tam eşleşme
            if norm in qwen_norms:
                continue

            # Fuzzy eşleşme (rapidfuzz varsa)
            if _has_rapidfuzz:
                is_dup = any(
                    WRatio(norm, qn) >= 85
                    for qn in qwen_norms
                )
                if is_dup:
                    continue

            # Qwen'de yok → ekle (kaynak bilgisini güncelle)
            try:
                extra = OCRLine(
                    text=line.text,
                    first_seen=line.first_seen,
                    last_seen=line.last_seen,
                    seen_count=line.seen_count,
                    avg_confidence=line.avg_confidence,
                    bbox=line.bbox,
                    frame_path=line.frame_path,
                    source="oneocr+hybrid",
                )
            except Exception:
                extra = line
            extra_lines.append(extra)

        if extra_lines:
            cb(f"  [Hybrid] Merge: Qwen={len(qwen_lines)}, "
               f"oneocr eklendi={len(extra_lines)}")

        merged = list(qwen_lines) + extra_lines
        merged.sort(key=lambda l: getattr(l, "first_seen", 0.0))
        return merged
