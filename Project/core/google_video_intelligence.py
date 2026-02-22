"""
google_video_intelligence.py — Google Cloud Video Intelligence TEXT_DETECTION fallback.

Tetikleme: Aşağıdaki 4 koşuldan HERHANGİ BİRİ sağlanıyorsa gönder.
  1. OCR conf ortalaması < 0.65
  2. (Satır sayısı / Aday frame sayısı) < 0.15
  3. cast + crew < 3
  4. NameDB onarım oranı > %30

Kota: Aylık 1000 dakika (Google ücretsiz tier). Dosya tabanlı sayaç.

Gönderim: Sadece jenerik segmentleri (entry + exit) — tüm video değil.
Birleştirme: Paddle + VI birleştirilir, normalize key üzerinden dedup.
             Paddle metnini koru (NameDB/Qwen geçmiş olabilir), VI doğrular.

Config anahtarları:
  google_vi_monthly_limit  : float  aylık dakika limiti (default: 1000.0)
  google_vi_min_conf       : float  VI sonuçları için min conf (default: 0.50)
  google_vi_timeout        : int    API timeout saniye (default: 600)
  google_credentials_json  : str    service account json yolu (opsiyonel)
  google_vi_cache_dir      : str    sayaç dosyası dizini (default: ~/.arsiv_cache)

Auth:
  GOOGLE_APPLICATION_CREDENTIALS env değişkeni
  veya google_credentials_json config key'i
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Tuple

try:
    from google.cloud import videointelligence
    HAS_VIDEO_INTELLIGENCE = True
except ImportError:
    HAS_VIDEO_INTELLIGENCE = False


# ═══════════════════════════════════════════════════════════════════
# VERİ YAPILARI
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GText:
    text: str
    confidence: float
    start_sec: float
    end_sec: float


@dataclass
class OCRLine:
    """PaddleOCR OCRLine ile aynı alan adları — merge için zorunlu."""
    text: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    seen_count: int = 1
    avg_confidence: float = 0.0
    bbox: list = field(default_factory=list)
    frame_path: str = ""
    source: str = "google_vi"


@dataclass
class GVIDecision:
    should_run: bool
    reason: str
    triggers: list = field(default_factory=list)  # hangi koşullar tetikledi


# ═══════════════════════════════════════════════════════════════════
# ANA MOTOR
# ═══════════════════════════════════════════════════════════════════

class GoogleVITextEngine:
    """
    Google Cloud Video Intelligence TEXT_DETECTION motoru.

    Pipeline entegrasyon noktaları:
        decide()            → bu video VI'ya gitmeli mi?
        process_segments()  → segmentleri gönder, OCRLine listesi döndür
        merge_with_paddle() → Paddle sonuçlarıyla birleştir
    """

    # ── Tetikleme eşikleri ──────────────────────────────────────────
    CONF_AVG_MIN         = 0.65   # Koşul 1
    LINE_FRAME_RATIO_MIN = 0.15   # Koşul 2
    CAST_CREW_MIN        = 3      # Koşul 3
    REPAIR_RATIO_MAX     = 0.30   # Koşul 4

    def __init__(self, config: dict = None, log_cb=None):
        self._cfg          = config or {}
        self._log          = log_cb or (lambda m: None)
        self.monthly_limit = float(self._cfg.get("google_vi_monthly_limit", 1000.0))
        self.min_conf      = float(self._cfg.get("google_vi_min_conf", 0.50))
        self.timeout_sec   = int(self._cfg.get("google_vi_timeout", 600))
        self._client       = None  # lazy init

        cred_path = (
            self._cfg.get("google_credentials_json") or
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or ""
        ).strip()
        if cred_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

    # ── Karar mekanizması ────────────────────────────────────────────

    def decide(
        self,
        ocr_avg_conf: float,
        total_ocr_lines: int,
        candidate_frame_count: int,
        cast_count: int,
        crew_count: int,
        repaired_lines: int,
        segment_duration_min: float,
    ) -> GVIDecision:
        """
        4 koşuldan herhangi biri sağlanıyorsa → gönder.

        Args:
            ocr_avg_conf          : PaddleOCR satırlarının conf ortalaması
            total_ocr_lines       : PaddleOCR toplam satır sayısı
            candidate_frame_count : TextFilter'dan geçen frame sayısı
            cast_count            : CreditsParser oyuncu sayısı
            crew_count            : CreditsParser ekip sayısı
            repaired_lines        : NameDB'nin düzelttiği satır sayısı
            segment_duration_min  : Gönderilecek segment süresi (dakika)
        """
        # ── Kota ve erişilebilirlik ──
        if not self.is_available():
            return GVIDecision(
                False,
                "google_vi kullanılamıyor — kütüphane veya credential eksik",
            )

        used_min = self._get_monthly_usage()
        remaining = self.monthly_limit - used_min
        if remaining < segment_duration_min:
            return GVIDecision(
                False,
                f"aylık kota yetersiz — kalan:{remaining:.1f}dk, "
                f"gereken:{segment_duration_min:.1f}dk",
            )

        # ── 4 koşulu değerlendir ──
        triggers = []

        # Koşul 1: OCR conf ortalaması
        if ocr_avg_conf < self.CONF_AVG_MIN:
            triggers.append(
                f"1) OCR conf ortalaması düşük "
                f"({ocr_avg_conf:.2f} < {self.CONF_AVG_MIN})"
            )

        # Koşul 2: Satır/frame oranı
        if candidate_frame_count > 0:
            ratio = total_ocr_lines / candidate_frame_count
            if ratio < self.LINE_FRAME_RATIO_MIN:
                triggers.append(
                    f"2) Az satır çıktı ({total_ocr_lines} satır / "
                    f"{candidate_frame_count} frame = "
                    f"{ratio:.2f} < {self.LINE_FRAME_RATIO_MIN})"
                )

        # Koşul 3: Bulunan kişi sayısı
        total_found = cast_count + crew_count
        if total_found < self.CAST_CREW_MIN:
            triggers.append(
                f"3) Çok az isim bulundu "
                f"({total_found} < {self.CAST_CREW_MIN})"
            )

        # Koşul 4: NameDB onarım oranı
        if total_ocr_lines > 0:
            repair_ratio = repaired_lines / total_ocr_lines
            if repair_ratio > self.REPAIR_RATIO_MAX:
                triggers.append(
                    f"4) OCR kalite düşük — onarım oranı "
                    f"{repair_ratio:.0%} > {self.REPAIR_RATIO_MAX:.0%}"
                )

        if triggers:
            return GVIDecision(
                True,
                f"{len(triggers)}/4 koşul tetiklendi → Google VI çalışacak",
                triggers=triggers,
            )

        return GVIDecision(
            False,
            f"tüm koşullar sağlandı — Google VI'ya gerek yok "
            f"(conf:{ocr_avg_conf:.2f}, "
            f"satır/frame:{total_ocr_lines}/{candidate_frame_count}, "
            f"kişi:{total_found}, "
            f"onarım:{repaired_lines}/{total_ocr_lines})",
        )

    # ── İşlem ────────────────────────────────────────────────────────

    def process_segments(
        self,
        video_path: str,
        entry_start: float,
        entry_end: float,
        exit_start: float,
        exit_end: float,
    ) -> List[OCRLine]:
        """
        Jenerik segmentlerini Google VI'ya gönder.
        entry + exit tek API çağrısında iki segment olarak işlenir.
        """
        if not self.is_available():
            self._log("  [GoogleVI] Kullanılamıyor — atlanıyor")
            return []

        segment_min = ((entry_end - entry_start) + (exit_end - exit_start)) / 60.0
        self._log(
            f"  [GoogleVI] TEXT_DETECTION başlıyor "
            f"(~{segment_min:.1f} dk gönderilecek)..."
        )
        self._log(f"  [GoogleVI] Giriş: {entry_start:.1f}s → {entry_end:.1f}s")
        self._log(f"  [GoogleVI] Çıkış: {exit_start:.1f}s → {exit_end:.1f}s")

        try:
            raw = self._call_api(
                video_path,
                segments=[(entry_start, entry_end), (exit_start, exit_end)],
            )
        except Exception as e:
            self._log(f"  [GoogleVI] API hatası: {e}")
            return []

        self._add_monthly_usage(segment_min)

        filtered = [g for g in raw if g.confidence >= self.min_conf]
        self._log(
            f"  [GoogleVI] {len(raw)} annotation → "
            f"{len(filtered)} tutuldu (conf≥{self.min_conf})"
        )

        lines = self._to_ocr_lines(filtered)
        self._log(f"  [GoogleVI] {len(lines)} benzersiz satır üretildi")
        return lines

    def merge_with_paddle(
        self,
        paddle_lines: list,
        vi_lines: List[OCRLine],
    ) -> list:
        """
        PaddleOCR + Google VI satırlarını birleştir.

        - Sadece VI'da var → ekle
        - Her ikisinde de var → Paddle metnini koru, source = "paddle+vi"
        - Sadece Paddle'da var → dokunma
        """
        if not vi_lines:
            return paddle_lines

        index: dict[str, object] = {}
        for line in paddle_lines:
            key = self._norm_key(self._get_text(line))
            if key:
                index[key] = line

        vi_only = 0
        confirmed = 0
        for vi_line in vi_lines:
            key = self._norm_key(vi_line.text)
            if not key:
                continue
            if key not in index:
                index[key] = vi_line
                vi_only += 1
            else:
                self._set_source(index[key], "paddle+vi")
                confirmed += 1

        if vi_only or confirmed:
            self._log(
                f"  [GoogleVI] Merge: +{vi_only} yeni satır, "
                f"{confirmed} satır VI ile doğrulandı"
            )

        merged = list(index.values())
        try:
            merged.sort(key=lambda l: self._get_first_seen(l))
        except Exception:
            pass
        return merged

    # ── Erişilebilirlik ──────────────────────────────────────────────

    def is_available(self) -> bool:
        if not HAS_VIDEO_INTELLIGENCE:
            return False
        cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        return bool(cred and os.path.isfile(cred))

    def get_usage_summary(self) -> dict:
        """Pipeline log'u için kullanım özeti."""
        used = self._get_monthly_usage()
        return {
            "month": date.today().strftime("%Y-%m"),
            "minutes_used": round(used, 1),
            "minutes_limit": self.monthly_limit,
            "minutes_remaining": round(self.monthly_limit - used, 1),
            "usage_pct": round(used / self.monthly_limit * 100, 1),
        }

    # ── Aylık dakika sayacı ──────────────────────────────────────────

    def _counter_file(self) -> Path:
        cache_dir = Path(
            self._cfg.get("google_vi_cache_dir", Path.home() / ".arsiv_cache")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "google_vi_monthly.json"

    def _get_monthly_usage(self) -> float:
        try:
            data = json.loads(self._counter_file().read_text(encoding="utf-8"))
            if data.get("month") == date.today().strftime("%Y-%m"):
                return float(data.get("minutes_used", 0.0))
        except Exception:
            pass
        return 0.0

    def _add_monthly_usage(self, minutes: float):
        try:
            this_month = date.today().strftime("%Y-%m")
            new_total = self._get_monthly_usage() + minutes
            self._counter_file().write_text(
                json.dumps({
                    "month": this_month,
                    "minutes_used": round(new_total, 2),
                    "limit": self.monthly_limit,
                }),
                encoding="utf-8",
            )
            remaining = self.monthly_limit - new_total
            self._log(
                f"  [GoogleVI] Aylık kullanım: "
                f"{new_total:.1f}/{self.monthly_limit:.0f} dk "
                f"(kalan: {remaining:.1f} dk)"
            )
        except Exception as e:
            self._log(f"  [GoogleVI] Sayaç güncellenemedi: {e}")

    # ── API çağrısı ──────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            self._client = videointelligence.VideoIntelligenceServiceClient()
        return self._client

    def _call_api(
        self,
        video_path: str,
        segments: List[Tuple[float, float]],
    ) -> List[GText]:
        with open(video_path, "rb") as f:
            content = f.read()

        vi_segments = [
            videointelligence.VideoSegment(
                start_time_offset={
                    "seconds": int(s),
                    "nanos": int((s - int(s)) * 1e9),
                },
                end_time_offset={
                    "seconds": int(e),
                    "nanos": int((e - int(e)) * 1e9),
                },
            )
            for s, e in segments if e > s
        ]

        video_context = (
            videointelligence.VideoContext(segments=vi_segments)
            if vi_segments else None
        )

        self._log("  [GoogleVI] API isteği gönderildi, yanıt bekleniyor...")
        operation = self._get_client().annotate_video(
            request={
                "features": [videointelligence.Feature.TEXT_DETECTION],
                "input_content": content,
                "video_context": video_context,
            }
        )
        result = operation.result(timeout=self.timeout_sec)

        out: List[GText] = []
        for ann in result.annotation_results:
            for ta in ann.text_annotations:
                txt = (ta.text or "").strip()
                if not txt:
                    continue
                conf = s0 = e0 = 0.0
                if ta.segments:
                    seg = ta.segments[0]
                    conf = float(getattr(seg, "confidence", 0.0) or 0.0)
                    s = seg.segment.start_time_offset
                    e = seg.segment.end_time_offset
                    s0 = (float(getattr(s, "seconds", 0) or 0)
                          + float(getattr(s, "nanos", 0) or 0) / 1e9)
                    e0 = (float(getattr(e, "seconds", 0) or 0)
                          + float(getattr(e, "nanos", 0) or 0) / 1e9)
                out.append(GText(text=txt, confidence=conf, start_sec=s0, end_sec=e0))
        return out

    # ── Dönüşüm ─────────────────────────────────────────────────────

    def _to_ocr_lines(self, items: List[GText]) -> List[OCRLine]:
        groups: dict[str, List[GText]] = {}
        for item in items:
            key = self._norm_key(item.text)
            if key:
                groups.setdefault(key, []).append(item)

        lines: List[OCRLine] = []
        for group in groups.values():
            best = max(group, key=lambda g: g.confidence)
            avg_conf = sum(g.confidence for g in group) / len(group)
            lines.append(OCRLine(
                text=best.text,
                first_seen=min(g.start_sec for g in group),
                last_seen=max(g.end_sec for g in group),
                seen_count=len(group),
                avg_confidence=round(avg_conf, 3),
                source="google_vi",
            ))

        lines.sort(key=lambda l: l.first_seen)
        return lines

    # ── Yardımcılar ──────────────────────────────────────────────────

    def _norm_key(self, text: str) -> str:
        if not text:
            return ""
        tr_map = str.maketrans({
            'ç':'c','ğ':'g','ı':'i','ö':'o','ş':'s','ü':'u',
            'Ç':'c','Ğ':'g','İ':'i','Ö':'o','Ş':'s','Ü':'u',
        })
        t = text.lower().strip().translate(tr_map)
        return re.sub(r'[^a-z0-9]', '', t)

    def _get_text(self, line) -> str:
        return (line.get("text", "") if isinstance(line, dict)
                else getattr(line, "text", ""))

    def _get_first_seen(self, line) -> float:
        return float(
            line.get("first_seen", 0.0) if isinstance(line, dict)
            else getattr(line, "first_seen", 0.0)
        )

    def _set_source(self, line, source: str):
        if isinstance(line, dict):
            line["source"] = source
        else:
            try:
                line.source = source
            except Exception:
                pass
