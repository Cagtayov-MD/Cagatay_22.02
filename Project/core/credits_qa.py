"""credits_qa.py — TMDB cast ve crew eksikliği kontrol modülü.

TMDB doğrulama başarılı olduğunda OCR'dan gelen satırları TMDB cast ve crew
listeleriyle karşılaştırır.  TMDB'de bulunmayan yüksek güvenilirlikli oyuncu
ve yapım ekibi isimlerini raporlar.

Hem cast hem crew kontrolü yapar.
LLM kullanmaz, tamamen deterministik.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

# ── Crew rol başlığı blacklist (tam eşleşme, küçük harf) ──────────────────────
_CREW_ROLE_BLACKLIST_EXACT = frozenset({
    "art director", "assistant director", "first assistant director",
    "director of photography", "film editor", "film edtor",
    "set decorations", "set decoration", "makeup supervision",
    "makeup supervisor", "special effects", "optical effects",
    "costume design", "costumes designed by", "costumes, designed by",
    "sound", "music", "screenplay by", "directed by",
    "with", "and", "presents", "certificate", "approved",
    "the end", "end", "production", "a republic production",
})

# ── Eşik değerleri ────────────────────────────────────────────────────────────
MIN_WORDS        = 2      # İsim + soyisim minimum
MIN_CONFIDENCE   = 0.88   # OCR güven eşiği
MIN_SEEN_FRAMES  = 3      # Minimum frame tekrarı
OPENING_CUTOFF_S = 45.0   # Açılış jeneriği sınırı (saniye)
MATCH_THRESHOLD  = 0.75   # TMDB eşleşme eşiği


# ── Benzerlik backend (rapidfuzz varsa, yoksa Jaccard fallback) ───────────────
def _jaccard_similarity(a: str, b: str) -> float:
    """Token-bazlı Jaccard benzerliği — rapidfuzz yokken fallback."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


try:
    from rapidfuzz.fuzz import token_sort_ratio as _tsr

    def _similarity(a: str, b: str) -> float:
        return _tsr(a, b) / 100.0

except ImportError:
    _similarity = _jaccard_similarity


# ── İsim normalizasyonu ───────────────────────────────────────────────────────
_CAMEL_RE = re.compile(r'([a-z])([A-Z])')


def _normalize(text: str) -> str:
    """Deterministic isim normalizasyonu.

    1. Unicode NFKD normalize
    2. ASCII'ye indir (aksan kaldır)
    3. Birleşik isim ayır: "ShannaReed" → "shanna reed"
    4. Lowercase
    5. Noktalama ve fazla boşluk temizle
    """
    # NFKD normalize + ASCII'ye çevir
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    # CamelCase ayır: "ShannaReed" → "Shanna Reed"
    spaced = _CAMEL_RE.sub(r'\1 \2', ascii_text)
    lower = spaced.lower()
    # Sadece alfanümerik + boşluk bırak
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', lower)
    return re.sub(r'\s+', ' ', cleaned).strip()


# ── Veri sınıfları ────────────────────────────────────────────────────────────
@dataclass
class MissingActor:
    name: str               # OCR'dan gelen ham isim
    confidence: float       # OCR güvenilirlik
    seen_count: int         # Kaç frame'de görüldü
    first_seen_sec: float   # İlk görüldüğü saniye
    is_opening_credit: bool # Açılış jeneriğinde mi
    best_tmdb_match: str    # En yakın TMDB eşleşmesi
    similarity: float       # Benzerlik skoru

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "confidence":     round(self.confidence, 4),
            "seen_frames":    self.seen_count,
            "first_seen_sec": round(self.first_seen_sec, 2),
            "opening_credit": self.is_opening_credit,
            "closest_tmdb":   self.best_tmdb_match,
            "similarity":     round(self.similarity, 4),
        }


@dataclass
class MissingCrewMember:
    name: str               # OCR'dan gelen ham isim
    job: str                # OCR'dan veya name_verify'dan gelen görev
    confidence: float       # OCR güvenilirlik
    seen_count: int         # Kaç frame'de görüldü
    first_seen_sec: float   # İlk görüldüğü saniye
    is_opening_credit: bool # Açılış jeneriğinde mi
    best_tmdb_match: str    # En yakın TMDB eşleşmesi
    similarity: float       # Benzerlik skoru

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "job":            self.job,
            "confidence":     round(self.confidence, 4),
            "seen_frames":    self.seen_count,
            "first_seen_sec": round(self.first_seen_sec, 2),
            "opening_credit": self.is_opening_credit,
            "closest_tmdb":   self.best_tmdb_match,
            "similarity":     round(self.similarity, 4),
        }


@dataclass
class CreditsQA:
    tmdb_cast_count: int
    ocr_actor_count: int
    tmdb_looks_incomplete: bool          # cast < 10 ve OCR > 20
    missing_actors: List[MissingActor] = field(default_factory=list)
    missing_crew: List[MissingCrewMember] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        d = {
            "tmdb_cast_count":       self.tmdb_cast_count,
            "ocr_actor_count":       self.ocr_actor_count,
            "tmdb_looks_incomplete": self.tmdb_looks_incomplete,
            "missing_actor_count":   len(self.missing_actors),
            "missing_actors":        [a.to_dict() for a in self.missing_actors],
            "summary":               self.summary,
        }
        if self.missing_crew:
            d["missing_crew_count"] = len(self.missing_crew)
            d["missing_crew"] = [c.to_dict() for c in self.missing_crew]
        return d


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────
def check_missing_actors(
    ocr_results: list,
    tmdb_cast: list,
) -> CreditsQA:
    """OCR satırlarından TMDB'de eksik oyuncuları tespit eder.

    Parameters
    ----------
    ocr_results:
        OCRLine nesneleri veya dict listesi.
        Beklenen alanlar: text, avg_confidence/confidence,
        seen_count/frame_count, first_seen/first_seen_sec.
    tmdb_cast:
        TMDB'den gelen oyuncu listesi; her eleman "actor_name" veya "name"
        içeren bir dict'tir.

    Returns
    -------
    CreditsQA
        Eksik oyuncuları içeren rapor nesnesi.
        ``missing_actors`` boşsa ``summary`` da boş bırakılır.
    """
    # ── TMDB cast isimlerini normalize et ──────────────────────────────────────
    tmdb_names_norm: list[str] = []
    for entry in (tmdb_cast or []):
        raw = entry.get("actor_name") or entry.get("name") or ""
        n = _normalize(raw)
        if n:
            tmdb_names_norm.append(n)

    tmdb_count = len(tmdb_names_norm)

    # ── OCR satırlarını filtrele ───────────────────────────────────────────────
    candidates: list[MissingActor] = []

    for line in (ocr_results or []):
        # Değerleri hem OCRLine dataclass hem dict formatından oku
        if hasattr(line, "text"):
            text       = line.text or ""
            conf       = getattr(line, "avg_confidence", 0.0)
            seen       = getattr(line, "seen_count", 1)
            first_sec  = getattr(line, "first_seen", 0.0)
        else:
            text       = line.get("text") or ""
            conf       = line.get("avg_confidence", line.get("confidence", 0.0))
            seen       = line.get("seen_count", line.get("frame_count", 1))
            first_sec  = line.get("first_seen", line.get("first_seen_sec", 0.0))

        text = text.strip()

        # Filtre 1: En az MIN_WORDS kelime
        if len(text.split()) < MIN_WORDS:
            continue

        # Filtre 2: OCR güveni >= MIN_CONFIDENCE
        if conf < MIN_CONFIDENCE:
            continue

        # Filtre 3: En az MIN_SEEN_FRAMES frame VEYA açılış jeneriği
        is_opening = first_sec <= OPENING_CUTOFF_S
        if seen < MIN_SEEN_FRAMES and not is_opening:
            continue

        # Filtre 4: TMDB cast listesiyle eşleşmiyor
        norm = _normalize(text)
        if not norm:
            continue

        best_match = ""
        best_sim   = 0.0
        for tmdb_norm in tmdb_names_norm:
            s = _similarity(norm, tmdb_norm)
            if s > best_sim:
                best_sim   = s
                best_match = tmdb_norm

        if best_sim >= MATCH_THRESHOLD:
            continue  # Zaten TMDB'de var

        candidates.append(MissingActor(
            name=text,
            confidence=conf,
            seen_count=seen,
            first_seen_sec=first_sec,
            is_opening_credit=is_opening,
            best_tmdb_match=best_match,
            similarity=round(best_sim, 4),
        ))

    # ── Sırala: açılış jeneriği önce, sonra seen_count azalan ─────────────────
    candidates.sort(key=lambda a: (not a.is_opening_credit, -a.seen_count))

    # ── tmdb_looks_incomplete flag ─────────────────────────────────────────────
    # ocr_actor_count: tüm filtrelerden geçen aday sayısı değil,
    # OCR satır sayısından basit bir tahmin; ancak spesifikasyon "OCR > 20" derken
    # ocr_results uzunluğunu kastediyor — anlamlı bir kıyaslama için ham sayıyı kullan.
    raw_ocr_count         = len(ocr_results) if ocr_results else 0
    tmdb_looks_incomplete = (tmdb_count < 10) and (raw_ocr_count >= 20)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(candidates)
    if n == 0:
        summary = ""
    elif n <= 3:
        names = ", ".join(a.name for a in candidates)
        summary = f"⚠️ {n} isim TMDB'de eksik görünüyor: {names}"
    else:
        summary = f"⚠️ {n} isim TMDB'de eksik. TMDB kaydı muhtemelen eksik doldurulmuş."

    return CreditsQA(
        tmdb_cast_count=tmdb_count,
        ocr_actor_count=raw_ocr_count,
        tmdb_looks_incomplete=tmdb_looks_incomplete,
        missing_actors=candidates,
        summary=summary,
    )


# ── Crew QA ───────────────────────────────────────────────────────────────────
def check_missing_crew(
    ocr_results: list,
    tmdb_crew: list,
) -> List[MissingCrewMember]:
    """OCR satırlarından TMDB'de eksik crew üyelerini tespit eder.

    Parameters
    ----------
    ocr_results:
        OCRLine nesneleri veya dict listesi (check_missing_actors ile aynı format).
    tmdb_crew:
        TMDB'den gelen crew listesi; her eleman "name" içeren bir dict.

    Returns
    -------
    List[MissingCrewMember]
    """
    # TMDB crew isimlerini normalize et
    tmdb_names_norm: list[str] = []
    for entry in (tmdb_crew or []):
        raw = entry.get("name") or ""
        n = _normalize(raw)
        if n:
            tmdb_names_norm.append(n)

    candidates: list[MissingCrewMember] = []

    for line in (ocr_results or []):
        if hasattr(line, "text"):
            text      = line.text or ""
            conf      = getattr(line, "avg_confidence", 0.0)
            seen      = getattr(line, "seen_count", 1)
            first_sec = getattr(line, "first_seen", 0.0)
        else:
            text      = line.get("text") or ""
            conf      = line.get("avg_confidence", line.get("confidence", 0.0))
            seen      = line.get("seen_count", line.get("frame_count", 1))
            first_sec = line.get("first_seen", line.get("first_seen_sec", 0.0))

        text = text.strip()

        # Filtre 1: En az MIN_WORDS kelime
        if len(text.split()) < MIN_WORDS:
            continue

        # Filtre 2: OCR güveni
        if conf < MIN_CONFIDENCE:
            continue

        # Filtre 3: Frame sayısı veya açılış jeneriği
        is_opening = first_sec <= OPENING_CUTOFF_S
        if seen < MIN_SEEN_FRAMES and not is_opening:
            continue

        # Filtre 4: Rol başlığı blacklist (crew'a özgü)
        if text.lower() in _CREW_ROLE_BLACKLIST_EXACT:
            continue

        norm = _normalize(text)
        if not norm:
            continue

        # Filtre 5: Zaten TMDB crew listesinde var
        best_match = ""
        best_sim   = 0.0
        for tmdb_norm in tmdb_names_norm:
            s = _similarity(norm, tmdb_norm)
            if s > best_sim:
                best_sim   = s
                best_match = tmdb_norm

        if best_sim >= MATCH_THRESHOLD:
            continue

        candidates.append(MissingCrewMember(
            name=text,
            job="",
            confidence=conf,
            seen_count=seen,
            first_seen_sec=first_sec,
            is_opening_credit=is_opening,
            best_tmdb_match=best_match,
            similarity=round(best_sim, 4),
        ))

    # Sırala: açılış jeneriği önce, sonra seen_count azalan
    candidates.sort(key=lambda c: (not c.is_opening_credit, -c.seen_count))
    return candidates
