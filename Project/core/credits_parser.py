"""credits_parser.py — OCR satırları ve layout pair'lerden jenerik verisi çıkarır."""
import re

# ── Türkçe rol anahtar kelimeleri (inline) ────────────────────────────────────
_CAST_KEYWORDS = frozenset({
    "oynayanlar", "oyuncular", "cast", "başroller", "basroller",
    "starring", "oynayan", "oyuncu",
})
_DIRECTOR_KEYWORDS = frozenset({
    "yönetmen", "yonetmen", "director", "yöneten", "yoneten",
})
_CREW_KEYWORDS = frozenset({
    "görüntü yönetmeni", "goruntu yonetmeni",
    "müzik", "muzik", "kurgu", "senaryo",
    "yapımcı", "yapimci", "producer",
    "görüntü", "goruntu", "montaj",
    "kostüm", "kostum", "sanat yönetmeni", "sanat yonetmeni",
    "ses", "sound", "efekt", "effects",
    "makyaj", "makeup", "kamera", "camera",
    "ışık", "isik", "lighting",
    "prodüksiyon", "produksiyon",
    "executive producer", "koordinatör", "koordinator",
})
_PRODUCTION_KEYWORDS = frozenset({
    "yapım", "yapim", "production", "yapımevi", "yapimevi",
})
_COMPANY_SUFFIXES = (
    "film", "films", "filmcilik", "medya", "yapım", "yapimevi",
    "production", "productions", "stüdyo", "studyo", "studio",
    "televizyon",
)

_NOISE_RE = re.compile(
    r'^[\W\d_]+$'
    r'|^.{0,1}$'
    r'|^[\d\s.\-:,]+$'
)
_EXCESSIVE_SPECIAL_RE = re.compile(r'[^\w\s\u00C0-\u024F\u011E\u011F\u0130\u0131\u015E\u015F\u00DC\u00FC\u00D6\u00F6\u00C7\u00E7]{3,}')
_YEAR_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')


def _get_text(line) -> str:
    """OCRLine nesnesi veya dict'ten text al."""
    if isinstance(line, dict):
        return line.get("text", "")
    return getattr(line, "text", "")


def _is_noise(text: str) -> bool:
    """Gürültülü OCR satırı mı?"""
    t = text.strip()
    if len(t) < 2:
        return True
    if _NOISE_RE.match(t):
        return True
    if _EXCESSIVE_SPECIAL_RE.search(t):
        return True
    return False


def _detect_role_category(text: str):
    """Metin bir rol başlığı mı? Kategori adını döndür, değilse None."""
    low = text.strip().lower()
    for kw in _CAST_KEYWORDS:
        if kw in low:
            return "cast"
    for kw in _DIRECTOR_KEYWORDS:
        if kw in low:
            return "director"
    for kw in _CREW_KEYWORDS:
        if low == kw or low.startswith(kw + " ") or low.startswith(kw + ":"):
            return "crew"
    for kw in _PRODUCTION_KEYWORDS:
        if kw in low:
            return "production"
    for suffix in _COMPANY_SUFFIXES:
        if low.endswith(suffix) and len(low) > len(suffix) + 2:
            return "company"
    return None


class CreditsParser:
    """
    OCR satırları ve layout pair'lerden jenerik verisi çıkarır.

    Kullanım:
        parser = CreditsParser(turkish_name_db=name_db)
        parsed = parser.parse(ocr_lines, layout_pairs=layout_pairs)
        cdata  = parser.to_report_dict(parsed)
    """

    def __init__(self, turkish_name_db=None):
        self._name_db = turkish_name_db

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def parse(self, ocr_lines, layout_pairs=None) -> dict:
        """
        OCR satırları ve layout pair'lerden jenerik verisi çıkar.

        Args:
            ocr_lines:     list of dict {"text": str, ...} or OCRLine objects
            layout_pairs:  list of CastPair objects (from layout_analyzer.py)

        Returns:
            dict with keys: cast, crew, directors, production_companies,
                            production_info, film_title, year
        """
        cast = []
        crew = []
        directors = []
        production_companies = []
        production_info = []
        film_title = ""
        year = None

        # ── 1. Layout pair'lerden oyuncu listesi ──────────────────────────
        layout_actor_keys = set()
        for pair in (layout_pairs or []):
            char_name = (getattr(pair, "character_name", "") or "").strip()
            actor_name = (getattr(pair, "actor_name", "") or "").strip()
            confidence = float(getattr(pair, "confidence", 0.5))
            method = getattr(pair, "method", "layout")

            if not actor_name and not char_name:
                continue
            if actor_name and _is_noise(actor_name):
                continue

            # Swap kontrolü
            if (char_name and actor_name and self._name_db
                    and self._name_db.check_swap_risk(char_name, actor_name)):
                char_name, actor_name = actor_name, char_name

            # NameDB ile güven artırımı
            if actor_name and self._name_db and self._name_db.is_name(actor_name):
                confidence = min(1.0, confidence + 0.1)

            cast.append({
                "actor_name": actor_name,
                "character_name": char_name,
                "role": "Cast",
                "role_category": "cast",
                "confidence": round(confidence, 3),
                "frame": method,
            })
            if actor_name:
                layout_actor_keys.add(actor_name.lower().strip())

        # ── 2. OCR satırlarından sıralı ayrıştırma ────────────────────────
        current_category = "cast"
        current_role = ""

        for line in (ocr_lines or []):
            text = _get_text(line).strip()
            if not text:
                continue
            if _is_noise(text):
                continue

            # Rol başlığı tespiti
            cat = _detect_role_category(text)
            if cat:
                current_category = cat
                current_role = text.strip()
                if cat == "company" and len(text.strip()) > 3:
                    production_companies.append(text.strip())
                continue

            # Layout pair'de zaten varsa atla
            if text.lower() in layout_actor_keys:
                continue

            # Yıl tespiti
            year_match = _YEAR_RE.search(text)
            if year_match and not year:
                year = int(year_match.group(1))

            # Film başlığı: henüz hiç içerik yoksa ve başlık büyük harfliyse
            if not film_title and not cast and not crew:
                words = text.split()
                if words and all(w[0].isupper() for w in words if w):
                    film_title = text

            # Kategori bazlı ekleme
            if current_category == "director":
                directors.append({"name": text})
                crew.append({
                    "name": text,
                    "job": "Yönetmen",
                    "role": "Yönetmen",
                    "role_category": "crew",
                    "confidence": 0.7,
                    "frame": "",
                })
            elif current_category == "crew":
                crew.append({
                    "name": text,
                    "job": current_role or "Ekip",
                    "role": current_role or "Ekip",
                    "role_category": "crew",
                    "confidence": 0.6,
                    "frame": "",
                })
            elif current_category == "production":
                production_info.append(text)
            elif current_category == "company":
                production_companies.append(text)
            else:
                cast.append({
                    "actor_name": text,
                    "character_name": "",
                    "role": current_role or "Cast",
                    "role_category": "cast",
                    "confidence": 0.6,
                    "frame": "",
                })

        return {
            "cast": cast,
            "crew": crew,
            "directors": directors,
            "production_companies": production_companies,
            "production_info": production_info,
            "film_title": film_title,
            "year": year,
        }

    def to_report_dict(self, parsed: dict) -> dict:
        """
        parse() çıktısını export_engine.py ve pipeline_runner.py'nin
        beklediği şemaya dönüştür.
        """
        cast = parsed.get("cast") or []
        crew = parsed.get("crew") or []
        directors = parsed.get("directors") or []
        production_companies = parsed.get("production_companies") or []
        production_info = parsed.get("production_info") or []

        # Şirket tekilleştirme
        seen_companies = []
        seen_set: set = set()
        for c in production_companies:
            key = c.strip().lower()
            if key and key not in seen_set:
                seen_set.add(key)
                seen_companies.append(c.strip())

        return {
            "film_title": parsed.get("film_title") or "",
            "year": parsed.get("year"),
            "cast": cast,
            "crew": crew,
            "technical_crew": list(crew),
            "directors": directors,
            "production_companies": seen_companies,
            "production_info": production_info,
            "total_actors": len(cast),
            "total_crew": len(crew),
            "total_companies": len(seen_companies),
            "verification_status": "ocr_parsed",
        }
