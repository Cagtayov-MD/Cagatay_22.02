"""credits_parser.py — OCR satırları ve layout pair'lerden jenerik verisi çıkarır."""
import json
import re
from pathlib import Path

# ── Türkçe rol anahtar kelimeleri (inline fallback) ───────────────────────────
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

# ── JSON alias tablosu (lazy yükleme) ─────────────────────────────────────────
_ROLE_ALIASES = None  # {aliases: {low_str: (category, canonical)}, patterns: [...]}


def _load_role_aliases() -> dict | None:
    """credits_role_alias_tr.json'dan rol alias'larını yükle.

    Returns:
        dict with keys 'aliases' (str→(category,canonical)) and 'patterns'
        (list of (compiled_re, category, canonical)), or None on failure.
    """
    json_path = Path(__file__).resolve().parent.parent / "config" / "credits_role_alias_tr.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        alias_map: dict[str, tuple[str, str]] = {}
        pattern_list: list[tuple] = []
        for role in data.get("roles", []):
            canonical = role.get("canonical", "")
            category = role.get("category", "")
            if not canonical or not category:
                continue
            for alias in role.get("aliases", []):
                alias_map[alias.lower()] = (category, canonical)
            for pat in role.get("patterns", []):
                try:
                    pattern_list.append((re.compile(pat, re.IGNORECASE), category, canonical))
                except re.error:
                    pass
        return {"aliases": alias_map, "patterns": pattern_list}
    except (json.JSONDecodeError, OSError, ValueError) as e:
        import warnings
        warnings.warn(f"credits_role_alias_tr.json yüklenemedi: {e}", RuntimeWarning)
        return None


def _get_role_aliases() -> dict | None:
    """Alias tablosunu lazy olarak yükle ve döndür."""
    global _ROLE_ALIASES
    if _ROLE_ALIASES is None:
        _ROLE_ALIASES = _load_role_aliases()
    return _ROLE_ALIASES
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


def _get_confidence(line) -> float:
    """OCRLine nesnesi veya dict'ten confidence al."""
    if isinstance(line, dict):
        return float(line.get("avg_confidence", line.get("confidence", 0.6)))
    return float(getattr(line, "avg_confidence", getattr(line, "confidence", 0.6)))


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


def _extract_inline_name(text: str) -> str:
    """'role → NAME' veya 'role: NAME' formatından isim çıkar."""
    for sep in ("→", "->", ":", " – ", " - "):
        if sep in text:
            parts = text.split(sep, 1)
            if len(parts) == 2:
                name = parts[1].strip()
                if len(name) >= 3 and any(c.isalpha() for c in name):
                    return name
    return ""


def _is_subtitle_like(text: str) -> bool:
    """Altyazı benzeri metinleri tespit et."""
    t = text.strip()
    words = t.split()
    # 6+ kelimeli metin → muhtemelen altyazı, jenerik ismi değil
    if len(words) > 6:
        return True
    # Yoğun noktalama
    punct_count = sum(1 for c in t if c in '!?;…"\'')
    if punct_count >= 2:
        return True
    # Türkçe/İngilizce fiil ekleri (isim benzeyen son ekler çıkarıldı)
    lower = t.lower()
    verb_suffixes = ('yor', 'mak', 'mek', 'dır', 'dir', 'lar', 'ler', 'miş', 'muş',
                     'ing')
    for suffix in verb_suffixes:
        if lower.endswith(suffix) and len(t) > 15:
            return True
    return False


def _detect_role_category(text: str):
    """Metin bir rol başlığı mı? Kategori adını döndür, değilse None.

    Önce credits_role_alias_tr.json'daki alias tablosu kullanılır;
    JSON bulunamazsa hardcoded set'lere fallback yapılır.
    """
    low = text.strip().lower()

    # ── JSON alias tablosundan eşleştir (çok dilli) ───────────────────────
    aliases = _get_role_aliases()
    if aliases:
        # Tam eşleşme veya "alias: " / "alias " ile başlayan
        if low in aliases["aliases"]:
            return aliases["aliases"][low][0]
        for alias, (category, _canonical) in aliases["aliases"].items():
            if low == alias or low.startswith(alias + " ") or low.startswith(alias + ":"):
                return category
        # Regex pattern eşleşmesi
        for compiled, category, _canonical in aliases["patterns"]:
            if compiled.search(low):
                return category
        # Şirket suffix kontrolü (JSON'dan bağımsız)
        for suffix in _COMPANY_SUFFIXES:
            if low.endswith(suffix) and len(low) > len(suffix) + 2:
                return "company"
        return None

    # ── Fallback: hardcoded set'ler ───────────────────────────────────────
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
        unrecognized_streak = 0  # ardışık tanınmayan satır sayacı (director sonrası)
        NOISE_STREAK_LIMIT = 3   # bu kadar ardışık tanınmayan satır → noise bloğu

        for line in (ocr_lines or []):
            text = _get_text(line).strip()
            if not text:
                continue
            if _is_noise(text):
                continue
            if _is_subtitle_like(text):
                continue

            # Rol başlığı tespiti
            cat = _detect_role_category(text)
            if cat:
                # Tanınan bir kategori geldi — noise streak sıfırla
                unrecognized_streak = 0
                # İnline isim var mı kontrol et: "yonetmen → TULAY ERATALAY" gibi
                inline_name = _extract_inline_name(text)
                if inline_name and cat in ("crew", "director"):
                    current_category = cat
                    # Rol adını ayırıcıdan önce gelen kısımdan al
                    for sep in ("→", "->", ":", " – ", " - "):
                        if sep in text:
                            current_role = text.split(sep, 1)[0].strip()
                            break
                    else:
                        current_role = text.strip()
                    if cat == "director":
                        directors.append({"name": inline_name})
                    crew.append({
                        "name": inline_name,
                        "job": current_role,
                        "role": current_role,
                        "role_category": "crew",
                        "confidence": 0.7,
                        "frame": "",
                    })
                else:
                    current_category = cat
                    current_role = text.strip()
                    if cat == "company" and len(text.strip()) > 3:
                        production_companies.append(text.strip())
                continue

            # ── Sponsor/marka bloğu tespiti (director kategorisinde) ──────────
            # "director" kategorisindeyken gelen ve tanınmayan satırlar sayılır.
            # 3+ ardışık tanınmayan satır → bu blok reklam/sponsor, "noise" yap.
            if current_category == "director" or current_category == "noise":
                in_name_db = self._name_db and self._name_db.is_name(text)
                if not in_name_db:
                    # Tek-kelimelik tamamen büyük harf veya kısa bağlantılı kelime → direkt noise
                    words = text.split()
                    is_brand_like = (
                        (len(words) == 1 and text.isupper() and len(text) > 3)
                        or (len(words) == 1 and len(text) <= 3)
                    )
                    if is_brand_like:
                        current_category = "noise"
                        unrecognized_streak = NOISE_STREAK_LIMIT  # sayacı hemen doldur
                        continue
                    unrecognized_streak += 1
                    if unrecognized_streak >= NOISE_STREAK_LIMIT:
                        current_category = "noise"
                else:
                    # NameDB'de bulundu → gerçek bir isim, sayacı sıfırla
                    unrecognized_streak = 0
                    if current_category == "noise":
                        current_category = "director"

            # "noise" kategorisindeyken hiçbir listeye ekleme
            if current_category == "noise":
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
                # OCR güven değerini al
                conf = _get_confidence(line)
                is_verified = False
                match_method = ""

                if self._name_db:
                    if self._name_db.is_name(text):
                        # Tam eşleşme → güven artır
                        conf = min(1.0, conf + 0.2)
                        is_verified = True
                        match_method = "exact_db"
                    else:
                        canonical, score, match_method = self._name_db.find_with_method(text)
                        if canonical and score >= 0.80:
                            # Fonetik/fuzzy eşleşme → metni düzelt
                            text = canonical
                            conf = min(1.0, score)
                            is_verified = True
                        else:
                            match_method = ""
                        # NameDB'de bulunamasa da cast'a ekle — kural filtre + LLM sonra filtreler

                cast.append({
                    "actor_name": text,
                    "character_name": "",
                    "role": current_role or "Cast",
                    "role_category": current_category,
                    "confidence": round(conf, 3),
                    "is_verified_name": is_verified,
                    "match_method": match_method,
                    "frame": "ocr_sequential",
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
