"""
tmdb_verify.py — TMDB arama bazlı doğrulama.

ID girmeden çalışır. Film adı + oyuncu listesiyle TMDB'de arama yapar.

Doğruluk mantığı:
  - film_adı + en az 2 oyuncu eşleşirse → %100 güven
  - film_adı yok ama 3 farklı oyuncu eşleşirse → %100 güven
  - Eşleşme sağlanırsa tüm cast TMDB'deki kanonik isimlerle güncellenir.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

_THROTTLE_INTERVAL = 0.15  # saniye — saniyede ~6 istek, 40 limitinin çok altında

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False


def _norm(s: str) -> str:
    """Karşılaştırma için normalize: küçük harf, sadece alfanumerik."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _fold_text(text: str) -> str:
    """Aksanları sadeleştirerek ASCII-benzeri karşılaştırma metni üret."""
    nfkd = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _name_tokens(text: str) -> List[str]:
    folded = _fold_text(text).lower()
    return [tok for tok in re.findall(r"[a-z0-9]+", folded) if tok]


_MATCH_CANDIDATE_EXACT = frozenset({
    "the", "end", "presents", "present", "featuring", "introducing",
    "copyright", "fin", "play", "foul", "thanks", "thank",
    "dolby", "thx", "panavision", "technicolor",
    "director", "writer", "producer", "editor", "yönetmen",
    "yonetmen", "oyuncu", "kamera", "kurgu", "senaryo",
})

_MATCH_CANDIDATE_CONTAINS = frozenset({
    "all rights reserved",
    "no animals were harmed",
    "no animal was harmed",
    "with the assistance of",
    "sincere thanks",
    "special thanks",
    "additional assistant directors",
    "district administration",
    "war memorial opera",
    "h.m.s. pinafore",
    "seoul olympics",
    "technicolor",
    "panavision",
    "dolby",
    "copyright",
})


def _strong_person_name_match(query: str, choice: str, threshold: int) -> bool:
    """OCR bozulmuş kişi isimleri için kontrollü fuzzy eşleşme."""
    if not HAS_RAPIDFUZZ:
        return False

    query_tokens = _name_tokens(query)
    choice_tokens = _name_tokens(choice)
    if len(query_tokens) < 2 or len(choice_tokens) < 2:
        return False

    query_first, query_last = query_tokens[0], query_tokens[-1]
    choice_first, choice_last = choice_tokens[0], choice_tokens[-1]

    if min(len(query_first), len(choice_first)) < 4:
        return False
    if min(len(query_last), len(choice_last)) < 5:
        return False
    if len("".join(query_tokens)) < 10 or len("".join(choice_tokens)) < 10:
        return False

    first_ratio = fuzz.ratio(query_first, choice_first)
    last_ratio = fuzz.ratio(query_last, choice_last)
    full_ratio = fuzz.ratio(" ".join(query_tokens), " ".join(choice_tokens))

    if first_ratio >= 92 and last_ratio >= 58 and full_ratio >= max(74, threshold - 8):
        return True

    if query_first == choice_first and last_ratio >= 62 and full_ratio >= max(76, threshold - 6):
        return True

    return False


def _accept_fuzzy_candidate(query: str, choice: str, threshold: int) -> bool:
    """Kısa/generic token'larda fuzzy sonucu biraz daha sıkı doğrula."""
    if not HAS_RAPIDFUZZ:
        return True

    query_tokens = _name_tokens(query)
    choice_tokens = _name_tokens(choice)
    if len(query_tokens) < 2 or len(choice_tokens) < 2:
        return True

    if query_tokens[0] != choice_tokens[0]:
        return True

    query_last = query_tokens[-1]
    choice_last = choice_tokens[-1]
    if min(len(query_last), len(choice_last)) >= 5:
        return True

    return fuzz.ratio(query_last, choice_last) >= max(88, threshold)


def parse_imdb_characters(characters_raw: str | None) -> list[str]:
    """IMDB characters alanını parse et: '["Malo"]' → ['Malo']"""
    if not characters_raw:
        return []
    try:
        parsed = json.loads(characters_raw)
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    return [characters_raw]


_TURKISH_CHARS = frozenset("çşğıöüÇŞĞİÖÜ")
_CYRILLIC_RANGE = (0x0400, 0x04FF)


def _is_turkish(text: str) -> bool:
    """Türkçe özel karakterler içeriyorsa True döner."""
    return bool(set(text) & _TURKISH_CHARS)


def _is_cyrillic(text: str) -> bool:
    """Kiril karakter içeriyorsa True döner."""
    return any(_CYRILLIC_RANGE[0] <= ord(c) <= _CYRILLIC_RANGE[1] for c in (text or ""))


# Şirket/organizasyon isimlerini tespit etmek için anahtar kelimeler
COMPANY_KEYWORDS = {
    'productions', 'production', 'sa', 'ltd', 'inc', 'corp',
    'musicales', 'musicale', 'musique', 'editions', 'edition',
    'records', 'music', 'films', 'film', 'pictures', 'picture',
    'entertainment', 'distribution', 'distributeur', 'international',
    'studios', 'studio', 'canal', 'mk2', 'channel', 'media',
    'télévision', 'television', 'tv', 'arte', 'tve', 'rai',
}


def _looks_like_company(name: str) -> bool:
    """İsim bir şirket/organizasyon gibi görünüyor mu?"""
    name_lower = name.lower()
    words = name_lower.replace('-', ' ').split()
    # Herhangi bir kelime company_keywords'te ise şirkettir
    if any(w in COMPANY_KEYWORDS for w in words):
        return True
    # Tüm büyük harf + rakam/özel karakter ağırlıklı → şirket kodu
    alpha = [c for c in name if c.isalpha()]
    if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.8 and len(name) > 10:
        # Boşluk yoksa tek kelime büyük harf blok → şirket
        if ' ' not in name.strip():
            return True
    # Boşluksuz birleşik kelime içinde şirket anahtar kelimesi varsa şirkettir
    # (örn. 'editionsmusicales' → 'editions' + 'musicales')
    if ' ' not in name.strip() and len(name) > 8:
        if any(kw in name_lower for kw in COMPANY_KEYWORDS if len(kw) > 4):
            return True
    return False


def _title_candidates(title: str) -> List[str]:
    """
    Verilen film başlığından olası TMDB arama varyantları üret.
    Örnek: "Madam Bovary" → ["Madam Bovary", "Madame Bovary"]
    """
    candidates = [title]
    words = title.split()

    # Tüm büyük harfli başlık → title-case varyantı (örn. "MADAM BOVARY" → "Madam Bovary")
    if title == title.upper() and any(c.isalpha() for c in title):
        title_cased = title.title()
        if title_cased != title:
            candidates.append(title_cased)
        # Bundan sonraki dönüşümleri title-case üzerinden de yap
        words = title_cased.split()

    # "Madam" → "Madame" (Fransızca unvan düzeltmesi)
    new_words = ["Madame" if w.lower() == "madam" else w for w in words]
    variant = " ".join(new_words)
    if variant != title and variant not in candidates:
        candidates.append(variant)

    return list(dict.fromkeys(candidates))  # sıralı tekrarsız


def _fuzzy_match(query: str, choices: List[str], threshold: int = 85) -> Optional[str]:
    """Fuzzy eşleştirme — en iyi sonucu döndür."""
    if not query or not choices:
        return None
    qn = _norm(query)
    for c in choices:
        if _norm(c) == qn:
            return c
    if HAS_RAPIDFUZZ:
        res = rf_process.extractOne(query, choices, scorer=fuzz.WRatio,
                                    score_cutoff=threshold)
        if res and _accept_fuzzy_candidate(query, res[0], threshold):
            return res[0]
        folded_query = _fold_text(query)
        folded_choices = [_fold_text(c) for c in choices]
        folded_res = rf_process.extractOne(
            folded_query,
            folded_choices,
            scorer=fuzz.WRatio,
            score_cutoff=max(75, threshold - 3),
        )
        if folded_res:
            candidate = choices[folded_choices.index(folded_res[0])]
            if _accept_fuzzy_candidate(query, candidate, threshold):
                return candidate
        for choice in choices:
            if _strong_person_name_match(query, choice, threshold):
                return choice
    return None


def _is_reasonable_match_candidate(name: str) -> bool:
    """TMDB/IMDb eşleşmesine gönderilecek aday ismi filtrele."""
    raw = (name or "").strip()
    if len(raw) < 3:
        return False
    words_raw = raw.split()
    if _looks_like_company(raw):
        looks_like_titlecase_person = (
            len(words_raw) >= 2
            and not raw.isupper()
            and all(word and word[0].isupper() for word in words_raw)
        )
        if not looks_like_titlecase_person:
            return False
    if any(ch.isdigit() for ch in raw):
        return False

    lowered = _fold_text(raw).lower().strip()
    if lowered in _MATCH_CANDIDATE_EXACT:
        return False
    if any(term in lowered for term in _MATCH_CANDIDATE_CONTAINS):
        return False

    try:
        from core.name_verify import is_valid_person_name as _is_valid_person_name
    except Exception:
        _is_valid_person_name = None

    if _is_valid_person_name is not None and _is_valid_person_name(raw):
        return True

    words = _name_tokens(raw)
    if not words or len(words) > 4:
        return False
    if len(words) == 1:
        return len(words[0]) >= 4 and words[0] not in _MATCH_CANDIDATE_EXACT
    if sum(1 for tok in words if tok in _MATCH_CANDIDATE_EXACT) >= len(words) - 1:
        return False
    return True


def _filter_match_candidates(
    names: List[str],
    *,
    label: str,
    log_cb=None,
) -> List[str]:
    """Ham OCR adayı listesinden sadece eşleşmede kullanılacak isimleri seç."""
    filtered: List[str] = []
    seen: set[str] = set()
    rejected: List[str] = []

    for raw_name in names or []:
        name = (raw_name or "").strip()
        if not name:
            continue
        norm_name = _norm(name)
        if not norm_name or norm_name in seen:
            continue
        seen.add(norm_name)
        if _is_reasonable_match_candidate(name):
            filtered.append(name)
        else:
            rejected.append(name)

    if log_cb and rejected:
        log_cb(
            f"  [TMDB] {label} aday filtresi: {len(names or [])} → {len(filtered)} "
            f"(elenen örnekler: {rejected[:5]})"
        )

    return filtered


@dataclass
class TMDBVerifyResult:
    updated: bool
    reason: str
    hits: int = 0
    misses: int = 0
    confidence: str = "low"
    matched_title: str = ""
    matched_id: int = 0
    cast: List[Dict[str, Any]] = None  # TMDB kanonik cast listesi (ASR speaker matching için)
    crew: List[Dict[str, Any]] = None  # TMDB kanonik crew listesi
    year: int = 0
    keywords: List[str] = None
    genres: List[str] = None
    original_title: str = ""
    matched_via: str = ""  # "title" veya "cast_only" — LOCK kararı için
    reverse_score: float = 0.0          # Ters doğrulama puanı
    reverse_breakdown: Dict[str, Any] = None  # Puan detayları
    rejected: bool = False              # Ters doğrulama tarafından reddedildi mi
    ocr_title: str = ""                 # accept öncesi OCR'dan gelen orijinal başlık
    person_evidence: List[Dict[str, Any]] = None  # Strateji C/D eşleşen kişi kanıtları

    def __post_init__(self):
        if self.cast is None:
            self.cast = []
        if self.crew is None:
            self.crew = []
        if self.keywords is None:
            self.keywords = []
        if self.genres is None:
            self.genres = []
        if self.reverse_breakdown is None:
            self.reverse_breakdown = {}
        if self.person_evidence is None:
            self.person_evidence = []


class TMDBClient:
    BASE = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str = "", bearer_token: str = "",
                 language: str = "tr-TR", timeout: int = 15,
                 log_cb=None):
        self.api_key  = (api_key or "").strip()
        self.bearer   = (bearer_token or "").strip()
        self.language = (language or "tr-TR").strip()
        self.timeout  = timeout
        self._log     = log_cb or (lambda m: None)
        self._last_request_time = 0.0
        self._request_lock = threading.Lock()

    def enabled(self) -> bool:
        return bool(self.api_key or self.bearer)

    def _headers(self) -> Dict[str, str]:
        if self.bearer:
            return {"Authorization": f"Bearer {self.bearer}"}
        return {}

    def _params(self, extra: dict = None) -> Dict[str, str]:
        p = {"language": self.language}
        if self.api_key and not self.bearer:
            p["api_key"] = self.api_key
        if extra:
            p.update(extra)
        return p

    def _throttle_sleep(self, extra: float = 0.0):
        """Son istekten bu yana _THROTTLE_INTERVAL geçmediyse bekle."""
        with self._request_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            wait = max(0.0, _THROTTLE_INTERVAL - elapsed) + extra
            if wait > 0:
                time.sleep(wait)
            self._last_request_time = time.time()

    def _request(self, url: str, params: dict) -> dict:
        """Rate limit korumalı GET isteği. 429 gelirse Retry-After kadar bekler, max 3 deneme."""
        for attempt in range(3):
            self._throttle_sleep()
            r = requests.get(url, headers=self._headers(), params=params, timeout=self.timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                self._log(f"  [TMDB] Rate limit (429) — {wait}sn bekleniyor...")
                self._throttle_sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise Exception("TMDB rate limit aşıldı, 3 denemede sonuç alınamadı")

    def search_multi(self, query: str) -> List[Dict[str, Any]]:
        return self._request(
            f"{self.BASE}/search/multi",
            self._params({"query": query, "page": "1"}),
        ).get("results") or []

    def get_tv_credits(self, tv_id: int) -> Dict[str, Any]:
        return self._request(
            f"{self.BASE}/tv/{tv_id}/credits",
            self._params(),
        )

    def get_tv_aggregate_credits(self, tv_id: int) -> Dict[str, Any]:
        return self._request(
            f"{self.BASE}/tv/{tv_id}/aggregate_credits",
            self._params(),
        )

    def get_movie_credits(self, movie_id: int) -> Dict[str, Any]:
        return self._request(
            f"{self.BASE}/movie/{movie_id}/credits",
            self._params(),
        )

    def search_person(self, query: str) -> List[Dict[str, Any]]:
        return self._request(
            f"{self.BASE}/search/person",
            self._params({"query": query}),
        ).get("results") or []

    def find_by_imdb_id(self, imdb_id: str) -> Optional[Dict[str, Any]]:
        """IMDB tconst → TMDB eşleştirmesi. /find/{imdb_id} endpoint'i kullanır."""
        data = self._request(
            f"{self.BASE}/find/{imdb_id}",
            self._params({"external_source": "imdb_id"}),
        )
        results = data.get("movie_results") or data.get("tv_results") or []
        return results[0] if results else None

    def get_tv_details(self, tv_id: int) -> Dict[str, Any]:
        return self._request(
            f"{self.BASE}/tv/{tv_id}",
            self._params(),
        )

    def get_movie_details(self, movie_id: int) -> Dict[str, Any]:
        return self._request(
            f"{self.BASE}/movie/{movie_id}",
            self._params(),
        )

    def get_tv_keywords(self, tv_id: int) -> List[str]:
        results = self._request(
            f"{self.BASE}/tv/{tv_id}/keywords",
            self._params(),
        ).get("results") or []
        return [kw.get("name", "") for kw in results if kw.get("name")]

    def get_movie_keywords(self, movie_id: int) -> List[str]:
        kws = self._request(
            f"{self.BASE}/movie/{movie_id}/keywords",
            self._params(),
        ).get("keywords") or []
        return [kw.get("name", "") for kw in kws if kw.get("name")]

    def get_person_combined_credits(self, person_id: int) -> Dict[str, Any]:
        return self._request(
            f"{self.BASE}/person/{person_id}/combined_credits",
            self._params(),
        )


class TMDBVerify:
    """
    TMDB arama bazlı doğrulama.
    Film adı + oyuncu listesiyle TMDB'yi bulur, cast'ı kanonikleştirir.
    """

    MIN_ACTOR_MATCH = 3  # Film adı yoksa en az kaç oyuncu eşleşmeli

    def __init__(self, work_dir: str, api_key: str = "",
                 bearer_token: str = "", language: str = "tr-TR",
                 log_cb=None):
        self.work_dir   = work_dir
        self.client     = TMDBClient(api_key=api_key, bearer_token=bearer_token,
                                     language=language, log_cb=log_cb)
        self._log       = log_cb or (lambda m: None)
        self._cache_dir = os.path.join(work_dir, ".cache")
        os.makedirs(self._cache_dir, exist_ok=True)

    # ── Cache ────────────────────────────────────────────────────────
    def _cache_path(self, key: str) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in key)
        return os.path.join(self._cache_dir, f"tmdb_{safe}.json")

    def _load_cache(self, key: str, ttl: int = 7 * 86400) -> Optional[dict]:
        p = self._cache_path(key)
        if os.path.isfile(p):
            try:
                if time.time() - os.stat(p).st_mtime < ttl:
                    with open(p, encoding="utf-8") as f:
                        return json.load(f)
            except Exception:
                pass
        return None

    def _save_cache(self, key: str, data):
        try:
            with open(self._cache_path(key), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ── Ana doğrulama ────────────────────────────────────────────────
    def verify_credits(self, cdata: Dict[str, Any],
                       tv_id=None, movie_id=None,
                       is_series: bool = False) -> TMDBVerifyResult:
        """
        cdata içinden film adını ve oyuncu listesini al,
        TMDB'de ara, eşleşme bul, cast'ı kanonikleştir.

        tv_id / movie_id parametreleri artık kullanılmıyor,
        geriye dönük uyumluluk için bırakıldı.
        """
        if not self.client.enabled():
            return TMDBVerifyResult(False, "no api key/token")

        film_title = (cdata.get("film_title") or "").strip()

        cast_with_conf = [
            (
                (row.get("actor_name") or row.get("actor") or "").strip(),
                float(row.get("confidence", 0.5)),
            )
            for row in (cdata.get("cast") or [])
            if isinstance(row, dict)
        ]
        cast_with_conf.sort(key=lambda x: x[1], reverse=True)  # en güvenilir önce
        cast_names = [n for n, _ in cast_with_conf if len(n) >= 3]

        # ── Yönetmen isimlerini çek ──────────────────────────────────────────
        director_names: List[str] = []
        for row in (cdata.get("directors") or []):
            if isinstance(row, dict):
                name = (row.get("name") or "").strip()
                if len(name) >= 3:
                    director_names.append(name)
        # crew listesinden de yönetmen rolündeki isimleri al
        for row in (cdata.get("crew") or []):
            if not isinstance(row, dict):
                continue
            job = (row.get("job") or row.get("role") or "").lower()
            if "yönetmen" in job or "yonetmen" in job or "director" in job:
                name = (row.get("name") or "").strip()
                if len(name) >= 3 and name not in director_names:
                    director_names.append(name)

        # Yönetmen dışındaki doğrulanmış crew isimleri (senarist, yapımcı, vb.)
        crew_names: List[str] = []
        for row in (cdata.get("crew") or []):
            if not isinstance(row, dict):
                continue
            job = (row.get("job") or row.get("role") or "").lower()
            # Zaten director_names'e girenleri atla
            if "yönetmen" in job or "yonetmen" in job or "director" in job:
                continue
            name = (row.get("name") or "").strip()
            if len(name) >= 3 and name not in director_names and name not in cast_names:
                crew_names.append(name)

        cast_names = _filter_match_candidates(cast_names, label="cast", log_cb=self._log)
        director_names = _filter_match_candidates(
            director_names, label="yönetmen", log_cb=self._log
        )
        crew_names = _filter_match_candidates(crew_names, label="crew", log_cb=self._log)

        cdata["_tmdb_match_candidates"] = {
            "cast": list(cast_names),
            "directors": list(director_names),
            "crew": list(crew_names),
        }

        original_title_input = (
            (cdata.get("original_title") or "").strip() or
            (cdata.get("original_name") or "").strip() or
            (cdata.get("_gemini_suggested_title") or "").strip()
        )

        if not cast_names and not director_names and not crew_names:
            if not film_title and not original_title_input:
                return TMDBVerifyResult(False, "no cast or directors to verify")
            self._log(
                "  [TMDB] Kişi adayları filtrede elendi; yalnızca başlık kanıtı ile arama deneniyor"
            )

        if not cast_names:
            if director_names or crew_names:
                self._log(
                    f"  [TMDB] Cast boş — yönetmen={len(director_names)}, diğer crew={len(crew_names)} kişiyle aranıyor"
                )

        # ── Anomali tespiti: gerçek dışı yüksek cast sayısı ──
        if len(cast_names) > 500:
            self._log(f"  [TMDB] ⚠️ Anormal cast sayısı: {len(cast_names)} — muhtemelen OCR çöpü")
            self._log(f"  [TMDB] Sadece en güvenilir ilk 50 isim kullanılıyor")
            cast_names = cast_names[:50]

        # ── OCR yılı (Strateji B crew kıyaslaması için) ──
        _ocr_year_pre = 0
        _raw_year_pre = cdata.get("year") or cdata.get("_ocr_year")
        if _raw_year_pre:
            try:
                _ocr_year_pre = int(str(_raw_year_pre)[:4])
            except (ValueError, TypeError):
                pass
        if not _ocr_year_pre:
            _fname_pre = str(cdata.get("filename") or cdata.get("_source_file") or "")
            _m_pre = re.search(r'\b(1[9][0-9]{2}|20[0-2][0-9])\b', _fname_pre)
            if _m_pre:
                _ocr_year_pre = int(_m_pre.group(1))

        # ── Crew dict'leri (Strateji B DoP/editor kıyaslaması için) ──
        _ocr_crew_dicts = [
            row for row in (cdata.get("crew") or [])
            if isinstance(row, dict) and (row.get("name") or "").strip()
        ]

        # TMDB'de eşleşme bul
        tmdb_entry, kind, matched_via, person_evidence = self._find_tmdb_entry(
            film_title, cast_names, director_names,
            original_title=original_title_input,
            crew_names=crew_names,
            is_series=is_series,
            ocr_year=_ocr_year_pre,
            ocr_crew_dicts=_ocr_crew_dicts,
        )

        if not tmdb_entry:
            if person_evidence:
                self._log(
                    f"  [TMDB] Eşleşme bulunamadı ama {len(person_evidence)} kişi kanıtı korunuyor"
                )
            else:
                self._log(f"  [TMDB] Eşleşme bulunamadı")
            return TMDBVerifyResult(False, "tmdb match not found",
                                    person_evidence=person_evidence)

        tmdb_id    = tmdb_entry.get("id", 0)
        tmdb_title = (tmdb_entry.get("name") or tmdb_entry.get("title") or "").strip()
        self._log(f"  [TMDB] ✓ '{tmdb_title}' (id:{tmdb_id}, tür:{kind})")

        # tmdb_entry'den yıl çek
        date_str = (tmdb_entry.get("first_air_date") or tmdb_entry.get("release_date") or "")
        tmdb_year = 0
        if date_str and len(date_str) >= 4:
            try:
                tmdb_year = int(date_str[:4])
            except ValueError:
                pass

        # Credits çek
        credits_data = self._fetch_credits(kind, tmdb_id)
        if not credits_data:
            return TMDBVerifyResult(False, "tmdb credits fetch failed",
                                    matched_title=tmdb_title, matched_id=tmdb_id)

        # ── İleri yön eşleşme sayısı (ters doğrulama için) ──
        _tmdb_cast_names = self._extract_names(credits_data, section="cast")
        _forward_hits    = self._count_matches(cast_names, _tmdb_cast_names)
        _forward_misses  = len(cast_names) - _forward_hits

        # ── OCR yılı kaynakları (öncelik sırasıyla) ──
        _ocr_year = 0
        _raw_year = cdata.get("year") or cdata.get("_ocr_year")
        if _raw_year:
            try:
                _ocr_year = int(str(_raw_year)[:4])
            except (ValueError, TypeError):
                pass
        if not _ocr_year:
            # Dosya adından parse: ilk 4 haneli sayı 1900-2030 arasında
            import re as _re
            _fname = str(cdata.get("filename") or cdata.get("_source_file") or "")
            _m = _re.search(r'\b(1[9][0-9]{2}|20[0-2][0-9])\b', _fname)
            if _m:
                try:
                    _ocr_year = int(_m.group(1))
                except ValueError:
                    pass

        # ── Ters doğrulama HER ZAMAN çalışacak ──
        _rv_accepted, _rv_score, _rv_breakdown = self._reverse_validate(
            ocr_title=film_title,
            ocr_cast_names=cast_names,
            ocr_director_names=director_names,
            ocr_year=_ocr_year,
            tmdb_entry=tmdb_entry,
            credits_data=credits_data,
            forward_hits=_forward_hits,
            forward_misses=_forward_misses,
        )

        if not _rv_accepted:
            if person_evidence:
                self._log(
                    f"  [TMDB] Ters doğrulama reddetti ama "
                    f"{len(person_evidence)} kişi kanıtı korunuyor"
                )
            return TMDBVerifyResult(
                updated=False,
                reason="reverse_validation_rejected",
                matched_title=tmdb_title,
                matched_id=0,
                reverse_score=_rv_score,
                reverse_breakdown=_rv_breakdown,
                rejected=True,
                person_evidence=person_evidence,
            )

        # Cast kanonikleştir
        updated, hits, misses = self._canonicalize(cdata, credits_data)

        # TMDB cast/crew listesini result'a ekle (ASR speaker matching için)
        tmdb_cast = [
            {"name": (item.get("name") or "").strip(),
             "character": (item.get("character") or "").strip(),
             "tmdb_id": item.get("id", 0),
             "order": item.get("order", 999)}
            for item in (credits_data.get("cast") or [])
            if (item.get("name") or "").strip()
        ]
        tmdb_crew = [
            {"name": (item.get("name") or "").strip(),
             "job": (item.get("job") or "").strip(),
             "department": (item.get("department") or "").strip(),
             "tmdb_id": item.get("id", 0)}
            for item in (credits_data.get("crew") or [])
            if (item.get("name") or "").strip()
        ]

        # TMDB keywords ve genres çek
        tmdb_keywords: List[str] = []
        tmdb_genres: List[str] = []
        details = {}
        try:
            if kind == "tv":
                details = self.client.get_tv_details(int(tmdb_id))
                tmdb_keywords = self.client.get_tv_keywords(int(tmdb_id))
            else:
                details = self.client.get_movie_details(int(tmdb_id))
                tmdb_keywords = self.client.get_movie_keywords(int(tmdb_id))
            tmdb_genres = [g.get("name", "") for g in (details.get("genres") or []) if g.get("name")]
        except Exception as e:
            self._log(f"  [TMDB] Keywords/genres çekme hatası: {e}")

        # original_title: TMDB'den orijinal dildeki film/dizi adı
        original_title = ""
        if details:
            original_title = (details.get("original_title") or details.get("original_name") or "").strip()
        if not original_title:
            original_title = (tmdb_entry.get("original_title") or tmdb_entry.get("original_name") or "").strip()

        # OCR'dan gelen orijinal başlığı koru — TMDB ile ezilmeden önce sakla
        # (Başarılı TMDB eşleşmesinde, canonicalization flag'ından bağımsız olarak)
        if tmdb_title:
            _ocr_film_title = (cdata.get("film_title") or "").strip()
            if _ocr_film_title and _ocr_film_title != tmdb_title:
                if not cdata.get("ocr_title"):
                    cdata["ocr_title"] = _ocr_film_title

        if updated:
            cdata["tmdb_verified"] = True
            cdata["tmdb_title"]    = tmdb_title
            cdata["tmdb_id"]       = tmdb_id
            cdata["tmdb_type"]     = kind
            cdata["film_title"]    = tmdb_title
            cdata["tmdb_original_title"] = original_title

        return TMDBVerifyResult(
            updated=updated, reason="ok",
            hits=hits, misses=misses,
            confidence="high",
            matched_title=tmdb_title, matched_id=tmdb_id,
            cast=tmdb_cast, crew=tmdb_crew,
            year=tmdb_year,
            keywords=tmdb_keywords,
            genres=tmdb_genres,
            original_title=original_title,
            matched_via=matched_via,
            reverse_score=_rv_score,
            reverse_breakdown=_rv_breakdown,
            ocr_title=film_title,
            person_evidence=person_evidence,
        )

    # ── TMDB'de eşleşme bul ─────────────────────────────────────────
    def _find_tmdb_entry(self, film_title: str,
                         cast_names: List[str],
                         director_names: List[str] = None,
                         original_title: str = "",
                         crew_names: List[str] = None,
                         is_series: bool = False,
                         ocr_year: int = 0,
                         ocr_crew_dicts: list = None) -> tuple:
        """
        Strateji A → B → C → D sırasıyla dene.

          A: Film adı + Yönetmen + 1-2 oyuncu ile ara
          B: Film adı + Yönetmen (oyuncular devre dışı)
             - DİZİYSE: direkt kabul
             - FİLMSE: yıl/DoP/editor crew kıyaslaması
          C: Oyuncu/yönetmen kişi araması (fuzzy isim doğrulaması yok)
          D: Oyuncu/yönetmen kişi araması (rapidfuzz >= 80 fuzzy isim doğrulaması)

        Return: (entry, kind, matched_via, person_evidence)
          - person_evidence: Strateji C/D sırasında TMDB'de doğrulanan kişilerin listesi;
            film eşleşmesi başarısız olsa bile korunur.
        """
        director_names = director_names or []
        crew_names = crew_names or []
        ocr_crew_dicts = ocr_crew_dicts or []
        cast_names = _filter_match_candidates(cast_names, label="cast", log_cb=self._log)
        director_names = _filter_match_candidates(
            director_names, label="yönetmen", log_cb=self._log
        )
        crew_names = _filter_match_candidates(crew_names, label="crew", log_cb=self._log)

        def _director_matches_crew(credits: dict) -> bool:
            """TMDB crew'unda yönetmen eşleşmesi var mı?"""
            tmdb_crew = self._extract_names(credits, section="crew")
            return any(
                _fuzzy_match(d, tmdb_crew, threshold=80)
                for d in director_names
            )

        def _search_by_title(title_attempt: str) -> list:
            """Başlıkla TMDB'de ara, sonuçları cache ile döndür."""
            self._log(f"  [TMDB] Aranan başlık: '{title_attempt}'")
            cache_key = f"search_multi_{_norm(title_attempt)}"
            results = self._load_cache(cache_key)
            if results is None:
                try:
                    results = self.client.search_multi(title_attempt)
                    self._save_cache(cache_key, results)
                except Exception as e:
                    self._log(f"  [TMDB] Arama hatası: {e}")
                    results = []
            self._log(f"  [TMDB] {len(results or [])} sonuç bulundu")
            return results or []

        def _b_crew_ok(entry: dict, credits: dict) -> bool:
            """Strateji B film kıyaslaması: yıl, DoP, editor eşleşmesi (en az 1 kriter)."""
            _DOP_JOBS = {
                "director of photography", "dop", "dp",
                "görüntü yönetmeni", "chef opérateur", "cinematographer",
            }
            _EDITOR_JOBS = {"editor", "kurgu", "film editor", "montaj", "monteur"}

            score = 0

            # Yıl karşılaştırması
            if ocr_year:
                date_str = (entry.get("first_air_date") or entry.get("release_date") or "")
                tmdb_year = 0
                if date_str and len(date_str) >= 4:
                    try:
                        tmdb_year = int(date_str[:4])
                    except ValueError:
                        pass
                if tmdb_year and abs(tmdb_year - ocr_year) <= 1:
                    self._log(
                        f"  [TMDB] Strateji B yıl eşleşti: OCR={ocr_year} TMDB={tmdb_year}"
                    )
                    score += 1

            # DoP ve Editor karşılaştırması
            for crew_item in (credits.get("crew") or []):
                job = (crew_item.get("job") or crew_item.get("department") or "").lower()
                name = (crew_item.get("name") or "").strip()
                if not name:
                    continue
                is_dop = any(j in job for j in _DOP_JOBS)
                is_editor = any(j in job for j in _EDITOR_JOBS)
                if not (is_dop or is_editor):
                    continue
                for ocr_row in ocr_crew_dicts:
                    if not isinstance(ocr_row, dict):
                        continue
                    ocr_job = (ocr_row.get("job") or ocr_row.get("role") or "").lower()
                    ocr_name = (ocr_row.get("name") or "").strip()
                    if not ocr_name:
                        continue
                    ocr_is_dop = any(j in ocr_job for j in _DOP_JOBS)
                    ocr_is_editor = any(j in ocr_job for j in _EDITOR_JOBS)
                    if (is_dop and not ocr_is_dop) or (is_editor and not ocr_is_editor):
                        continue
                    if HAS_RAPIDFUZZ:
                        ratio = fuzz.ratio(ocr_name.lower(), name.lower())
                        if ratio >= 80:
                            self._log(
                                f"  [TMDB] Strateji B crew eşleşti: "
                                f"'{ocr_name}' ≈ '{name}' ({ratio:.0f}%)"
                            )
                            score += 1
                            break
                    else:
                        if _norm(ocr_name) == _norm(name):
                            self._log(f"  [TMDB] Strateji B crew eşleşti: '{name}'")
                            score += 1
                            break
                if score >= 1:
                    break

            # Karşılaştırılacak OCR verisi yoksa reddetme — kanıt eksikliği kabul sayılır
            if score == 0 and not ocr_year and not ocr_crew_dicts:
                self._log(
                    f"  [TMDB] Strateji B: kıyaslanacak OCR verisi yok — "
                    f"başlık + yönetmen eşleşmesi yeterli kabul ediliyor"
                )
                return True

            return score >= 1

        # Başlık kaynakları: önce birincil başlık, sonra orijinal başlık
        _norm_film = _norm(film_title or "")
        _norm_orig = _norm(original_title or "")
        _orig_usable = bool(original_title and _norm_orig and _norm_orig != _norm_film)
        _SMALL_CAST_LIMIT = 5    # bu değerin altında küçük cast eşiği (2) kullanılır
        _ORIG_MIN_CAST_ABS = 3   # orijinal başlık için mutlak minimum cast eşleşmesi
        _ORIG_CAST_DIVISOR = 3   # orijinal başlık için dinamik eşik: cast / bu sayı

        title_sources = []
        if _orig_usable:
            orig_min_cast = (
                max(_ORIG_MIN_CAST_ABS, len(cast_names) // _ORIG_CAST_DIVISOR)
                if len(cast_names) >= _SMALL_CAST_LIMIT
                else 2
            )
            title_sources.append(("original", original_title, orig_min_cast))
        if film_title:
            title_sources.append(("primary", film_title, 2))
        elif original_title and not _orig_usable:
            self._log(f"  [TMDB] Orijinal başlık yerel başlıkla aynı, tekrar aranmıyor")

        # ── Strateji A: Film adı + Yönetmen + 1-2 oyuncu ──────────────────────
        self._log(f"  [TMDB] Strateji A test ediliyor...")
        for _src_label, _src_title, _min_cast in title_sources:
            if _src_label == "original":
                self._log(
                    f"  [TMDB] Orijinal başlık TMDB'ye iletiliyor: '{_src_title}' "
                    f"(min cast eşleşme: {_min_cast})"
                )
            for attempt in _title_candidates(_src_title):
                results = _search_by_title(attempt)
                for r in results[:10]:
                    kind = r.get("media_type", "")
                    title = r.get("name") or r.get("title") or "?"
                    if kind not in ("tv", "movie"):
                        continue
                    self._log(f"  [TMDB] Kontrol: '{title}' ({kind}, id:{r['id']})")
                    credits = self._fetch_credits(kind, r["id"])
                    if not credits:
                        continue
                    tmdb_names = self._extract_names(credits)
                    matched = self._count_matches(cast_names, tmdb_names)
                    self._log(f"  [TMDB]   → {matched}/{len(cast_names)} oyuncu eşleşti")
                    _via = "title" if _src_label == "primary" else "original_title"
                    # Strateji A: sadece-cast eşleşmesi devre dışı —
                    # başlık + yönetmen zorunlu (Sorun 5: Flashdance %64 kayıp giderme)
                    if matched >= 1 and director_names and _director_matches_crew(credits):
                        self._log(
                            f"  [TMDB] Strateji A başarılı: '{title}' — "
                            f"başlık + {matched} oyuncu + yönetmen eşleşti"
                        )
                        return r, kind, _via, []
                    if _src_label == "original" and not director_names and matched >= _min_cast:
                        self._log(
                            f"  [TMDB] Strateji A başarılı: '{title}' — "
                            f"orijinal başlık + güçlü cast eşleşmesi ({matched}/{len(cast_names)})"
                        )
                        return r, kind, _via, []
        self._log(f"  [TMDB] Strateji A başarısız")

        # ── Strateji B: Sadece Film adı + Yönetmen (oyuncular devre dışı) ──────
        self._log(f"  [TMDB] Strateji B test ediliyor...")
        if director_names:
            for _src_label, _src_title, _ in title_sources:
                if _src_label == "original":
                    self._log(
                        f"  [TMDB] Strateji B: orijinal başlık deneniyor: '{_src_title}'"
                    )
                for attempt in _title_candidates(_src_title):
                    results = _search_by_title(attempt)
                    for r in results[:10]:
                        kind = r.get("media_type", "")
                        title = r.get("name") or r.get("title") or "?"
                        if kind not in ("tv", "movie"):
                            continue
                        self._log(f"  [TMDB] Kontrol: '{title}' ({kind}, id:{r['id']})")
                        credits = self._fetch_credits(kind, r["id"])
                        if not credits:
                            continue
                        if _director_matches_crew(credits):
                            _via = "title" if _src_label == "primary" else "original_title"
                            if is_series:
                                self._log(
                                    f"  [TMDB] Strateji B başarılı: DİZİ — "
                                    f"'{title}' başlık + yönetmen eşleşti (is_series=True)"
                                )
                                return r, kind, _via, []
                            else:
                                if _b_crew_ok(r, credits):
                                    self._log(
                                        f"  [TMDB] Strateji B başarılı: FİLM — "
                                        f"'{title}' crew kıyaslaması geçti"
                                    )
                                    return r, kind, _via, []
                                else:
                                    self._log(
                                        f"  [TMDB] Strateji B: '{title}' — "
                                        f"crew kıyaslaması yetersiz, devam ediliyor"
                                    )
        else:
            self._log(f"  [TMDB] Strateji B: yönetmen bilgisi yok — atlandı")
        self._log(f"  [TMDB] Strateji B başarısız")

        # ── Ortak kişi araması mantığı (C ve D için) ──────────────────────────
        def _run_person_search(fuzzy_validate: bool) -> tuple:
            """Kişi aramasıyla ortak film/dizi bul.

            fuzzy_validate=False — Strateji C: isim doğrulaması yok
            fuzzy_validate=True  — Strateji D: rapidfuzz ratio >= 80 gerekli

            Dördüncü eleman: kişi kanıtı listesi (eşleşen kişilerin yapılandırılmış kaydı)
            """
            work_matches: Dict[int, dict] = {}
            matched_persons: List[Dict[str, Any]] = []  # kişi kanıtı biriktirici

            cast_names_set = set(cast_names)
            actor_candidates = [n for n in cast_names if not _looks_like_company(n)]
            filtered_directors = [d for d in director_names if d not in cast_names_set][:5]

            all_known = cast_names_set | set(director_names)
            filtered_crew = [
                n for n in crew_names
                if n not in all_known and not _looks_like_company(n)
            ][:10]

            _lbl = "C" if not fuzzy_validate else "D"
            self._log(
                f"  [TMDB] Strateji {_lbl} aranan kişiler: oyuncu={actor_candidates[:5]}, "
                f"yönetmen={filtered_directors}, diğer_crew={filtered_crew[:3]}"
            )

            _cast_set_in_search = set(actor_candidates[:20])
            _director_set_in_search = set(d for d in director_names if d not in cast_names_set)

            persons_to_search = (
                actor_candidates[:20]
                + [d for d in director_names if d not in cast_names_set]
                + filtered_crew
            )

            for idx, actor in enumerate(persons_to_search):
                if idx > 0 and idx % 5 == 0:
                    self._log(
                        f"  [TMDB] Strateji {_lbl} throttle: "
                        f"{idx}/{len(persons_to_search)} kişi işlendi, bekleniyor..."
                    )
                    self.client._throttle_sleep(0.5)

                # Rol tespiti: kişi arama kanıtında kullanılır
                if actor in _cast_set_in_search:
                    _actor_role = "cast"
                elif actor in _director_set_in_search:
                    _actor_role = "director"
                else:
                    _actor_role = "crew"

                cache_key = f"search_person_{_norm(actor)}"
                persons = self._load_cache(cache_key)
                if persons is None:
                    try:
                        persons = self.client.search_person(actor)
                        self._save_cache(cache_key, persons)
                    except Exception:
                        continue

                for person in (persons or [])[:2]:
                    if fuzzy_validate:
                        person_name = (person.get("name") or "").strip()
                        if not person_name:
                            continue
                        if HAS_RAPIDFUZZ:
                            ratio = fuzz.ratio(actor.lower(), person_name.lower())
                            if ratio < 80:
                                name_words = [_norm(w) for w in person_name.split()]
                                surname_match = len(_norm(actor)) >= 4 and _norm(actor) in name_words
                                # Prefix match: OCR adı TMDB adının öneki
                                # (e.g. "Anne Pernod" → "Anne Pernod-Sawada")
                                ocr_lc = actor.lower().strip()
                                tmdb_lc = person_name.lower().strip()
                                prefix_match = (
                                    len(ocr_lc) >= 8
                                    and tmdb_lc.startswith(ocr_lc)
                                    and (len(tmdb_lc) == len(ocr_lc)
                                         or tmdb_lc[len(ocr_lc)] in (' ', '-'))
                                )
                                if not surname_match and not prefix_match:
                                    self._log(
                                        f"  [TMDB] Strateji D: '{person_name}' OCR ismiyle "
                                        f"('{actor}') eşleşmiyor ({ratio:.0f}%) — atlanıyor"
                                    )
                                    continue
                        else:
                            if _norm(actor) != _norm(person_name):
                                continue

                    person_id = person.get("id")
                    if not person_id:
                        continue
                    _tmdb_person_name = (person.get("name") or "").strip()
                    cache_key_credits = f"person_combined_{person_id}"
                    person_credits = self._load_cache(cache_key_credits)
                    if person_credits is None:
                        try:
                            person_credits = self.client.get_person_combined_credits(person_id)
                            self._save_cache(cache_key_credits, person_credits)
                        except Exception:
                            # combined_credits başarısız olursa known_for fallback
                            _contributed_via_known_for = False
                            for work in (person.get("known_for") or []):
                                kind = work.get("media_type", "")
                                if kind not in ("tv", "movie"):
                                    continue
                                wid = work.get("id", 0)
                                if wid:
                                    if wid not in work_matches:
                                        work_matches[wid] = {"entry": work, "kind": kind, "count": 0}
                                    work_matches[wid]["count"] += 1
                                    _contributed_via_known_for = True
                            # Kişi kanıtına ekle (known_for ile katkı yaptıysa)
                            if _contributed_via_known_for and _tmdb_person_name:
                                if not any(pe["ocr_name"] == actor for pe in matched_persons):
                                    matched_persons.append({
                                        "ocr_name": actor,
                                        "tmdb_name": _tmdb_person_name,
                                        "tmdb_id": person_id,
                                        "role": _actor_role,
                                        "source_strategy": _lbl,
                                    })
                            continue

                    # combined_credits cast + crew bölümlerini tara
                    _contributed_via_combined = False
                    seen_work_ids: set = set()
                    for section in ("cast", "crew"):
                        for work in (person_credits.get(section) or []):
                            kind = work.get("media_type", "")
                            if kind not in ("tv", "movie"):
                                continue
                            wid = work.get("id", 0)
                            if not wid or wid in seen_work_ids:
                                continue
                            seen_work_ids.add(wid)
                            if wid not in work_matches:
                                work_matches[wid] = {"entry": work, "kind": kind, "count": 0}
                            work_matches[wid]["count"] += 1
                            _contributed_via_combined = True

                    # Kişi kanıtına ekle (combined_credits ile katkı yaptıysa)
                    if _contributed_via_combined and _tmdb_person_name:
                        if not any(pe["ocr_name"] == actor for pe in matched_persons):
                            matched_persons.append({
                                "ocr_name": actor,
                                "tmdb_name": _tmdb_person_name,
                                "tmdb_id": person_id,
                                "role": _actor_role,
                                "source_strategy": _lbl,
                            })

            if work_matches:
                best = max(work_matches.values(), key=lambda x: x["count"])
                min_match_threshold = 1 if not cast_names else self.MIN_ACTOR_MATCH
                if best["count"] >= min_match_threshold:
                    credits = self._fetch_credits(best["kind"], best["entry"]["id"])
                    if credits:
                        tmdb_cast_names = self._extract_names(credits, section="cast")
                        tmdb_crew_names = self._extract_names(credits, section="crew")
                        matched = self._count_matches(cast_names, tmdb_cast_names)
                        director_matched = sum(
                            1 for d in director_names
                            if _fuzzy_match(d, tmdb_crew_names, threshold=82)
                        )
                        total_matched = matched + director_matched
                        if total_matched >= min_match_threshold:
                            self._log(
                                f"  [TMDB] {matched} oyuncu + {director_matched} yönetmen "
                                f"eşleşti → %100 güven"
                            )
                            return best["entry"], best["kind"], "cast_only", matched_persons

            return None, "", "", matched_persons

        # ── Strateji C: Oyuncu isimleri + Yönetmen (başlık yok, fuzzy doğrulama yok) ──
        self._log(f"  [TMDB] Strateji C test ediliyor...")
        result_c = _run_person_search(fuzzy_validate=False)
        if result_c[0] is not None:
            self._log(f"  [TMDB] Strateji C başarılı")
            return result_c
        self._log(f"  [TMDB] Strateji C başarısız")

        # ── Strateji D: Oyuncuları tek tek varlık kontrolü (fuzzy >= 80) ──────
        self._log(f"  [TMDB] Strateji D test ediliyor...")
        result_d = _run_person_search(fuzzy_validate=True)
        if result_d[0] is not None:
            self._log(f"  [TMDB] Strateji D başarılı")
            return result_d

        # Stratejiler başarısız — C ve D'nin kişi kanıtlarını birleştir
        _evidence_c = result_c[3]
        _evidence_d = result_d[3]
        _c_ocr_names = {pe["ocr_name"] for pe in _evidence_c}
        _combined_evidence = list(_evidence_c) + [
            pe for pe in _evidence_d if pe["ocr_name"] not in _c_ocr_names
        ]

        self._log(f"  [TMDB] Tüm stratejiler başarısız — ADIM 4'e geçiliyor")
        return None, "", "", _combined_evidence

    # ── Credits çek ─────────────────────────────────────────────────
    def _fetch_credits(self, kind: str, mid: int) -> Optional[dict]:
        cache_key = f"credits_{kind}_{mid}_{_norm(self.client.language)}"
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        try:
            if kind == "tv":
                data = self.client.get_tv_credits(int(mid))
            else:
                data = self.client.get_movie_credits(int(mid))
            self._save_cache(cache_key, data)
            return data
        except Exception as e:
            self._log(f"  [TMDB] Credits çekme hatası: {e}")
            return None

    # ── Yardımcılar ─────────────────────────────────────────────────
    def _extract_names(self, credits_data: dict, section: str = "cast") -> List[str]:
        return [
            (item.get("name") or "").strip()
            for item in (credits_data.get(section) or [])
            if (item.get("name") or "").strip()
        ]

    def _count_matches(self, ocr_names: List[str],
                       tmdb_names: List[str]) -> int:
        """Kaç OCR ismi TMDB'de eşleşiyor?"""
        return sum(
            1 for name in ocr_names
            if _fuzzy_match(name, tmdb_names, threshold=82)
        )

    def _canonicalize(self, cdata: dict,
                      credits_data: dict) -> tuple:
        """cdata cast'ını TMDB kanonik isimleriyle güncelle."""
        tmdb_cast_names = self._extract_names(credits_data, section="cast")
        tmdb_crew_names = self._extract_names(credits_data, section="crew")
        if not tmdb_cast_names and not tmdb_crew_names:
            return False, 0, 0

        updated = False
        hits = misses = 0

        for row in (cdata.get("cast") or []):
            if not isinstance(row, dict):
                continue
            actor = (row.get("actor_name") or row.get("actor") or "").strip()
            if not actor:
                continue
            canonical = _fuzzy_match(actor, tmdb_cast_names, threshold=82)
            if canonical:
                if canonical != actor:
                    row["actor_name"] = canonical
                    if "actor" in row:
                        row["actor"] = canonical
                    updated = True
                row["is_verified_name"] = True
                row["is_tmdb_verified"] = True
                hits += 1
            else:
                misses += 1

        for key in ("crew", "technical_crew"):
            for row in (cdata.get(key) or []):
                if not isinstance(row, dict):
                    continue
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                canonical = _fuzzy_match(name, tmdb_crew_names, threshold=82)
                if canonical and canonical != name:
                    row["name"] = canonical
                    updated = True

        return updated, hits, misses

    # ── Ters Doğrulama ──────────────────────────────────────────────
    def _reverse_validate(
        self,
        ocr_title: str,
        ocr_cast_names: List[str],
        ocr_director_names: List[str],
        ocr_year: int,
        tmdb_entry: dict,
        credits_data: dict,
        forward_hits: int,
        forward_misses: int,
    ) -> tuple:
        """
        Ters doğrulama: TMDB'den gelen filmin OCR verimizle ne kadar uyuştuğunu ölçer.

        İleri yönde film bulunduktan ve credits çekildikten sonra çağrılır.
        "Bu TMDB filmi gerçekten bizim filmimiz mi?" sorusunu sorar.

        Return: (accepted: bool, score: float, breakdown: dict)
        """
        tmdb_title = (tmdb_entry.get("name") or tmdb_entry.get("title") or "").strip()
        tmdb_original_title = (
            tmdb_entry.get("original_title") or tmdb_entry.get("original_name") or ""
        ).strip()
        date_str = (tmdb_entry.get("first_air_date") or tmdb_entry.get("release_date") or "")
        tmdb_year = 0
        if date_str and len(date_str) >= 4:
            try:
                tmdb_year = int(date_str[:4])
            except ValueError:
                pass

        score = 0.0
        breakdown: Dict[str, Any] = {}

        self._log(
            f"  [TMDB] Ters doğrulama başlatılıyor: '{tmdb_title}' (id:{tmdb_entry.get('id', 0)})"
        )

        # ── 1. Başlık eşleşme kalitesi (max +2.5 / max -4.0) ──────────
        title_pos = 0.0
        title_neg = 0.0
        fuzzy_score = 0
        title_active = False  # dinamik eşik hesabı için

        compare_title: Optional[str] = None
        if ocr_title and tmdb_title:
            # Kiril başlıkla kıyaslama yapma — farklı alfabe, fuzzy her zaman sıfır çıkar
            _tmdb_cyrillic = _is_cyrillic(tmdb_title)
            _orig_cyrillic = _is_cyrillic(tmdb_original_title) if tmdb_original_title else True

            if _tmdb_cyrillic:
                # TMDB ana başlığı Kiril → orijinal başlık Latin mi diye bak
                if tmdb_original_title and not _orig_cyrillic:
                    # Orijinal başlık Latin: dil eşleşmesi kontrol et
                    ocr_is_tr  = _is_turkish(ocr_title)
                    orig_is_tr = _is_turkish(tmdb_original_title)
                    if ocr_is_tr == orig_is_tr:
                        compare_title = tmdb_original_title
                # else: her iki başlık da Kiril veya alternatif yok → compare_title=None, atla
            else:
                ocr_is_tr  = _is_turkish(ocr_title)
                tmdb_is_tr = _is_turkish(tmdb_title)
                orig_is_tr = _is_turkish(tmdb_original_title) if tmdb_original_title else False

                # Aynı dildeki TMDB başlığını seç
                if ocr_is_tr == tmdb_is_tr:
                    compare_title = tmdb_title
                elif tmdb_original_title and (ocr_is_tr == orig_is_tr) and not _orig_cyrillic:
                    compare_title = tmdb_original_title
                # else: farklı dil → atla (title_active=False, 0.0/0.0)

            if compare_title is not None:
                title_active = True
                # Tam eşleşme (normalize edilmiş)
                if _norm(ocr_title) == _norm(compare_title):
                    title_pos = 2.5
                    fuzzy_score = 100
                else:
                    if HAS_RAPIDFUZZ:
                        fuzzy_score = fuzz.WRatio(ocr_title, compare_title)

                    if fuzzy_score >= 90:
                        title_pos = 2.0
                    elif fuzzy_score >= 80:
                        title_pos = 1.0
                    else:
                        # Kısmi eşleşme: başlık birinin içinde geçiyor mu?
                        ocr_lower = ocr_title.lower()
                        cmp_lower = compare_title.lower()
                        if ocr_lower in cmp_lower or cmp_lower in ocr_lower:
                            title_pos = 0.5
                        else:
                            title_pos = 0.0

        title_net = title_pos + title_neg
        score += title_net
        breakdown["title"] = {
            "ocr": ocr_title,
            "tmdb": tmdb_title,
            "compare": compare_title,
            "fuzzy": fuzzy_score,
            "pos": title_pos,
            "neg": title_neg,
            "net": title_net,
        }
        self._log(
            f"  [TMDB]   Başlık: '{ocr_title}' vs "
            f"'{compare_title if title_active else '(atlandı)'}' → "
            f"fuzzy={fuzzy_score} → +{title_pos} / {title_neg}"
        )

        # ── 2. Yönetmen çapraz kontrolü (max +2.5 / max -2.5) ─────────
        director_pos = 0.0
        director_neg = 0.0

        if ocr_director_names:
            # Kiril yönetmen adları TMDB Latin isimlerle kıyaslanamaz → ceza verme, atla
            if all(_is_cyrillic(d) for d in ocr_director_names if d):
                self._log(
                    f"  [TMDB]   Yönetmen: {ocr_director_names!r} — Kiril alfabe, kıyaslama atlandı → +0.0 / 0.0"
                )
            else:
                tmdb_crew_names = self._extract_names(credits_data, section="crew")
                tmdb_director_names = [
                    (item.get("name") or "").strip()
                    for item in (credits_data.get("crew") or [])
                    if (item.get("job") or "").lower() == "director"
                    and (item.get("name") or "").strip()
                ]

                director_found   = False
                director_similar = False

                for d in ocr_director_names:
                    if _fuzzy_match(d, tmdb_director_names, threshold=95):
                        director_found = True
                        break
                    if _fuzzy_match(d, tmdb_director_names, threshold=80):
                        director_similar = True
                    elif _fuzzy_match(d, tmdb_crew_names, threshold=82):
                        director_similar = True

                if director_found:
                    director_pos = 2.5
                elif director_similar:
                    director_pos = 1.5
                else:
                    director_neg = -2.5

                self._log(
                    f"  [TMDB]   Yönetmen: {ocr_director_names!r} vs crew → "
                    f"{'eşleşme var' if director_found or director_similar else 'eşleşme yok'} → "
                    f"+{director_pos} / {director_neg}"
                )
        else:
            self._log("  [TMDB]   Yönetmen: bilgi yok → +0.0 / 0.0")

        director_net = director_pos + director_neg
        score += director_net
        breakdown["director"] = {
            "ocr": ocr_director_names,
            "pos": director_pos,
            "neg": director_neg,
            "net": director_net,
        }

        # ── 3. Cast eşleşme ORANI + MUTLAK BONUS (max +6.0 / max -3.0) ──────────────
        cast_pos = 0.0
        cast_neg = 0.0
        total = forward_hits + forward_misses
        cast_active = total > 0
        ratio = (forward_hits / total) if total > 0 else 0.0
        ratio_pct = ratio * 100

        # Pozitif (oran bazlı)
        if ratio >= 0.50:
            cast_pos = 3.0
        elif ratio >= 0.30:
            cast_pos = 2.0
        elif ratio >= 0.15:
            cast_pos = 1.0
        elif ratio >= 0.05:
            cast_pos = 0.5
        else:
            cast_pos = 0.0

        # Mutlak eşleşme bonusu
        if forward_hits >= 8:
            cast_pos += 3.0
        elif forward_hits >= 5:
            cast_pos += 2.0
        elif forward_hits >= 3:
            cast_pos += 1.0

        # Negatif: sadece forward_hits < 3 ise uygulanır
        if forward_hits < 3:
            if ratio < 0.05:
                cast_neg = -3.0
            elif ratio < 0.10:
                cast_neg = -2.0
            elif ratio < 0.15:
                cast_neg = -1.0

        cast_net = cast_pos + cast_neg
        score += cast_net
        breakdown["cast"] = {
            "hits": forward_hits,
            "total": total,
            "ratio_pct": round(ratio_pct, 1),
            "pos": cast_pos,
            "neg": cast_neg,
            "net": cast_net,
        }
        self._log(
            f"  [TMDB]   Cast oranı: {forward_hits}/{total} = {ratio_pct:.1f}% → "
            f"+{cast_pos} / {cast_neg}"
        )

        # ── 4. Yıl uyumu (max +2.0 / max -3.0) ───────────────────────
        year_pos = 0.0
        year_neg = 0.0

        if ocr_year and tmdb_year:
            year_diff = abs(ocr_year - tmdb_year)

            # Pozitif
            if year_diff <= 2:
                year_pos = 2.0
            elif year_diff <= 5:
                year_pos = 1.0
            elif year_diff <= 10:
                year_pos = 0.5

            # Negatif
            if year_diff > 20:
                year_neg = -3.0
            elif year_diff > 10:
                year_neg = -2.0
            elif year_diff > 5:
                year_neg = -1.0

            self._log(
                f"  [TMDB]   Yıl: {ocr_year} vs {tmdb_year} = {year_diff} yıl fark → "
                f"+{year_pos} / {year_neg}"
            )
        else:
            year_diff = None
            self._log("  [TMDB]   Yıl: bilinmiyor → +0.0 / 0.0")

        year_net = year_pos + year_neg
        score += year_net
        breakdown["year"] = {
            "ocr": ocr_year,
            "tmdb": tmdb_year,
            "diff": year_diff,
            "pos": year_pos,
            "neg": year_neg,
            "net": year_net,
        }

        # ── Dinamik eşik: aktif kategorilerin max pozitif puan toplamı × %40 ──
        _MAX_TITLE   = 2.5
        _MAX_DIRECTOR = 2.5
        _MAX_CAST    = 6.0
        _MAX_YEAR    = 2.0

        max_pos = 0.0
        if title_active:
            max_pos += _MAX_TITLE
        if ocr_director_names:
            max_pos += _MAX_DIRECTOR
        if cast_active:
            max_pos += _MAX_CAST
        if ocr_year and tmdb_year:
            max_pos += _MAX_YEAR

        _THRESHOLD = max(1.0, round(max_pos * 0.40, 2))

        # ── Karar ─────────────────────────────────────────────────────
        accepted = score >= _THRESHOLD
        breakdown["total"]     = round(score, 2)
        breakdown["threshold"] = _THRESHOLD

        status = "✅ ACCEPT" if accepted else "❌ REJECT"
        self._log(
            f"  [TMDB]   Toplam: {score:.1f} / {max_pos:.1f} → {status} (eşik: {_THRESHOLD})"
        )

        return accepted, round(score, 2), breakdown
