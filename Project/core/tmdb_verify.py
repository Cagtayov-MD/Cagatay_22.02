"""
tmdb_verify.py — TMDB arama bazlı doğrulama.

ID girmeden çalışır. Film adı + oyuncu listesiyle TMDB'de arama yapar.

Doğruluk mantığı:
  - film_adı + herhangi bir oyuncu eşleşirse → %100 güven
  - film_adı yok ama 3 farklı oyuncu eşleşirse → %100 güven
  - Eşleşme sağlanırsa tüm cast TMDB'deki kanonik isimlerle güncellenir.
"""
from __future__ import annotations

import json
import os
import threading
import time
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
    # "Madam" → "Madame" (Fransızca unvan düzeltmesi)
    new_words = ["Madame" if w.lower() == "madam" else w for w in words]
    variant = " ".join(new_words)
    if variant != title:
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
        if res:
            return res[0]
    return None


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

    def __post_init__(self):
        if self.cast is None:
            self.cast = []
        if self.crew is None:
            self.crew = []
        if self.keywords is None:
            self.keywords = []
        if self.genres is None:
            self.genres = []


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
                       tv_id=None, movie_id=None) -> TMDBVerifyResult:
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

        if not cast_names and not director_names:
            return TMDBVerifyResult(False, "no cast or directors to verify")

        if not cast_names:
            self._log(f"  [TMDB] Cast boş ama {len(director_names)} yönetmen var — yönetmenle aramaya devam ediliyor")

        # ── Anomali tespiti: gerçek dışı yüksek cast sayısı ──
        if len(cast_names) > 500:
            self._log(f"  [TMDB] ⚠️ Anormal cast sayısı: {len(cast_names)} — muhtemelen OCR çöpü")
            self._log(f"  [TMDB] Sadece en güvenilir ilk 50 isim kullanılıyor")
            cast_names = cast_names[:50]

        # ── cast_names ön-filtreleme: Şirket/organizasyon isimlerini çıkar ──
        company_filtered = [n for n in cast_names if not _looks_like_company(n)]
        if len(cast_names) != len(company_filtered):
            self._log(f"  [TMDB] Şirket filtresi: {len(cast_names)} → {len(company_filtered)} isim")
            removed = [n for n in cast_names if n not in set(company_filtered)]
            if removed[:5]:
                self._log(f"  [TMDB] Şirket olarak elenenler: {removed[:5]}")
        cast_names = company_filtered

        # ── İsim kalite filtresi: OCR çöpünü TMDB'ye göndermeden ele ──
        def _is_plausible_name(name: str) -> bool:
            """İsim olarak makul mü? Sesli harf, kelime yapısı kontrolü."""
            alpha_chars = [c for c in name.lower() if c.isalpha()]
            if not alpha_chars:
                return False
            vowels = sum(1 for c in alpha_chars if c in 'aeıioöuü')
            vowel_ratio = vowels / len(alpha_chars)
            if vowel_ratio < 0.15:
                return False
            if vowel_ratio > 0.80:
                return False
            if name.isupper() and ' ' not in name and len(name) > 12:
                return False
            noise_words = {'filme', 'filmer', 'cinema', 'telecinco', 'production', 'studio', 'channel'}
            if name.lower().strip() in noise_words:
                return False
            return True

        qualified_names = [n for n in cast_names if _is_plausible_name(n)]

        if len(cast_names) != len(qualified_names):
            self._log(f"  [TMDB] İsim kalite filtresi: {len(cast_names)} → {len(qualified_names)} isim")
            qualified_set = set(qualified_names)
            rejected = [n for n in cast_names if n not in qualified_set]
            if rejected[:5]:
                self._log(f"  [TMDB] Elenen örnekler: {rejected[:5]}")

        cast_names = qualified_names

        if not cast_names and not director_names:
            return TMDBVerifyResult(False, "no cast or directors to verify")

        # TMDB'de eşleşme bul
        tmdb_entry, kind = self._find_tmdb_entry(film_title, cast_names, director_names)

        if not tmdb_entry:
            self._log(f"  [TMDB] Eşleşme bulunamadı")
            return TMDBVerifyResult(False, "tmdb match not found")

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

        if updated:
            cdata["tmdb_verified"] = True
            cdata["tmdb_title"]    = tmdb_title
            cdata["tmdb_id"]       = tmdb_id
            cdata["tmdb_type"]     = kind
            cdata["film_title"]    = tmdb_title

        # TMDB keywords ve genres çek
        tmdb_keywords: List[str] = []
        tmdb_genres: List[str] = []
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

        return TMDBVerifyResult(
            updated=updated, reason="ok",
            hits=hits, misses=misses,
            confidence="high",
            matched_title=tmdb_title, matched_id=tmdb_id,
            cast=tmdb_cast, crew=tmdb_crew,
            year=tmdb_year,
            keywords=tmdb_keywords,
            genres=tmdb_genres,
        )

    # ── TMDB'de eşleşme bul ─────────────────────────────────────────
    def _find_tmdb_entry(self, film_title: str,
                         cast_names: List[str],
                         director_names: List[str] = None) -> tuple:
        """
        Strateji:
          1. Film adıyla search/multi → top 10 sonucu oyuncularla doğrula
             (birden fazla başlık varyantı denenir; örn. "Madam" → "Madame")
             - film_adı + 1 oyuncu → %100 güven
             - film_adı + yönetmen (TMDB crew'unda) → %100 güven (cast yoksa)
          2. Film adı yoksa her oyuncuyla search/person → combined_credits doğrula
             (combined_credits başarısız olursa known_for fallback)
             - 3 oyuncu eşleşirse → %100 güven
             - yönetmenler de search_person ile aranır
        """
        director_names = director_names or []

        def _director_matches_crew(credits: dict) -> bool:
            """TMDB crew'unda yönetmen eşleşmesi var mı?"""
            tmdb_crew = self._extract_names(credits, section="crew")
            return any(
                _fuzzy_match(d, tmdb_crew, threshold=82)
                for d in director_names
            )

        # ── Strateji 1: Film adıyla ara (çoklu başlık denemesi) ──
        for film_title_attempt in _title_candidates(film_title) if film_title else []:
            self._log(f"  [TMDB] Aranan başlık: '{film_title_attempt}'")
            cache_key = f"search_multi_{_norm(film_title_attempt)}"
            results = self._load_cache(cache_key)
            if results is None:
                try:
                    results = self.client.search_multi(film_title_attempt)
                    self._save_cache(cache_key, results)
                except Exception as e:
                    self._log(f"  [TMDB] Arama hatası: {e}")
                    results = []

            self._log(f"  [TMDB] {len(results or [])} sonuç bulundu")
            for r in (results or [])[:10]:
                kind = r.get("media_type", "")
                title = r.get("name") or r.get("title") or "?"
                if kind not in ("tv", "movie"):
                    continue
                self._log(f"  [TMDB] Kontrol: '{title}' ({kind}, id:{r['id']})")
                credits = self._fetch_credits(kind, r["id"])
                if not credits:
                    continue
                tmdb_names = self._extract_names(credits)
                matched    = self._count_matches(cast_names, tmdb_names)
                self._log(f"  [TMDB]   → {matched}/{len(cast_names)} oyuncu eşleşti")
                if matched >= 1:
                    self._log(f"  [TMDB] film adı + {matched} oyuncu eşleşti → %100 güven")
                    return r, kind
                # Cast eşleşmesi yoksa yönetmen eşleşmesini kontrol et
                if matched == 0 and director_names and _director_matches_crew(credits):
                    self._log(f"  [TMDB] film adı + yönetmen eşleşti → %100 güven")
                    return r, kind
            # Bu başlık denemesinde eşleşme yoksa sonraki varyantı dene

        # ── Strateji 2: Oyuncularla ara ──
        self._log(f"  [TMDB] Film adıyla eşleşme yok veya film adı yanlış, oyuncularla aranıyor...")
        work_matches: Dict[int, dict] = {}  # tmdb_id → {entry, kind, count}

        # Oyuncular + yönetmenler birlikte aranır; şirket isimleri önceden elenir
        cast_names_set = set(cast_names)
        actor_candidates = [n for n in cast_names if not _looks_like_company(n)]
        self._log(f"  [TMDB] Aranan oyuncular (filtreli): {actor_candidates[:5]}")
        persons_to_search = actor_candidates[:8] + [d for d in director_names if d not in cast_names_set]

        for actor in persons_to_search:
            cache_key = f"search_person_{_norm(actor)}"
            persons = self._load_cache(cache_key)
            if persons is None:
                try:
                    persons = self.client.search_person(actor)
                    self._save_cache(cache_key, persons)
                except Exception:
                    continue

            for person in (persons or [])[:2]:  # en iyi 2 eşleşme yeterli
                person_id = person.get("id")
                if not person_id:
                    continue
                cache_key_credits = f"person_combined_{person_id}"
                person_credits = self._load_cache(cache_key_credits)
                if person_credits is None:
                    try:
                        person_credits = self.client.get_person_combined_credits(person_id)
                        self._save_cache(cache_key_credits, person_credits)
                    except Exception:
                        # combined_credits başarısız olursa known_for fallback
                        for work in (person.get("known_for") or []):
                            kind = work.get("media_type", "")
                            if kind not in ("tv", "movie"):
                                continue
                            wid = work.get("id", 0)
                            if wid:
                                if wid not in work_matches:
                                    work_matches[wid] = {"entry": work, "kind": kind, "count": 0}
                                work_matches[wid]["count"] += 1
                        continue

                # combined_credits'ten cast bölümünü kullan
                for work in (person_credits.get("cast") or []):
                    kind = work.get("media_type", "")
                    if kind not in ("tv", "movie"):
                        continue
                    wid = work.get("id", 0)
                    if not wid:
                        continue
                    if wid not in work_matches:
                        work_matches[wid] = {"entry": work, "kind": kind, "count": 0}
                    work_matches[wid]["count"] += 1

        # En çok eşleşen yapıtı bul
        if work_matches:
            best = max(work_matches.values(), key=lambda x: x["count"])
            if best["count"] >= self.MIN_ACTOR_MATCH:
                # Credits ile kesin doğrula
                credits = self._fetch_credits(best["kind"], best["entry"]["id"])
                if credits:
                    tmdb_names = self._extract_names(credits)
                    matched    = self._count_matches(cast_names, tmdb_names)
                    if matched >= self.MIN_ACTOR_MATCH:
                        self._log(f"  [TMDB] {matched} oyuncu eşleşti → %100 güven")
                        return best["entry"], best["kind"]

        return None, ""

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
