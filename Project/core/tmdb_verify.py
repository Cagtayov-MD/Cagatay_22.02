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
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False


def _norm(s: str) -> str:
    """Karşılaştırma için normalize: küçük harf, sadece alfanumerik."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


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

    def __post_init__(self):
        if self.cast is None:
            self.cast = []
        if self.crew is None:
            self.crew = []


class TMDBClient:
    BASE = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str = "", bearer_token: str = "",
                 language: str = "tr-TR", timeout: int = 15):
        self.api_key  = (api_key or "").strip()
        self.bearer   = (bearer_token or "").strip()
        self.language = (language or "tr-TR").strip()
        self.timeout  = timeout

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

    def search_multi(self, query: str) -> List[Dict[str, Any]]:
        r = requests.get(f"{self.BASE}/search/multi",
                         headers=self._headers(),
                         params=self._params({"query": query, "page": "1"}),
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("results") or []

    def get_tv_credits(self, tv_id: int) -> Dict[str, Any]:
        r = requests.get(f"{self.BASE}/tv/{tv_id}/credits",
                         headers=self._headers(),
                         params=self._params(),
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_movie_credits(self, movie_id: int) -> Dict[str, Any]:
        r = requests.get(f"{self.BASE}/movie/{movie_id}/credits",
                         headers=self._headers(),
                         params=self._params(),
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def search_person(self, query: str) -> List[Dict[str, Any]]:
        r = requests.get(f"{self.BASE}/search/person",
                         headers=self._headers(),
                         params=self._params({"query": query}),
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("results") or []


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
                                     language=language)
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

        cast_names = [
            (row.get("actor_name") or row.get("actor") or "").strip()
            for row in (cdata.get("cast") or [])
            if isinstance(row, dict)
        ]
        cast_names = [n for n in cast_names if len(n) >= 3]

        if not cast_names:
            return TMDBVerifyResult(False, "no cast to verify")

        # TMDB'de eşleşme bul
        tmdb_entry, kind = self._find_tmdb_entry(film_title, cast_names)

        if not tmdb_entry:
            self._log(f"  [TMDB] Eşleşme bulunamadı")
            return TMDBVerifyResult(False, "tmdb match not found")

        tmdb_id    = tmdb_entry.get("id", 0)
        tmdb_title = (tmdb_entry.get("name") or tmdb_entry.get("title") or "").strip()
        self._log(f"  [TMDB] ✓ '{tmdb_title}' (id:{tmdb_id}, tür:{kind})")

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

        return TMDBVerifyResult(
            updated=updated, reason="ok",
            hits=hits, misses=misses,
            confidence="high",
            matched_title=tmdb_title, matched_id=tmdb_id,
            cast=tmdb_cast, crew=tmdb_crew,
        )

    # ── TMDB'de eşleşme bul ─────────────────────────────────────────
    def _find_tmdb_entry(self, film_title: str,
                         cast_names: List[str]) -> tuple:
        """
        Strateji:
          1. Film adıyla search/multi → top 5 sonucu oyuncularla doğrula
             - film_adı + 1 oyuncu → %100 güven
          2. Film adı yoksa her oyuncuyla search/person → known_for'u doğrula
             - 3 oyuncu eşleşirse → %100 güven
        """
        # ── Strateji 1: Film adıyla ara ──
        if film_title:
            self._log(f"  [TMDB] Aranan başlık: '{film_title}'")
            cache_key = f"search_multi_{_norm(film_title)}"
            results = self._load_cache(cache_key)
            if results is None:
                try:
                    results = self.client.search_multi(film_title)
                    self._save_cache(cache_key, results)
                except Exception as e:
                    self._log(f"  [TMDB] Arama hatası: {e}")
                    results = []

            self._log(f"  [TMDB] {len(results or [])} sonuç bulundu")
            for r in (results or [])[:5]:
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

        # ── Strateji 2: Oyuncularla ara ──
        self._log(f"  [TMDB] Film adıyla eşleşme yok, oyuncularla aranıyor...")
        self._log(f"  [TMDB] Aranan oyuncular: {cast_names[:5]}")
        work_matches: Dict[int, dict] = {}  # tmdb_id → {entry, kind, count}

        for actor in cast_names[:8]:
            cache_key = f"search_person_{_norm(actor)}"
            persons = self._load_cache(cache_key)
            if persons is None:
                try:
                    persons = self.client.search_person(actor)
                    self._save_cache(cache_key, persons)
                except Exception:
                    continue

            for person in (persons or [])[:3]:
                for work in (person.get("known_for") or []):
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
