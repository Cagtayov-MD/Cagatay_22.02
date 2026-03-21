"""
imdb_lookup.py — Yerel IMDb DuckDB üzerinden doğrulama.

Tablo yapısı:
  titles     : tconst, primaryTitle, originalTitle, titleType, startYear
  names      : nconst, primaryName
  principals : tconst, nconst, ordering, category, job, characters
  crew       : tconst, directors, writers
  episodes   : tconst, parentTconst, seasonNumber, episodeNumber
  akas       : tconst, title, region, language
  ratings    : tconst, averageRating, numVotes
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Fuzzy eşleştirme — rapidfuzz varsa kullan, yoksa basit karşılaştırma
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _fuzz

    def _fuzzy_score(a: str, b: str) -> float:
        return _fuzz.token_sort_ratio(a, b)

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

    def _fuzzy_score(a: str, b: str) -> float:
        a, b = a.lower().strip(), b.lower().strip()
        if a == b:
            return 100.0
        if a in b or b in a:
            return 80.0
        return 0.0


_FUZZY_THRESHOLD = 80.0  # minimum benzerlik skoru (0-100)


# ---------------------------------------------------------------------------
# Sonuç dataclass
# ---------------------------------------------------------------------------

@dataclass
class IMDBLookupResult:
    matched: bool
    reason: str
    tconst: str = ""
    title: str = ""
    title_type: str = ""        # "movie" | "tvSeries" | vb.
    year: int = 0
    cast: list = field(default_factory=list)
    crew: list = field(default_factory=list)
    directors: list = field(default_factory=list)
    matched_via: str = ""       # "strat_a" | "strat_b_series" | "strat_b_film" | "strat_c" | "strat_d"


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Karşılaştırma için normalize et."""
    return (s or "").lower().strip()


try:
    from unidecode import unidecode as _unidecode
    _HAS_UNIDECODE = True
except ImportError:
    _HAS_UNIDECODE = False

def _norm_ascii(s: str) -> str:
    """Başlığı ASCII'ye normalize et: unidecode + apostrof/tire temizliği.

    IMDb başlıklarda iyelik ekleri apostrofa dönüşür (Hanımın → Hanim'in).
    unidecode her iki tarafı da aynı forma getirir:
      "TÜRKAN_HANIMIN_KONAĞI" → "turkan hanimin konagi"
      "Türkan Hanim'in Konagi" → "turkan hanimin konagi"  ✓
    """
    s = (s or "").strip()
    if _HAS_UNIDECODE:
        s = _unidecode(s)
    s = s.lower().replace("'", "").replace("_", " ").replace("-", " ")
    return s


def _fuzzy_match(a: str, b: str, threshold: float = _FUZZY_THRESHOLD) -> bool:
    return _fuzzy_score(_norm(a), _norm(b)) >= threshold


def _best_fuzzy(query: str, candidates: list[str], threshold: float = _FUZZY_THRESHOLD) -> Optional[str]:
    """candidates listesinden en iyi eşleşeni döndür."""
    best_score = 0.0
    best = None
    q = _norm(query)
    for c in candidates:
        score = _fuzzy_score(q, _norm(c))
        if score > best_score:
            best_score = score
            best = c
    if best_score >= threshold:
        return best
    return None


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class IMDBLookup:
    DB_PATH = r"F:\IMDB\db\imdb.duckdb"  # env: IMDB_DB_PATH ile override edilebilir

    def __init__(self, log_cb: Optional[Callable] = None):
        self._log_cb = log_cb
        self._db_path = os.environ.get("IMDB_DB_PATH", "").strip() or self.DB_PATH

    def _log(self, msg: str) -> None:
        if self._log_cb:
            self._log_cb(msg)

    def enabled(self) -> bool:
        """DB dosyası var mı ve duckdb kurulu mu?"""
        try:
            import duckdb  # noqa: F401
        except ImportError:
            return False
        return os.path.isfile(self._db_path)

    # ------------------------------------------------------------------
    # Ana giriş noktası
    # ------------------------------------------------------------------

    def lookup(self, cdata: dict) -> IMDBLookupResult:
        """
        Ana giriş noktası. cdata'dan film_title, cast, directors, crew çeker
        ve STRAT-A → B → C → D sırasıyla arama yapar.
        İlk eşleşmede durur.
        """
        film_title = _norm(cdata.get("film_title") or "")
        if not film_title:
            return IMDBLookupResult(matched=False, reason="no film title")

        # OCR'dan oyuncu isimlerini çek
        ocr_cast_names: List[str] = []
        for item in (cdata.get("cast") or []):
            if isinstance(item, dict):
                name = (item.get("actor_name") or item.get("name") or "").strip()
            else:
                name = str(item).strip()
            if name:
                ocr_cast_names.append(name)

        # Yönetmen isimlerini çek
        director_names: List[str] = []
        for item in (cdata.get("directors") or []):
            if isinstance(item, dict):
                name = (item.get("name") or "").strip()
            else:
                name = str(item).strip()
            if name:
                director_names.append(name)

        # Crew içinden de yönetmen çek
        for item in (cdata.get("crew") or []):
            if isinstance(item, dict):
                job = _norm(item.get("job") or item.get("role") or "")
                if "director" in job or "yönetmen" in job or "yonetmen" in job:
                    name = (item.get("name") or "").strip()
                    if name and name not in director_names:
                        director_names.append(name)

        # Kameraman isimlerini çek
        cinematographer_names: List[str] = []
        for item in (cdata.get("crew") or []):
            if isinstance(item, dict):
                job = _norm(item.get("job") or item.get("role") or "")
                if "cinemat" in job or "görüntü" in job or "kameraman" in job or "photography" in job:
                    name = (item.get("name") or "").strip()
                    if name:
                        cinematographer_names.append(name)

        ocr_year = int(cdata.get("year") or 0)

        try:
            import duckdb
            con = duckdb.connect(self._db_path, read_only=True)
            try:
                result = self._run_strategies(
                    con=con,
                    film_title=film_title,
                    ocr_cast_names=ocr_cast_names,
                    director_names=director_names,
                    cinematographer_names=cinematographer_names,
                    ocr_year=ocr_year,
                )
            finally:
                con.close()
            return result
        except Exception as e:
            self._log(f"  [IMDb] DuckDB hatası: {e}")
            return IMDBLookupResult(matched=False, reason=f"duckdb error: {e}")

    # ------------------------------------------------------------------
    # Strateji orchestrator
    # ------------------------------------------------------------------

    def _run_strategies(
        self,
        con,
        film_title: str,
        ocr_cast_names: List[str],
        director_names: List[str],
        cinematographer_names: List[str],
        ocr_year: int,
    ) -> IMDBLookupResult:

        # Başlık adaylarını bir kere çek (tüm stratejiler kullanır)
        candidates = self._find_title_candidates(con, film_title)

        # STRAT-A
        if ocr_cast_names:
            r = self._strat_a(con, candidates, ocr_cast_names, director_names)
            if r.matched:
                return r

        # STRAT-B
        if director_names or ocr_cast_names:
            r = self._strat_b(
                con, candidates, director_names,
                ocr_cast_names, cinematographer_names, ocr_year
            )
            if r.matched:
                return r

        # STRAT-C
        if ocr_cast_names:
            r = self._strat_c(con, film_title, ocr_cast_names, director_names)
            if r.matched:
                return r

        # STRAT-D
        if ocr_cast_names:
            r = self._strat_d(con, film_title, ocr_cast_names, ocr_year)
            if r.matched:
                return r

        return IMDBLookupResult(matched=False, reason="no match in any strategy")

    # ------------------------------------------------------------------
    # Başlık araması (STRAT-A ve B için ortak)
    # ------------------------------------------------------------------

    def _find_title_candidates(self, con, film_title: str) -> list[dict]:
        """
        primaryTitle / originalTitle / akas.title üzerinden fuzzy başlık araması.
        Sonuç: [{"tconst": ..., "primaryTitle": ..., "titleType": ..., "startYear": ...}]
        """
        # LIKE için tam başlık kullan (orijinal karakterlerle — ü, ö vb. DB'de aynen saklanır).
        # Türkçe iyelik/çekim ekleri DB'de apostrofa dönüştüğünden tam eşleşme olmayabilir;
        # fuzzy score aşamasında _norm_ascii (unidecode) her iki tarafı normalize ederek yakalar.
        like_q = f"%{film_title}%"
        # İlk kelime LIKE: "türkan hanımın konağı" → "%türkan%" → DB'deki "Türkan Hanim'in Konagi" eşleşir
        first_word = film_title.split()[0] if film_title else film_title
        like_q_first = f"%{first_word}%"

        try:
            rows = con.execute("""
                SELECT DISTINCT t.tconst, t.primaryTitle, t.originalTitle,
                                t.titleType, t.startYear
                FROM titles t
                WHERE lower(t.primaryTitle) LIKE ?
                   OR lower(t.originalTitle) LIKE ?
                   OR lower(t.primaryTitle) LIKE ?
                   OR lower(t.originalTitle) LIKE ?
                LIMIT 50
            """, [like_q, like_q, like_q_first, like_q_first]).fetchall()
        except Exception:
            rows = []

        # akas üzerinden de ara
        try:
            aka_rows = con.execute("""
                SELECT DISTINCT t.tconst, t.primaryTitle, t.originalTitle,
                                t.titleType, t.startYear
                FROM titles t
                JOIN akas a ON a.tconst = t.tconst
                WHERE lower(a.title) LIKE ?
                   OR lower(a.title) LIKE ?
                LIMIT 50
            """, [like_q, like_q_first]).fetchall()
        except Exception:
            aka_rows = []

        seen = set()
        result = []
        for row in list(rows) + list(aka_rows):
            tconst = row[0]
            if tconst in seen:
                continue
            seen.add(tconst)
            primary = (row[1] or "")
            original = (row[2] or "")
            title_type = (row[3] or "")
            start_year = row[4]
            # unidecode her iki tarafı da aynı ASCII forma getirir → doğru karşılaştırma
            score_p = _fuzzy_score(_norm_ascii(film_title), _norm_ascii(primary))
            score_o = _fuzzy_score(_norm_ascii(film_title), _norm_ascii(original))
            if max(score_p, score_o) >= _FUZZY_THRESHOLD:
                result.append({
                    "tconst": tconst,
                    "primaryTitle": primary,
                    "originalTitle": original,
                    "titleType": title_type,
                    "startYear": int(start_year) if start_year else 0,
                })
        return result

    # ------------------------------------------------------------------
    # Cast yardımcıları
    # ------------------------------------------------------------------

    def _fetch_cast(self, con, tconst: str) -> list[dict]:
        """principals + names join ile cast/crew listesi döndür."""
        try:
            rows = con.execute("""
                SELECT n.primaryName, p.category, p.job, p.characters, p.ordering
                FROM principals p
                JOIN names n ON n.nconst = p.nconst
                WHERE p.tconst = ?
                ORDER BY p.ordering
            """, [tconst]).fetchall()
        except Exception:
            return []
        result = []
        for row in rows:
            result.append({
                "name": row[0] or "",
                "category": row[1] or "",
                "job": row[2] or "",
                "characters": row[3] or "",
                "ordering": row[4] or 0,
            })
        return result

    def _fetch_directors(self, con, tconst: str) -> list[str]:
        """crew.directors nconst → names.primaryName listesi."""
        try:
            row = con.execute(
                "SELECT directors FROM crew WHERE tconst = ?", [tconst]
            ).fetchone()
        except Exception:
            return []
        if not row or not row[0]:
            return []
        nconsts = [n.strip() for n in str(row[0]).split(",") if n.strip()]
        names = []
        for nc in nconsts:
            try:
                nr = con.execute(
                    "SELECT primaryName FROM names WHERE nconst = ?", [nc]
                ).fetchone()
                if nr and nr[0]:
                    names.append(nr[0])
            except Exception:
                continue
        return names

    def _count_cast_matches(self, ocr_names: List[str], imdb_cast: list[dict]) -> int:
        imdb_names = [r["name"] for r in imdb_cast]
        count = 0
        for ocr_name in ocr_names:
            if _best_fuzzy(ocr_name, imdb_names) is not None:
                count += 1
        return count

    def _director_matches(self, director_names: List[str], imdb_directors: List[str]) -> bool:
        for dn in director_names:
            if _best_fuzzy(dn, imdb_directors) is not None:
                return True
        return False

    def _build_result(self, con, cand: dict, imdb_cast: list[dict], matched_via: str) -> IMDBLookupResult:
        tconst = cand["tconst"]
        imdb_directors = self._fetch_directors(con, tconst)

        cast_out = []
        crew_out = []
        director_out = [{"name": n} for n in imdb_directors]

        for item in imdb_cast:
            category = (item.get("category") or "").lower()
            name = item.get("name") or ""
            job = item.get("job") or ""
            chars = item.get("characters") or ""

            if category in ("actor", "actress", "self"):
                cast_out.append({
                    "actor_name": name,
                    "character_name": chars,
                    "role": "Cast",
                    "confidence": 1.0,
                    "raw": "imdb",
                    "is_imdb_verified": True,
                })
            else:
                crew_out.append({
                    "name": name,
                    "job": job or category,
                    "role": job or category,
                    "raw": "imdb",
                    "is_imdb_verified": True,
                })

        # Yönetmenleri crew'a da ekle
        for dn in imdb_directors:
            if not any(_norm(c["name"]) == _norm(dn) for c in crew_out):
                crew_out.insert(0, {
                    "name": dn,
                    "job": "Director",
                    "role": "Director",
                    "raw": "imdb",
                    "is_imdb_verified": True,
                })

        return IMDBLookupResult(
            matched=True,
            reason=f"matched via {matched_via}",
            tconst=tconst,
            title=cand.get("primaryTitle") or "",
            title_type=cand.get("titleType") or "",
            year=cand.get("startYear") or 0,
            cast=cast_out,
            crew=crew_out,
            directors=director_out,
            matched_via=matched_via,
        )

    # ------------------------------------------------------------------
    # STRAT-A: Film adı + yönetmen + oyuncu
    # ------------------------------------------------------------------

    def _strat_a(
        self,
        con,
        candidates: list[dict],
        ocr_cast_names: List[str],
        director_names: List[str],
    ) -> IMDBLookupResult:
        for cand in candidates:
            tconst = cand["tconst"]
            imdb_cast = self._fetch_cast(con, tconst)
            if not imdb_cast:
                continue

            match_count = self._count_cast_matches(ocr_cast_names, imdb_cast)

            # En az 2 oyuncu eşleşmesi → KABUL
            if match_count >= 2:
                self._log(f"  [IMDb] STRAT-A kabul: {cand['primaryTitle']} — {match_count} oyuncu eşleşmesi")
                return self._build_result(con, cand, imdb_cast, "strat_a")

            # 1 oyuncu + yönetmen eşleşmesi → KABUL
            if match_count >= 1 and director_names:
                imdb_directors = self._fetch_directors(con, tconst)
                if self._director_matches(director_names, imdb_directors):
                    self._log(f"  [IMDb] STRAT-A kabul: {cand['primaryTitle']} — 1 oyuncu + yönetmen eşleşmesi")
                    return self._build_result(con, cand, imdb_cast, "strat_a")

        return IMDBLookupResult(matched=False, reason="strat_a: no match")

    # ------------------------------------------------------------------
    # STRAT-B: Film adı + yönetmen
    # ------------------------------------------------------------------

    def _strat_b(
        self,
        con,
        candidates: list[dict],
        director_names: List[str],
        ocr_cast_names: List[str],
        cinematographer_names: List[str],
        ocr_year: int,
    ) -> IMDBLookupResult:
        for cand in candidates:
            tconst = cand["tconst"]
            title_type = (cand.get("titleType") or "").lower()
            imdb_directors = self._fetch_directors(con, tconst)

            director_ok = self._director_matches(director_names, imdb_directors) if director_names else False

            if title_type == "tvseries":
                # Dizi: yönetmen eşleşmesi yeterli
                if director_ok:
                    self._log(f"  [IMDb] STRAT-B dizi kabul: {cand['primaryTitle']}")
                    imdb_cast = self._fetch_cast(con, tconst)
                    return self._build_result(con, cand, imdb_cast, "strat_b_series")
            else:
                # Film: yönetmen + ek koşullar
                if not director_ok:
                    continue

                imdb_cast = self._fetch_cast(con, tconst)

                # Yıl eşleşmesi (±1 yıl tolerans)
                year_ok = bool(ocr_year and cand["startYear"] and abs(ocr_year - cand["startYear"]) <= 1)
                # Oyuncu eşleşmesi
                cast_ok = bool(ocr_cast_names and self._count_cast_matches(ocr_cast_names, imdb_cast) >= 1)
                # Kameraman eşleşmesi
                cine_ok = False
                if cinematographer_names:
                    imdb_crew = [r for r in imdb_cast if "cinemat" in (r.get("category") or "").lower() or "photography" in (r.get("job") or "").lower()]
                    imdb_cine_names = [r["name"] for r in imdb_crew]
                    cine_ok = any(_best_fuzzy(cn, imdb_cine_names) for cn in cinematographer_names)

                if year_ok or cast_ok or cine_ok:
                    self._log(
                        f"  [IMDb] STRAT-B film kabul: {cand['primaryTitle']} "
                        f"(yıl:{year_ok}, oyuncu:{cast_ok}, kameraman:{cine_ok})"
                    )
                    return self._build_result(con, cand, imdb_cast, "strat_b_film")

        return IMDBLookupResult(matched=False, reason="strat_b: no match")

    # ------------------------------------------------------------------
    # STRAT-C: Oyuncu isimleri + yönetmen → ortak film bul
    # ------------------------------------------------------------------

    def _strat_c(
        self,
        con,
        film_title: str,
        ocr_cast_names: List[str],
        director_names: List[str],
    ) -> IMDBLookupResult:
        # Her OCR oyuncusu için nconst listesi bul
        person_tconsts: dict[str, set] = {}  # nconst → set of tconsts
        for ocr_name in ocr_cast_names:
            try:
                rows = con.execute("""
                    SELECT nconst, primaryName FROM names
                    WHERE lower(primaryName) LIKE ?
                    LIMIT 10
                """, [f"%{_norm(ocr_name)}%"]).fetchall()
            except Exception:
                continue
            for row in rows:
                nconst, db_name = row[0], row[1] or ""
                if not _fuzzy_match(ocr_name, db_name):
                    continue
                try:
                    tc_rows = con.execute(
                        "SELECT tconst FROM principals WHERE nconst = ? LIMIT 100",
                        [nconst]
                    ).fetchall()
                except Exception:
                    continue
                for tc_row in tc_rows:
                    tconst = tc_row[0]
                    if tconst not in person_tconsts:
                        person_tconsts[tconst] = set()
                    person_tconsts[tconst].add(nconst)

        # En az 2 OCR oyuncusunun eşleştiği tconst'ler
        common = {tc: nc_set for tc, nc_set in person_tconsts.items() if len(nc_set) >= 2}
        if not common:
            return IMDBLookupResult(matched=False, reason="strat_c: no common tconst")

        # Yönetmen kısıtı
        if director_names:
            filtered = {}
            for tconst, nc_set in common.items():
                imdb_directors = self._fetch_directors(con, tconst)
                if self._director_matches(director_names, imdb_directors):
                    filtered[tconst] = nc_set
            if filtered:
                common = filtered

        # En az 3 kişi (oyuncu + yönetmen)
        best_tconst = None
        best_count = 0
        for tconst, nc_set in common.items():
            count = len(nc_set)
            if director_names:
                imdb_directors = self._fetch_directors(con, tconst)
                if self._director_matches(director_names, imdb_directors):
                    count += 1
            if count >= 3 and count > best_count:
                best_count = count
                best_tconst = tconst

        if not best_tconst:
            return IMDBLookupResult(matched=False, reason="strat_c: fewer than 3 matches")

        # Başlık bilgisi çek
        try:
            row = con.execute(
                "SELECT tconst, primaryTitle, originalTitle, titleType, startYear FROM titles WHERE tconst = ?",
                [best_tconst]
            ).fetchone()
        except Exception:
            return IMDBLookupResult(matched=False, reason="strat_c: title fetch failed")

        if not row:
            return IMDBLookupResult(matched=False, reason="strat_c: tconst not in titles")

        cand = {
            "tconst": row[0],
            "primaryTitle": row[1] or "",
            "originalTitle": row[2] or "",
            "titleType": row[3] or "",
            "startYear": int(row[4]) if row[4] else 0,
        }
        imdb_cast = self._fetch_cast(con, best_tconst)
        self._log(f"  [IMDb] STRAT-C kabul: {cand['primaryTitle']} — {best_count} kişi eşleşmesi")
        return self._build_result(con, cand, imdb_cast, "strat_c")

    # ------------------------------------------------------------------
    # STRAT-D: Oyuncuları tek tek kontrol et
    # ------------------------------------------------------------------

    def _strat_d(
        self,
        con,
        film_title: str,
        ocr_cast_names: List[str],
        ocr_year: int,
    ) -> IMDBLookupResult:
        # OCR'dan gelen her oyuncuyu names tablosunda ara
        verified_tconsts: dict[str, int] = {}  # tconst → eşleşen kişi sayısı

        for ocr_name in ocr_cast_names:
            try:
                rows = con.execute("""
                    SELECT nconst, primaryName FROM names
                    WHERE lower(primaryName) LIKE ?
                    LIMIT 10
                """, [f"%{_norm(ocr_name)}%"]).fetchall()
            except Exception:
                continue
            for row in rows:
                nconst, db_name = row[0], row[1] or ""
                if not _fuzzy_match(ocr_name, db_name):
                    continue
                # Gerçek kişi doğrulandı — principals üzerinden tconst'ler
                try:
                    tc_rows = con.execute(
                        "SELECT tconst FROM principals WHERE nconst = ? LIMIT 50",
                        [nconst]
                    ).fetchall()
                except Exception:
                    continue
                for tc_row in tc_rows:
                    tconst = tc_row[0]
                    verified_tconsts[tconst] = verified_tconsts.get(tconst, 0) + 1

        if not verified_tconsts:
            return IMDBLookupResult(matched=False, reason="strat_d: no real actors found")

        # En çok ortak kişiye sahip tconst'u al
        best_tconst = max(verified_tconsts, key=lambda tc: verified_tconsts[tc])
        best_count = verified_tconsts[best_tconst]

        if best_count < 2:
            return IMDBLookupResult(matched=False, reason="strat_d: fewer than 2 common persons")

        # titles ile doğrula
        try:
            row = con.execute(
                "SELECT tconst, primaryTitle, originalTitle, titleType, startYear FROM titles WHERE tconst = ?",
                [best_tconst]
            ).fetchone()
        except Exception:
            return IMDBLookupResult(matched=False, reason="strat_d: title fetch failed")

        if not row:
            return IMDBLookupResult(matched=False, reason="strat_d: tconst not in titles")

        cand = {
            "tconst": row[0],
            "primaryTitle": row[1] or "",
            "originalTitle": row[2] or "",
            "titleType": row[3] or "",
            "startYear": int(row[4]) if row[4] else 0,
        }

        # Başlık benzerliği kontrol
        title_score = max(
            _fuzzy_score(_norm(film_title), _norm(cand["primaryTitle"])),
            _fuzzy_score(_norm(film_title), _norm(cand["originalTitle"])),
        )
        year_ok = bool(ocr_year and cand["startYear"] and abs(ocr_year - cand["startYear"]) <= 1)

        if title_score >= _FUZZY_THRESHOLD or year_ok:
            imdb_cast = self._fetch_cast(con, best_tconst)
            self._log(f"  [IMDb] STRAT-D kabul: {cand['primaryTitle']} — {best_count} ortak kişi")
            return self._build_result(con, cand, imdb_cast, "strat_d")

        return IMDBLookupResult(matched=False, reason="strat_d: title/year mismatch")
