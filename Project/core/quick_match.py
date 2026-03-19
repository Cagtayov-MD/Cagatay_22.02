"""
quick_match.py — Hızlı eşleşme stratejisi (CREDITS_PARSE → NAME_VERIFY arası).

Strateji (3 katmanlı):
  Katman 1: CREDITS_PARSE çıktısından yönetmen + film adı + top-2 oyuncu çıkar.
  Katman 2: Bu az bilgiyle TMDB'de hızlı arama yap.
            Eşleşme varsa → TMDB'den tam cast/crew çek, cdata'yı güncelle,
                            NAME_VERIFY'ı atla.
  Katman 3: (Sadece filmler için) Eşleşme yoksa tüm OCR listesini
            GeminiCastExtractor.extract() ile temizle, tekrar Katman 2'yi dene.
            Hâlâ eşleşme yoksa → NAME_VERIFY akışına devam et (fallback).

Dizi vs Film ayrımı (film_id 3. bloğuna göre):
  3. blok = 1 → Film → Katman 1-2-3 uygula
  3. blok = 0 → Dizi → Sadece Katman 1-2 uygula
"""
from __future__ import annotations

import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Opsiyonel bağımlılıklar — yoksa ilgili katman devre dışı kalır.
try:
    from core.tmdb_verify import TMDBVerify
except ImportError:  # pragma: no cover
    TMDBVerify = None  # type: ignore

try:
    from core.gemini_cast_extractor import GeminiCastExtractor
except ImportError:  # pragma: no cover
    GeminiCastExtractor = None  # type: ignore


def _ocr_lines_to_text(ocr_lines: list) -> List[str]:
    """OCR satır listesinden metin dizisi çıkar."""
    texts = []
    for line in ocr_lines:
        if isinstance(line, dict):
            texts.append(line.get("text", ""))
        elif hasattr(line, "text"):
            texts.append(line.text)
        else:
            texts.append(str(line))
    return texts


class QuickMatcher:
    """Hızlı eşleşme stratejisi: az veri ile TMDB eşleşme dene."""

    def __init__(self, tmdb_client=None, gemini_api_key: str = "", log_cb=None):
        """
        Args:
            tmdb_client: TMDBClient örneği (None ise TMDB devre dışı).
            gemini_api_key: Gemini API anahtarı (Katman 3 için).
            log_cb: Opsiyonel log geri çağırma fonksiyonu.
        """
        self._tmdb_client = tmdb_client
        self._gemini_api_key = (gemini_api_key or "").strip()
        self._log_cb = log_cb

    def _log(self, msg: str) -> None:
        if self._log_cb:
            self._log_cb(msg)

    # ── Yardımcılar ──────────────────────────────────────────────────

    def _is_film(self, film_id: str) -> bool:
        """Film ID'nin 3. bloğuna göre film mi (True) dizi mi (False) belirle.

        3. blok = 0 → Dizi, aksi hâlde Film.
        """
        parts = film_id.split("-")
        if len(parts) < 3:
            return True  # Varsayılan: film
        return parts[2].strip() != "0"

    def _extract_key_info(self, cdata: dict) -> Tuple[str, List[str], List[str]]:
        """cdata'dan film adı, yönetmen listesi ve top-2 oyuncu çıkar.

        Returns:
            (film_title, director_names, top2_cast_names)
        """
        film_title = (cdata.get("film_title") or "").strip()

        directors: List[str] = []
        for d in (cdata.get("directors") or []):
            if isinstance(d, str):
                name = d.strip()
            elif isinstance(d, dict):
                name = (d.get("name") or "").strip()
            else:
                continue
            if name:
                directors.append(name)

        cast_with_conf: List[Tuple[str, float]] = []
        for row in (cdata.get("cast") or []):
            if not isinstance(row, dict):
                continue
            name = (row.get("actor_name") or row.get("actor") or "").strip()
            conf = float(row.get("confidence", 0.0))
            if name:
                cast_with_conf.append((name, conf))
        cast_with_conf.sort(key=lambda x: -x[1])
        top_cast = [name for name, _ in cast_with_conf[:2]]

        return film_title, directors, top_cast

    # ── TMDB ─────────────────────────────────────────────────────────

    def _try_tmdb(
        self,
        film_title: str,
        cast_names: List[str],
        director_names: List[str],
        work_dir: str,
        cdata: dict,
    ) -> Tuple[bool, dict]:
        """TMDB'de hızlı eşleşme dene.

        Returns:
            (matched: bool, updated_cdata: dict)
        """
        if not self._tmdb_client or not self._tmdb_client.enabled():
            return False, cdata

        if TMDBVerify is None:
            self._log("  [QUICK_MATCH] tmdb_verify modülü yüklenemedi")
            return False, cdata

        verifier = TMDBVerify(
            work_dir=work_dir,
            api_key=self._tmdb_client.api_key,
            bearer_token=getattr(self._tmdb_client, "bearer", ""),
            language=getattr(self._tmdb_client, "language", "tr-TR"),
            log_cb=self._log_cb,
        )

        entry, kind, _ = verifier._find_tmdb_entry(
            film_title=film_title,
            cast_names=cast_names,
            director_names=director_names,
        )
        if not entry:
            return False, cdata

        tmdb_id = entry.get("id", 0)
        if not tmdb_id:
            return False, cdata

        credits = verifier._fetch_credits(kind, tmdb_id)
        if not credits:
            return False, cdata

        tmdb_title = (entry.get("name") or entry.get("title") or "").strip()
        self._log(
            f"  [QUICK_MATCH] ✓ TMDB eşleşme: '{tmdb_title}' "
            f"(id:{tmdb_id}, tür:{kind})"
        )

        updated = self._apply_tmdb_credits(cdata, entry, kind, credits)
        return True, updated

    def _apply_tmdb_credits(
        self,
        cdata: dict,
        entry: dict,
        kind: str,
        credits: dict,
    ) -> dict:
        """TMDB cast/crew verisiyle cdata'yı güncelle."""
        tmdb_id = entry.get("id", 0)
        tmdb_title = (entry.get("name") or entry.get("title") or "").strip()

        # Cast güncelle
        tmdb_cast: List[Dict[str, Any]] = []
        for item in (credits.get("cast") or []):
            name = (item.get("name") or "").strip()
            if not name:
                continue
            tmdb_cast.append({
                "actor_name": name,
                "character_name": (item.get("character") or "").strip(),
                "role": "Cast",
                "role_category": "cast",
                "raw": "tmdb",
                "confidence": 1.0,
                "frame": "tmdb",
                "is_verified_name": True,
                "is_tmdb_verified": True,
                "tmdb_order": item.get("order", 999),
            })

        # Crew güncelle
        tmdb_crew: List[Dict[str, Any]] = []
        for item in (credits.get("crew") or []):
            name = (item.get("name") or "").strip()
            job = (item.get("job") or "").strip()
            if not name or not job:
                continue
            tmdb_crew.append({
                "name": name,
                "job": job,
                "role": job,
                "raw": "tmdb",
                "confidence": 1.0,
                "is_verified_name": True,
                "is_tmdb_verified": True,
            })

        cdata["cast"] = tmdb_cast
        cdata["technical_crew"] = tmdb_crew
        cdata["crew"] = tmdb_crew
        cdata["total_actors"] = len(tmdb_cast)
        cdata["total_crew"] = len(tmdb_crew)
        cdata["verification_status"] = "tmdb_verified"
        if tmdb_title:
            _current = (cdata.get("film_title") or "").strip()
            if _current and _current != tmdb_title and not cdata.get("ocr_title"):
                cdata["ocr_title"] = _current
            cdata["film_title"] = tmdb_title
        cdata["tmdb_id"] = tmdb_id
        cdata["tmdb_type"] = kind

        return cdata

    # ── Gemini ───────────────────────────────────────────────────────

    def _run_gemini(self, ocr_lines: list, cdata: dict) -> bool:
        """Gemini ile OCR satırlarından cast/crew ayıkla ve cdata'yı güncelle.

        Returns:
            True — Gemini sonuç döndürdü ve cdata güncellendi.
        """
        if not self._gemini_api_key:
            return False

        if GeminiCastExtractor is None:
            self._log("  [QUICK_MATCH] gemini_cast_extractor modülü yüklenemedi")
            return False

        extractor = GeminiCastExtractor(
            api_key=self._gemini_api_key,
            log_cb=self._log_cb,
        )
        ocr_text_list = _ocr_lines_to_text(ocr_lines)
        result = extractor.extract(
            ocr_lines=ocr_text_list,
            film_title=cdata.get("film_title", ""),
        )
        if not result or not (result.get("cast") or result.get("crew")):
            return False

        self._log(
            f"  [QUICK_MATCH] Gemini: "
            f"{len(result.get('cast', []))} oyuncu, "
            f"{len(result.get('crew', []))} ekip"
        )
        cdata["cast"] = result.get("cast", [])
        cdata["crew"] = result.get("crew", [])
        cdata["total_actors"] = len(cdata["cast"])
        cdata["total_crew"] = len(cdata["crew"])
        return True

    # ── Ana metot ────────────────────────────────────────────────────

    def match(self, cdata: dict, ocr_lines: list, film_id: str) -> dict:
        """Hızlı eşleşme stratejisini çalıştır.

        Args:
            cdata: CREDITS_PARSE aşamasının çıktısı.
            ocr_lines: Ham OCR satır listesi.
            film_id: Dosya adından çıkarılan film ID (örn. "1955-0019-1-0000-00-1").

        Returns:
            {
                "matched": bool,
                "method": "quick_tmdb" | "gemini_then_tmdb" | "fallback_name_verify",
                "cdata": dict,  # Güncellenmiş cdata
            }
        """
        is_film = self._is_film(film_id)
        self._log(
            f"  [QUICK_MATCH] Başlıyor — "
            f"{'Film' if is_film else 'Dizi'} (film_id: {film_id!r})"
        )

        # Katman 1: Anahtar bilgileri çıkar
        film_title, directors, top_cast = self._extract_key_info(cdata)
        self._log(
            f"  [QUICK_MATCH] Başlık: {film_title!r}  "
            f"Yönetmen: {directors}  Top-2 Oyuncu: {top_cast}"
        )

        work_dir = tempfile.mkdtemp(prefix="qmatch_")
        try:
            # Katman 2: TMDB hızlı eşleşme
            matched, updated_cdata = self._try_tmdb(
                film_title, top_cast, directors, work_dir, cdata
            )
            if matched:
                updated_cdata["quick_match_method"] = "quick_tmdb"
                updated_cdata["quick_match_skipped_name_verify"] = True
                return {"matched": True, "method": "quick_tmdb", "cdata": updated_cdata}

            # Katman 3 sadece filmler için
            if not is_film:
                self._log("  [QUICK_MATCH] Dizi — Katman 3 atlanıyor, NAME_VERIFY'a devam")
                cdata["quick_match_method"] = "none"
                cdata["quick_match_skipped_name_verify"] = False
                return {"matched": False, "method": "fallback_name_verify", "cdata": cdata}

            # Katman 3: Gemini fallback → tekrar TMDB
            gemini_ok = self._run_gemini(ocr_lines, cdata)
            if gemini_ok:
                film_title2, directors2, top_cast2 = self._extract_key_info(cdata)
                matched2, updated_cdata2 = self._try_tmdb(
                    film_title2, top_cast2, directors2, work_dir, cdata
                )
                if matched2:
                    updated_cdata2["quick_match_method"] = "gemini_then_tmdb"
                    updated_cdata2["quick_match_skipped_name_verify"] = True
                    return {
                        "matched": True,
                        "method": "gemini_then_tmdb",
                        "cdata": updated_cdata2,
                    }

            self._log("  [QUICK_MATCH] Eşleşme bulunamadı → NAME_VERIFY'a devam")
            cdata["quick_match_method"] = "none"
            cdata["quick_match_skipped_name_verify"] = False
            return {"matched": False, "method": "fallback_name_verify", "cdata": cdata}

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
