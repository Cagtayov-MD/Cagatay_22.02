"""
match_verify.py — Film/Dizi eşleştirme motoru.

Pipeline NAME_VERIFY sonrasında çalışır (EXPORT'tan önce).
Mevcut pipeline aşamalarına (INGEST → NAME_VERIFY) dokunmaz.

İçerik tipine göre iki strateji:
  - Film (3. blok = 1): başlık + yönetmen + top-2 oyuncu → TMDB teyit
  - Dizi (3. blok = 0): başlık + yönetmen → TMDB teyit
Eşleşme yoksa: OCR isim listesini Gemini 2.5 Flash ile doğrula (fallback).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz, process as rf_process
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

_TIMEOUT_SEC = 60


def _norm(s: str) -> str:
    """Karşılaştırma için normalize: küçük harf, sadece alfanumerik."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _fuzzy_match(query: str, choices: List[str], threshold: int = 82) -> Optional[str]:
    """Fuzzy eşleştirme — en iyi sonucu döndür."""
    if not query or not choices:
        return None
    qn = _norm(query)
    for c in choices:
        if _norm(c) == qn:
            return c
    if _HAS_RAPIDFUZZ:
        res = rf_process.extractOne(query, choices, scorer=fuzz.WRatio,
                                    score_cutoff=threshold)
        if res:
            return res[0]
    return None


_GEMINI_VERIFY_PROMPT = """Sen bir film/dizi jenerikleri uzmanısın. Sana OCR ile okunmuş bir isim listesi verilecek.
Görevin: Bu isimleri gerçek kişi isimleri olarak doğrulamak ve filtrelemek.

KATİ KURALLAR:
- SADECE JSON döndür. JSON dışında tek bir karakter bile yazma.
- Gerçek insan isimleri gibi görünmeyen metinleri eler (şirket adları, yer isimleri, teknik terimler, çöp OCR metni vb.)
- Eğer bir isim bozuk OCR'dan kaynaklanıyorsa ve düzeltilebiliyorsa düzelt.
- Emin değilsen o ismi EKLEME.

ÇIKTI ŞEMASI (kesin):
{
  "verified_names": [
    {"name": "Ad Soyad", "role": "cast|crew|director|unknown", "confidence": 0.0-1.0}
  ]
}

GİRDİ:
"""


class MatchVerifier:
    """Film/Dizi eşleştirme motoru.

    Pipeline NAME_VERIFY sonrasında çalışır.
    Mevcut pipeline aşamalarına dokunmaz.
    """

    def __init__(self, tmdb_client=None, imdb_client=None,
                 gemini_api_key: str = "", log_cb=None):
        """
        Args:
            tmdb_client: Mevcut TMDBClient örneği (pipeline_runner'dan geçirilir).
            imdb_client: Opsiyonel IMDB client (şu an kullanılmıyor).
            gemini_api_key: Gemini API anahtarı (fallback için).
            log_cb: Log callback fonksiyonu.
        """
        self._tmdb = tmdb_client
        self._imdb = imdb_client
        self._gemini_api_key = gemini_api_key or ""
        self._log_cb = log_cb

    def _log(self, msg: str) -> None:
        if self._log_cb:
            self._log_cb(msg)

    # ──────────────────────────────────────────────────────────────────────
    # Ana giriş noktası
    # ──────────────────────────────────────────────────────────────────────

    def verify(self, cdata: dict, film_id: str) -> dict:
        """Ana eşleştirme mantığı.

        Args:
            cdata: credits_data (pipeline çıktısı).
            film_id: Dosya adından parse edilen ID (ör: 1993-0466-1-0000-00-1).

        Returns:
            Güncellenmiş cdata dict'i.
        """
        is_series = self._is_series(film_id)
        media_label = "Dizi" if is_series else "Film"
        self._log(f"  [MATCH_VERIFY] Tür: {media_label} (film_id={film_id!r})")

        if is_series:
            return self._verify_series(cdata)
        else:
            return self._verify_film(cdata)

    # ──────────────────────────────────────────────────────────────────────
    # Tür belirleme
    # ──────────────────────────────────────────────────────────────────────

    def _is_series(self, film_id: str) -> bool:
        """Film ID'deki 3. bloğa bakarak dizi mi film mi belirle.

        3. blok = 1 → Film, 3. blok = 0 → Dizi.
        Örnek: '1993-0466-1-0000-00-1' → 3. blok = '1' → Film
               '1993-0466-0-0000-00-1' → 3. blok = '0' → Dizi
        """
        parts = (film_id or "").split("-")
        if len(parts) >= 3:
            return parts[2].strip() == "0"
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Film eşleştirme
    # ──────────────────────────────────────────────────────────────────────

    def _verify_film(self, cdata: dict) -> dict:
        """Film eşleştirme: yönetmen + film adı + 1-2 oyuncu → TMDB/IMDB teyit."""
        title = (cdata.get("film_title") or "").strip()
        directors = self._get_directors(cdata)
        top_actors = self._get_top_actors(cdata, count=2)

        self._log(
            f"  [MATCH_VERIFY] Film: başlık='{title}', "
            f"yönetmen={directors}, oyuncu={top_actors}"
        )

        # Adım 1: TMDB'de film adı + yönetmen + oyuncularla ara
        tmdb_match = self._search_tmdb(title, directors, top_actors, media_type="movie")
        if tmdb_match:
            self._log(f"  [MATCH_VERIFY] TMDB eşleşmesi bulundu (film)")
            return self._apply_match(cdata, tmdb_match)

        # Adım 2: Dizi olarak da dene (TMDB'de movie yerine tv kayıtlı olabilir)
        tmdb_match = self._search_tmdb(title, directors, top_actors, media_type="tv")
        if tmdb_match:
            self._log(f"  [MATCH_VERIFY] TMDB eşleşmesi bulundu (tv olarak kayıtlı)")
            return self._apply_match(cdata, tmdb_match)

        # Adım 3: Gemini fallback — OCR listesini Gemini ile kontrol et
        self._log(f"  [MATCH_VERIFY] TMDB eşleşmesi yok → Gemini fallback")
        return self._gemini_fallback(cdata)

    # ──────────────────────────────────────────────────────────────────────
    # Dizi eşleştirme
    # ──────────────────────────────────────────────────────────────────────

    def _verify_series(self, cdata: dict) -> dict:
        """Dizi eşleştirme: yönetmen + dizi adı → TMDB/IMDB teyit."""
        title = (cdata.get("film_title") or "").strip()
        directors = self._get_directors(cdata)

        self._log(
            f"  [MATCH_VERIFY] Dizi: başlık='{title}', yönetmen={directors}"
        )

        # Dizilerde oyuncu kullanılmıyor — sadece başlık + yönetmen
        tmdb_match = self._search_tmdb(title, directors, [], media_type="tv")
        if tmdb_match:
            self._log(f"  [MATCH_VERIFY] TMDB eşleşmesi bulundu (dizi)")
            return self._apply_match(cdata, tmdb_match)

        # Film olarak da kayıtlı olabilir
        tmdb_match = self._search_tmdb(title, directors, [], media_type="movie")
        if tmdb_match:
            self._log(f"  [MATCH_VERIFY] TMDB eşleşmesi bulundu (film olarak kayıtlı)")
            return self._apply_match(cdata, tmdb_match)

        # Gemini fallback
        self._log(f"  [MATCH_VERIFY] TMDB eşleşmesi yok → Gemini fallback")
        return self._gemini_fallback(cdata)

    # ──────────────────────────────────────────────────────────────────────
    # TMDB arama
    # ──────────────────────────────────────────────────────────────────────

    def _search_tmdb(
        self,
        title: str,
        directors: List[str],
        actors: List[str],
        media_type: str = "movie",
    ) -> Optional[Dict[str, Any]]:
        """TMDB'de başlık + yönetmen + oyuncularla ara.

        Args:
            title: Film/dizi başlığı.
            directors: Yönetmen isimleri listesi.
            actors: Oyuncu isimleri listesi (film için 1-2, dizi için boş).
            media_type: "movie" veya "tv".

        Returns:
            Eşleşme bulunursa TMDB entry dict'i, bulunamazsa None.
        """
        if not self._tmdb or not self._tmdb.enabled():
            self._log("  [MATCH_VERIFY] TMDB client mevcut değil — atlanıyor")
            return None

        if not title:
            self._log("  [MATCH_VERIFY] Başlık boş — TMDB araması atlanıyor")
            return None

        try:
            results = self._tmdb.search_multi(title)
        except Exception as e:
            self._log(f"  [MATCH_VERIFY] TMDB arama hatası: {e}")
            return None

        if not results:
            self._log(f"  [MATCH_VERIFY] '{title}' için TMDB sonucu yok")
            return None

        self._log(f"  [MATCH_VERIFY] '{title}' → {len(results)} TMDB sonucu")

        for r in results[:10]:
            kind = r.get("media_type", "")
            if kind != media_type:
                continue

            entry_title = (r.get("name") or r.get("title") or "?")
            entry_id = r.get("id", 0)
            self._log(f"  [MATCH_VERIFY]   Kontrol: '{entry_title}' ({kind}, id:{entry_id})")

            # Credits çek
            try:
                if kind == "tv":
                    credits = self._tmdb.get_tv_credits(entry_id)
                else:
                    credits = self._tmdb.get_movie_credits(entry_id)
            except Exception as e:
                self._log(f"  [MATCH_VERIFY]   Credits hatası: {e}")
                continue

            if not credits:
                continue

            # Yönetmen eşleşmesi kontrolü
            tmdb_crew = [
                (c.get("name") or "").strip()
                for c in (credits.get("crew") or [])
                if (c.get("name") or "").strip()
            ]
            director_match = any(
                _fuzzy_match(d, tmdb_crew, threshold=82)
                for d in directors
            ) if directors else False

            # Oyuncu eşleşmesi kontrolü
            tmdb_cast = [
                (c.get("name") or "").strip()
                for c in (credits.get("cast") or [])
                if (c.get("name") or "").strip()
            ]
            actor_matches = sum(
                1 for a in actors if _fuzzy_match(a, tmdb_cast, threshold=82)
            )

            self._log(
                f"  [MATCH_VERIFY]   Yönetmen eşleşmesi: {director_match}, "
                f"oyuncu eşleşmesi: {actor_matches}/{len(actors)}"
            )

            # Eşleşme kriterleri:
            # - Başlık + 2 oyuncu → kabul
            # - Başlık + 1 oyuncu + yönetmen → kabul
            # - Başlık + yönetmen → kabul (oyuncu yoksa — dizi durumu)
            if actor_matches >= 2:
                self._log(f"  [MATCH_VERIFY]   → Kabul: başlık + {actor_matches} oyuncu")
                return {"entry": r, "kind": kind, "credits": credits}
            if actor_matches >= 1 and director_match:
                self._log(f"  [MATCH_VERIFY]   → Kabul: başlık + 1 oyuncu + yönetmen")
                return {"entry": r, "kind": kind, "credits": credits}
            if not actors and director_match:
                self._log(f"  [MATCH_VERIFY]   → Kabul: başlık + yönetmen (oyuncu yok)")
                return {"entry": r, "kind": kind, "credits": credits}
            if not actors and not directors:
                # Başlık + hiç doğrulama verisi yoksa eşleşme yapılmaz.
                # Bu blok yalnızca hem oyuncu hem yönetmen bilgisi OCR'da bulunamamışsa
                # ve arama sadece başlıkla yapılmışsa aktif olur — false positive riski
                # yüksektir, bu nedenle bu durumda None dönülerek Gemini fallback devreye girer.
                pass

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Eşleşme uygulama
    # ──────────────────────────────────────────────────────────────────────

    def _apply_match(self, cdata: dict, match_result: Dict[str, Any]) -> dict:
        """TMDB eşleşmesini cdata'ya yaz — kanonik veriyi güncelle.

        Args:
            cdata: Mevcut credits data.
            match_result: _search_tmdb'den dönen dict (entry, kind, credits).

        Returns:
            Güncellenmiş cdata.
        """
        entry = match_result.get("entry", {})
        kind = match_result.get("kind", "")
        credits = match_result.get("credits", {})

        matched_title = (entry.get("name") or entry.get("title") or "").strip()
        matched_id = entry.get("id", 0)
        date_str = (entry.get("first_air_date") or entry.get("release_date") or "")
        matched_year = 0
        if date_str and len(date_str) >= 4:
            try:
                matched_year = int(date_str[:4])
            except ValueError:
                pass

        self._log(
            f"  [MATCH_VERIFY] Kanonik veri yazılıyor: '{matched_title}' "
            f"({kind}, id:{matched_id}, yıl:{matched_year})"
        )

        # TMDB eşleşme bilgisini cdata'ya ekle
        cdata["_match_verify_result"] = {
            "matched": True,
            "title": matched_title,
            "id": matched_id,
            "kind": kind,
            "year": matched_year,
            "method": "tmdb",
        }
        cdata["tmdb_verified"] = True
        cdata["tmdb_title"] = matched_title
        cdata["tmdb_id"] = matched_id
        if matched_year and not cdata.get("year"):
            cdata["year"] = matched_year

        # OCR cast'ını TMDB kanonik isimleriyle güncelle (soft update — sadece isim düzelt)
        tmdb_cast_names = [
            (c.get("name") or "").strip()
            for c in (credits.get("cast") or [])
            if (c.get("name") or "").strip()
        ]
        if tmdb_cast_names:
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
                    row["is_tmdb_verified"] = True
                    row["is_verified_name"] = True

        return cdata

    # ──────────────────────────────────────────────────────────────────────
    # Gemini fallback
    # ──────────────────────────────────────────────────────────────────────

    def _gemini_fallback(self, cdata: dict) -> dict:
        """Eşleşme yoksa: OCR'dan okunan isimleri Gemini 2.5 Flash ile kontrol et.

        Cast listesindeki tüm isimleri + crew isimlerini Gemini'ye gönderir.
        Gemini gerçek insan isimlerini doğrular, çöp OCR metinlerini eler.
        Sonuç cdata'ya yazılır.
        """
        if not self._gemini_api_key:
            self._log("  [MATCH_VERIFY] Gemini API key yok — fallback atlanıyor")
            cdata["_match_verify_result"] = {"matched": False, "method": "no_gemini_key"}
            return cdata

        # Tüm isimleri topla
        all_names: List[str] = []

        for row in (cdata.get("cast") or []):
            if isinstance(row, dict):
                name = (row.get("actor_name") or row.get("actor") or "").strip()
                if name and len(name) >= 3:
                    all_names.append(name)

        for row in (cdata.get("crew") or []):
            if isinstance(row, dict):
                name = (row.get("name") or "").strip()
                if name and len(name) >= 3:
                    all_names.append(name)

        for d in (cdata.get("directors") or []):
            if isinstance(d, str) and len(d.strip()) >= 3:
                all_names.append(d.strip())
            elif isinstance(d, dict):
                name = (d.get("name") or "").strip()
                if len(name) >= 3:
                    all_names.append(name)

        # Tekrarları kaldır
        seen: set = set()
        unique_names = []
        for n in all_names:
            if n not in seen:
                seen.add(n)
                unique_names.append(n)

        if not unique_names:
            self._log("  [MATCH_VERIFY] Gemini fallback: isim listesi boş — atlanıyor")
            cdata["_match_verify_result"] = {"matched": False, "method": "empty_names"}
            return cdata

        self._log(f"  [MATCH_VERIFY] Gemini fallback: {len(unique_names)} isim doğrulanıyor")

        # Prompt'a isimleri ekle
        names_text = "\n".join(f"- {n}" for n in unique_names)
        prompt = _GEMINI_VERIFY_PROMPT + names_text

        try:
            import core.llm_provider as _llm
            response = _llm._gemini_generate(
                prompt,
                api_key=self._gemini_api_key,
                model="gemini-2.5-flash",
                timeout=_TIMEOUT_SEC,
                log_cb=self._log_cb,
            )
        except Exception as e:
            self._log(f"  [MATCH_VERIFY] Gemini API hatası: {e}")
            cdata["_match_verify_result"] = {"matched": False, "method": "gemini_error",
                                              "error": str(e)}
            return cdata

        if not response:
            self._log("  [MATCH_VERIFY] Gemini boş yanıt döndürdü")
            cdata["_match_verify_result"] = {"matched": False, "method": "gemini_empty"}
            return cdata

        # Yanıtı parse et
        verified_names = self._parse_gemini_response(response)
        if not verified_names:
            self._log("  [MATCH_VERIFY] Gemini yanıtı parse edilemedi")
            cdata["_match_verify_result"] = {"matched": False, "method": "gemini_parse_error"}
            return cdata

        self._log(
            f"  [MATCH_VERIFY] Gemini doğruladı: {len(verified_names)} isim "
            f"(başlangıçta {len(unique_names)} vardı)"
        )

        # Doğrulanan isimlerle cast listesini güncelle
        verified_name_set = {_norm(v.get("name", "")) for v in verified_names}
        original_cast_count = len(cdata.get("cast") or [])

        filtered_cast = [
            row for row in (cdata.get("cast") or [])
            if isinstance(row, dict) and _norm(
                row.get("actor_name") or row.get("actor") or ""
            ) in verified_name_set
        ]

        if filtered_cast:
            cdata["cast"] = filtered_cast
            cdata["total_actors"] = len(filtered_cast)
            self._log(
                f"  [MATCH_VERIFY] Cast güncellendi: {original_cast_count} → {len(filtered_cast)}"
            )

        # Gemini'nin önerdiği yeni isimleri ekle (cast'ta olmayanlar)
        existing_cast_names = {
            _norm(row.get("actor_name") or row.get("actor") or "")
            for row in (cdata.get("cast") or [])
            if isinstance(row, dict)
        }
        for v in verified_names:
            vn = _norm(v.get("name", ""))
            vname_str = v.get("name", "").strip()
            if vn not in existing_cast_names and vname_str and v.get("role") in ("cast", "unknown"):
                cdata.setdefault("cast", []).append({
                    "actor_name": vname_str,
                    "character_name": "",
                    "role": "Cast",
                    "role_category": "cast",
                    "confidence": float(v.get("confidence", 0.7)),
                    "raw": "gemini_match_verify",
                    "is_verified_name": True,
                })
                existing_cast_names.add(vn)

        cdata["total_actors"] = len(cdata.get("cast") or [])
        cdata["_match_verify_result"] = {
            "matched": False,
            "method": "gemini_fallback",
            "verified_count": len(verified_names),
            "original_count": len(unique_names),
        }
        return cdata

    def _parse_gemini_response(self, response: str) -> List[Dict[str, Any]]:
        """Gemini yanıtından doğrulanmış isim listesini çıkar."""
        text = response.strip()
        if "```" in text:
            start = text.find("{", text.find("```"))
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
        else:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            self._log(f"  [MATCH_VERIFY] Gemini JSON parse hatası: {e}")
            return []

        if not isinstance(data, dict):
            return []

        items = data.get("verified_names", [])
        if not isinstance(items, list):
            return []

        result = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or "").strip()
            if name and len(name) >= 3:
                result.append({
                    "name": name,
                    "role": (item.get("role") or "unknown").strip(),
                    "confidence": float(item.get("confidence", 0.7)),
                })
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Yardımcı metotlar
    # ──────────────────────────────────────────────────────────────────────

    def _get_directors(self, cdata: dict) -> List[str]:
        """cdata'dan yönetmen isimlerini çıkar."""
        directors: List[str] = []

        for d in (cdata.get("directors") or []):
            if isinstance(d, str):
                name = d.strip()
            elif isinstance(d, dict):
                name = (d.get("name") or "").strip()
            else:
                continue
            if name and len(name) >= 3 and name not in directors:
                directors.append(name)

        # crew listesinden de yönetmen rolündeki isimleri al
        for row in (cdata.get("crew") or []):
            if not isinstance(row, dict):
                continue
            job = (row.get("job") or row.get("role") or "").lower()
            if "yönetmen" in job or "yonetmen" in job or "director" in job:
                name = (row.get("name") or "").strip()
                if name and len(name) >= 3 and name not in directors:
                    directors.append(name)

        return directors

    def _get_top_actors(self, cdata: dict, count: int = 2) -> List[str]:
        """Cast listesinden en yüksek confidence'lı N oyuncuyu döndür."""
        cast = [
            row for row in (cdata.get("cast") or [])
            if isinstance(row, dict)
        ]
        # confidence'a göre sırala (en yüksek önce)
        cast.sort(key=lambda r: float(r.get("confidence", 0.0)), reverse=True)

        names: List[str] = []
        for row in cast[:count]:
            name = (row.get("actor_name") or row.get("actor") or "").strip()
            if name and len(name) >= 3:
                names.append(name)
        return names
