"""person_verify.py — İsim bazlı TMDB Person Search doğrulama.

Film TMDB/IMDB'de bulunamadığında bile her OCR ismini tek tek doğrular.
Tiyatro gösterileri, nadir filmler, hiçbir veritabanında olmayan yapımlar
için kritik. 30.000 filme ölçeklenebilir olması için cache mekanizması içerir.

Kullanım:
    from core.person_verify import PersonVerifier
    pv = PersonVerifier(tmdb_client=client, log_cb=self._log)
    result = pv.verify_name("Joseph Guerin")
    crew = pv.verify_crew_list(crew_entries)
"""

import re

# TMDB department → VİTOS rol eşleştirmesi
_TMDB_DEPT_TO_ROLE: dict[str, str | None] = {
    "directing": "YÖNETMEN",
    "writing": "SENARYO",
    "camera": "GÖRÜNTÜ YÖNETMENİ",
    "editing": "KURGU",
    "production": "YAPIMCI",
    "acting": "OYUNCU",
    "sound": None,
    "art": None,
    "costume & make-up": None,
    "visual effects": None,
    "lighting": None,
    "crew": None,
}

# Asgari kişi ismi kriterleri
_MIN_NAME_LEN = 3
_MAX_NAME_WORDS = 5

# Kişi ismi olmayan terimlerin hızlı filtresi (küçük harf)
_NON_PERSON_EXACT: frozenset[str] = frozenset({
    # Fransızca rol başlıkları
    "cadreur", "son", "scripte", "régie", "regie", "bruitage",
    "mixage", "maquillage", "habilleur", "maintenance", "groupiste",
    "machinerie", "chauffeurs", "stagiaires", "laboratoire", "repiquage",
    "générique", "coordination", "avec", "avee",
    "coproduction", "une coproduction", "coopérative",
    "assistant monteur", "assistant opérateur", "assistant operateur",
    "assistant à la production", "assistant a la production",
    "assistant réalisateur", "assistant realisateur",
    "directeur de production", "supervision technique",
    "chef électricien", "chef electricien",
    "régisseur général", "regisseur général",
    "production exécutive", "production executive",
    # Ülke/coğrafya
    "france", "cameroun", "cameroon", "paris", "london",
    "allemagne", "belgique", "canada", "senegal", "mali",
    "burkina", "niger", "maroc", "tunisie", "algerie",
    "italia", "espana", "portugal", "suisse", "suede",
    # Kurum tipleri
    "ministere", "ministre", "ministry",
    "ecole", "universite", "lycee", "publique",
    "editions", "edition", "editeur",
    # Yapım / sunum
    "fodic", "presentent", "presente", "presenten",
    "les films", "les eleves",
    "hotel", "makonee", "makomee",
    "un film de", "a film by",
})

_NON_PERSON_CONTAINS: frozenset[str] = frozenset({
    "ministere", "ministre", "les eleves", "ecole publique",
    "une coproduction", "les films", "copyright",
})

# İngilizce/Türkçe/Fransızca jenerik etiketleri (tek veya çok kelime)
_ROLE_TITLE_PATTERNS = re.compile(
    r"^(assistant|directeur|director|producer|editor|operator|"
    r"yönetmen|yonetmen|senaryo|kamera|kurgu|yapımcı|yapimci|"
    r"producteur|réalisateur|realisateur|opérateur|operateur|"
    r"monteur|ingénieur|ingenieur|technicien|superviseur)\b",
    re.IGNORECASE,
)


def _is_likely_non_person(name: str) -> bool:
    """Verilen metnin gerçek bir kişi ismi olamayacağını hızlıca kontrol et."""
    if not name or len(name.strip()) < _MIN_NAME_LEN:
        return True

    low = name.strip().lower()

    if low in _NON_PERSON_EXACT:
        return True

    for term in _NON_PERSON_CONTAINS:
        if term in low:
            return True

    words = low.split()
    if len(words) > _MAX_NAME_WORDS:
        return True

    if _ROLE_TITLE_PATTERNS.match(low):
        return True

    return False


def _is_likely_person_name(name: str) -> bool:
    """Metnin yapısal olarak kişi ismine benzeyip benzemediğini kontrol et.

    En az 2 kelime, her kelime büyük harfle başlıyor veya yeterli uzunlukta.
    """
    if not name or len(name.strip()) < _MIN_NAME_LEN:
        return False
    t = name.strip()
    words = t.split()
    # Tek kelime → soyadı olabilir (ör. TAKOUKAM)
    if len(words) == 1:
        return len(t) >= 4 and not t.isupper() or len(t) >= 5
    # 2+ kelime ve her biri büyük harfle başlıyorsa → yüksek ihtimalle isim
    if all(w and w[0].isupper() for w in words):
        return True
    # İlk harfin büyük olması yeterli
    return t[0].isupper()


class PersonVerifier:
    """İsim bazlı TMDB Person Search doğrulama motoru.

    Film TMDB'de bulunamadığında bile her ismi tek tek doğrular.
    Cache mekanizması ile 30.000 filmde aynı kişi için tekrar sorgu yapmaz.

    Kullanım:
        pv = PersonVerifier(tmdb_client=client, log_cb=log)
        result = pv.verify_name("Joseph Guerin")
        # {"name": "Joseph Guerin", "found": True, "tmdb_person_id": ...,
        #  "known_for_department": "camera", "mapped_role": "GÖRÜNTÜ YÖNETMENİ",
        #  "confidence": 0.9, "source": "tmdb_person"}

        verified_crew = pv.verify_crew_list(crew_entries)
    """

    def __init__(self, tmdb_client=None, log_cb=None):
        self._tmdb = tmdb_client
        self._log = log_cb or (lambda x: None)
        # İsim → sonuç cache (aynı film içinde tekrar sorgulamayı önle)
        self._cache: dict[str, dict] = {}

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def verify_name(self, name: str) -> dict:
        """Tek bir ismi TMDB Person Search ile doğrula.

        Önce hızlı filtreden geçirir (rol başlığı/mekan/kısa):
          - Eğer non-person → found=False, reason="non_person_term"
        Sonra TMDB'de arar:
          - Bulunduysa → found=True + department + mapped_role
          - Bulunamadıysa → found=False, reason="not_found"

        Returns:
            {
                "name": str,
                "found": bool,
                "tmdb_person_id": int | None,
                "known_for_department": str | None,
                "mapped_role": str | None,   # _TMDB_DEPT_TO_ROLE'den
                "confidence": float,
                "source": str,               # "tmdb_person" | "non_person_filter" | "no_client"
                "reason": str,
            }
        """
        name = (name or "").strip()

        # Cache'de var mı?
        cache_key = name.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Hızlı filtre: kişi olamaz
        if _is_likely_non_person(name):
            result = {
                "name": name,
                "found": False,
                "tmdb_person_id": None,
                "known_for_department": None,
                "mapped_role": None,
                "confidence": 0.0,
                "source": "non_person_filter",
                "reason": "non_person_term",
            }
            self._cache[cache_key] = result
            return result

        # TMDB client yoksa yapısal tahminde bulun
        if not self._tmdb:
            is_person = _is_likely_person_name(name)
            result = {
                "name": name,
                "found": is_person,
                "tmdb_person_id": None,
                "known_for_department": None,
                "mapped_role": None,
                "confidence": 0.4 if is_person else 0.1,
                "source": "no_client",
                "reason": "structural_guess",
            }
            self._cache[cache_key] = result
            return result

        # TMDB Person Search
        try:
            results = self._tmdb.search_person(name)
        except Exception as exc:
            self._log(f"    [PERSON] TMDB person search hatası ({name}): {exc}")
            result = {
                "name": name,
                "found": False,
                "tmdb_person_id": None,
                "known_for_department": None,
                "mapped_role": None,
                "confidence": 0.3,
                "source": "tmdb_error",
                "reason": f"api_error:{exc}",
            }
            self._cache[cache_key] = result
            return result

        if not results:
            result = {
                "name": name,
                "found": False,
                "tmdb_person_id": None,
                "known_for_department": None,
                "mapped_role": None,
                "confidence": 0.2,
                "source": "tmdb_person",
                "reason": "not_found",
            }
            self._cache[cache_key] = result
            return result

        best = results[0]
        dept_raw = (best.get("known_for_department") or "").lower()
        mapped_role = _TMDB_DEPT_TO_ROLE.get(dept_raw)
        result = {
            "name": best.get("name", name),
            "found": True,
            "tmdb_person_id": best.get("id"),
            "known_for_department": dept_raw or None,
            "mapped_role": mapped_role,
            "confidence": 0.9,
            "source": "tmdb_person",
            "reason": "found",
        }
        self._cache[cache_key] = result
        return result

    def verify_crew_list(self, crew_entries: list[dict]) -> list[dict]:
        """Crew listesindeki tüm isimleri doğrula.

        Rol başlıkları ve non-person terimler otomatik filtrelenir.
        Gerçek kişiler TMDB person search ile doğrulanır.

        Args:
            crew_entries: list of dicts with at least "name" key.

        Returns:
            Filtrelenmiş ve doğrulanmış crew entry listesi (aynı format,
            "_person_verify" ek alanı eklenir).
        """
        verified: list[dict] = []
        for entry in (crew_entries or []):
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            result = self.verify_name(name)
            if result["found"]:
                enriched = dict(entry)
                enriched["_person_verify"] = result
                verified.append(enriched)
                self._log(
                    f"    [PERSON] ✅ {name} → dept={result.get('known_for_department')}"
                )
            elif _is_likely_person_name(name):
                # TMDB'de bulunamadı ama yapısal olarak isim gibi görünüyor → düşük güvenle koru
                enriched = dict(entry)
                enriched["_person_verify"] = result
                enriched["confidence"] = min(
                    float(entry.get("confidence", 0.6)) * 0.7, 0.5
                )
                verified.append(enriched)
                self._log(f"    [PERSON] ⚠️ {name} → bulunamadı, yapısal tahmin")
            else:
                self._log(
                    f"    [PERSON] ❌ {name} → REJECT "
                    f"({result.get('reason', 'unknown')})"
                )
        return verified

    def verify_cast_list(self, cast_entries: list[dict]) -> list[dict]:
        """Cast listesindeki tüm isimleri doğrula.

        verify_crew_list ile aynı mantık; actor_name alanını da okur.

        Args:
            cast_entries: list of dicts with "actor_name" or "name" key.

        Returns:
            Filtrelenmiş ve doğrulanmış cast entry listesi.
        """
        verified: list[dict] = []
        for entry in (cast_entries or []):
            name = (
                entry.get("actor_name") or entry.get("name") or ""
            ).strip()
            if not name:
                continue
            result = self.verify_name(name)
            if result["found"]:
                enriched = dict(entry)
                enriched["_person_verify"] = result
                verified.append(enriched)
                self._log(
                    f"    [PERSON/CAST] ✅ {name} → "
                    f"dept={result.get('known_for_department')}"
                )
            elif _is_likely_person_name(name):
                enriched = dict(entry)
                enriched["_person_verify"] = result
                enriched["confidence"] = min(
                    float(entry.get("confidence", 0.6)) * 0.7, 0.5
                )
                verified.append(enriched)
                self._log(f"    [PERSON/CAST] ⚠️ {name} → bulunamadı, yapısal tahmin")
            else:
                self._log(
                    f"    [PERSON/CAST] ❌ {name} → REJECT "
                    f"({result.get('reason', 'unknown')})"
                )
        return verified
