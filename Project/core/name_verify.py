"""
name_verify.py — Katmanlı isim doğrulama motoru + verification log.

Export engine'e veri girmeden önce çalışır.
7 yıldızlı YAPIM EKİBİ ve OYUNCU listesini katman katman temizler/doğrular.

Katmanlar (ucuzdan pahalıya):
  1. Hard Blacklist + Yapısal Kontrol  (maliyet: 0)
  2. NameDB doğrulaması               (maliyet: 0, lokal)
  3. TMDB Person Search               (maliyet: düşük, HTTP)
  4. Gemini YES/NO doğrulama          (maliyet: yüksek, son çare)
     Pass 2 akışı: doğrulanamamış adaylar için tam-isim fuzzy top-2 →
     minimum fuzzy gate (≥72) → Gemini YES/NO validator (aday 1) →
     reddedilirse ve aday 2 de gate'i geçiyorsa Gemini YES/NO (aday 2) →
     aksi hâlde unresolved (raw OCR korunur, verified=False).
  5. Bulunamadı → unresolved olarak flag ile rapora yaz (maliyet: 0)

Her aşama bir verification_log JSON'a yazılır (D:\\DATABASE altına).
Log formatı: her katmanda her isim için ne oldu (kept/dropped/corrected/flagged).

Kullanım:
    verifier = NameVerifier(name_db=name_db, tmdb_client=tmdb_client, log_cb=log)
    result = verifier.verify_crew(crew_roles_dict)
    result = verifier.verify_cast(cast_list)
    log_data = verifier.get_log()
"""

import re
import unicodedata

try:
    from rapidfuzz import fuzz as _rf_fuzz, process as _rf_process
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# Minimum fuzzy skor eşiği — bu puanın altındaki adaylar Gemini'ye gönderilmez.
# Agresif filtreleme önlemek için 60–65 gibi çok düşük bir değerden kaçınılır;
# mevcut doğrudan-kabul eşiğinden (85) belirgin şekilde düşük tutulur.
_GEMINI_FUZZY_GATE = 72


def _norm_name(s: str) -> str:
    """Karşılaştırma için normalize: küçük harf, sadece alfanumerik."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _fuzzy_name_match(query: str, choices: list, threshold: int = 82) -> bool:
    """İsim fuzzy eşleşmesi — query, choices içinde eşleşiyor mu?"""
    if not query or not choices:
        return False
    qn = _norm_name(query)
    for c in choices:
        if _norm_name(c) == qn:
            return True
    if _HAS_RAPIDFUZZ:
        res = _rf_process.extractOne(
            query, choices, scorer=_rf_fuzz.WRatio, score_cutoff=threshold
        )
        return res is not None
    return False

# ═══════════════════════════════════════════════════════════════════
# KATMAN 1 — HARD BLACKLIST
# ═══════════════════════════════════════════════════════════════════

_CREW_BLACKLIST_EXACT = frozenset({
    # ── Teknik terimler / Rol başlıkları ──
    "a.s.c.", "a.s.c", "asc", "a.c.e.", "a.c.e", "ace",
    "additional assistant directors", "adr mixer",
    "assistant director", "b camera operator",
    "best boy electric", "best boy grip", "best boy grips",
    "camera operator", "certificate",
    "chief financial officer", "color by", "colour by",
    "color consultant", "craft service",
    "dialogue editor", "director of photography",
    "drivers", "drink mixer", "driak mixer",
    "editorial supervision", "editorial supervision.",
    "film editor", "film edtor",
    "first assistant camera", "first assistant director",
    "gaffer", "hair styles by",
    "key grip", "location manager",
    "main title design by", "makeup",
    "music direction", "mustc direction",
    "of photography", "post production",
    "production designer", "production manager",
    "prop master", "prop master.",
    "property master", "property master.",
    "recording system", "script supervisor",
    "seript supervisor", "second unit",
    "set costumers", "set decorations", "set medic",
    "special effects coordinator", "special elfects coordinator",
    "stage managers", "steadicam operator",
    "still photographer", "still photooranher",
    "stunt coordinator", "stunt coordinators", "stunt players",
    "swing gang", "technical advisor", "technical manager",
    "transportation captain", "transportation coordinator",
    "wardrobe", "wirdrobe",
    # ── Marka / Şirket / Teknoloji ──
    "technicolor", "technicolor'",
    "technicolor color consultant",
    "lechnicolor color consultant",
    "colsy technicolor", "techongplor",
    "eastman color", "panavision",
    "kodak", "dolby", "thx",
    "metro", "goldwyn", "mayer", "metro goldwyn mayer",
    "universal", "universal-international picture",
    # ── Jenerik kalıpları ──
    "the end", "theend", "the", "end",
    "presents", "presente", "featuring", "introducing",
    "with", "and", "and introducing",
    "screen", "trade", "reward", "rewward",
    "all rights reserved", "all rights reserved.",
    "copyright", "fin",
    # ── Arapça jenerik / program bilgisi ──
    "على طريق العشق", "الحلقة", "بطولة",
    "مصمم المشروع", "المشروع مستشار", "فريدة",
    # ── Kazakça / diğer dil rol başlıkları ──
    "qoiyshy rejissery", "dybys rejissery",
    "ctsenarii avtorlary:", "ctsenarii avtorlary",
    "kasting:", "grim syretshist:",
    "kostiym syretshisi", "kompozitory",
    "kompiyterlik", "bas prodiyser:",
    "grafikanyn sypervaizeri:",
    "tapsyrysy boiynsha",
    # ── Sıfatlar / kısa bağlaçlar ──
    "re", "cr", "ir", "pa", "or", "and",
    # ── Fransızca jenerik etiketleri ──
    "cadreur",
    "son",
    "musique de",
    "musiques additionnelles",
    "musiques additionelles",  # OCR varyantı
    "scripte",
    "seripte",  # OCR varyantı
    "régie", "regie",
    "bruitage", "brultage",  # OCR varyantı
    "mixage", "minage", "mirage",  # OCR varyantları
    "maquillage",
    "habilleur", "habllleur",  # OCR varyantı
    "maintenance",
    "groupiste",
    "machinerie",
    "chauffeurs",
    "stagiaires", "staniaires",  # OCR varyantı
    "laboratoire",
    "repiquage",
    "génériques", "genériques",  # OCR varyantı
    "coordination",
    "avec le concours de",
    "avec la participation de",
    "avec", "avee",  # OCR varyantı
    "nous remercions",
    "une coproduction",
    "coproduction",
    "coopérative",
    "hotel",
    "makonee", "makomee",  # lokasyon/otel adı + OCR
    # ── Ülke / şehir / coğrafi ifadeler ──
    "france", "cameroun", "cameroon", "paris", "london",
    "allemagne", "belgique", "canada", "senegal", "mali",
    "burkina", "niger", "maroc", "tunisie", "algerie",
    "italia", "espana", "portugal", "suisse", "suede",
    # ── Kurum tipleri / resmi terimler ──
    "ministere", "ministre", "ministry",
    "ecole", "universite", "lycee", "publique",
    "editions", "edition", "editeur",
    # ── Yapım şirketi / sunum ifadeleri ──
    "fodic", "presentent", "presente", "presenten",
    "les films", "les eleves",
    "assistant à la production",
    "assistant a la production",
    "assistant à la produetion",  # OCR varyantı
    "assistant monteur",
    "assistant opérateur",
    "assistant operateur",
    "assistants au son",
    "secrétaires de production",
    "secretaires de production",
    "directeur de production",
    "directeur de produetion",  # OCR varyantı
    "supervision technique",
    "supervision technlque",  # OCR varyantı
    "chef électricien", "chef electricien",
    "chef dlectricien", "chef eleetricien",  # OCR varyantları
    "régisseur général", "regisseur général",
    "régísseur général",  # OCR varyantı
    "production exécutive", "production executive",
    "produetion exécutive",
    "assistant réalisateur", "assistant realisateur",
})

# Blacklist'te kelime içeren ifadeler (contains match)
_CREW_BLACKLIST_CONTAINS = frozenset({
    "copyright", "all rights reserved",
    "technicolor", "eastman color",
    "color consultant", "colour consultant",
    "certificate no", "certificate",
    "recording system",
    "filmed at", "filmed in",
    "panavision", "in hollywood",
    "bolum",  # bölüm bilgisi
    "ministere", "ministre",  # resmi kurum
    "les eleves",              # öğrenci grubu
    "ecole publique",          # okul adı
    "une coproduction",        # yapım ifadesi
    "les films",               # yapım şirketi öneki
})

# Regex ile yakalanacak kalıplar
_BLACKLIST_PATTERNS = [
    re.compile(r"^\W+$"),                          # sadece özel karakter
    re.compile(r"^\d[\d\s.\-:,]*$"),               # sadece rakam/noktalama
    re.compile(r"^.{0,2}$"),                        # 2 karakter veya daha kısa
    re.compile(r"copyright\s+\w", re.IGNORECASE),   # copyright metinleri
    re.compile(r"^no\.\s*\d+", re.IGNORECASE),      # sertifika numaraları
    re.compile(r"^\d+\s+(lr|rr|mm)\d+", re.IGNORECASE),  # film kodu (34 LR505)
    re.compile(r"styles?\s+by", re.IGNORECASE),     # hair styles by
    re.compile(r"color\s+by", re.IGNORECASE),       # color by
    re.compile(r"recorded\s+in", re.IGNORECASE),    # recorded in
    re.compile(r"filmed\s+(at|in)", re.IGNORECASE),  # filmed at/in
    re.compile(r"\b\d{2,}\b"),                        # Fix-B: 2+ haneli rakam içeriyor → sertifika no, teknik kod (MPAA # 45977)
]


def _is_latin(text: str) -> bool:
    """Metnin çoğunluğu Latin alfabesi mi?"""
    if not text:
        return False
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    latin_count = sum(1 for c in alpha_chars
                      if unicodedata.category(c).startswith("L")
                      and ord(c) < 0x0400)  # Kiril/Ermeni/Arap dışı Latin
    return latin_count / len(alpha_chars) > 0.5


def _structural_check(name: str) -> tuple[bool, str]:
    """Yapısal kontrol — isim formatına uygun mu?

    Returns:
        (passed: bool, reason: str)
    """
    t = name.strip()

    # Boş veya çok kısa
    if len(t) < 3:
        return False, "too_short"

    # 4+ kelimeli metinler genellikle isim değil
    words = t.split()
    if len(words) > 4:
        return False, "too_many_words"

    # Tamamen büyük harf + 8+ karakter + boşluk yok → birleşik OCR çöpü
    if t.isupper() and len(t) >= 8 and " " not in t:
        # Ama tek kelimelik soyad olabilir (STALLONE gibi)
        if len(t) > 15:
            return False, "concatenated_noise"

    # Latin olmayan metin (Arapça, Kiril vs.) — Türkçe karakterler OK
    if not _is_latin(t):
        return False, "non_latin"

    # Diyalog satırı — tire/uzun tire ile başlıyor
    if t[0] in ('-', '—', '–'):
        return False, "dialogue_marker"

    # Soru veya ünlem işareti → cümle, isim değil
    if '?' in t or '!' in t:
        return False, "sentence_punctuation"

    # Virgül içeriyor → cümle parçası, isim değil
    if ',' in t:
        return False, "sentence_fragment"

    # Slash içeriyor → alıntı veya referans, isim değil
    if '/' in t:
        return False, "slash_reference"

    # Nokta ile biten + 3+ kelime → cümle, isim değil
    if t.endswith('.') and len(words) >= 3:
        return False, "sentence_ending"

    # Noktalama yoğunluğu
    punct_count = sum(1 for c in t if c in ";…[]{}()")
    if punct_count >= 2:
        return False, "excessive_punctuation"

    # Sadece büyük harfli tek kelime + noktalama → teknik terim
    if len(words) == 1 and t.endswith(".") and t[:-1].isupper():
        return False, "technical_abbreviation"

    return True, "ok"


def _blacklist_check(name: str) -> tuple[bool, str]:
    """Blacklist kontrolü.

    Returns:
        (is_blacklisted: bool, reason: str)
    """
    low = name.strip().lower()

    # Exact match
    if low in _CREW_BLACKLIST_EXACT:
        return True, f"blacklist_exact:{low}"

    # Contains match
    for term in _CREW_BLACKLIST_CONTAINS:
        if term in low:
            return True, f"blacklist_contains:{term}"

    # Regex match
    for pattern in _BLACKLIST_PATTERNS:
        if pattern.search(name.strip()):
            return True, f"blacklist_pattern:{pattern.pattern}"

    return False, "ok"


def is_valid_person_name(name: str) -> bool:
    """Bir metnin gerçek kişi ismi olup olmadığını kontrol et.

    Kontroller (öncelik sırasıyla):
      1. CJK / CJK-benzeri karakterler → reddet
      2. Blacklist (teknik terimler, departman adları, şirketler)
      3. Yapısal kontrol (çok kısa, Latin olmayan, aşırı noktalama vb.)
      4. Tek-kelimelik isim → şüpheli (soyadı yok)
      5. 12+ karakter büyük harfli tek token → OCR birleştirme çöpü (VIDEOTAPEMUSIC vb.)

    Returns:
        bool — True ise geçerli kişi ismi.
    """
    t = name.strip()

    # 1. CJK ve benzeri Unicode blokları
    for ch in t:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF    # CJK Unified
                or 0x3040 <= cp <= 0x30FF   # Hiragana / Katakana
                or 0xAC00 <= cp <= 0xD7AF   # Hangul
                or 0x0400 <= cp <= 0x04FF   # Kiril
                or 0x0600 <= cp <= 0x06FF): # Arapça
            return False

    # 2. Blacklist
    is_bl, _ = _blacklist_check(t)
    if is_bl:
        return False

    # 3. Yapısal kontrol
    passed, _ = _structural_check(t)
    if not passed:
        return False

    # 4. Tek-kelimelik isim (soyadı eksik — OCR çöpü veya kısaltma: "CHARLES")
    words = t.split()
    if len(words) < 2:
        return False

    # 5. 12+ karakter büyük harfli tek token → birleşik OCR çöpü (VIDEOTAPEMUSIC = 14)
    if any(len(w) > 12 and w.isupper() and w.isalpha() for w in words):
        return False

    return True


# ═══════════════════════════════════════════════════════════════════
# KATMAN 3 — TMDB Person Search (kişi merkezli doğrulama)
# ═══════════════════════════════════════════════════════════════════

def _tmdb_person_verify(name: str, expected_role: str, tmdb_client, log_cb=None) -> dict:
    """TMDB person search ile isim doğrulama.

    Film aramıyor, kişi arıyor. Nisa Serezli gibi TMDB'de filmi olmayan
    ama kişi olarak kayıtlı olan isimler doğrulanır.

    Args:
        name: Doğrulanacak isim
        expected_role: Geriye dönük uyumluluk için korunur; rol türetmede kullanılmaz.
        tmdb_client: TMDBClient instance (search_person metodu olan)
        log_cb: Log callback

    Returns:
        dict: {verified, tmdb_name, tmdb_department, tmdb_id, reason}
    """
    if not tmdb_client or not name:
        return {"verified": False, "reason": "no_client_or_name"}

    try:
        results = tmdb_client.search_person(name.strip())
    except Exception as e:
        if log_cb:
            log_cb(f"    [TMDB] Person search hatası: {e}")
        return {"verified": False, "reason": f"api_error:{e}"}

    if not results:
        return {"verified": False, "reason": "not_found"}

    # En az bir sonuç var → isim doğrulandı
    # İlk sonucu al (en yüksek popularity)
    best = results[0]
    tmdb_name = best.get("name", "")
    tmdb_dept = (best.get("known_for_department") or "").lower()
    tmdb_id = best.get("id")

    return {
        "verified": True,
        "tmdb_name": tmdb_name,
        "tmdb_department": tmdb_dept,
        "tmdb_id": tmdb_id,
        "reason": "found",
    }


# ═══════════════════════════════════════════════════════════════════
# ANA SINIF — NameVerifier
# ═══════════════════════════════════════════════════════════════════

class NameVerifier:
    """Katmanlı isim doğrulama motoru.

    Her doğrulama adımını loglar. Log, D:\\DATABASE altına JSON olarak yazılır.

    Kullanım:
        verifier = NameVerifier(name_db=name_db, tmdb_client=tmdb_client)
        crew_roles = verifier.verify_crew(crew_roles)
        cast_list = verifier.verify_cast(cast_list)
        log = verifier.get_log()
    """

    def __init__(self, name_db=None, tmdb_client=None, log_cb=None):
        self._name_db = name_db
        self._tmdb_client = tmdb_client
        self._log = log_cb or (lambda m: None)
        self._verification_log = []  # Her adım burada

    def _add_log(self, layer: str, role: str, name_in: str,
                 name_out: str, action: str, reason: str, extra: dict = None):
        """Log satırı ekle."""
        entry = {
            "layer": layer,
            "role": role,
            "name_in": name_in,
            "name_out": name_out,
            "action": action,   # kept / dropped / corrected / flagged
            "reason": reason,
        }
        if extra:
            entry.update(extra)
        self._verification_log.append(entry)

    def get_log(self) -> list:
        """Tüm doğrulama logunu döndür."""
        return self._verification_log

    def get_log_text(self) -> str:
        """Okunabilir metin log — rapor dosyasına yazılabilir."""
        lines = []
        current_layer = ""
        for entry in self._verification_log:
            layer = entry["layer"]
            if layer != current_layer:
                lines.append("")
                lines.append(f"{'─' * 60}")
                lines.append(f"  {layer}")
                lines.append(f"{'─' * 60}")
                current_layer = layer

            role = entry.get("role", "")
            name_in = entry.get("name_in", "")
            name_out = entry.get("name_out", "")
            action = entry.get("action", "")
            reason = entry.get("reason", "")

            if action == "dropped":
                lines.append(f"  [{role}] ✗ {name_in}  →  ÇIKARILDI ({reason})")
            elif action == "corrected":
                lines.append(f"  [{role}] ↻ {name_in}  →  {name_out} ({reason})")
            elif action == "kept":
                lines.append(f"  [{role}] ✓ {name_in} ({reason})")
            elif action == "flagged":
                lines.append(f"  [{role}] ? {name_in} ({reason})")

            # TMDB ek bilgileri
            if entry.get("tmdb_department"):
                lines.append(f"          TMDB: {entry.get('tmdb_name','')} "
                             f"[{entry.get('tmdb_department','')}] "
                             f"id={entry.get('tmdb_id','')}")
            if entry.get("namedb_score"):
                lines.append(f"          NameDB: skor={entry['namedb_score']:.2f}")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────
    # VERIFY CREW — 7 yıldızlı yapım ekibi doğrulama
    # ─────────────────────────────────────────────────────

    def verify_crew(self, crew_roles: dict) -> dict:
        """7 yıldızlı yapım ekibini katman katman doğrula.

        Args:
            crew_roles: dict mapping role → list of names
                        Örnek: {"YÖNETMEN": ["CHARLES WALTERS", "KUL KEDİSİ"], ...}

        Returns:
            Temizlenmiş crew_roles dict (aynı format).
        """
        self._log("\n  ── NAME VERIFY: YAPIM EKİBİ ──")
        self._add_log("ONEOCR_RAW", "", "", "", "section_start",
                       "Yapım ekibi ham veri")

        # Log raw state
        for role, names in crew_roles.items():
            for name in names:
                self._add_log("ONEOCR_RAW", role, name, name, "raw", "ocr_output")

        result = {}
        for role, names in crew_roles.items():
            verified_names = []
            unverified_candidates = []

            # ── GEÇİŞ 1: Tüm isimleri işle, verified ve unverified'ı ayır ──
            for name in names:
                name = name.strip()
                if not name or name == "VERİ YOK":
                    continue

                # ── KATMAN 1: Blacklist + Yapısal Kontrol ──
                is_bl, bl_reason = _blacklist_check(name)
                if is_bl:
                    self._add_log("BLACKLIST", role, name, "", "dropped", bl_reason)
                    self._log(f"    [BL] ✗ {name} ({bl_reason})")
                    continue

                passed, struct_reason = _structural_check(name)
                if not passed:
                    self._add_log("STRUCTURAL", role, name, "", "dropped", struct_reason)
                    self._log(f"    [ST] ✗ {name} ({struct_reason})")
                    continue

                # ── KATMAN 2: NameDB ──
                namedb_verified = False
                corrected_name = name
                if self._name_db:
                    if self._name_db.is_name(name):
                        namedb_verified = True
                        self._add_log("NAMEDB", role, name, name, "kept",
                                       "exact_match", {"namedb_score": 1.0})
                        self._log(f"    [NDB] ✓ {name}")
                    else:
                        result_name, score, method = self._name_db.find_with_method(name)
                        if result_name and score >= 0.80:
                            namedb_verified = True
                            corrected_name = result_name
                            self._add_log("NAMEDB", role, name, corrected_name, "corrected",
                                           f"fuzzy_{method}", {"namedb_score": score})
                            self._log(f"    [NDB] ↻ {name} → {corrected_name} (skor:{score:.2f})")

                # ── KATMAN 3: TMDB Person Search ──
                tmdb_verified = False
                if self._tmdb_client:
                    search_name = corrected_name if namedb_verified else name
                    tmdb_result = _tmdb_person_verify(
                        search_name, role, self._tmdb_client, self._log)

                    if tmdb_result["verified"]:
                        tmdb_verified = True
                        self._add_log("TMDB_PERSON", role, search_name,
                                       tmdb_result.get("tmdb_name", search_name),
                                       "kept", "person_found", {
                                           "tmdb_name": tmdb_result.get("tmdb_name"),
                                           "tmdb_department": tmdb_result.get("tmdb_department"),
                                           "tmdb_id": tmdb_result.get("tmdb_id"),
                                       })
                        self._log(f"    [TMDB] ✓ {search_name} "
                                  f"[{tmdb_result.get('tmdb_department','')}]")

                        # TMDB ismi daha doğru olabilir
                        if tmdb_result.get("tmdb_name"):
                            corrected_name = tmdb_result["tmdb_name"]
                    else:
                        self._add_log("TMDB_PERSON", role, search_name, "",
                                       "not_found", tmdb_result.get("reason", ""))

                # ── KARAR (geçiş 1): verified → hemen ekle, unverified → beklet ──
                if namedb_verified or tmdb_verified:
                    if corrected_name not in verified_names:
                        verified_names.append(corrected_name)
                        self._add_log("FINAL", role, name, corrected_name,
                                       "kept", "verified")
                else:
                    unverified_candidates.append((name, corrected_name))

            # ── GEÇİŞ 2: unverified adayları değerlendir ──
            for orig_name, cand_name in unverified_candidates:
                if verified_names:
                    # Aynı rol için verified isim var → unverified'ı düşür
                    self._add_log("FINAL", role, orig_name, "",
                                   "dropped", "unverified_has_alternative")
                    self._log(f"    [DROP] ✗ {orig_name} — doğrulanamadı, alternatif var")
                else:
                    # Bu rol için hiç verified isim yok → Gemini pass2 dene
                    resolved = self._gemini_pass2(role, orig_name)
                    if resolved is not None:
                        if resolved not in verified_names:
                            verified_names.append(resolved)
                    else:
                        # Gemini de başarısız → unresolved olarak flag ile ekle
                        self._add_log("FINAL", role, orig_name, orig_name,
                                       "flagged", "unresolved",
                                       {"verified": False, "resolution": "unresolved"})
                        self._log(f"    [?] {orig_name} — çözümsüz (unresolved)")
                        if orig_name not in verified_names:
                            verified_names.append(orig_name)

            result[role] = verified_names

        return result

    # ─────────────────────────────────────────────────────
    # KATMAN 4 — Gemini Pass 2 (tam isim fuzzy top-2 + YES/NO)
    # ─────────────────────────────────────────────────────

    def _gemini_pass2(self, role: str, ocr_name: str) -> str | None:
        """Katman 4: doğrulanamamış aday için fuzzy top-2 + Gemini YES/NO.

        Akış:
          1. _fuzzy_find_top2() ile en fazla 2 aday al.
          2. Aday 1 _GEMINI_FUZZY_GATE (≥72) eşiğini geçmiyorsa unresolved.
          3. Aday 1 eşiği geçiyorsa Gemini YES/NO sor.
             - YES → aday 1'i döndür.
             - NO / hata → Aday 2 de eşiği geçiyorsa Gemini YES/NO sor.
               * YES → aday 2'yi döndür.
               * NO / hata → unresolved (None döndür).
          4. Kesinlikle 3. veya daha fazla adaya geçilmez.

        Gemini hataları fail-closed: timeout/network/parse/geçersiz yanıt
        her zaman reddetme sayılır, hiçbir zaman isim üretmez.

        Args:
            role: Doğrulama rolü (log için).
            ocr_name: Ham OCR ismi (raw değer korunur, log için).

        Returns:
            Onaylanan canonical isim (str), bulunamazsa None.
        """
        if self._name_db is None:
            self._log(f"    [GEMINI_P2] NameDB yok — {ocr_name!r} atlanıyor")
            return None

        try:
            from core.gemini_crew_validator import verify_single_name as _vsn
        except ImportError:
            self._log("[GEMINI_P2] gemini_crew_validator import edilemedi")
            return None

        # Skor eşiği: _GEMINI_FUZZY_GATE (0–100 arası) → _fuzzy_find_top2 0–100 bekleniyor
        gate_100 = _GEMINI_FUZZY_GATE  # _fuzzy_find_top2 threshold 0–100
        candidates = self._name_db._fuzzy_find_top2(ocr_name, threshold=gate_100)

        self._log(
            f"    [GEMINI_P2] {ocr_name!r} → top2 adaylar: {candidates}"
        )

        if not candidates:
            self._add_log("GEMINI_PASS2", role, ocr_name, ocr_name,
                           "flagged", "no_fuzzy_candidate_above_gate",
                           {"verified": False, "resolution": "unresolved"})
            return None

        def _try_gemini(candidate_name: str, cand_score: float, idx: int) -> bool:
            """Gemini'ye sor; True = YES, False = diğer her şey."""
            self._log(
                f"    [GEMINI_P2] Aday {idx}: {candidate_name!r} "
                f"(skor:{cand_score:.3f}) — Gemini'ye soruluyor"
            )
            verdict = _vsn(candidate_name)
            self._add_log(
                "GEMINI_PASS2", role, ocr_name, candidate_name,
                "kept" if verdict == "YES" else "dropped",
                f"gemini_{verdict.lower()}",
                {
                    "ocr_raw": ocr_name,
                    f"fuzzy_cand{idx}": candidate_name,
                    f"fuzzy_score{idx}": cand_score,
                    "gemini_verdict": verdict,
                    "verified": verdict == "YES",
                    "resolution": "gemini_verified" if verdict == "YES" else "unresolved",
                },
            )
            return verdict == "YES"

        # Aday 1
        cand1_name, cand1_score = candidates[0]
        if _try_gemini(cand1_name, cand1_score, 1):
            self._log(f"    [GEMINI_P2] ✓ {cand1_name!r} Gemini tarafından onaylandı")
            return cand1_name

        # Aday 2 (sadece gate'i geçiyorsa dene)
        # Not: _fuzzy_find_top2 zaten threshold=gate_100 (0–100 skala) ile
        # filtrelediğinden cand2_score (0–1 skala) her zaman >= gate/100 olur.
        # Bu kontrol, üstten gelen skalayla tutarlılığı açıkça belgeler.
        if len(candidates) >= 2:
            cand2_name, cand2_score = candidates[1]
            if cand2_score >= _GEMINI_FUZZY_GATE / 100.0:
                if _try_gemini(cand2_name, cand2_score, 2):
                    self._log(
                        f"    [GEMINI_P2] ✓ {cand2_name!r} Gemini tarafından onaylandı (aday 2)"
                    )
                    return cand2_name
            else:
                self._log(
                    f"    [GEMINI_P2] Aday 2 {cand2_name!r} gate'i geçemedi "
                    f"(skor:{cand2_score:.3f} < {_GEMINI_FUZZY_GATE / 100.0:.2f})"
                )

        # Her iki aday da başarısız
        self._add_log("GEMINI_PASS2", role, ocr_name, ocr_name,
                       "flagged", "unresolved",
                       {"verified": False, "resolution": "unresolved", "ocr_raw": ocr_name})
        return None

    # ─────────────────────────────────────────────────────
    # VERIFY CAST — Oyuncu listesi doğrulama
    # ─────────────────────────────────────────────────────

    def verify_cast(self, cast_list: list) -> list:
        """Oyuncu listesini katman katman doğrula.

        Args:
            cast_list: list of dict with "actor_name", "confidence", etc.

        Returns:
            Temizlenmiş cast listesi (aynı format).
        """
        self._log("\n  ── NAME VERIFY: OYUNCULAR ──")
        self._add_log("ONEOCR_RAW", "OYUNCU", "", "", "section_start",
                       "Oyuncu listesi ham veri")

        verified_cast = []

        for entry in cast_list:
            name = (entry.get("actor_name") or "").strip()
            if not name:
                continue

            # Log raw
            self._add_log("ONEOCR_RAW", "OYUNCU", name, name, "raw", "ocr_output")

            # ── KATMAN 1: Blacklist + Yapısal Kontrol ──
            is_bl, bl_reason = _blacklist_check(name)
            if is_bl:
                self._add_log("BLACKLIST", "OYUNCU", name, "", "dropped", bl_reason)
                continue

            passed, struct_reason = _structural_check(name)
            if not passed:
                self._add_log("STRUCTURAL", "OYUNCU", name, "", "dropped", struct_reason)
                continue

            # ── KATMAN 2: NameDB ──
            corrected_name = name
            is_verified = entry.get("is_verified_name", False)

            if self._name_db and not is_verified:
                if self._name_db.is_name(name):
                    is_verified = True
                    self._add_log("NAMEDB", "OYUNCU", name, name, "kept", "exact_match")
                else:
                    result_name, score, method = self._name_db.find_with_method(name)
                    if result_name and score >= 0.80:
                        is_verified = True
                        corrected_name = result_name
                        self._add_log("NAMEDB", "OYUNCU", name, corrected_name,
                                       "corrected", f"fuzzy_{method}",
                                       {"namedb_score": score})

            # ── KATMAN 3: TMDB Person Search (sadece doğrulanmayanlar için) ──
            tmdb_verified = False
            if not is_verified and self._tmdb_client:
                tmdb_result = _tmdb_person_verify(
                    corrected_name, "OYUNCU", self._tmdb_client, self._log)
                if tmdb_result["verified"]:
                    tmdb_verified = True
                    is_verified = True
                    if tmdb_result.get("tmdb_name"):
                        corrected_name = tmdb_result["tmdb_name"]
                    self._add_log("TMDB_PERSON", "OYUNCU", name,
                                   corrected_name, "kept", "person_found", {
                                       "tmdb_department": tmdb_result.get("tmdb_department"),
                                       "tmdb_id": tmdb_result.get("tmdb_id"),
                                   })

            # Sonucu güncelle
            new_entry = dict(entry)
            new_entry["actor_name"] = corrected_name
            new_entry["is_verified_name"] = is_verified
            if not is_verified:
                new_entry["needs_review"] = True
                self._add_log("FINAL", "OYUNCU", name, corrected_name,
                               "flagged", "unverified")
            else:
                self._add_log("FINAL", "OYUNCU", name, corrected_name,
                               "kept", "verified")

            verified_cast.append(new_entry)

        # İstatistik
        total = len(cast_list)
        kept = len(verified_cast)
        dropped = total - kept
        verified_count = sum(1 for c in verified_cast if c.get("is_verified_name"))
        flagged_count = sum(1 for c in verified_cast if c.get("needs_review"))

        self._log(f"    Toplam: {total} → Kalan: {kept} "
                  f"(doğrulanan: {verified_count}, flag: {flagged_count}, "
                  f"çıkarılan: {dropped})")

        return verified_cast

    # ─────────────────────────────────────────────────────
    # VERIFY AS SERIES — Dizi bazlı TMDB doğrulama
    # ─────────────────────────────────────────────────────

    def verify_as_series(self, title: str, director_names: list,
                         top_actors: list = None) -> dict | None:
        """Dizi adı + yönetmen (+ oyuncu) kombinasyonu ile TMDB'de doğrulama yapar.

        Strateji A: başlık + yönetmen + ≥1 oyuncu eşleşmesi (güçlü eşleşme)
        Strateji B: başlık + yönetmen (yönetmen bilgisi yoksa başlık yeterli)

        Args:
            title: Dizi adı (dosya adından çıkarılmış).
            director_names: Yönetmen adları listesi (OCR/credits_parse çıktısından).
            top_actors: En yüksek skorlu 1-2 oyuncu (Strateji A için).

        Returns:
            dict with {tmdb_entry, credits, matched_via, tmdb_id, tmdb_title, media_type}
            veya eşleşme bulunamazsa None.
        """
        if not self._tmdb_client or not title:
            self._log("  [NAME_VERIFY/Dizi] TMDB client yok veya başlık boş — atlanıyor")
            return None

        self._log(f"\n  ── NAME VERIFY: DİZİ MODU ──")
        self._log(
            f"  [NAME_VERIFY/Dizi] Aranan: '{title}' | "
            f"Yönetmenler: {director_names} | Oyuncular: {top_actors or []}"
        )

        try:
            results = self._tmdb_client.search_multi(title)
        except Exception as e:
            self._log(f"  [NAME_VERIFY/Dizi] TMDB arama hatası: {e}")
            return None

        self._log(f"  [NAME_VERIFY/Dizi] {len(results or [])} TMDB sonucu")

        for r in (results or [])[:10]:
            if r.get("media_type") != "tv":
                continue

            tmdb_title = r.get("name") or r.get("title") or "?"
            tmdb_id = r.get("id")

            try:
                credits = self._tmdb_client.get_tv_credits(int(tmdb_id))
            except Exception as e:
                self._log(f"  [NAME_VERIFY/Dizi] Credits çekme hatası (id:{tmdb_id}): {e}")
                continue

            if not credits:
                continue

            tmdb_cast_names = [
                item.get("name", "") for item in (credits.get("cast") or [])
                if item.get("name")
            ]
            tmdb_crew_names = [
                item.get("name", "") for item in (credits.get("crew") or [])
                if item.get("name")
            ]

            # ── Strateji A: yönetmen + en az 1 oyuncu (güçlü eşleşme) ──
            if director_names and top_actors:
                _dir_match_a = any(
                    _fuzzy_name_match(d, tmdb_crew_names) for d in director_names if d
                )
                _actor_matches = sum(
                    1 for a in top_actors if a and _fuzzy_name_match(a, tmdb_cast_names)
                )
                if _dir_match_a and _actor_matches >= 1:
                    self._log(
                        f"  [NAME_VERIFY/Dizi] '{tmdb_title}' (id:{tmdb_id}) — "
                        f"Strateji A: yönetmen ✓ | {_actor_matches}/{len(top_actors)} oyuncu ✓"
                    )
                    self._add_log(
                        "TMDB_SERIES", "DİZİ", title, tmdb_title, "kept",
                        "series_strat_a_director_cast",
                        {"tmdb_id": tmdb_id, "media_type": "tv",
                         "actor_matches": _actor_matches},
                    )
                    return {
                        "tmdb_entry": r, "credits": credits,
                        "matched_via": "series_strat_a_director_cast",
                        "tmdb_id": tmdb_id, "tmdb_title": tmdb_title, "media_type": "tv",
                    }

            # ── Strateji B: yönetmen eşleşmesi — hiç yönetmen bilgisi yoksa başlık yeterli ──
            if director_names:
                director_match = any(
                    _fuzzy_name_match(d, tmdb_crew_names) for d in director_names if d
                )
                # /tv/{id}/credits dizilerde yönetmeni nadiren döndürür (bölüm bazlı atama).
                # Eşleşme yoksa aggregate_credits ile bir kez daha dene.
                if not director_match:
                    try:
                        agg = self._tmdb_client.get_tv_aggregate_credits(int(tmdb_id))
                        # aggregate_credits crew yapısı: {name, department, jobs:[{job, episode_count}]}
                        # Regular credits formatına düzleştir: her jobs girişi ayrı crew kaydı olur.
                        agg_crew_flat = []
                        for item in (agg.get("crew") or []):
                            name = item.get("name", "")
                            department = item.get("department", "")
                            if not name:
                                continue
                            for job_entry in (item.get("jobs") or [{"job": department}]):
                                agg_crew_flat.append({
                                    "name": name,
                                    "job": job_entry.get("job", department),
                                    "department": department,
                                    "episode_count": job_entry.get("episode_count", 0),
                                })
                        agg_crew_names = [c["name"] for c in agg_crew_flat]
                        director_match = any(
                            _fuzzy_name_match(d, agg_crew_names) for d in director_names if d
                        )
                        if director_match:
                            # Mevcut crew listesine aggregate crew'u birleştir (isim tekrarını önle)
                            existing_names = {
                                item.get("name", "") for item in (credits.get("crew") or [])
                            }
                            for c in agg_crew_flat:
                                if c["name"] not in existing_names:
                                    credits.setdefault("crew", []).append(c)
                                    existing_names.add(c["name"])
                            tmdb_crew_names = agg_crew_names
                            self._log(
                                f"  [NAME_VERIFY/Dizi] '{tmdb_title}' aggregate_credits ile "
                                f"yönetmen eşleşti, {len(agg_crew_flat)} crew eklendi ✓"
                            )
                    except Exception as e:
                        self._log(f"  [NAME_VERIFY/Dizi] aggregate_credits hatası (id:{tmdb_id}): {e}")
                if not director_match:
                    self._log(
                        f"  [NAME_VERIFY/Dizi] '{tmdb_title}' (id:{tmdb_id}) — yönetmen eşleşmedi"
                    )
                    continue
                self._log(
                    f"  [NAME_VERIFY/Dizi] '{tmdb_title}' (id:{tmdb_id}) — yönetmen eşleşti ✓"
                )
            else:
                self._log(
                    f"  [NAME_VERIFY/Dizi] '{tmdb_title}' (id:{tmdb_id}) — yönetmen bilgisi yok, başlık eşleşmesi kabul edildi"
                )

            self._add_log(
                "TMDB_SERIES", "DİZİ", title, tmdb_title, "kept", "series_strat_b_director",
                {"tmdb_id": tmdb_id, "media_type": "tv"},
            )
            return {
                "tmdb_entry": r,
                "credits": credits,
                "matched_via": "series_strat_b_director",
                "tmdb_id": tmdb_id,
                "tmdb_title": tmdb_title,
                "media_type": "tv",
            }

        self._log(f"  [NAME_VERIFY/Dizi] Eşleşme bulunamadı")
        return None

    # ─────────────────────────────────────────────────────
    # VERIFY AS FILM — Film bazlı TMDB doğrulama
    # ─────────────────────────────────────────────────────

    def verify_as_film(self, title: str, director_names: list,
                       top_actors: list) -> dict | None:
        """Film adı + yönetmen + 1-2 oyuncu kombinasyonu ile TMDB'de doğrulama yapar.

        Tek tek isim aramak yerine film adıyla arama yapar, ardından yönetmen
        ve en az 1 oyuncu eşleşmesini kontrol eder.

        Args:
            title: Film adı (dosya adından çıkarılmış).
            director_names: Yönetmen adları listesi (OCR/credits_parse çıktısından).
            top_actors: En yüksek confidence'lı 1-2 oyuncu adı (OCR'dan).

        Returns:
            dict with {tmdb_entry, credits, matched_via, tmdb_id, tmdb_title, media_type}
            veya eşleşme bulunamazsa None.
        """
        if not self._tmdb_client or not title:
            self._log("  [NAME_VERIFY/Film] TMDB client yok veya başlık boş — atlanıyor")
            return None

        self._log(f"\n  ── NAME VERIFY: FİLM MODU ──")
        self._log(
            f"  [NAME_VERIFY/Film] Aranan: '{title}' | "
            f"Yönetmenler: {director_names} | Oyuncular: {top_actors}"
        )

        try:
            results = self._tmdb_client.search_multi(title)
        except Exception as e:
            self._log(f"  [NAME_VERIFY/Film] TMDB arama hatası: {e}")
            return None

        self._log(f"  [NAME_VERIFY/Film] {len(results or [])} TMDB sonucu")

        for r in (results or [])[:10]:
            if r.get("media_type") != "movie":
                continue

            tmdb_title = r.get("title") or r.get("name") or "?"
            tmdb_id = r.get("id")

            try:
                credits = self._tmdb_client.get_movie_credits(int(tmdb_id))
            except Exception as e:
                self._log(f"  [NAME_VERIFY/Film] Credits çekme hatası (id:{tmdb_id}): {e}")
                continue

            if not credits:
                continue

            tmdb_cast_names = [
                item.get("name", "") for item in (credits.get("cast") or [])
                if item.get("name")
            ]
            tmdb_crew_names = [
                item.get("name", "") for item in (credits.get("crew") or [])
                if item.get("name")
            ]

            # Yönetmen eşleşmesi
            director_match = True
            if director_names:
                director_match = any(
                    _fuzzy_name_match(d, tmdb_crew_names) for d in director_names if d
                )

            # Oyuncu eşleşmesi — top_actors'dan en az 1 eşleşmeli
            actor_matches = sum(
                1 for a in top_actors if a and _fuzzy_name_match(a, tmdb_cast_names)
            )

            if director_match and actor_matches >= 1:
                self._log(
                    f"  [NAME_VERIFY/Film] '{tmdb_title}' (id:{tmdb_id}) — "
                    f"yönetmen ✓ | {actor_matches}/{len(top_actors)} oyuncu eşleşti ✓"
                )
                self._add_log(
                    "TMDB_FILM", "FİLM", title, tmdb_title, "kept",
                    "film_title_director_cast",
                    {"tmdb_id": tmdb_id, "media_type": "movie",
                     "actor_matches": actor_matches},
                )
                return {
                    "tmdb_entry": r,
                    "credits": credits,
                    "matched_via": "film_title_director_cast",
                    "tmdb_id": tmdb_id,
                    "tmdb_title": tmdb_title,
                    "media_type": "movie",
                }

            self._log(
                f"  [NAME_VERIFY/Film] '{tmdb_title}' (id:{tmdb_id}) — "
                f"yönetmen={'✓' if director_match else '✗'} | "
                f"{actor_matches}/{len(top_actors)} oyuncu eşleşti"
            )

        self._log(f"  [NAME_VERIFY/Film] Eşleşme bulunamadı")
        return None
