"""
name_verify.py — Katmanlı isim doğrulama motoru + verification log.

Export engine'e veri girmeden önce çalışır.
7 yıldızlı YAPIM EKİBİ ve OYUNCU listesini katman katman temizler/doğrular.

Katmanlar (ucuzdan pahalıya):
  1. Hard Blacklist + Yapısal Kontrol  (maliyet: 0)
  2. NameDB doğrulaması               (maliyet: 0, lokal)
  3. TMDB Person Search               (maliyet: düşük, HTTP)
  4. Gemini API                        (maliyet: yüksek, son çare)
  5. Bulunamadı → flag ile rapora yaz  (maliyet: 0)

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
                      and ord(c) < 0x0600)  # Arap alfabesi öncesi
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

    # Noktalama yoğunluğu
    punct_count = sum(1 for c in t if c in "!?;…[]{}()")
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


# ═══════════════════════════════════════════════════════════════════
# KATMAN 3 — TMDB Person Search (kişi merkezli doğrulama)
# ═══════════════════════════════════════════════════════════════════

# TMDB department → VİTOS rol eşleştirmesi
_TMDB_DEPT_TO_ROLE = {
    "directing": "YÖNETMEN",
    "writing": "SENARYO",
    "camera": "GÖRÜNTÜ YÖNETMENİ",
    "editing": "KURGU",
    "production": "YAPIMCI",
    "acting": "OYUNCU",
    "sound": None,      # 7 yıldızlı rolde yok
    "art": None,
    "costume & make-up": None,
    "visual effects": None,
    "lighting": None,
    "crew": None,
}


def _tmdb_person_verify(name: str, expected_role: str, tmdb_client, log_cb=None) -> dict:
    """TMDB person search ile isim doğrulama.

    Film aramıyor, kişi arıyor. Nisa Serezli gibi TMDB'de filmi olmayan
    ama kişi olarak kayıtlı olan isimler doğrulanır.

    Args:
        name: Doğrulanacak isim
        expected_role: Beklenen rol (YÖNETMEN, OYUNCU, vs.)
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

    # Meslek eşleşmesi bonus bilgi
    expected_dept = None
    for dept, role in _TMDB_DEPT_TO_ROLE.items():
        if role == expected_role:
            expected_dept = dept
            break

    role_match = (expected_dept == tmdb_dept) if expected_dept else None

    return {
        "verified": True,
        "tmdb_name": tmdb_name,
        "tmdb_department": tmdb_dept,
        "tmdb_id": tmdb_id,
        "role_match": role_match,
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
                                           "role_match": tmdb_result.get("role_match"),
                                       })
                        self._log(f"    [TMDB] ✓ {search_name} "
                                  f"[{tmdb_result.get('tmdb_department','')}]")

                        # TMDB ismi daha doğru olabilir
                        if tmdb_result.get("tmdb_name"):
                            corrected_name = tmdb_result["tmdb_name"]
                    else:
                        self._add_log("TMDB_PERSON", role, search_name, "",
                                       "not_found", tmdb_result.get("reason", ""))

                # ── KARAR ──
                if namedb_verified or tmdb_verified:
                    if corrected_name not in verified_names:
                        verified_names.append(corrected_name)
                        self._add_log("FINAL", role, name, corrected_name,
                                       "kept", "verified")
                else:
                    # Ne NameDB ne TMDB doğruladı — flag ile geç
                    self._add_log("FINAL", role, name, name,
                                   "flagged", "unverified")
                    self._log(f"    [?] {name} — doğrulanamadı")
                    # Yine de ekle — Gemini ayıkladıysa güvenilir
                    if name not in verified_names:
                        verified_names.append(name)

            result[role] = verified_names

        return result

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
