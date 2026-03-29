"""
gemini_summarizer.py — Transcript'ten Türkçe özet çıkarma.

Model Stratejisi: Gemini 2.5 Pro (birincil) → Flash (fallback)

Dil Stratejisi (v2):
  • Türkçe transcript → Pro ile doğrudan Türkçe özet (tek adım, mevcut davranış)
  • Yabancı transcript → Pro ile orijinal dilde özet → Flash ile Türkçeye çevir
    + TMDB cast varsa isimleri doğrula + Türkçe karakter kuralları uygula

summarize_transcript(transcript_text, api_key, log_cb, detected_language, tmdb_cast) -> dict | None
  → {"en": "...", "model_used": "gemini-2.5-pro"|"gemini-2.5-flash"}
"""

import core.llm_provider as _llm

_TIMEOUT_SEC = 90
_MAX_CHARS = 120000

# ─────────────────────────────────────────────────────────────────────────────
# DİL ETİKETLERİ
# ─────────────────────────────────────────────────────────────────────────────
LANG_LABELS = {
    "tr": "TR", "en": "İNG", "de": "ALM", "fr": "FRA",
    "es": "İSP", "it": "İTA", "ru": "RUS", "ar": "ARA",
    "ja": "JAP", "ko": "KOR", "zh": "ÇİN", "pt": "POR",
    "nl": "HOL", "pl": "POL", "sv": "İSV", "da": "DAN",
    "fi": "FİN", "no": "NOR", "el": "YUN", "hu": "MAC",
    "cs": "ÇEK", "ro": "ROM", "hi": "HİN",
    "ku": "KÜR", "fa": "FAR",
}


def get_language_label(lang_code: str) -> str:
    """ISO dil kodundan kısa Türkçe etiket döndür."""
    if not lang_code:
        return "VERİ YOK"
    return LANG_LABELS.get(lang_code.lower(), lang_code.upper())


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Türkçe özet (Türkçe transcript için)
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_TR = (
    "You are an analytical summarizer working for a video archive database. "
    "You will be given a transcript (ASR) of a film or video.\n"
    "The transcript is in Turkish. Analyze the content and write the summary in Turkish.\n\n"
    "Your task: analyze the story in the transcript and produce a concise, "
    "direct summary in narrative prose format.\n\n"
    "STRICT RULES — THESE MUST BE FOLLOWED WITHOUT EXCEPTION:\n\n"
    "LENGTH: Target 80 words, maximum 100 words.\n\n"
    "ZERO SUSPENSE: Attempting to make the reader curious about the film is STRICTLY FORBIDDEN. "
    "Do NOT use phrases like 'faces life's challenges', 'a big decision awaits', "
    "'what will happen next?' or any marketing/teaser language.\n\n"
    "SPOILER (ENDING) IS MANDATORY: You MUST include the film's final outcome and "
    "each main character's ultimate fate as explicitly as possible. "
    "(e.g., 'X quits her job, Y dies, Z leaves the city.')\n\n"
    "FORMAT: No bullet points. Write a single, flowing paragraph.\n\n"
    "FOCUS: Focus only on the main character's key turning point and their final decision. "
    "Completely ignore subplots and secondary characters.\n\n"
    "NO CHARACTER INTRODUCTIONS: Do NOT introduce or describe characters by name/age/job "
    "at the start. Just tell the story — who they are will emerge from the events.\n\n"
    "Output language: Turkish. No title. Start directly with the story.\n\n"
    "FOREIGN NAMES: Write all foreign proper names (character names, place names, "
    "person names) using their original Latin spelling. NEVER use Turkish-specific "
    "characters (ç, ğ, ı, ö, ş, ü, İ) in foreign names. "
    "Example: Write 'Vichita' not 'Viçita', 'Tom' not 'Töm'.\n\n"
    "FOREIGN NAME TAGGING: Wrap EVERY foreign proper name with double square brackets. "
    "Turkish words and common Turkish names (Mehmet, Ayşe, Fatma, Ali, Hasan, etc.) "
    "must NOT be wrapped.\n"
    "  Multi-word names: wrap the full name together: [[Jack Sparrow]], [[New York]].\n"
    "  Single-word names: [[Amy]], [[Charles]], [[Oliver]].\n"
    "  CORRECT: [[Amy]] şehre geldi. Kadın [[Charles]]'a aşık olur.\n"
    "  WRONG: Amy şehre geldi.  (brackets missing)\n"
    "  WRONG: [[Mehmet]] eve döndü.  (Turkish name, no brackets)\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Orijinal dilde özet (yabancı dil transcript için, adım 1)
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_FOREIGN = (
    "You are an analytical summarizer working for a video archive database. "
    "You will be given a transcript (ASR) of a film or video.\n\n"
    "IMPORTANT: Write the summary in the SAME LANGUAGE as the transcript. "
    "Do NOT translate to any other language.\n\n"
    "Your task: analyze the story in the transcript and produce a concise, "
    "direct summary in narrative prose format.\n\n"
    "STRICT RULES — THESE MUST BE FOLLOWED WITHOUT EXCEPTION:\n\n"
    "LENGTH: Target 80 words, maximum 100 words.\n\n"
    "ZERO SUSPENSE: Do NOT use teaser/marketing language.\n\n"
    "SPOILER (ENDING) IS MANDATORY: Include the final outcome.\n\n"
    "FORMAT: No bullet points. Write a single, flowing paragraph.\n\n"
    "FOCUS: Main character's key turning point and final decision only.\n\n"
    "NO CHARACTER INTRODUCTIONS: Just tell the story.\n\n"
    "No title. Start directly with the story."
)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Çeviri + isim doğrulama (yabancı dil, adım 2)
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_TRANSLATE = (
    "You are a professional translator and name verification specialist "
    "for a Turkish video archive database.\n\n"
    "Your task: Translate the given summary to Turkish.\n\n"
    "RULES:\n"
    "1. Translate accurately and naturally to Turkish.\n"
    "2. FOREIGN NAMES: Keep all foreign proper names (character names, person names, "
    "   place names) in their ORIGINAL spelling. Do NOT turkify them.\n"
    "   Example: 'Jack Sparrow' stays 'Jack Sparrow', NOT 'Cek Sparov'.\n"
    "3. TURKISH NAME RULES: In Turkish text, 'i' uppercases to 'İ' and 'I' lowercases "
    "   to 'ı'. But this rule applies ONLY to Turkish words, NOT to foreign names.\n"
    "4. FOREIGN NAME TAGGING: Wrap EVERY foreign proper name with double square brackets. "
    "   Turkish words and Turkish names must NOT be wrapped.\n"
    "   Example: [[Jack Sparrow]] gemiden kaçtı. [[Elizabeth]] onu takip etti.\n"
    "5. Output ONLY the Turkish translation. No explanations, no notes.\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# Few-shot örnek (Türkçe özet için)
# ─────────────────────────────────────────────────────────────────────────────
_FEW_SHOT = (
    "Aşağıdaki örnek DOĞRU formattır — tam olarak böyle yaz:\n\n"
    "ÖRNEK:\n"
    "Başarılı bir reklamcı olan [[David]], önemli bir müşteriyle duygusal bir ilişki "
    "içindeyken, babasının şeker hastalığı yüzünden bacağının kesilmesi gerektiğini "
    "öğrenir. Kariyeri için hayati önem taşıyan bir sunum ile babasının ameliyatı "
    "aynı güne denk gelir. İşi ve sevgilisi yerine babasını tercih eden [[David]], "
    "her şeyi elinin tersiyle iterek hastanede onun yanında kalır. Annesi de "
    "hastabakıcılık yapmayı reddedince, ameliyattan sakat çıkan babasının tüm "
    "sorumluluğu [[David]]'e kalır. Film, yıllarca birbirinden uzak kalan baba-oğulun "
    "geçmişi geride bırakıp birlikte yeni bir hayata başlamasıyla biter.\n\n"
    "Şimdi aşağıdaki transcript için aynı formatta özet yaz:\n"
)

# Türkçe'ye özgü karakterler (kontrol için)
_TR_CHARS = set("çÇğĞıİöÖşŞüÜ")


def _is_turkish_text(text: str) -> bool:
    """Metnin Türkçe olup olmadığını kontrol et.

    Arapça/Kiril gibi Latin-dışı metinler (ör. yanlış dil tespiti sonucu gelen özetler)
    Latin karakter oranı kontrolüyle reddedilir.
    """
    if not text or len(text) < 20:
        return False
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count == 0:
        return False
    latin_count = sum(1 for c in text if c.isalpha() and ord(c) < 0x0250)
    if latin_count / alpha_count < 0.7:
        return False  # Latin oranı %70 altında → Türkçe değil (Arapça, Kiril vb.)
    return any(c in _TR_CHARS for c in text)


def _try_summarize(prompt: str, system: str, model: str,
                   api_key: str, log_cb) -> str | None:
    """Tek model ile özet denemesi yap."""
    try:
        result = _llm._gemini_generate(
            prompt,
            system=system,
            api_key=api_key or None,
            model=model,
            timeout=_TIMEOUT_SEC,
            log_cb=log_cb,
        )
        return result if result and result.strip() else None
    except Exception as e:
        if log_cb:
            log_cb(f"  [Summarizer] {model} hatası: {e}")
        return None


def _build_cast_reference(tmdb_cast: list) -> str:
    """TMDB cast listesinden çeviri prompt'u için referans metni oluştur."""
    if not tmdb_cast:
        return ""

    lines = []
    for entry in tmdb_cast[:20]:  # max 20 kişi
        actor = ""
        char = ""
        if isinstance(entry, dict):
            actor = (entry.get("actor_name") or entry.get("name") or "").strip()
            char = (entry.get("character_name") or entry.get("character") or "").strip()
        if actor and char:
            lines.append(f"  {actor} → {char}")
        elif actor:
            lines.append(f"  {actor}")

    if not lines:
        return ""

    return (
        "\n\nIMPORTANT — CAST REFERENCE LIST (use these exact spellings):\n"
        + "\n".join(lines)
        + "\n\nIf the summary mentions any character that matches this list, "
        "use the EXACT spelling from this list. "
        "For example, if transcript says 'Cek Sparov' but the list says "
        "'Jack Sparrow', use 'Jack Sparrow'.\n"
    )


def _translate_and_verify(
    foreign_summary: str,
    api_key: str,
    tmdb_cast: list | None,
    detected_language: str,
    log_cb,
) -> str:
    """
    Flash ile yabancı dildeki özeti Türkçeye çevir.
    TMDB cast varsa isimleri doğrula.
    """
    cast_ref = _build_cast_reference(tmdb_cast or [])
    lang_label = get_language_label(detected_language)

    prompt = (
        f"Source language: {detected_language.upper()} ({lang_label})\n\n"
        f"Summary to translate:\n{foreign_summary}"
        f"{cast_ref}"
    )

    try:
        result = _llm._gemini_generate(
            prompt,
            system=_SYSTEM_PROMPT_TRANSLATE,
            api_key=api_key or None,
            model="gemini-2.5-flash",
            timeout=60,
            log_cb=log_cb,
        )
        if result and result.strip():
            if log_cb:
                has_cast = "evet" if cast_ref else "hayır"
                log_cb(
                    f"  [Summarizer] Flash çeviri tamamlandı "
                    f"({len(result)} kar, cast_ref={has_cast})"
                )
            return result.strip()
    except Exception as e:
        if log_cb:
            log_cb(f"  [Summarizer] Flash çeviri hatası: {e}")

    # Fallback: çeviri başarısızsa orijinali döndür
    return foreign_summary


def summarize_transcript(
    transcript_text: str,
    api_key: str = "",
    model: str = "",
    log_cb=None,
    variant: str = "en",
    detected_language: str = "tr",
    tmdb_cast: list | None = None,
) -> dict | None:
    """Transcript metninden Gemini ile Türkçe özet üret.

    Dil Stratejisi (v2):
      • Türkçe → Pro ile doğrudan Türkçe özet (tek adım)
      • Yabancı → Pro ile orijinal dilde özet → Flash ile Türkçeye çevir + isim doğrula

    Args:
        transcript_text:  ASR transcript metni
        api_key:          Gemini API key
        model:            Belirli model (boşsa Pro→Flash fallback)
        log_cb:           Log callback
        variant:          "en" (varsayılan)
        detected_language: Tespit edilen dil kodu ("tr", "en", "ar", ...)
        tmdb_cast:        TMDB cast listesi (opsiyonel, yabancı dil çevirisinde kullanılır)

    Returns:
        {"en": "<Türkçe özet>", "model_used": "gemini-2.5-pro"|"gemini-2.5-flash"}
        Hata durumunda None.
    """
    if not transcript_text or not transcript_text.strip():
        return None

    snippet = transcript_text.strip()[:_MAX_CHARS]
    is_turkish = (detected_language == "tr" or not detected_language)

    results = {}

    if variant in ("en", "both"):
        summary_text = None
        model_used = ""

        if is_turkish:
            # ══════════════════════════════════════════════════════
            # TÜRKÇE AKIŞ — tek adım, mevcut davranış
            # ══════════════════════════════════════════════════════
            if log_cb:
                log_cb("  [Summarizer] Türkçe transcript → tek adım özet")

            system = _SYSTEM_PROMPT_TR
            prompt = _FEW_SHOT + f"Transcript:\n{snippet}"

            models_to_try = [model] if model else ["gemini-2.5-pro", "gemini-2.5-flash"]
            for m in models_to_try:
                if log_cb:
                    log_cb(f"  [Summarizer] Türkçe özet oluşturuluyor ({m})...")
                summary_text = _try_summarize(prompt, system, m, api_key, log_cb)
                if summary_text:
                    model_used = m
                    if log_cb:
                        log_cb(f"  [Summarizer] Türkçe özet alındı ({len(summary_text)} kar, {m})")
                    break
                else:
                    if log_cb:
                        log_cb(f"  [Summarizer] {m} başarısız — sonraki modele geçiliyor")

            # Güvenlik: Türkçe demiştik ama özet Türkçe gelmezse Flash ile çevir
            if summary_text and not _is_turkish_text(summary_text):
                if log_cb:
                    log_cb("  [Summarizer] Özet Türkçe değil — Flash ile çeviri yapılıyor")
                summary_text = _translate_and_verify(
                    summary_text, api_key, tmdb_cast, detected_language, log_cb
                )

        else:
            # ══════════════════════════════════════════════════════
            # YABANCI DİL AKIŞI — iki adım
            # Adım 1: Orijinal dilde özet (Pro)
            # Adım 2: Türkçeye çevir + isim doğrula (Flash)
            # ══════════════════════════════════════════════════════
            lang_label = get_language_label(detected_language)
            if log_cb:
                log_cb(
                    f"  [Summarizer] Yabancı dil ({detected_language.upper()}/{lang_label}) "
                    f"→ iki adımlı akış"
                )

            # ── Adım 1: Orijinal dilde özet ──
            system = _SYSTEM_PROMPT_FOREIGN
            lang_info = f"\nTranscript language: {detected_language.upper()} ({lang_label})\n"
            prompt = lang_info + f"Transcript:\n{snippet}"

            models_to_try = [model] if model else ["gemini-2.5-pro", "gemini-2.5-flash"]
            foreign_summary = None

            for m in models_to_try:
                if log_cb:
                    log_cb(
                        f"  [Summarizer] Adım 1: {detected_language.upper()} özet "
                        f"oluşturuluyor ({m})..."
                    )
                foreign_summary = _try_summarize(prompt, system, m, api_key, log_cb)
                if foreign_summary:
                    model_used = m
                    if log_cb:
                        log_cb(
                            f"  [Summarizer] Adım 1 tamamlandı: {detected_language.upper()} "
                            f"özet ({len(foreign_summary)} kar, {m})"
                        )
                    break
                else:
                    if log_cb:
                        log_cb(f"  [Summarizer] {m} başarısız — sonraki modele geçiliyor")

            if not foreign_summary:
                if log_cb:
                    log_cb("  [Summarizer] Adım 1 başarısız — hiçbir modelden özet alınamadı")
                return None

            # ── Adım 2: Türkçeye çevir + isim doğrula ──
            if log_cb:
                cast_count = len(tmdb_cast) if tmdb_cast else 0
                log_cb(
                    f"  [Summarizer] Adım 2: Flash ile Türkçeye çeviri "
                    f"(cast_ref={cast_count} kişi)..."
                )

            summary_text = _translate_and_verify(
                foreign_summary, api_key, tmdb_cast, detected_language, log_cb
            )

            if log_cb:
                log_cb(f"  [Summarizer] Adım 2 tamamlandı: Türkçe özet ({len(summary_text)} kar)")

        # Sonuç
        if not summary_text:
            if log_cb:
                log_cb("  [Summarizer] Hiçbir modelden özet alınamadı")
            return None

        results["en"] = summary_text
        results["model_used"] = model_used

    return results if results else None
