"""
gemini_summarizer.py — Transcript'ten Türkçe özet çıkarma.

Model Stratejisi: Gemini 2.5 Pro (birincil) → Flash (fallback)
Çok dilli destek: Transcript dil bilgisi prompt'a eklenir, çıktı her zaman Türkçe.

summarize_transcript(transcript_text, api_key, log_cb, detected_language) -> dict | None
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
}


def get_language_label(lang_code: str) -> str:
    """ISO dil kodundan kısa Türkçe etiket döndür."""
    if not lang_code:
        return "VERİ YOK"
    return LANG_LABELS.get(lang_code.lower(), lang_code.upper())


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Çok dilli destek
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_EN = (
    "You are an analytical summarizer working for a video archive database. "
    "You will be given a transcript (ASR) of a film or video.\n"
    "The transcript may be in any language. Analyze the content in its original "
    "language, then write the summary in Turkish.\n\n"
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
    "Example: Write 'Vichita' not 'Viçita', 'Tom' not 'Töm'."
)

# ─────────────────────────────────────────────────────────────────────────────
# Few-shot örnek
# ─────────────────────────────────────────────────────────────────────────────
_FEW_SHOT = (
    "Aşağıdaki örnek DOĞRU formattır — tam olarak böyle yaz:\n\n"
    "ÖRNEK:\n"
    "Başarılı bir reklamcı olan David, önemli bir müşteriyle duygusal bir ilişki "
    "içindeyken, babasının şeker hastalığı yüzünden bacağının kesilmesi gerektiğini "
    "öğrenir. Kariyeri için hayati önem taşıyan bir sunum ile babasının ameliyatı "
    "aynı güne denk gelir. İşi ve sevgilisi yerine babasını tercih eden David, "
    "her şeyi elinin tersiyle iterek hastanede onun yanında kalır. Annesi de "
    "hastabakıcılık yapmayı reddedince, ameliyattan sakat çıkan babasının tüm "
    "sorumluluğu David'e kalır. Film, yıllarca birbirinden uzak kalan baba-oğulun "
    "geçmişi geride bırakıp birlikte yeni bir hayata başlamasıyla biter.\n\n"
    "Şimdi aşağıdaki transcript için aynı formatta özet yaz:\n"
)

# Türkçe'ye özgü karakterler (kontrol için)
_TR_CHARS = set("çÇğĞıİöÖşŞüÜ")


def _is_turkish_text(text: str) -> bool:
    """Metnin Türkçe olup olmadığını kontrol et."""
    if not text:
        return False
    return any(c in _TR_CHARS for c in text)


def _try_summarize(prompt: str, model: str, api_key: str, log_cb) -> str | None:
    """Tek model ile özet denemesi yap."""
    try:
        result = _llm._gemini_generate(
            prompt,
            system=_SYSTEM_PROMPT_EN,
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


def _translate_to_turkish(text: str, api_key: str, log_cb) -> str:
    """Flash ile Türkçeye çevir (edge case: özet yabancı dilde geldiğinde)."""
    try:
        result = _llm._gemini_generate(
            f"Translate the following text to Turkish. Output only the Turkish translation, nothing else.\n\n{text}",
            system="You are a professional translator. Translate accurately to Turkish.",
            api_key=api_key or None,
            model="gemini-2.5-flash",
            timeout=60,
            log_cb=log_cb,
        )
        return result.strip() if result else text
    except Exception:
        return text


def summarize_transcript(
    transcript_text: str,
    api_key: str = "",
    model: str = "",
    log_cb=None,
    variant: str = "en",
    detected_language: str = "tr",
) -> dict | None:
    """Transcript metninden Gemini ile Türkçe özet üret.

    Model stratejisi: Pro → Flash fallback.

    Returns:
        {"en": "<özet>", "model_used": "gemini-2.5-pro"|"gemini-2.5-flash"}
        Hata durumunda None.
    """
    if not transcript_text or not transcript_text.strip():
        return None

    snippet = transcript_text.strip()[:_MAX_CHARS]

    # Dil bilgisini prompt'a ekle
    lang_label = get_language_label(detected_language)
    lang_info = f"\nTranscript language: {detected_language.upper()} ({lang_label})\n" if detected_language != "tr" else ""

    prompt = _FEW_SHOT + lang_info + f"Transcript:\n{snippet}"

    results = {}

    if variant in ("en", "both"):
        summary_text = None
        model_used = ""

        # Strateji: belirli model verilmişse onu kullan, yoksa Pro→Flash
        models_to_try = [model] if model else ["gemini-2.5-pro", "gemini-2.5-flash"]

        for m in models_to_try:
            if log_cb:
                log_cb(f"  [Summarizer] EN varyant özeti oluşturuluyor ({m})...")
            summary_text = _try_summarize(prompt, m, api_key, log_cb)
            if summary_text:
                model_used = m
                if log_cb:
                    log_cb(f"  [Summarizer] EN özet alındı ({len(summary_text)} karakter, {m})")
                break
            else:
                if log_cb:
                    log_cb(f"  [Summarizer] {m} başarısız — sonraki modele geçiliyor")

        if not summary_text:
            if log_cb:
                log_cb("  [Summarizer] Hiçbir modelden özet alınamadı")
            return None

        # Türkçe kontrolü — yabancı dilde gelirse Flash ile çevir
        if not _is_turkish_text(summary_text):
            if log_cb:
                log_cb("  [Summarizer] Özet Türkçe değil — Flash ile çeviri yapılıyor")
            summary_text = _translate_to_turkish(summary_text, api_key, log_cb)

        results["en"] = summary_text
        results["model_used"] = model_used

    return results if results else None
