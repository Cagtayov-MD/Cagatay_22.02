"""
gemini_summarizer.py — Transcript'ten Türkçe özet çıkarma (Gemini 2.5 Pro).

summarize_transcript(transcript_text, api_key, log_cb) -> dict | None
  - Transcript metnini Gemini'ye gönderir
  - İngilizce prompt ile Türkçe özet üretir: {"en": ".."}

VARYANT SEÇİMİ:
  - summarize_transcript(..., variant="en") -> EN prompt (varsayılan)
"""

import core.llm_provider as _llm

_TIMEOUT_SEC = 90
_MAX_CHARS = 120000

# ─────────────────────────────────────────────────────────────────────────────
# VARYANT EN — İngilizce kurallar, Türkçe çıktı
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_EN = (
    "You are an analytical summarizer working for a video archive database. "
    "You will be given a transcript (ASR) of a film or video.\n"
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
    "Output language: Turkish. No title. Start directly with the story."
)

# ─────────────────────────────────────────────────────────────────────────────
# Few-shot örnek (her iki varyanta eklenir)
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


def summarize_transcript(
    transcript_text: str,
    api_key: str = "",
    model: str = "gemini-2.5-pro",
    log_cb=None,
    variant: str = "en",
) -> dict | None:
    """Transcript metninden Gemini 2.5 Pro ile Türkçe özet üret.

    Args:
        transcript_text: Ham transcript metni.
        api_key: Gemini API anahtarı. Boşsa env'den okunur.
        model: Kullanılacak Gemini model adı.
        log_cb: İsteğe bağlı log callback fonksiyonu.
        variant: "en" (varsayılan) — İngilizce prompt ile Türkçe özet üretir.

    Returns:
        variant="en"   -> {"en": "<özet>"}
        Hata durumunda None.
    """
    if not transcript_text or not transcript_text.strip():
        return None

    snippet = transcript_text.strip()[:_MAX_CHARS]
    prompt = _FEW_SHOT + f"Transcript:\n{snippet}"

    results = {}

    # ── Varyant EN ──────────────────────────────────────────────────────────
    if variant in ("en", "both"):
        if log_cb:
            log_cb("  [Summarizer] EN varyant özeti oluşturuluyor...")
        try:
            en_result = _llm._gemini_generate(
                prompt,
                system=_SYSTEM_PROMPT_EN,
                api_key=api_key or None,
                model=model,
                timeout=_TIMEOUT_SEC,
                log_cb=log_cb,
            )
            if en_result:
                results["en"] = en_result
                if log_cb:
                    log_cb(f"  [Summarizer] EN özet alındı ({len(en_result)} karakter)")
            else:
                if log_cb:
                    log_cb("  [Summarizer] EN varyant boş yanıt döndü")
        except Exception as e:
            if log_cb:
                log_cb(f"  [Summarizer] EN varyant API hatası: {e}")

    return results if results else None
