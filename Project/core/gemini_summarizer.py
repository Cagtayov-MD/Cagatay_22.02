"""
gemini_summarizer.py — Transcript'ten Türkçe özet çıkarma (Gemini 2.5 Flash).

summarize_transcript(transcript_text, api_key, log_cb) -> str | None
  - Transcript metninin ilk ~4000 karakterini Gemini'ye gönderir
  - 5-8 cümlelik Türkçe özet döndürür
"""

import core.llm_provider as _llm

_TIMEOUT_SEC = 30
_MAX_CHARS = 4000

_SYSTEM_PROMPT = (
    "Sen bir film/dizi analistisın. Sana verilen transcript'ten 5-8 cümlelik Türkçe özet çıkar. "
    "Önemli kişileri, olayları ve konuları vurgula. Sadece özeti yaz."
)


def summarize_transcript(
    transcript_text: str,
    api_key: str = "",
    model: str = "gemini-2.5-flash",
    log_cb=None,
) -> str | None:
    """Transcript metninden Gemini 2.5 Flash ile Türkçe özet üret.

    Args:
        transcript_text: Ham transcript metni.
        api_key: Gemini API anahtarı. Boşsa env'den okunur.
        model: Kullanılacak Gemini model adı.
        log_cb: İsteğe bağlı log callback fonksiyonu.

    Returns:
        5-8 cümlelik Türkçe özet string'i veya hata durumunda None.
    """
    if not transcript_text or not transcript_text.strip():
        return None

    # İlk ~4000 karakteri al
    snippet = transcript_text.strip()[:_MAX_CHARS]
    prompt = f"Transcript:\n{snippet}"

    if log_cb:
        log_cb("  [Summarizer] Transcript özeti oluşturuluyor...")

    try:
        result = _llm._gemini_generate(
            prompt,
            system=_SYSTEM_PROMPT,
            api_key=api_key or None,
            model=model,
            timeout=_TIMEOUT_SEC,
            log_cb=log_cb,
        )
    except Exception as e:
        if log_cb:
            log_cb(f"  [Summarizer] API hatası: {e}")
        return None

    if result:
        if log_cb:
            log_cb(f"  [Summarizer] Özet alındı ({len(result)} karakter)")
    else:
        if log_cb:
            log_cb("  [Summarizer] Boş yanıt — özet oluşturulamadı")

    return result or None
