"""
gemini_summarizer.py — Transcript'ten Türkçe özet çıkarma (Gemini 2.5 Flash).

summarize_transcript(transcript_text, api_key, log_cb) -> str | None
  - Transcript metnini Gemini'ye gönderir
  - 8-12 cümlelik Türkçe özet döndürür
"""

import core.llm_provider as _llm

_TIMEOUT_SEC = 90
_MAX_CHARS = 120000

_SYSTEM_PROMPT = (
    "Sen profesyonel bir içerik çözümleyicisisin. Sana bir filmin veya dizi bölümünün "
    "otomatik oluşturulmuş transcripti verilecektir.\n\n"
    "GÖREVİN:\n"
    "Bu transcriptten içeriğin SPOILER DAHİL ÖZETİNİ yaz.\n\n"
    "KURALLAR:\n"
    "- Önce ana karakterleri kısaca tanıt (örn: AHMETİN OĞLU MEHMET, MAHALLENİN KASABI AYŞE)\n"
    "- Sonra olayları GİRİŞ-GELİŞME-SONUÇ örgüsüyle TOPLAM 2 CÜMLEYİ GEÇMEYECEK şekilde anlat\n"
    "- SPOILER VER — Bu özet birinin filmi izleyip merak etmesi için DEĞİL, "
    "okuyanın içerik hakkında NET bir fikir sahibi olması için\n"
    "- Örnek: 'Ahmetin babası bu bölümde öldü' gibi direkt bilgi ver\n"
    "- Kibar veya merak uyandırıcı ifadeler KULLANMA (örn: 'büyük zorluklar beklemektedir' KULLANMA)\n"
    "- Yalnızca transcriptteki bilgileri kullan, uydurma YAPMA\n"
    "- Türkçe yaz"
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
        8-12 cümlelik Türkçe özet string'i veya hata durumunda None.
    """
    if not transcript_text or not transcript_text.strip():
        return None

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
