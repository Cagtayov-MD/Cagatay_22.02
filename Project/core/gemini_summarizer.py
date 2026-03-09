"""
gemini_summarizer.py — Transcript'ten Türkçe özet çıkarma (Gemini 2.5 Flash).

summarize_transcript(transcript_text, api_key, log_cb) -> str | None
  - Transcript metnini Gemini'ye gönderir
  - Kısa spoiler+özet (max 2 cümle) döndürür
"""

import core.llm_provider as _llm

_TIMEOUT_SEC = 90
_MAX_CHARS = 120000

_SYSTEM_PROMPT = (
    "Sen bir video arşiv veri tabanı için çalışan analitik bir özetleyicisin. "
    "Sana bir filmin veya videonun deşifre (ASR/transcript) metni verilecek.\n"
    "Görevin: Metindeki hikayeyi analiz edip, doğrudan sonuca giden, "
    "hikaye formatında (düz metin) net bir özet çıkarmaktır.\n\n"
    "KESİN KURALLAR (BUNLARA KESİNLİKLE UYULACAK):\n\n"
    "UZUNLUK: Ortalama 80, en fazla 100 kelime.\n\n"
    "SIFIR MERAK UNSURU: Okuyucuyu filme çekmeye çalışmak KESİNLİKLE YASAKTIR. "
    "\"Acaba ne olacak?\", \"Hayatın zorluklarıyla yüzleşir\", "
    "\"Büyük bir karar beklemektedir\" gibi pazarlama ağzı ve klişe cümleler KULLANMA.\n\n"
    "SPOILER (SONUÇ) ZORUNLUDUR: Hikayenin veya filmin en sonunu, "
    "karakterlerin nihai kaderini EN NET haliyle metne ekleyeceksin. "
    "(Örn: \"X işi bırakır, Y ölür, Z şehri terk eder.\")\n\n"
    "FORMAT: Madde işareti (bullet point) kullanma. Akıcı, tek parça bir paragraf yaz.\n\n"
    "ODAK: Yalnızca ana karakterin yaşadığı temel kırılma noktasına ve finalde verdiği "
    "kesin karara odaklan. Yan hikayeleri tamamen görmezden gel."
)


def summarize_transcript(
    transcript_text: str,
    api_key: str = "",
    model: str = "gemini-2.5-flash",
    log_cb=None,
) -> str | None:
    """Transcript metninden Gemini 2.5 Flash ile Türkçe kısa spoiler+özet üret.

    Args:
        transcript_text: Ham transcript metni.
        api_key: Gemini API anahtarı. Boşsa env'den okunur.
        model: Kullanılacak Gemini model adı.
        log_cb: İsteğe bağlı log callback fonksiyonu.

    Returns:
        Kesin sonuç odaklı spoiler özeti (ort. 80, max 100 kelime, tek paragraf) veya hata durumunda None.
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
