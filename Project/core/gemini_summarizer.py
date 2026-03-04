"""
gemini_summarizer.py — Transcript'ten Türkçe özet çıkarma (Gemini 2.5 Flash).

summarize_transcript(transcript_text, api_key, log_cb) -> str | None
  - Transcript metnini Gemini'ye gönderir
  - 8-12 cümlelik Türkçe özet döndürür
"""

import core.llm_provider as _llm

_TIMEOUT_SEC = 30
_MAX_CHARS = 120000

_SYSTEM_PROMPT = (
    "Sen profesyonel bir senaryo analistisin. Sana bir filmin veya dizi bölümünün "
    "otomatik oluşturulmuş, hatalı yazımlar içeren bir transcripti verilecektir.\n\n"
    "GÖREVİN:\n"
    "Bu transcripti BAŞTAN SONA analiz ederek içeriğin olay örgüsünü özetle.\n\n"
    "TEMEL KURAL:\n"
    "YALNIZCA transcript'i kullan. İçerik tanıdık gelse bile "
    "olayları transcriptten çıkar, asla uydurma yapma.\n"
    "Transcriptte geçmeyen bir olay veya bilgiyi kesinlikle ekleme.\n\n"
    "İSİM ve HATA DÜZELTMESİ:\n"
    "Otomatik transkripsiyondan kaynaklanan fonetik hataları bağlamdan çıkar ve düzelt.\n"
    "Aynı karakterin farklı yazımlarını (örn: 'Met'/'Mert'/'Matt') birleştir.\n"
    "Karakter isimlerini, mekânları ve rolleri tutarlı hale getir.\n\n"
    "KAPSAM:\n"
    "Transcriptin başından SONUNA kadar tüm ana olayları kapsa.\n"
    "Hiçbir kilit sahneyi atlama; her ana olay en az bir cümleyle temsil edilmeli.\n"
    "Özellikle SON SAHNE mutlaka dahil edilmeli — hikayenin nasıl bittiğini yaz.\n\n"
    "GÜRÜLTÜ AYIKLAMA:\n"
    "Tekrarlayan altyazı etiketleri, selamlaşmalar ve "
    "önemsiz diyalogları yoksay.\n"
    "Uzun sessiz bölümleri dikkate alma.\n\n"
    "ÇIKTI FORMATI:\n"
    "- Tek paragraf, akıcı ve edebi Türkçe\n"
    "- Giriş → Gelişme → Sonuç akışını koru\n"
    "- 8-12 cümle\n"
    "- Başlık kullanma, doğrudan hikayeyi anlatmaya başla"
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
