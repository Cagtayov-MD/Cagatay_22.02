"""gemini_crew_validator.py — Gemini ile crew ve tekil isim doğrulama.

İki bağımsız işlev içerir:

1. validate_crew_with_gemini()
   TMDB bulunamadığında tüm OCR crew'unu JSON formatında Gemini'ye gönderir.
   Sadece TMDB miss durumunda pipeline_runner tarafından çağrılır.

2. verify_single_name()
   name_verify.py'nin Geçiş-2 Gemini katmanı için tekil isim doğrulayıcısı.
   OCR adayını Gemini'ye sorar; yalnızca YES veya NO kabul eder.
   Fail-closed: timeout/network/parse hataları False (unresolved) döndürür.
"""
import json
import logging

_log = logging.getLogger(__name__)


def validate_crew_with_gemini(
    film_title: str,
    ocr_crew: list[dict],
    ocr_lines: list | None = None,
    gemini_model: str = "gemini-2.5-flash",
) -> dict | None:
    """OCR crew verisini Gemini ile doğrula.

    Args:
        film_title: Film adı (OCR veya filename'den)
        ocr_crew: OCR'dan parse edilmiş crew listesi [{name, role, job, ...}]
        ocr_lines: Ham OCR satırları (opsiyonel, ek context için)
        gemini_model: Kullanılacak Gemini model adı

    Returns:
        dict: {verified_roles: {YAPIMCI: [...], YÖNETMEN: [...], ...}, source: "gemini"}
        None: Gemini erişilemezse veya hata olursa
    """
    try:
        import core.llm_provider as _llm
    except ImportError:
        _log.warning("[GeminiCrewValidator] llm_provider import edilemedi")
        return None

    # OCR crew'u okunabilir formata çevir
    crew_text = []
    for entry in (ocr_crew or []):
        name = (entry.get("name") or "").strip()
        role = (entry.get("role") or entry.get("job") or "").strip()
        if name:
            crew_text.append(f"  - {role}: {name}" if role else f"  - {name}")

    if not crew_text:
        return None

    # Opsiyonel: Ham OCR satırlarından credits kısmını ekle (son 100 satır)
    ocr_context = ""
    if ocr_lines:
        credit_lines = []
        for line in ocr_lines[-100:]:
            text = line.text if hasattr(line, "text") else (line.get("text") or "")
            conf = getattr(line, "avg_confidence", line.get("confidence", 0))
            if text and conf > 0.5:
                credit_lines.append(text)
        if credit_lines:
            ocr_context = "\n\nHam OCR credits satırları:\n" + "\n".join(credit_lines[:50])

    prompt = f"""Film: "{film_title}"

Aşağıda bu filmin jenerik yazılarından OCR ile okunan yapım ekibi bilgileri var.
Bu bilgilerde OCR hataları, kurum adlarının kişi adı olarak okunması, rol başlıklarının isim olarak yazılması gibi sorunlar olabilir.

OCR Crew Verisi:
{chr(10).join(crew_text)}
{ocr_context}

GÖREV:
1. Kurum adlarını (ör: "Cameroun Radio and Television", "MINISTERE DE LA COOPERATION") kişi olarak YAZMA
2. Rol başlıklarını (ör: "Assistant réalisateur", "Cadreur", "Son") kişi adı olarak YAZMA
3. OCR hatalarını düzelt (ör: "Camamnun Radin and Tolouielnn" → bu bir OCR hatası, gerçek metin "Cameroun Radio and Television")
4. Her kişiyi doğru role ata

Yanıtı SADECE şu JSON formatında ver, başka hiçbir şey yazma:
{{
  "YAPIMCI": ["isim1", "isim2"],
  "YÖNETMEN": ["isim1"],
  "YÖNETMEN YARDIMCISI": ["isim1"],
  "GÖRÜNTÜ YÖNETMENİ": ["isim1"],
  "SENARYO": ["isim1"],
  "KAMERA": ["isim1"],
  "KURGU": ["isim1"]
}}

Emin olmadığın rolleri boş bırak: []
Filmde bu bilgi yoksa "VERİ YOK" yazma, sadece boş liste [] kullan.
"""

    try:
        response = _llm.generate(
            prompt=prompt,
            provider="gemini",
            model=gemini_model,
        )

        if not response:
            return None

        # JSON parse — markdown code block varsa temizle
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

        result = json.loads(text)

        if not isinstance(result, dict):
            _log.warning("[GeminiCrewValidator] Geçersiz JSON formatı")
            return None

        # Eksik rolleri boş liste ile doldur
        expected_roles = {
            "YAPIMCI", "YÖNETMEN", "YÖNETMEN YARDIMCISI",
            "GÖRÜNTÜ YÖNETMENİ", "SENARYO", "KAMERA", "KURGU",
        }
        for role in expected_roles:
            if role not in result:
                result[role] = []
            elif not isinstance(result[role], list):
                result[role] = []

        total = sum(len(v) for v in result.values())
        _log.info(f"[GeminiCrewValidator] Başarılı: {total} kişi doğrulandı")
        return {"verified_roles": result, "source": "gemini"}

    except json.JSONDecodeError as e:
        _log.warning(f"[GeminiCrewValidator] JSON parse hatası: {e}")
        return None
    except Exception as e:
        _log.warning(f"[GeminiCrewValidator] Hata: {e}")
        return None


def verify_single_name(
    ocr_raw: str,
    candidate: str,
    gemini_model: str = "gemini-2.5-flash",
) -> tuple[bool, str]:
    """Tek bir isim adayını Gemini ile YES/NO doğrula.

    name_verify.py Geçiş-2 Gemini katmanı için tasarlanmıştır.
    validate_crew_with_gemini() işlevinden bağımsızdır ve onu değiştirmez.

    Davranış kuralları:
    - Gemini yanıtı response.strip().upper() sonrası tam "YES" veya "NO" olmalıdır.
    - Başka her yanıt (YES., Yes, NO - uncertain, vb.) reject/False sayılır ve
      "invalid_response" olarak loglanır.
    - Timeout / ağ hatası / parse hatası → fail-closed (False döner).

    Args:
        ocr_raw: OCR'dan gelen ham metin (doğrulama bağlamı için)
        candidate: Fuzzy matcher'ın önerdiği aday isim
        gemini_model: Kullanılacak Gemini model adı

    Returns:
        (accepted: bool, reason: str)
        reason değerleri: "yes", "no", "invalid_response",
                          "gemini_timeout", "gemini_network_error",
                          "gemini_parse_error", "gemini_error",
                          "llm_unavailable"
    """
    try:
        import core.llm_provider as _llm
    except ImportError:
        _log.warning("[GeminiSingleName] llm_provider import edilemedi")
        return False, "llm_unavailable"

    prompt = (
        f'OCR metni: "{ocr_raw}"\n'
        f'Aday isim: "{candidate}"\n\n'
        "Bu OCR metni bu aday ismin bozulmuş bir yazımı olabilir mi?\n"
        "Sadece YES veya NO yaz, başka hiçbir şey yazma."
    )

    try:
        response = _llm.generate(
            prompt=prompt,
            provider="gemini",
            model=gemini_model,
        )
    except TimeoutError as e:
        _log.warning(f"[GeminiSingleName] Timeout: ocr='{ocr_raw}' candidate='{candidate}' err={e}")
        return False, "gemini_timeout"
    except OSError as e:
        # socket/network hataları OSError alt sınıflarıdır
        _log.warning(f"[GeminiSingleName] Ağ hatası: ocr='{ocr_raw}' candidate='{candidate}' err={e}")
        return False, "gemini_network_error"
    except Exception as e:
        err_name = type(e).__name__.lower()
        if "timeout" in err_name:
            _log.warning(f"[GeminiSingleName] Timeout: ocr='{ocr_raw}' candidate='{candidate}' err={e}")
            return False, "gemini_timeout"
        if any(k in err_name for k in ("network", "connect", "socket", "http")):
            _log.warning(f"[GeminiSingleName] Ağ hatası: ocr='{ocr_raw}' candidate='{candidate}' err={e}")
            return False, "gemini_network_error"
        _log.warning(f"[GeminiSingleName] Hata: ocr='{ocr_raw}' candidate='{candidate}' err={e}")
        return False, "gemini_error"

    if not response:
        _log.warning(f"[GeminiSingleName] Boş yanıt: ocr='{ocr_raw}' candidate='{candidate}'")
        return False, "gemini_parse_error"

    verdict = response.strip().upper()
    if verdict == "YES":
        _log.info(f"[GeminiSingleName] YES: ocr='{ocr_raw}' → candidate='{candidate}'")
        return True, "yes"
    if verdict == "NO":
        _log.info(f"[GeminiSingleName] NO: ocr='{ocr_raw}' candidate='{candidate}'")
        return False, "no"

    # Model geçersiz/beklenmedik yanıt verdi — fail-closed
    _log.warning(
        f"[GeminiSingleName] Geçersiz yanıt: ocr='{ocr_raw}' candidate='{candidate}' "
        f"response='{response.strip()[:80]}'"
    )
    return False, "invalid_response"
