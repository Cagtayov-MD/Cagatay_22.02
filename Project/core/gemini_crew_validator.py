"""gemini_crew_validator.py — TMDB bulunamadığında Gemini ile crew doğrulama.

OCR'dan okunan crew bilgilerini Gemini 2.5 Flash'a gönderir.
Gemini, kurum adlarını / rol başlıklarını / OCR hatalarını ayıklar
ve doğru kişi-rol eşleştirmesi döndürür.

Sadece TMDB miss durumunda çağrılır — 30K film ölçeğinde gereksiz
API çağrısı yapılmaması için bu guard pipeline_runner'da uygulanır.
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
