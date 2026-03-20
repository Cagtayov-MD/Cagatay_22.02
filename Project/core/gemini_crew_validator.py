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
    ocr_scores: list[dict] | None = None,
    gemini_model: str = "gemini-2.5-flash",
) -> dict | None:
    """OCR crew verisini Gemini ile doğrula.

    Yalnızca Yönetmen (YÖNETMEN), Yapımcı (YAPIMCI) ve Yazar (SENARYO)
    rollerine odaklanır. OCR skor verisi varsa (ocr_scores) bunu tercih et;
    skor verisi yoksa ocr_lines ile fallback yap.

    Kişi olmayan öğeleri (ülke, bakanlık, kanal, şirket, fon, teşekkür,
    ortak yapım satırı vb.) agresif şekilde reddeder.

    Args:
        film_title: Film adı (OCR veya filename'den)
        ocr_crew: OCR'dan parse edilmiş crew listesi [{name, role, job, ...}]
        ocr_lines: Ham OCR satırları (opsiyonel, skor yoksa fallback context)
        ocr_scores: OCR skor dict listesi (opsiyonel; tercih edilen kaynak).
                    Her eleman: {text, ocr_confidence, seen_count, verdict,
                                 name_db_match, llm_verified, pipeline_confidence}
        gemini_model: Kullanılacak Gemini model adı

    Returns:
        dict: {verified_roles: {YAPIMCI: [...], YÖNETMEN: [...], SENARYO: [...]}, source: "gemini"}
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

    # OCR skor verisi varsa — Gemini'ye yapılandırılmış sinyal gönder
    score_context = ""
    if ocr_scores:
        import json as _json
        # Sadece KEEP + high-confidence REJECTED satırları gönder (kalabalık önle)
        relevant = [
            s for s in ocr_scores
            if s.get("verdict") == "KEEP"
            or (
                s.get("ocr_confidence", 0) >= 0.5
                and s.get("seen_count", 1) >= 2
            )
        ]
        if relevant:
            score_context = (
                "\n\nOCR Score Data (text + confidence + verdict — REJECTED entries excluded by default):\n"
                + _json.dumps(relevant[:100], ensure_ascii=False)
            )
    elif ocr_lines:
        # Fallback: ham OCR satırlarından credits kısmını ekle (son 100 satır)
        credit_lines = []
        for line in ocr_lines[-100:]:
            text = line.text if hasattr(line, "text") else (line.get("text") or "")
            conf = getattr(line, "avg_confidence", line.get("confidence", 0))
            if text and conf > 0.5:
                credit_lines.append(text)
        if credit_lines:
            score_context = "\n\nHam OCR credits satırları:\n" + "\n".join(credit_lines[:50])

    prompt = f"""Film: "{film_title}"

You are a strict crew validator. From the OCR crew data below, extract ONLY real HUMAN persons
for these three roles: Director (YÖNETMEN), Producer (YAPIMCI), Writer (SENARYO).

MULTILINGUAL ROLE VOCABULARY (case-insensitive):
  Director  : director, directed by, a film by, réalisateur, réalisatrice, réalisation, un film de,
              regisseur, regie, regia, yönetmen, yöneten, مخرج, निर्देशक
  Producer  : producer, produced by, executive producer, produzent, producteur, productrice,
              produttore, productor, yapımcı, yapım yönetmeni, منتج, निर्माता
  Writer    : writer, written by, screenplay, screenwriter, drehbuchautor, scénariste,
              sceneggiatore, guionista, senarist, senaryo, كاتب السيناريو, पटकथा

HARD REJECTION RULES — NEVER return these as persons:
- Country names (France, Germany, Cameroon, Cameroun, Italy, etc.)
- Government ministries / agencies (MINISTERE DE LA COOPERATION, Ministry of Culture, etc.)
- TV/Radio channels (CRTV, Cameroun Radio and Television, RAI, ARD, etc.)
- Production companies / studios (names ending in Films, Productions, Studio, Corp, Inc, Ltd, GmbH, S.A., SARL, etc.)
- Schools / universities / funds / foundations (FODIC, CNC, Fonds Sud, etc.)
- Generic role headers without a person name (e.g. "Réalisateur" alone)
- Acknowledgement / participation blocks (avec le soutien de, en coproduction avec, remerciements, special thanks, etc.)
- Co-production credit lines, locations, addresses, years, slogans
- Any OCR errors that are clearly not human names

OCR Crew Data:
{chr(10).join(crew_text)}
{score_context}

Return ONLY this JSON, nothing else:
{{
  "YAPIMCI": ["full name"],
  "YÖNETMEN": ["full name"],
  "SENARYO": ["full name"]
}}

Leave empty [] for any role you are not confident about.
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

        # Sadece 3 temel rol — eksikleri boş liste ile doldur
        expected_roles = {"YAPIMCI", "YÖNETMEN", "SENARYO"}
        for role in expected_roles:
            if role not in result:
                result[role] = []
            elif not isinstance(result[role], list):
                result[role] = []

        # Bilinmeyen rolleri temizle
        result = {k: v for k, v in result.items() if k in expected_roles}

        total = sum(len(v) for v in result.values())
        _log.info(f"[GeminiCrewValidator] Başarılı: {total} kişi doğrulandı")
        return {"verified_roles": result, "source": "gemini"}

    except json.JSONDecodeError as e:
        _log.warning(f"[GeminiCrewValidator] JSON parse hatası: {e}")
        return None
    except Exception as e:
        _log.warning(f"[GeminiCrewValidator] Hata: {e}")
        return None
