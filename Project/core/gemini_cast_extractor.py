"""
gemini_cast_extractor.py — Cast/crew ayıklama için Gemini 2.5 Flash kullanır.

TMDB eşleşmediğinde, PaddleOCR + 8'li filtre sonrası temizlenmiş
OCR metin listesini Gemini API'ye gönderir. Gemini metin bazlı olarak:
- Gerçek kişi isimlerini ayıklar
- Çöp/gürültü metinleri eler
- Cast (oyuncu) ve Crew (teknik ekip) ayrımı yapar
- Bozuk isimleri düzeltir (OCR hataları)

Görsel/frame göndermez — sadece metin bazlı çalışır.
API: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent
"""

import json

import core.llm_provider as _llm

_TIMEOUT_SEC = 60

_EXTRACT_PROMPT = """Sen bir film/dizi jeneriği OCR temizleme ve cast/crew ayıklama asistanısın.
Girdi: OCR ile okunmuş ham metin satırları (gürültü içerebilir).
Amaç: YALNIZCA gerçek kişi isimlerini ayıklamak, cast/crew ayırmak ve OCR kaynaklı isim hatalarını düzeltmek.

KATİ KURALLAR:
- SADECE JSON döndür. JSON dışında tek bir karakter bile yazma.
- Emin değilsen öğeyi EKLEME (false positive istemiyoruz).
- Şirket/marka/platfrom/teknik terim/başlık/yıl/ülke/slogan vb. kişi olmayan her şeyi ele.
- Tamamı büyük harfli uzun başlıklar, “DIRECTED BY”, “CAST”, “PRODUCER”, “SPECIAL THANKS”, “WITH” gibi kategori satırları kişi değildir (kategori olabilir ama kişi listesine ekleme).
- Aynı kişi farklı yazımlarla geçiyorsa tekilleştir: en doğru/standart yazımı kullan.
- OCR düzeltmesi yap: harf hataları, aksanlar (Gonxalez→González), birleşik/ayrık kelimeler vb.

ÇOKLU KOLON DÜZENİ (KRİTİK):
- Film jeneriğinde isimler bazen yatayda 2-3 kolonda gösterilir.
  OCR bu kolonları soldan sağa okur ve YANLIŞ birleştirir.
  
  ÖRNEK HATA:
    OCR satırları: ["Tommy", "Murvyn", "Douglas", "Rettig", "Vye", "Spencer"]
    Yanlış okuma:  Tommy Murvyn Douglas + Rettig Vye Spencer
    Doğru okuma:   Tommy Rettig | Murvyn Vye | Douglas Spencer
    (her kolondaki üst kelime + alt kelime = bir isim)

- Bu örüntüyü tanı: Ardışık satırlarda N adet tekil kelime varsa
  ve bu kelimeler bilinen isim parçalarıysa, kolonlara göre eşleştir.
- Tek başına "Tommy", "Murvyn", "Rettig", "Vye", "Spencer", "Douglas" 
  gibi kelimeler geliyorsa bunların çok kolonlu bir isim listesinin 
  parçası olduğunu anla ve doğru şekilde birleştir.
- Genel kural: üst satır sırasıyla ad, alt satır sırasıyla soyad
  (veya tam tersi) olabilir. Bilgi tabanındaki gerçek oyuncu isimlerini
  kullanarak doğru eşleştirmeyi yap.
- Eğer film adı verilmişse, o filme ait bilinen oyuncu isimlerini referans al
  ve OCR bozukluklarını bu bilgiyle düzelt.

ÇIKTI ŞEMASI (kesin):
{{
  "cast": [
    {{"actor_name": "Ad Soyad", "character_name": "Karakter" }}
  ],
  "crew": [
    {{"name": "Ad Soyad", "role": "Görev/Rol" }}
  ]
}}

CAST kuralları:
- Actor (oyuncu) isimlerini "cast" içine koy.
- Karakter adı görünmüyorsa "character_name" boş string olsun "".
- Karakter↔oyuncu eşleşmesi açıkça anlaşılıyorsa koru.
  Örn formatlar: "CHARACTER — ACTOR", "ACTOR as CHARACTER", "CHARACTER / ACTOR".
  Belirsizse eşleştirme yapma (character_name="").

CREW kuralları:
- Crew'e sadece kişi isimlerini ekle.
- "role": mümkünse kısa ve standart tut (örn: Director, Producer, Writer, Cinematography, Editor, Music, Costume, Makeup, Sound, Production Design).
- Rol belli değilse crew'e ekleme.
- Stunt veya dublör rolleri (Stunts, Stunt Double, Stunt Coordinator, Stunt Driver, Stunt Performer, Stunt Actor, Utility Stunts, Dublör, Cascadeur, Stuntkoordinator vb.) varsa bunları tamamen atla — cast'a veya crew'e ekleme.
"""


# ---------------------------------------------------------------------------
# Crew-only score-aware prompt (Director / Producer / Writer only)
# ---------------------------------------------------------------------------
_CREW_SCORE_PROMPT = """You are a strict film-credits extraction assistant.
Your ONLY task: identify real HUMAN persons serving as Director, Producer, or Writer in the provided OCR score data.

INPUT FORMAT
Each entry has these fields:
  text            — OCR-read text
  ocr_confidence  — OCR engine confidence (0.0–1.0); higher = more reliable reading
  seen_count      — how many frames this text appeared in; higher = more persistent, less noise
  verdict         — pipeline pre-classification: "KEEP" (likely a real person/role line) or "REJECTED" (likely noise/non-person)
  name_db_match   — true if matched in a known-names database
  llm_verified    — true if a previous LLM pass already verified this as a person name
  pipeline_confidence — overall pipeline confidence score (null if unavailable)

MULTILINGUAL ROLE VOCABULARY — look for these labels/headers (case-insensitive) immediately before or near a person name:
  Director  : director, directed by, a film by, film by, réalisateur, réalisatrice, réalisation, un film de,
              regisseur, regie, regia, yönetmen, yöneten, مخرج, निर्देशक, निर्देशन
  Producer  : producer, produced by, executive producer, produzent, producteur, productrice,
              produttore, productor, yapımcı, yapım yönetmeni, منتج, निर्माता
  Writer    : writer, written by, screenplay, screenwriter, story by, drehbuchautor, scénariste,
              sceneggiatore, guionista, senarist, senaryo, كاتب السيناريو, पटकथा

HARD REJECTION RULES — return an empty list if ALL entries are rejected by any of these:
1. verdict == "REJECTED" entries are excluded by default unless they contain a strong role label
   AND have ocr_confidence >= 0.50 AND seen_count >= 2.
2. NEVER return any of the following as a person:
   - Country names (France, Germany, Cameroon, Cameroun, Italy, Spain, etc.)
   - Government ministries / agencies (MINISTERE DE LA COOPERATION, Ministry of Culture, etc.)
   - TV/Radio channels (CRTV, Cameroun Radio and Television, RAI, ARD, etc.)
   - Production companies / studios (any text ending in "Films", "Productions", "Studio", "Corp", "Inc", "Ltd", "GmbH", "S.A.", "SARL", etc.)
   - Schools / universities / funds / foundations (FODIC, CNC, Fonds Sud, etc.)
   - Generic role headers WITHOUT a person name following (e.g. "Réalisateur" alone)
   - Acknowledgement / participation blocks ("avec le soutien de", "en coproduction avec", "remerciements", "special thanks", etc.)
   - Co-production credit lines
   - Locations / addresses / years / slogans
3. Be conservative: if you cannot confidently identify a real human first+last name, return empty list for that role.
4. Prefer entries where name_db_match=true or llm_verified=true or high seen_count + high ocr_confidence.
5. ONLY return JSON. No explanations.

OUTPUT FORMAT (strict JSON, no markdown):
{
  "directors": ["Full Name"],
  "producers": ["Full Name"],
  "writers":   ["Full Name"]
}

If no confident person found for a role, use an empty list [].
"""

class GeminiCastExtractor:
    """Cast/crew ayıklama için LLM kullanır (varsayılan: Gemini; openai da desteklenir)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        log_cb=None,
        provider: str = "gemini",   # "gemini" | "openai"
    ):
        self._api_key = api_key
        self._model = model
        self._log_cb = log_cb
        self._provider = provider.lower()
        self.timed_out: bool = False

    def _log(self, msg: str) -> None:
        if self._log_cb:
            self._log_cb(msg)

    def extract(self, ocr_lines: list[str], film_title: str = "", max_lines: int = 200) -> dict:
        """OCR satırlarından cast/crew ayıkla.

        Args:
            ocr_lines: Ham OCR metin satırları listesi.
            film_title: Film/dizi başlığı (isteğe bağlı, log için).
            max_lines: İşlenecek maksimum OCR satırı sayısı (varsayılan: 200).

        Returns:
            {"cast": [...], "crew": [...]} veya hata durumunda boş dict.
        """
        self.timed_out = False
        if not ocr_lines:
            return {}

        if not self._api_key:
            self._log(f"  [LLM/{self._provider}] API key yok — atlanıyor")
            return {}

        # Çok büyük OCR listesi → ilk max_lines satırla sınırla
        if len(ocr_lines) > max_lines:
            ocr_lines = ocr_lines[:max_lines]
            self._log(f"  [LLM/{self._provider}] OCR satırı sınırlandı: {max_lines}")

        ocr_text = "\n".join(line.strip() for line in ocr_lines if line and line.strip())
        if not ocr_text:
            return {}

        if film_title:
            title_section = f"""DOSYA ADI BAŞLIĞI (ANCHOR — bu başlık dosya adından çıkarılmıştır ve güvenilirdir):
"{film_title}"
Bu başlığı referans al. OCR satırlarında farklı bir başlık görünse bile, dosya adından gelen başlık önceliklidir.
OCR'dan okunan kısa/anlamsız metin parçaları (örn. "XX", "AB", "CD") film başlığı DEĞİLDİR.

"""
            self._log(f"  [LLM/{self._provider}] Cast ayıklama: '{film_title}'")
        else:
            title_section = "Film başlığı bilinmiyor.\n\n"
            self._log(f"  [LLM/{self._provider}] Cast ayıklama başlatılıyor...")

        prompt = _EXTRACT_PROMPT + title_section + "GİRDİ:\n" + ocr_text

        _timed_out_flag: list[bool] = []
        try:
            if self._provider == "openai":
                response = _llm._openai_generate(
                    prompt,
                    api_key=self._api_key,
                    model=self._model,
                    timeout=_TIMEOUT_SEC,
                    log_cb=self._log_cb,
                )
            else:
                response = _llm._gemini_generate(
                    prompt,
                    api_key=self._api_key,
                    model=self._model,
                    timeout=_TIMEOUT_SEC,
                    log_cb=self._log_cb,
                    timeout_flag=_timed_out_flag,
                )
        except Exception as e:
            self._log(f"  [LLM/{self._provider}] API hatası: {e}")
            return {}
        finally:
            if _timed_out_flag:
                self.timed_out = True

        if not response:
            self._log(f"  [LLM/{self._provider}] Boş yanıt")
            return {}

        return self._parse_response(response)

    def extract_crew_from_scores(
        self,
        ocr_scores: list[dict],
        film_title: str = "",
        max_entries: int = 200,
    ) -> dict:
        """OCR skor verisinden yalnızca Yönetmen/Yapımcı/Yazar ayıkla.

        Ham OCR metin satırları yerine OCR score JSON yapısını kullanır.
        Bu sayede verdict/confidence/seen_count filtreleri Gemini'ye
        sinyal olarak iletilir ve gürültülü satırlar (REJECTED) varsayılan
        olarak dışlanır.

        Args:
            ocr_scores: OCR score dict listesi. Her eleman:
                        {text, ocr_confidence, seen_count, verdict,
                         name_db_match, llm_verified, pipeline_confidence}
            film_title: Film başlığı (anchor, isteğe bağlı).
            max_entries: Gönderilecek maksimum skor kaydı sayısı.

        Returns:
            {"directors": [...], "producers": [...], "writers": [...]}
            veya hata durumunda boş dict.
        """
        self.timed_out = False
        if not ocr_scores:
            return {}
        if not self._api_key:
            self._log(f"  [LLM/{self._provider}] API key yok — crew skor ayıklama atlanıyor")
            return {}

        # Büyük listeler için sınırla
        entries = ocr_scores[:max_entries]
        if len(ocr_scores) > max_entries:
            self._log(f"  [LLM/{self._provider}] OCR skor sınırlandı: {max_entries}")

        # Compact JSON payload — sadece gerekli alanlar
        payload = json.dumps(entries, ensure_ascii=False)

        title_section = (
            f'Film title (trusted anchor): "{film_title}"\n\n'
            if film_title
            else ""
        )
        prompt = (
            _CREW_SCORE_PROMPT
            + title_section
            + "OCR SCORE DATA (JSON array):\n"
            + payload
        )

        if film_title:
            self._log(f"  [LLM/{self._provider}] Crew skor ayıklama: '{film_title}'")
        else:
            self._log(f"  [LLM/{self._provider}] Crew skor ayıklama başlatılıyor...")

        _timed_out_flag: list[bool] = []
        try:
            if self._provider == "openai":
                response = _llm._openai_generate(
                    prompt,
                    api_key=self._api_key,
                    model=self._model,
                    timeout=_TIMEOUT_SEC,
                    log_cb=self._log_cb,
                )
            else:
                response = _llm._gemini_generate(
                    prompt,
                    api_key=self._api_key,
                    model=self._model,
                    timeout=_TIMEOUT_SEC,
                    log_cb=self._log_cb,
                    timeout_flag=_timed_out_flag,
                )
        except Exception as e:
            self._log(f"  [LLM/{self._provider}] API hatası (crew skor): {e}")
            return {}
        finally:
            if _timed_out_flag:
                self.timed_out = True

        if not response:
            self._log(f"  [LLM/{self._provider}] Boş yanıt (crew skor)")
            return {}

        return self._parse_crew_score_response(response)

    def _parse_crew_score_response(self, response: str) -> dict:
        """Crew skor Gemini yanıtından JSON ayıkla.

        Beklenen format: {"directors": [...], "producers": [...], "writers": [...]}
        """
        text = response.strip()
        if "```" in text:
            start = text.find("{", text.find("```"))
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
        else:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            self._log(f"  [LLM/{self._provider}] JSON parse hatası (crew skor): {e}")
            return {}

        if not isinstance(data, dict):
            return {}

        result: dict[str, list[str]] = {}
        for key in ("directors", "producers", "writers"):
            raw = data.get(key)
            if isinstance(raw, list):
                result[key] = [
                    str(name).strip()
                    for name in raw
                    if name and str(name).strip()
                ]
            else:
                result[key] = []
        return result


    def _parse_response(self, response: str) -> dict:
        """Gemini yanıtından JSON cast/crew verisi çıkar."""
        # JSON bloğunu bul (```json ... ``` veya ham JSON)
        text = response.strip()
        # Markdown code fences temizle
        if "```" in text:
            start = text.find("{", text.find("```"))
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
        else:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            self._log(f"  [LLM/{self._provider}] JSON parse hatası: {e}")
            return {}

        if not isinstance(data, dict):
            return {}

        result = {}
        if "cast" in data and isinstance(data["cast"], list):
            result["cast"] = data["cast"]
        if "crew" in data and isinstance(data["crew"], list):
            result["crew"] = data["crew"]
        return result
