"""
gemini_cast_extractor.py — Gemini 2.5 Flash ile cast/crew ayıklama.

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

GİRDİ (Film Adı: {film_title}):
{ocr_text}
"""

class GeminiCastExtractor:
    """Gemini 2.5 Flash ile OCR metinden cast/crew ayıkla."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", log_cb=None):
        self._api_key = api_key
        self._model = model
        self._log_cb = log_cb

    def _log(self, msg: str) -> None:
        if self._log_cb:
            self._log_cb(msg)

    def extract(self, ocr_lines: list[str], film_title: str = "") -> dict:
        """OCR satırlarından cast/crew ayıkla.

        Args:
            ocr_lines: Ham OCR metin satırları listesi.
            film_title: Film/dizi başlığı (isteğe bağlı, log için).

        Returns:
            {"cast": [...], "crew": [...]} veya hata durumunda boş dict.
        """
        if not ocr_lines:
            return {}

        if not self._api_key:
            self._log("  [Gemini] API key yok — atlanıyor")
            return {}

        ocr_text = "\n".join(line.strip() for line in ocr_lines if line and line.strip())
        if not ocr_text:
            return {}

        prompt = _EXTRACT_PROMPT.format(ocr_text=ocr_text, film_title=film_title)
        if film_title:
            self._log(f"  [Gemini] Cast ayıklama: '{film_title}'")
        else:
            self._log("  [Gemini] Cast ayıklama başlatılıyor...")

        try:
            response = _llm._gemini_generate(
                prompt,
                api_key=self._api_key,
                model=self._model,
                timeout=_TIMEOUT_SEC,
                log_cb=self._log_cb,
            )
        except Exception as e:
            self._log(f"  [Gemini] API hatası: {e}")
            return {}

        if not response:
            self._log("  [Gemini] Boş yanıt")
            return {}

        return self._parse_response(response)

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
            self._log(f"  [Gemini] JSON parse hatası: {e}")
            return {}

        if not isinstance(data, dict):
            return {}

        result = {}
        if "cast" in data and isinstance(data["cast"], list):
            result["cast"] = data["cast"]
        if "crew" in data and isinstance(data["crew"], list):
            result["crew"] = data["crew"]
        return result
