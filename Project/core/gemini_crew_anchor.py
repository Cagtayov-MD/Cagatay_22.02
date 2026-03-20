"""
gemini_crew_anchor.py — Gemini 2.5 Flash ile OCR verisinden temel film metadata ankrajı.

Film TMDB/IMDb'de bulunamadığında, OCR ham satırları + mevcut credits_parser çıktısını
Gemini 2.5 Flash'a gönderir ve şu soruları sorar:
  1. Filmin/dizinin adı nedir?
  2. Yönetmen kim?
  3. En az 2 oyuncu söyle.
  4. Yapım ekibi kimlerden oluşuyor? (yapımcı, görüntü yönetmeni, senaryo, kurgu vb.)

Dönen yapılandırılmış veri TMDB/IMDb aramasında anchor olarak kullanılır:
  verify_as_film(title=anchor.film_title, director_names=anchor.directors,
                 top_actors=anchor.actors[:2])

Bu modül yalnızca metin bazlı çalışır; görsel/frame göndermez.
API: Gemini 2.5 Flash (gemini-2.5-flash)
"""

import json

import core.llm_provider as _llm

_TIMEOUT_SEC = 60

# ─── Sistem Promptu ─────────────────────────────────────────────────────────
_ANCHOR_SYSTEM_PROMPT = """Sen bir film jenerik analiz uzmanısın.
Aşağıda sana iki veri kaynağı verilecek:
1. OCR SATIRLARI: Filmin kapanış/açılış jeneriklerinden ham OCR çıktısı (gürültü içerebilir)
2. CREDITS PARSE ÇIKTISI (JSON): Otomatik ayrıştırıcının ham tahmini

KATİ KURALLAR:
- SADECE JSON döndür. JSON dışında tek bir karakter bile yazma.
- Rol başlıkları kişi ismi DEĞİLDİR: "CADREUR", "SON", "MONTAGE", "UN FILM DE",
  "DIRECTEUR DE LA PHOTOGRAPHIE", "RÉALISATEUR", "ASSISTANT REALISATEUR" vb.
- Kurum/mekan/ülke adları kişi ismi DEĞİLDİR: "HOTEL", "FRANCE", "CAMEROUN",
  "MINISTERE DE LA COOPERATION", "ECOLE", "LES ELEVES DE..." vb.
- OCR sırasında rol başlığı → kişi ismi sıralaması takip et:
  "Directeur de la photographie" → sonraki isim → görüntü yönetmeni
  "Cadreur" → sonraki isim → kamera operatörü
  "Son" (Fransızca ses) → sonraki isim → ses teknisyeni
- Emin değilsen o kişiyi EKLEME.
- Yönetmen için "UN FILM DE", "RÉALISÉ PAR", "A FILM BY", "DIRECTED BY",
  "YÖNETEN", "YÖNETİM" gibi giriş ifadelerinin ardından gelen ismi al.

ÇIKTI ŞEMASI (kesin, her alan doldurulmalı):
{
  "film_title": "string — filmin/dizinin en muhtemel adı (OCR veya credits_parse'dan)",
  "directors": ["string", ...],
  "actors": ["string", ...],
  "producer": ["string", ...],
  "cinematographer": ["string", ...],
  "screenplay": ["string", ...],
  "editor": ["string", ...],
  "assistant_director": ["string", ...],
  "camera_operator": ["string", ...],
  "sound": ["string", ...],
  "music": ["string", ...]
}

Boş listeler [] kabul edilir. film_title bulunamazsa "" kullan.
actors listesinde EN AZ 2 oyuncu adı olmalı (varsa).
"""


class GeminiCrewAnchor:
    """Gemini 2.5 Flash ile OCR verisinden temel film metadata ankrajı çıkarır.

    NAME_VERIFY aşamasında TMDB/IMDb eşleşmesi başarısız olduğunda devreye girer.
    credits_parser çıktısı (cdata) + ham OCR satırlarını Gemini'ye gönderir ve
    film_title, directors, actors gibi temel metadata'yı yapılandırılmış JSON olarak alır.

    Bu metadata daha sonra verify_as_film() / verify_as_series() aramasında
    anchor olarak kullanılır.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", log_cb=None):
        self._api_key = api_key
        self._model = model
        self._log_cb = log_cb
        self.timed_out: bool = False

    def _log(self, msg: str) -> None:
        if self._log_cb:
            self._log_cb(msg)

    def anchor(
        self,
        ocr_lines: list[str],
        cdata: dict,
        film_title_hint: str = "",
        max_ocr_lines: int = 150,
    ) -> dict:
        """OCR satırları + credits_parser çıktısından temel metadata ankrajı çıkar.

        Args:
            ocr_lines: Ham OCR metin satırları (str listesi).
            cdata: credits_parser.to_report_dict() çıktısı (film_title, cast, crew, directors).
            film_title_hint: Dosya adından gelen film başlığı (anchor için güvenilir).
            max_ocr_lines: Gemini'ye gönderilecek maksimum OCR satırı sayısı.

        Returns:
            dict: {
                "film_title": str,
                "directors": [str],
                "actors": [str],
                "producer": [str],
                "cinematographer": [str],
                "screenplay": [str],
                "editor": [str],
                "assistant_director": [str],
                "camera_operator": [str],
                "sound": [str],
                "music": [str],
                "_source": "gemini_anchor"
            }
            Hata durumunda boş dict.
        """
        self.timed_out = False

        if not self._api_key:
            self._log("  [GeminiAnchor] API key yok — atlanıyor")
            return {}

        # OCR satırlarını sınırla
        lines_to_send = ocr_lines[:max_ocr_lines] if ocr_lines else []
        ocr_text = "\n".join(
            (line.get("text", "") if isinstance(line, dict) else str(line)).strip()
            for line in lines_to_send
            if line
        )
        ocr_text = "\n".join(l for l in ocr_text.splitlines() if l.strip())

        # Credits parse çıktısını özetle (kısa tutmak için)
        credits_summary = self._build_credits_summary(cdata)

        # Film başlığı ipucu
        title_hint_section = ""
        if film_title_hint:
            title_hint_section = (
                f"\nDOSYA ADI BAŞLIĞI (güvenilir anchor — OCR'dan farklı görünse bile bu doğrudur):\n"
                f'"{film_title_hint}"\n'
            )

        prompt = (
            _ANCHOR_SYSTEM_PROMPT
            + title_hint_section
            + "\n\n─── OCR SATIRLARI ───\n"
            + (ocr_text or "(boş)")
            + "\n\n─── CREDITS PARSE ÇIKTISI (JSON özeti) ───\n"
            + credits_summary
            + "\n\nGÖREV: Yukarıdaki verilere dayanarak çıktı şemasını JSON olarak doldur."
        )

        self._log(
            f"  [GeminiAnchor] Anchor sorgusu başlatılıyor "
            f"('{film_title_hint or cdata.get('film_title', '?')}')"
        )

        _timed_out_flag: list[bool] = []
        try:
            response = _llm._gemini_generate(
                prompt,
                api_key=self._api_key,
                model=self._model,
                timeout=_TIMEOUT_SEC,
                log_cb=self._log_cb,
                timeout_flag=_timed_out_flag,
            )
        except Exception as e:
            self._log(f"  [GeminiAnchor] API hatası: {e}")
            return {}
        finally:
            if _timed_out_flag:
                self.timed_out = True

        if not response:
            self._log("  [GeminiAnchor] Boş yanıt")
            return {}

        parsed = self._parse_response(response)
        if parsed:
            parsed["_source"] = "gemini_anchor"
            directors = parsed.get("directors") or []
            actors = parsed.get("actors") or []
            self._log(
                f"  [GeminiAnchor] ✓ "
                f"Başlık:'{parsed.get('film_title', '')}' "
                f"Yönetmen:{directors} "
                f"Oyuncu:{actors[:3]}"
            )
        return parsed

    # ─── Yardımcı Metodlar ─────────────────────────────────────────────────

    def _build_credits_summary(self, cdata: dict) -> str:
        """credits_parser çıktısını kısa bir JSON özetine dönüştür."""
        if not cdata:
            return "{}"

        summary = {}

        # Film başlığı
        if cdata.get("film_title"):
            summary["film_title"] = cdata["film_title"]

        # Yönetmenler
        directors_raw = cdata.get("directors") or []
        director_names = []
        for d in directors_raw[:5]:
            if isinstance(d, str):
                director_names.append(d.strip())
            elif isinstance(d, dict):
                n = (d.get("name") or "").strip()
                if n:
                    director_names.append(n)
        if director_names:
            summary["directors"] = director_names

        # Oyuncular (ilk 10, sadece isim)
        cast_raw = cdata.get("cast") or []
        actor_names = []
        for entry in cast_raw[:10]:
            if isinstance(entry, dict):
                n = (entry.get("actor_name") or "").strip()
            else:
                n = str(entry).strip()
            if n:
                actor_names.append(n)
        if actor_names:
            summary["actors"] = actor_names

        # Crew (ilk 20, isim + rol)
        crew_raw = cdata.get("crew") or []
        crew_list = []
        for entry in crew_raw[:20]:
            if isinstance(entry, dict):
                n = (entry.get("name") or "").strip()
                job = (entry.get("job") or entry.get("role") or "").strip()
                if n:
                    crew_list.append({"name": n, "job": job})
        if crew_list:
            summary["crew"] = crew_list

        try:
            return json.dumps(summary, ensure_ascii=False, indent=2)
        except Exception:
            return "{}"

    def _parse_response(self, response: str) -> dict:
        """Gemini yanıtından JSON metadata çıkar."""
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
            self._log(f"  [GeminiAnchor] JSON parse hatası: {e}")
            return {}

        if not isinstance(data, dict):
            return {}

        # Normalize: her alan bir liste veya string olmalı
        result: dict = {}

        # film_title: string
        film_title = data.get("film_title", "")
        if isinstance(film_title, list):
            film_title = film_title[0] if film_title else ""
        result["film_title"] = str(film_title).strip()

        # Liste alanları
        _LIST_FIELDS = [
            "directors", "actors", "producer", "cinematographer",
            "screenplay", "editor", "assistant_director",
            "camera_operator", "sound", "music",
        ]
        for field in _LIST_FIELDS:
            raw = data.get(field, [])
            if isinstance(raw, str):
                raw = [raw] if raw.strip() else []
            if isinstance(raw, list):
                result[field] = [str(v).strip() for v in raw if str(v).strip()]
            else:
                result[field] = []

        return result
