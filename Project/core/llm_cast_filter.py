"""llm_cast_filter.py — Ollama LLM ile cast listesini filtrele.

OCR çöpünü at, gerçek kişi isimlerini (tüm diller) koru.
Görsel gerektirmez — sadece metin bazlı.
"""

import re
import urllib.request
import urllib.error
import json

CAST_FILTER_PROMPT = """Aşağıdaki liste bir film/dizi jenerik ekranından OCR ile okunmuştur.
Her satırda bir metin var. Bu metinlerden hangileri gerçek kişi isimleridir?

KATÎ KURALLAR (bu kurallar kesindir, istisna yoktur):
1. 1-3 harfli metinler ASLA isim değildir (Alt, Cp, Co, JC, CA, TA, Lp, L0 vb.)
2. Tamamen küçük harfle yazılmış metinler isim değildir (oulon, nibeu, etiol, algw vb.)
3. Aşağıdaki kelimeler/ifadeler bir ROL BAŞLIĞIDIR — kişi adı DEĞİLDİR:
   Yönetmen, Director, Regie, Réalisation, Realisation, Regia, Regia di,
   Dirección, Direccion, Dirigido por, Dirigé par, Film by, A film by, Directed by,
   Ein Film von, Réalisé par,
   Senaryo, Screenplay, Drehbuch, Scénario, Scenario, Sceneggiatura, Guión, Written by,
   Yapımcı, Producer, Produzent, Producteur, Executive producer,
   Görüntü Yönetmeni, Director of Photography, DOP, D.O.P., Kamera, Camera,
   Müzik, Music, Musik, Musique, Musica, Original Score,
   Kurgu, Montaj, Editor, Editing, Schnitt, Montage,
   Sanat Yönetmeni, Art Director, Production Designer,
   Kostüm, Costume, Kostümdesign, Ses, Sound, Efekt, Effects, VFX, SFX,
   Makyaj, Makeup, Işık, Lighting, Gaffer,
   Prodüksiyon, Production manager, Line producer,
   sahneye koyan, yazanlar, yazaniar, çeviren, ceviren, cevirtli, cevirla,
   yonetmen, yapimci, senarist, muzik,
   goruntu yonetmeni, oyuncular, oynayanlar, oynayan, basroller, başroller
   Bu kelimeleri kişi adı olarak İŞARETLEME.
4. Anlamsız harf dizileri isim değildir (COEACE, OCOCAOR, ocococ, matCwuda, Ozoquty vb.)
5. Noktalı/özel karakterli metinler isim değildir (t..Hacti, Pierot.Fali vb.)

İSİM OLAN metinlerin özellikleri:
- Ad Soyad formatında (iki veya üç kelime, her kelime büyük harfle başlar)
- Bilinen bir dilde gerçek bir isim (Türkçe, Fransızca, İngilizce, İtalyanca, Almanca vb.)
- OCR hataları olabilir ama temel yapı isim formatında olmalı

Metin listesi (numaralı):
{numbered_list}

Yanıt formatı — sadece isim olan satırların numaralarını yaz, her satıra bir numara:
ISIM: 1
ISIM: 5
ISIM: 12
...

Emin olmadığın metinleri EKLEME. Sadece kesin isim olanları yaz."""

_ROLE_KEYWORDS = {
    # Türkçe
    "sahneye koyan", "sahneyekoyan", "yazanlar", "yazaniar", "yazantar",
    "çeviren", "ceviren", "cevirtli", "cevirla", "cevirtln",
    "yönetmen", "yonetmen", "yapımcı", "yapimci",
    "oyuncular", "oynayanlar", "oynayan", "olayaniar",
    "senaryo", "senarist", "müzik", "muzik", "kurgu", "montaj",
    "görüntü yönetmeni", "goruntu yonetmeni",
    # İngilizce
    "director", "directed by", "a film by", "film by",
    "screenplay", "written by", "story by",
    "producer", "executive producer", "line producer",
    "director of photography", "dop", "d.o.p.",
    "music", "original score", "editor", "editing",
    "art director", "production designer", "costume design",
    "sound", "sound design", "visual effects", "vfx",
    # Almanca
    "regie", "drehbuch", "produzent", "kamera", "schnitt",
    "musik", "ausstattung", "kostümdesign",
    # Fransızca
    "réalisation", "realisation", "scénario", "scenario",
    "producteur", "directeur de la photographie",
    "montage", "musique", "décorateur",
    # İtalyanca
    "regia", "sceneggiatura", "produttore",
    "direttore della fotografia", "montaggio", "musica",
    # İspanyolca / Portekizce
    "dirección", "direccion", "guión", "guion",
    "productor", "director de fotografía",
}

_BATCH_SIZE = 50
_MIN_NAME_LEN = 3   # actor_name must have at least this many characters
_TIMEOUT_SEC = 60
_LLM_CONFIDENCE_BOOST = 0.3   # approved entries get +0.3 confidence
_REJECTED_CONFIDENCE  = 0.2   # rejected entries get confidence set to 0.2

# Parse: yalnızca "ISIM: 5" veya "5" gibi saf satırları kabul et (false-positive'leri önle)
_LINE_NUM_RE = re.compile(r'^(?:ISIM\s*:\s*)?(\d+)\s*$', re.IGNORECASE)


class LLMCastFilter:
    """
    Ollama LLM ile cast listesini filtrele.
    OCR çöpünü at, gerçek kişi isimlerini (tüm diller) koru.
    Görsel gerektirmez — sadece metin bazlı.
    """

    def __init__(self, ollama_url="http://localhost:11434",
                 model="llama3.1:8b", enabled=True, log_cb=None,
                 name_checker=None):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self._log_cb = log_cb
        self.name_checker = name_checker  # callable(text) -> bool, optional

    def _log(self, msg, log_cb=None):
        cb = log_cb or self._log_cb
        if cb:
            cb(msg)

    def _check_availability(self) -> bool:
        """Ollama erişilebilir mi kontrol et."""
        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                # model prefix eşleşmesi (örn. "llama3.1:8b" veya "llama3.1")
                model_base = self.model.split(":")[0]
                for m in models:
                    if m.startswith(model_base):
                        return True
                if models:
                    self._log(
                        f"  [LLM] Uyarı: '{self.model}' bulunamadı. "
                        f"Mevcut modeller: {', '.join(models[:3])}"
                    )
                return False
        except Exception as e:
            self._log(f"  [LLM] Ollama erişilemiyor: {e}")
            return False

    def _query_ollama(self, prompt: str) -> str | None:
        """Ollama'ya prompt gönder, yanıt al."""
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")
        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "")
        except urllib.error.URLError as e:
            self._log(f"  [LLM] Sorgu hatası: {e}")
            return None
        except Exception as e:
            self._log(f"  [LLM] Beklenmedik hata: {e}")
            return None

    def _parse_response(self, response: str, total: int) -> set[int]:
        """LLM yanıtından 1-tabanlı satır numaralarını çıkar."""
        approved: set[int] = set()
        for line in response.splitlines():
            m = _LINE_NUM_RE.match(line.strip())
            if m:
                n = int(m.group(1))
                if 1 <= n <= total:
                    approved.add(n)
        return approved

    def _is_namedb_verified(self, entry: dict) -> bool:
        """actor_name'in NameDB korumasına uygun olup olmadığını güçlü kriterlerle kontrol et.

        Kurallar:
        - En az 2 kelime olmalı (tek kelime isimleri koruma kapsamı dışı).
        - Rol/görev anahtar kelimeleri içeriyorsa NameDB koruması devre dışı.
        - Kelimelerin en az %80'i NameDB'de bulunmalı.
        """
        if not self.name_checker:
            return False
        name = (entry.get("actor_name") or "").strip()
        if not name:
            return False
        words = name.split()
        # Tek kelimeli isimler korunmaz (örn. sadece "Ali")
        if len(words) < 2:
            return False
        # Rol/görev anahtar kelimesi içeriyorsa NameDB koruması devre dışı
        name_lower = name.lower()
        for role_kw in _ROLE_KEYWORDS:
            if role_kw in name_lower:
                return False
        # En az %80 kelime NameDB'de bulunmalı
        matched = sum(1 for w in words if self.name_checker(w))
        return matched / len(words) >= 0.8

    def _pre_filter_obvious_junk(self, entry: dict) -> bool:
        """LLM'e göndermeden önce kesin çöpleri ele. True = çöp, atılmalı."""
        name = (entry.get("actor_name") or "").strip()
        if not name:
            return True
        # 3 harften kısa
        if len(name) < _MIN_NAME_LEN:
            return True
        # Tamamen küçük harf tek kelime
        if ' ' not in name and name.islower():
            return True
        # Bilinen Türkçe rol başlıkları
        if name.lower() in _ROLE_KEYWORDS:
            return True

        # Sesli harf oranı kontrolü
        alpha_chars = [c for c in name.lower() if c.isalpha()]
        if alpha_chars:
            vowels = sum(1 for c in alpha_chars if c in 'aeıioöuü')
            if len(alpha_chars) > 4 and vowels == 0:
                return True  # Hiç sesli harf yok → çöp
            if len(alpha_chars) > 5 and vowels / len(alpha_chars) < 0.15:
                return True  # Sesli harf oranı çok düşük → çöp

        # Özel karakter kontrolü
        if re.search(r'[.@#$%&{}\[\]]', name):
            return True

        # Rakam ağırlıklı
        digits = sum(1 for c in name if c.isdigit())
        if len(name) > 2 and digits / len(name) > 0.4:
            return True

        return False

    def _filter_batch(self, batch: list[dict]) -> list[dict]:
        """Tek bir batch'i filtrele."""
        # Pre-filter: kesin çöpleri LLM'e göndermeden işaretle
        pre_filtered_flags = [self._pre_filter_obvious_junk(e) for e in batch]

        # Sadece pre-filter'dan geçenleri LLM'e gönder
        llm_batch = [e for e, junk in zip(batch, pre_filtered_flags) if not junk]

        if llm_batch:
            names = [(e.get("actor_name") or "").strip() for e in llm_batch]
            numbered_list = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(names))
            prompt = CAST_FILTER_PROMPT.format(numbered_list=numbered_list)

            response = self._query_ollama(prompt)
            if response is None:
                # Ollama hatası → fail-safe: hiçbir satırı onaylama
                self._log("  [LLM] Ollama yanıtı None — batch onaylanmadı (fail-safe).")
                llm_approved: set[int] = set()
            else:
                llm_approved = self._parse_response(response, len(llm_batch))
        else:
            llm_approved = set()

        # Sonuçları orijinal batch sırasına göre yeniden birleştir
        result = []
        llm_idx = 0
        for i, entry in enumerate(batch):
            entry = dict(entry)  # kopya
            if pre_filtered_flags[i]:
                # Kesin çöp: doğrudan reddet
                entry["is_llm_verified"] = False
                entry["confidence"] = _REJECTED_CONFIDENCE
            else:
                one_based = llm_idx + 1
                llm_idx += 1
                if one_based in llm_approved:
                    entry["is_llm_verified"] = True
                    existing = float(entry.get("confidence", 0.6))
                    entry["confidence"] = round(min(1.0, existing + _LLM_CONFIDENCE_BOOST), 3)
                else:
                    entry["is_llm_verified"] = False
                    entry["confidence"] = _REJECTED_CONFIDENCE
            result.append(entry)
        return result

    def filter_cast(self, cast: list[dict], log_cb=None) -> list[dict]:
        """
        Cast listesini filtrele.

        Args:
            cast: [{"actor_name": "...", "character_name": "...", ...}, ...]

        Returns:
            Filtrelenmiş cast listesi — sadece LLM'in isim olarak onayladıkları
            + confidence ve is_llm_verified alanları eklenmiş
        """
        if not self.enabled:
            self._log("  [LLM] Cast filtresi devre dışı, atlanıyor.", log_cb)
            return cast

        if not cast:
            return cast

        if not self._check_availability():
            self._log(
                "  [LLM] Ollama erişilemiyor — cast listesi değiştirilmeden döndürülüyor.",
                log_cb,
            )
            return cast

        self._log(
            f"  [LLM] {len(cast)} giriş, model={self.model}, "
            f"batch_size={_BATCH_SIZE}",
            log_cb,
        )

        # Orijinal confidence değerlerini sıra ile sakla (NameDB koruması için)
        orig_confidences = [e.get("confidence", 0.5) for e in cast]

        filtered: list[dict] = []
        for start in range(0, len(cast), _BATCH_SIZE):
            batch = cast[start: start + _BATCH_SIZE]
            batch_result = self._filter_batch(batch)
            filtered.extend(batch_result)

        approved = []
        for i, e in enumerate(filtered):
            if e.get("is_llm_verified"):
                approved.append(e)
            elif self.name_checker and self._is_namedb_verified(e):
                # LLM reddetmiş ama NameDB biliyor → koru, orijinal confidence'ı geri yükle
                e["is_name_db_protected"] = True
                if e.get("confidence") == _REJECTED_CONFIDENCE:
                    e["confidence"] = orig_confidences[i]
                approved.append(e)
        self._log(
            f"  [LLM] {len(approved)}/{len(cast)} giriş onaylandı.",
            log_cb,
        )

        # Sadece onaylananları döndür
        return approved
