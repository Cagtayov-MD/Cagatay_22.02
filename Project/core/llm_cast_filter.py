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

Kurallar:
- İsimler herhangi bir dilde olabilir (Türkçe, Fransızca, İngilizce, İtalyanca, Almanca vb.)
- OCR hataları olabilir — küçük yazım hataları olan isimler yine de isimdir
- 1-3 harfli anlamsız kısaltmalar (OC, CX, AV, AM) isim DEĞİLDİR
- Rastgele harf kombinasyonları (COCACO, alqu, DCJV) isim DEĞİLDİR
- Teknik terimler, rol başlıkları (ceviren, sahneye koyan, yazanlar) isim DEĞİLDİR — bunlar rol başlığıdır

Metin listesi (numaralı):
{numbered_list}

Yanıt formatı — sadece isim olan satırların numaralarını yaz, her satıra bir numara:
ISIM: 1
ISIM: 5
ISIM: 12
...

Sadece numaraları yaz, açıklama ekleme."""

_BATCH_SIZE = 50
_TIMEOUT_SEC = 60
_LLM_CONFIDENCE_BOOST = 0.3   # approved entries get +0.3 confidence
_REJECTED_CONFIDENCE  = 0.2   # rejected entries get confidence set to 0.2

# Parse: "ISIM: 5", "5", "5. Ali" gibi çeşitli formatları kabul et
_LINE_NUM_RE = re.compile(r'(?:ISIM\s*:\s*)?(\d+)', re.IGNORECASE)


class LLMCastFilter:
    """
    Ollama LLM ile cast listesini filtrele.
    OCR çöpünü at, gerçek kişi isimlerini (tüm diller) koru.
    Görsel gerektirmez — sadece metin bazlı.
    """

    def __init__(self, ollama_url="http://localhost:11434",
                 model="llama3.1:8b", enabled=True, log_cb=None):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self._log_cb = log_cb

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
            m = _LINE_NUM_RE.search(line.strip())
            if m:
                n = int(m.group(1))
                if 1 <= n <= total:
                    approved.add(n)
        return approved

    def _filter_batch(self, batch: list[dict]) -> list[dict]:
        """Tek bir batch'i filtrele."""
        names = [
            (e.get("actor_name") or "").strip()
            for e in batch
        ]
        numbered_list = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(names))
        prompt = CAST_FILTER_PROMPT.format(numbered_list=numbered_list)

        response = self._query_ollama(prompt)
        if response is None:
            # Ollama hatası → batch'i olduğu gibi döndür
            return batch

        approved_indices = self._parse_response(response, len(batch))

        result = []
        for i, entry in enumerate(batch):
            one_based = i + 1
            entry = dict(entry)  # kopya
            if one_based in approved_indices:
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

        filtered: list[dict] = []
        for start in range(0, len(cast), _BATCH_SIZE):
            batch = cast[start: start + _BATCH_SIZE]
            batch_result = self._filter_batch(batch)
            filtered.extend(batch_result)

        approved = [e for e in filtered if e.get("is_llm_verified")]
        self._log(
            f"  [LLM] {len(approved)}/{len(cast)} giriş onaylandı.",
            log_cb,
        )

        # Sadece onaylananları döndür
        return approved
