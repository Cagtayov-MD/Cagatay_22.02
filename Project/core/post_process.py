"""
post_process.py — [E] LLM (Gemini veya Ollama) ile transcript post-process.

Girdi:  ham transcript segments
Çıktı:  düzeltilmiş transcript segments

İşler:
  - Türkçe noktalama düzeltme
  - Cümle bütünlüğü restore (faster-whisper kırık cümleler üretir)
  - Düşük confidence → [anlaşılamadı] işaretle
  - Konuşmacı adlandırma: SPEAKER_00 → TMDB cast'tan isim (varsa)

Varsayılan sağlayıcı: Gemini (LLM_PROVIDER=gemini).
Geri dönüş: Ollama (LLM_PROVIDER=ollama).
"""

import re
import time
import urllib.error
import urllib.parse
import urllib.request

from core._ollama_url import normalize_ollama_url
import core.llm_provider as _llm


class PostProcessStage:
    """LLM (Gemini/Ollama) ile Türkçe transcript düzeltme + özet."""

    # Confidence eşikleri
    LOW_CONF_THRESHOLD = 0.40      # altında → [anlaşılamadı]
    FIX_CONF_THRESHOLD = 0.75      # altında → Ollama düzeltme dene

    # Batch boyutu: kaç segment birden Ollama'ya gönderilsin
    BATCH_SIZE = 10

    def __init__(self, log_cb=None):
        self._log = log_cb or print

    def run(self, segments: list, **opts) -> dict:
        """
        Transcript segment'lerini düzelt.

        Args:
            segments: TranscribeStage çıktısı (segments listesi)
            opts:
                llm_provider: "gemini" veya "ollama" (default: LLM_PROVIDER env veya "gemini")
                ollama_url: "http://localhost:11434" (default, Ollama sağlayıcı için)
                ollama_model: "llama3.1:8b" (default, Ollama sağlayıcı için)
                tmdb_cast: TMDB cast listesi (konuşmacı eşleştirme)

        Returns:
            {
                "status": "ok",
                "segments": [...],
                "corrections": 12,
                "summary_tr": "...",
                "stage_time_sec": 30.0
            }
        """
        t0 = time.time()
        provider = opts.get("llm_provider") or _llm.get_provider()
        base_url = normalize_ollama_url(opts.get("ollama_url", "http://localhost:11434"))
        model = opts.get("ollama_model") or None  # None → provider default
        tmdb_cast = opts.get("tmdb_cast", [])

        # For Ollama provider: validate URL and check availability
        if provider == "ollama":
            if not self._validate_ollama_url(base_url):
                self._log(f"  [PostProcess] Geçersiz ollama_url={base_url!r} — post-process atlanıyor")
                processed = self._mark_low_confidence(segments) if segments else []
                return {
                    "status": "skipped",
                    "segments": processed,
                    "corrections": 0,
                    "summary_tr": "",
                    "stage_time_sec": round(time.time() - t0, 2),
                    "error": "invalid_ollama_url",
                }

            if not self._check_ollama(base_url):
                self._log("  [PostProcess] Ollama bağlantısı yok — düzeltme atlanıyor")
                processed = self._mark_low_confidence(segments) if segments else []
                return {
                    "status": "partial",
                    "segments": processed,
                    "corrections": 0,
                    "summary_tr": "",
                    "stage_time_sec": round(time.time() - t0, 2),
                    "error": "ollama_unavailable",
                }

        if not segments:
            return {
                "status": "ok",
                "segments": [],
                "corrections": 0,
                "summary_tr": "",
                "stage_time_sec": 0.0,
            }

        # ── Aşama 1: Düşük confidence işaretleme ──
        segments = self._mark_low_confidence(segments)

        # ── Aşama 2: Batch Türkçe düzeltme ──
        corrections = 0
        to_fix = [
            (i, s) for i, s in enumerate(segments)
            if s.get("confidence", 1.0) < self.FIX_CONF_THRESHOLD
            and s.get("confidence", 0) >= self.LOW_CONF_THRESHOLD
            and s.get("text", "").strip()
        ]

        if to_fix:
            self._log(f"  [PostProcess] {len(to_fix)} segment LLM düzeltmesine gidiyor...")

            for batch_start in range(0, len(to_fix), self.BATCH_SIZE):
                batch = to_fix[batch_start:batch_start + self.BATCH_SIZE]
                batch_texts = [s["text"] for _, s in batch]
                fixed_texts = self._batch_fix_turkish(
                    batch_texts, provider, base_url, model
                )

                for (idx, seg), fixed in zip(batch, fixed_texts):
                    if fixed and fixed != seg["text"]:
                        seg["text_original"] = seg["text"]
                        seg["text"] = fixed
                        corrections += 1

        self._log(f"  [PostProcess] {corrections} segment düzeltildi")

        # ── Aşama 3: Konuşmacı adlandırma (TMDB cast varsa) ──
        if tmdb_cast:
            self._resolve_speakers(segments, tmdb_cast)

        # ── Aşama 4: Özet üret (sadece açıkça istenirse) ──
        summary = ""
        summarize_enabled = opts.get("summarize_enabled", False)
        if summarize_enabled:
            full_text = " ".join(s.get("text", "") for s in segments if s.get("text"))
            if len(full_text) > 100:
                summary = self._summarize(full_text, provider, base_url, model)
                if summary:
                    self._log(f"  [PostProcess] Özet üretildi ({len(summary)} karakter)")
        else:
            self._log("  [PostProcess] Özet üretimi atlandı (summarize_enabled=False)")

        elapsed = round(time.time() - t0, 2)
        return {
            "status": "ok",
            "segments": segments,
            "corrections": corrections,
            "summary_tr": summary,
            "stage_time_sec": elapsed,
        }

    def _mark_low_confidence(self, segments: list) -> list:
        """Düşük confidence segment'leri [anlaşılamadı] ile işaretle."""
        for seg in segments:
            conf = seg.get("confidence", 1.0)
            if conf < self.LOW_CONF_THRESHOLD and seg.get("text", "").strip():
                seg["text_original"] = seg["text"]
                seg["text"] = "[anlaşılamadı]"
                seg["low_confidence"] = True
        return segments

    def _batch_fix_turkish(self, texts: list, provider: str,
                            base_url: str, model: str | None) -> list:
        """Birden fazla metni tek LLM çağrısında düzelt."""
        if not texts:
            return texts

        # Batch prompt: numaralı satırlar
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = (
            f"Aşağıdaki {len(texts)} Türkçe cümleyi dilbilgisi ve noktalama "
            "açısından düzelt. Her satır için sadece düzeltilmiş halini yaz. "
            "Anlamı değiştirme. Numara formatını koru.\n\n" + numbered
        )
        system = (
            "Sen bir Türkçe metin editörüsün. Sadece düzeltilmiş metinleri yaz, "
            "açıklama ekleme."
        )

        response = self._chat(prompt, system, provider, base_url, model)
        if not response:
            return texts

        # Parse: numaralı satırları geri çıkar
        result = list(texts)  # kopyala, parse başarısızsa orijinal döner
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # "1. düzeltilmiş metin" formatını parse et
            match = re.match(r'^(\d+)\.\s*(.+)$', line)
            if match:
                idx = int(match.group(1)) - 1
                fixed = match.group(2).strip()
                if 0 <= idx < len(result) and fixed:
                    result[idx] = fixed

        return result

    def _summarize(self, transcript: str, provider: str,
                   base_url: str, model: str | None) -> str:
        """5-8 cümle Türkçe özet üret."""
        system = (
            "Sen bir film/dizi analistisın. "
            "Sana verilen transcript'ten 5-8 cümlelik Türkçe özet çıkar. "
            "Önemli kişileri, olayları ve konuları vurgula. "
            "Sadece özeti yaz."
        )
        return self._chat(
            f"Transcript:\n{transcript[:4000]}", system,
            provider, base_url, model
        )

    def _resolve_speakers(self, segments: list, tmdb_cast: list):
        """
        TMDB cast listesi ile konuşmacı eşleştirmesi.

        ISSUE-08 FIX (post_process): Eski bag-of-words yaklaşımı yanlış pozitif
        üretiyordu — "Ali Kaya" → "ali dedi ki kaya gibi" ifadesinde her iki
        kelime ayrı ayrı geçtiği için yanlış eşleşiyordu.

        Çözüm: Bigram sliding-window — isim kelimelerinin ardışık geçmesi gerekiyor.
        """
        speaker_names = {}

        for seg in segments:
            speaker = seg.get("speaker", "")
            if not speaker or speaker in speaker_names:
                continue

            text = seg.get("text", "").lower()
            words = text.split()

            for person in tmdb_cast:
                name = person.get("name", "") or person.get("character", "")
                if not name:
                    continue
                name_parts = name.lower().split()
                if not name_parts:
                    continue

                if self._sliding_window_match(name_parts, words):
                    speaker_names[speaker] = name
                    break

        for seg in segments:
            speaker = seg.get("speaker", "")
            if speaker in speaker_names:
                seg["speaker_label"] = speaker_names[speaker]

    def _sliding_window_match(self, name_parts: list, context_words: list) -> bool:
        """
        İsim parçalarının context içinde ardışık (veya 1 kelime arayla)
        geçip geçmediğini kontrol et.
        """
        if not name_parts or not context_words:
            return False

        if len(name_parts) == 1:
            return name_parts[0] in context_words

        # Sliding window
        # Allow up to 1 extra word gap between name parts
        max_span = len(name_parts) + 1  # +1 allows 1 word gap
        for i in range(len(context_words) - len(name_parts) + 1):
            # Extract window large enough to check for gaps
            window_end = min(i + max_span, len(context_words))
            window = context_words[i:window_end]
            positions = []
            for part in name_parts:
                for j, w in enumerate(window):
                    if w == part and j not in positions:
                        positions.append(j)
                        break
            # Check if all parts found in order with acceptable span
            if (len(positions) == len(name_parts) and
                    positions == sorted(positions) and
                    positions[-1] - positions[0] <= len(name_parts)):
                return True
        return False

    def _validate_ollama_url(self, url: str) -> bool:
        """URL format doğrulama: http/https scheme ve host gereklidir."""
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    def _check_ollama(self, base_url: str) -> bool:
        """Ollama çalışıyor mu?"""
        try:
            req = urllib.request.Request(
                f"{base_url}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as e:
            self._log(f"  [Ollama] Erişim kontrolü HTTP hatası: {e.code} {e.reason}")
            return False
        except urllib.error.URLError as e:
            self._log(f"  [Ollama] Erişim kontrolü bağlantı hatası: {e.reason}")
            return False
        except Exception as e:
            self._log(f"  [Ollama] Erişim kontrolü hatası: {e}")
            return False

    def _chat(self, prompt: str, system: str,
              provider: str, base_url: str, model: str | None) -> str:
        """LLM API çağrısı (Gemini veya Ollama)."""
        kwargs = {}
        if provider == "ollama":
            kwargs["ollama_url"] = base_url
        if model:
            kwargs["model"] = model
        result = _llm.generate(
            prompt,
            system=system or None,
            provider=provider,
            log_cb=self._log,
            timeout=30,
            **kwargs,
        )
        return result or ""
