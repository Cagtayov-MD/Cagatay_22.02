"""
post_process.py — [E] Ollama llama3.1:8b ile transcript post-process.

Girdi:  ham transcript segments
Çıktı:  düzeltilmiş transcript segments

İşler:
  - Türkçe noktalama düzeltme
  - Cümle bütünlüğü restore (WhisperX kırık cümleler üretir)
  - Düşük confidence → [anlaşılamadı] işaretle
  - Konuşmacı adlandırma: SPEAKER_00 → TMDB cast'tan isim (varsa)

Ollama harici süreç olarak çalışır — VRAM yönetimi kendine ait.
"""

import json
import re
import time
import urllib.request
import urllib.error


class PostProcessStage:
    """Ollama ile Türkçe transcript düzeltme + özet."""

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
                ollama_url: "http://localhost:11434" (default)
                ollama_model: "llama3.1:8b" (default)
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
        base_url = opts.get("ollama_url", "http://localhost:11434")
        model = opts.get("ollama_model", "llama3.1:8b")
        tmdb_cast = opts.get("tmdb_cast", [])

        if not segments:
            return {
                "status": "ok",
                "segments": [],
                "corrections": 0,
                "summary_tr": "",
                "stage_time_sec": 0.0,
            }

        # Ollama erişilebilir mi?
        if not self._check_ollama(base_url):
            self._log("  [PostProcess] Ollama bağlantısı yok — düzeltme atlanıyor")
            # Düzeltme yapamıyoruz ama düşük confidence işaretleme yapabiliriz
            processed = self._mark_low_confidence(segments)
            return {
                "status": "partial",
                "segments": processed,
                "corrections": 0,
                "summary_tr": "",
                "stage_time_sec": round(time.time() - t0, 2),
                "error": "ollama_unavailable",
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
            self._log(f"  [PostProcess] {len(to_fix)} segment Ollama düzeltmesine gidiyor...")

            for batch_start in range(0, len(to_fix), self.BATCH_SIZE):
                batch = to_fix[batch_start:batch_start + self.BATCH_SIZE]
                batch_texts = [s["text"] for _, s in batch]
                fixed_texts = self._batch_fix_turkish(
                    batch_texts, base_url, model
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

        # ── Aşama 4: Özet üret ──
        summary = ""
        full_text = " ".join(s.get("text", "") for s in segments if s.get("text"))
        if len(full_text) > 100:
            summary = self._summarize(full_text, base_url, model)
            if summary:
                self._log(f"  [PostProcess] Özet üretildi ({len(summary)} karakter)")

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

    def _batch_fix_turkish(self, texts: list, base_url: str,
                            model: str) -> list:
        """Birden fazla metni tek Ollama çağrısında düzelt."""
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

        response = self._chat(prompt, system, base_url, model)
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

    def _summarize(self, transcript: str, base_url: str,
                    model: str) -> str:
        """5-8 cümle Türkçe özet üret."""
        system = (
            "Sen bir film/dizi analistisın. "
            "Sana verilen transcript'ten 5-8 cümlelik Türkçe özet çıkar. "
            "Önemli kişileri, olayları ve konuları vurgula. "
            "Sadece özeti yaz."
        )
        return self._chat(
            f"Transcript:\n{transcript[:4000]}", system,
            base_url, model
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
        window_size = len(name_parts) + 1  # +1 ara kelimeye izin ver
        for i in range(len(context_words) - len(name_parts) + 1):
            window = context_words[i:i + window_size]
            positions = []
            for part in name_parts:
                for j, w in enumerate(window):
                    if w == part and j not in positions:
                        positions.append(j)
                        break
            if (len(positions) == len(name_parts) and
                    positions == sorted(positions) and
                    positions[-1] - positions[0] <= len(name_parts)):
                return True
        return False

    def _check_ollama(self, base_url: str) -> bool:
        """Ollama çalışıyor mu?"""
        try:
            req = urllib.request.Request(
                f"{base_url}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _chat(self, prompt: str, system: str,
              base_url: str, model: str) -> str:
        """Ollama API çağrısı."""
        payload = {
            "model": model,
            "messages": [],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].append({"role": "user", "content": prompt})

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result.get("message", {}).get("content", "").strip()
                # Thinking mode temizliği
                content = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                ).strip()
                return content
        except Exception as e:
            self._log(f"  [Ollama] İstek hatası: {e}")
            return ""
