"""match_parser.py — Spor maçı transcript analizi (LLM: Gemini veya Ollama)."""

import json
import re

from core._ollama_url import normalize_ollama_url
import core.llm_provider as _llm


class MatchParser:
    def __init__(self, ollama_url="http://localhost:11434", model=None,
                 log_cb=None, provider=None):
        self.ollama_url = normalize_ollama_url(ollama_url)
        self.model = model  # None → provider default
        self._log = log_cb or print
        self._provider = provider  # None → read from env

    def _active_provider(self) -> str:
        return (self._provider or _llm.get_provider()).lower()

    def parse(self, transcript_segments, video_duration_sec):
        """İlk 10 + Son 10 dakika transcript'ini parse et."""
        first_10 = [s for s in transcript_segments if s.get("start", 0) < 600]
        last_10 = [s for s in transcript_segments if s.get("start", 0) > (video_duration_sec - 600)]

        combined_text = self._segments_to_text(first_10 + last_10)
        if not combined_text.strip():
            return self._empty_result()

        return self._llm_parse(combined_text)

    def _segments_to_text(self, segments):
        return "\n".join(s.get("text", "") for s in segments)

    def _llm_parse(self, text):
        """LLM'e transcript gönder, maç bilgilerini JSON olarak al."""
        prompt = f"""Aşağıdaki spor maçı yayını transkriptini analiz et ve JSON formatında yanıtla:

Transcript:
{text[:3000]}

JSON formatı:
{{
  "spor_turu": "futbol/basketbol/voleybol/diger",
  "lig": "",
  "sehir": "",
  "takimlar": [
    {{"isim": "", "skor": 0}},
    {{"isim": "", "skor": 0}}
  ],
  "teknik_direktorler": [],
  "olaylar": [
    {{"dakika": 0, "olay": "gol/sari_kart/kirmizi_kart", "oyuncu": "", "takim": ""}}
  ]
}}

Sadece JSON döndür, başka açıklama yapma."""

        provider = self._active_provider()
        kwargs = {}
        if provider == "ollama":
            kwargs["ollama_url"] = self.ollama_url
        if self.model:
            kwargs["model"] = self.model

        raw = _llm.generate(
            prompt,
            provider=provider,
            log_cb=self._log,
            timeout=120,
            **kwargs,
        )

        if not raw:
            return self._empty_result()

        # Thinking mode temizliği
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # JSON bloğunu regex ile çıkar
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                self._log("  [MatchParser] JSON parse edilemedi")
        return self._empty_result()

    # Keep _ollama_parse as alias for backward compatibility
    def _ollama_parse(self, text):
        return self._llm_parse(text)

    @staticmethod
    def _empty_result():
        return {
            "spor_turu": "",
            "lig": "",
            "sehir": "",
            "takimlar": [],
            "teknik_direktorler": [],
            "olaylar": [],
        }
