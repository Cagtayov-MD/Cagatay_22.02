"""match_parser.py — Spor maçı transcript analizi (Ollama)."""

import json
import re
import urllib.request
import urllib.error

from core._ollama_url import normalize_ollama_url


class MatchParser:
    def __init__(self, ollama_url="http://localhost:11434", model="llama3.1:8b", log_cb=None):
        self.ollama_url = normalize_ollama_url(ollama_url)
        self.model = model
        self._log = log_cb or print

    def parse(self, transcript_segments, video_duration_sec):
        """İlk 10 + Son 10 dakika transcript'ini parse et."""
        first_10 = [s for s in transcript_segments if s.get("start", 0) < 600]
        last_10 = [s for s in transcript_segments if s.get("start", 0) > (video_duration_sec - 600)]

        combined_text = self._segments_to_text(first_10 + last_10)
        if not combined_text.strip():
            return self._empty_result()

        return self._ollama_parse(combined_text)

    def _segments_to_text(self, segments):
        return "\n".join(s.get("text", "") for s in segments)

    def _ollama_parse(self, text):
        """Ollama'ya transcript gönder, maç bilgilerini JSON olarak al."""
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

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw = result.get("response", "")

                # Thinking mode temizliği
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

                # JSON bloğunu regex ile çıkar
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except urllib.error.HTTPError as e:
            self._log(f"  [MatchParser] Ollama HTTP hatası: {e.code} {e.reason}")
        except json.JSONDecodeError:
            self._log("  [MatchParser] Ollama JSON parse edilemedi")
        except Exception as e:
            self._log(f"  [MatchParser] Ollama hatası: {e}")
        return self._empty_result()

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
