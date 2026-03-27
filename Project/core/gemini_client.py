"""
gemini_client.py — sport_analyzer için ince Gemini sarmalayıcı.

llm_provider._gemini_generate üzerine GeminiClient arayüzü sunar.
"""

import core.llm_provider as _llm


class GeminiClient:
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    def generate(self, prompt: str, system: str | None = None) -> str:
        result = _llm._gemini_generate(
            prompt,
            system=system,
            api_key=self.api_key,
            model=self.model,
        )
        return result or ""

    def generate_with_search(self, prompt: str) -> tuple[str, list[str]]:
        """Google Search grounding ile Gemini sorgusu.

        Returns:
            (cevap_metni, benzersiz_domain_listesi)
            Hata durumunda ("", [])
        """
        import os, json, urllib.request, urllib.error
        from urllib.parse import urlparse

        api_key = self.api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return ("", [])

        base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
        url = f"{base_url}/v1beta/models/{self.model}:generateContent?key={api_key}"

        payload = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            candidate = data.get("candidates", [{}])[0]
            text = (candidate.get("content", {})
                             .get("parts", [{}])[0]
                             .get("text", ""))

            chunks = (candidate.get("grounding_metadata", {})
                                .get("grounding_chunks", []))
            domains = list({
                urlparse(c["web"]["uri"]).netloc
                for c in chunks
                if c.get("web", {}).get("uri")
            })

            return (text.strip(), domains)
        except Exception:
            return ("", [])

    def generate_with_audio(self, prompt: str, audio_b64: str, mime_type: str = "audio/wav") -> str:
        """Ses + metin içeren Gemini isteği — multimodal."""
        import os, json, urllib.request, urllib.error

        api_key = self.api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return ""

        base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
        url = f"{base_url}/v1beta/models/{self.model}:generateContent?key={api_key}"

        payload = json.dumps({
            "contents": [{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                    {"text": prompt},
                ],
            }]
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            return (data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", ""))
        except Exception:
            return ""
