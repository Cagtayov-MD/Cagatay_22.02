"""film_ozet_generator.py — İki varyantlı film özet üreticisi (TR ve EN prompt).

Her çalıştırmada aynı transcript'e hem Türkçe (TR) hem İngilizce (EN) prompt
uygulanır; iki özet yan yana karşılaştırmalı biçimde sunulur.

Config: PROJECT_ROOT/config/film_ozet_prompt.json
Bağımlılık: yalnızca Python stdlib (urllib, json, pathlib, argparse, re)

Kullanım (CLI):
    python film_ozet_generator.py transcript.txt
    python film_ozet_generator.py transcript.txt --model qwen2.5:7b --url http://localhost:11434
"""

import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

# Config dosyasının yolu bu dosyanın konumuna göre hesaplanır.
_THIS_DIR = pathlib.Path(__file__).resolve().parent          # Project/src/ai/
_PROJECT_ROOT = _THIS_DIR.parent.parent                      # Project/
_CONFIG_PATH = _PROJECT_ROOT / "config" / "film_ozet_prompt.json"

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:7b"
_TIMEOUT_SEC = 120


class FilmOzetGenerator:
    """İki varyantlı (TR / EN prompt) film özet üreticisi.

    Args:
        ollama_base_url: Ollama sunucusunun temel URL'si.
        model:           Kullanılacak Ollama model adı.
    """

    def __init__(
        self,
        ollama_base_url: str = _DEFAULT_OLLAMA_URL,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model = model
        self._config = self._load_config()

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def generate(self, transcript: str) -> dict:
        """Her iki varyantı çalıştırır ve sonuçları döndürür.

        Args:
            transcript: Film/video deşifre metni.

        Returns:
            {"tr": "<Türkçe özet>", "en": "<İngilizce-prompt özet>"}
            Hata durumunda ilgili değer "[HATA: <açıklama>]" olur.
        """
        few_shot = self._config.get("few_shot_example", "")
        variants = self._config.get("variants", {})
        results = {}

        for key in ("tr", "en"):
            variant = variants.get(key, {})
            system = variant.get("system", "")
            user_tpl = variant.get("user_template", "{transcript}")
            user = user_tpl.replace("{few_shot}", few_shot).replace(
                "{transcript}", transcript
            )
            result = self._call_ollama(system=system, user=user)
            results[key] = result

        return results

    def format_comparison(self, results: dict) -> str:
        """İki özeti yan yana karşılaştırmalı metin olarak formatlar.

        Args:
            results: ``generate()`` dönüşü dict; "tr" ve "en" anahtarları beklenir.

        Returns:
            Karşılaştırmalı metin bloğu.
        """
        separator = "=" * 72
        half_sep = "-" * 72
        lines = [
            separator,
            "  VİTOS — FİLM ÖZET KARŞILAŞTIRMASI",
            separator,
            "",
            "▌ VARYANT TR  (Türkçe prompt)",
            half_sep,
            results.get("tr") or "[boş yanıt]",
            "",
            "▌ VARYANT EN  (İngilizce prompt, Türkçe çıktı)",
            half_sep,
            results.get("en") or "[boş yanıt]",
            "",
            separator,
        ]
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        """Config JSON dosyasını okur ve döndürür."""
        if not _CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Config dosyası bulunamadı: {_CONFIG_PATH}\n"
                "Beklenen konum: Project/config/film_ozet_prompt.json"
            )
        with _CONFIG_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)

    def _call_ollama(self, system: str, user: str) -> str:
        """Ollama /api/chat endpoint'ine istek atar ve yanıt metnini döndürür.

        Args:
            system: Sistem mesajı.
            user:   Kullanıcı mesajı.

        Returns:
            Model yanıt metni; hata durumunda "[HATA: <açıklama>]".
        """
        endpoint = f"{self.ollama_base_url}/api/chat"
        ollama_cfg = self._config.get("ollama", {})

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": ollama_cfg.get("temperature", 0.4),
                    "top_p": ollama_cfg.get("top_p", 0.9),
                    "num_predict": ollama_cfg.get("num_predict", 512),
                    "repeat_penalty": ollama_cfg.get("repeat_penalty", 1.1),
                },
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("message", {}).get("content", "").strip()
                # Düşünce etiketlerini temizle (bazı modeller ekler)
                content = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                ).strip()
                if not content:
                    return "[HATA: Model boş yanıt döndürdü]"
                return content
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            return f"[HATA: HTTP {exc.code} {exc.reason} — {body}]"
        except urllib.error.URLError as exc:
            return f"[HATA: Ollama bağlantı hatası — {exc.reason}]"
        except TimeoutError:
            return f"[HATA: Ollama zaman aşımı ({_TIMEOUT_SEC}s)]"
        except Exception as exc:  # noqa: BLE001
            return f"[HATA: Beklenmedik hata — {exc}]"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="İki varyantlı film özet üreticisi (TR & EN prompt).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  python film_ozet_generator.py transcript.txt\n"
            "  python film_ozet_generator.py transcript.txt --model qwen2.5:7b\n"
            "  python film_ozet_generator.py transcript.txt "
            "--url http://localhost:11434 --model llama3.1:8b\n"
        ),
    )
    parser.add_argument("transcript_file", help="Transcript metin dosyası (.txt)")
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Ollama model adı (varsayılan: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--url",
        default=_DEFAULT_OLLAMA_URL,
        help=f"Ollama temel URL'si (varsayılan: {_DEFAULT_OLLAMA_URL})",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    transcript_path = pathlib.Path(args.transcript_file)
    if not transcript_path.exists():
        print(f"[HATA] Dosya bulunamadı: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    transcript = transcript_path.read_text(encoding="utf-8").strip()
    if not transcript:
        print("[HATA] Transcript dosyası boş.", file=sys.stderr)
        sys.exit(1)

    print(f"Model   : {args.model}")
    print(f"Ollama  : {args.url}")
    print(f"Config  : {_CONFIG_PATH}")
    print(f"Dosya   : {transcript_path} ({len(transcript)} karakter)")
    print()

    generator = FilmOzetGenerator(ollama_base_url=args.url, model=args.model)
    results = generator.generate(transcript)
    print(generator.format_comparison(results))


if __name__ == "__main__":
    main()
