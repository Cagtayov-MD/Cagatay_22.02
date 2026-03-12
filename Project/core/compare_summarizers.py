"""compare_summarizers.py — Aynı transcript'i hem Gemini hem local Ollama ile özetler.

İki model aynı anda (paralel thread) çalıştırılır; sonuçlar yan yana karşılaştırılır.

Kullanım (pipeline'dan):
    from core.compare_summarizers import compare_summaries, format_comparison

    result = compare_summaries(
        transcript_text,
        gemini_api_key="...",
        gemini_model="gemini-2.5-flash",
        ollama_url="http://localhost:11434",
        ollama_model="qwen2.5:7b",
        log_cb=self._log,
    )
    # result = {
    #   "gemini": "<özet metni veya None>",
    #   "local":  "<özet metni veya hata açıklaması>",
    #   "gemini_model": "gemini-2.5-flash",
    #   "local_model":  "qwen2.5:7b",
    # }

Kullanım (bağımsız CLI):
    python compare_summarizers.py transcript.txt
    python compare_summarizers.py transcript.txt --gemini-model gemini-2.5-flash \\
        --ollama-model qwen2.5:7b --ollama-url http://localhost:11434
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import core.llm_provider as _llm

# ─────────────────────────────────────────────────────────────────────────────
# Prompt — her iki modele de aynı sistem + kullanıcı mesajı gönderilir
# (gemini_summarizer.py ile birebir aynı prompt)
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are an analytical summarizer working for a video archive database. "
    "You will be given a transcript (ASR) of a film or video.\n"
    "Your task: analyze the story in the transcript and produce a concise, "
    "direct summary in narrative prose format.\n\n"
    "STRICT RULES — THESE MUST BE FOLLOWED WITHOUT EXCEPTION:\n\n"
    "LENGTH: Target 80 words, maximum 100 words.\n\n"
    "ZERO SUSPENSE: Attempting to make the reader curious about the film is STRICTLY FORBIDDEN. "
    "Do NOT use phrases like 'faces life's challenges', 'a big decision awaits', "
    "'what will happen next?' or any marketing/teaser language.\n\n"
    "SPOILER (ENDING) IS MANDATORY: You MUST include the film's final outcome and "
    "each main character's ultimate fate as explicitly as possible. "
    "(e.g., 'X quits her job, Y dies, Z leaves the city.')\n\n"
    "FORMAT: No bullet points. Write a single, flowing paragraph.\n\n"
    "FOCUS: Focus only on the main character's key turning point and their final decision. "
    "Completely ignore subplots and secondary characters.\n\n"
    "NO CHARACTER INTRODUCTIONS: Do NOT introduce or describe characters by name/age/job "
    "at the start. Just tell the story — who they are will emerge from the events.\n\n"
    "Output language: Turkish. No title. Start directly with the story."
)

_FEW_SHOT = (
    "Aşağıdaki örnek DOĞRU formattır — tam olarak böyle yaz:\n\n"
    "ÖRNEK:\n"
    "Başarılı bir reklamcı olan David, önemli bir müşteriyle duygusal bir ilişki "
    "içindeyken, babasının şeker hastalığı yüzünden bacağının kesilmesi gerektiğini "
    "öğrenir. Kariyeri için hayati önem taşıyan bir sunum ile babasının ameliyatı "
    "aynı güne denk gelir. İşi ve sevgilisi yerine babasını tercih eden David, "
    "her şeyi elinin tersiyle iterek hastanede onun yanında kalır. Annesi de "
    "hastabakıcılık yapmayı reddedince, ameliyattan sakat çıkan babasının tüm "
    "sorumluluğu David'e kalır. Film, yıllarca birbirinden uzak kalan baba-oğulun "
    "geçmişi geride bırakıp birlikte yeni bir hayata başlamasıyla biter.\n\n"
    "Şimdi aşağıdaki transcript için aynı formatta özet yaz:\n"
)

_MAX_CHARS = 120_000
_GEMINI_TIMEOUT = 90
_OLLAMA_TIMEOUT = 120


# ─────────────────────────────────────────────────────────────────────────────
# İç yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def _run_gemini(prompt: str, system: str, api_key: str, model: str, log_cb) -> str | None:
    """Gemini API'ye istek atar; başarılı metin veya None döndürür."""
    return _llm._gemini_generate(
        prompt,
        system=system,
        api_key=api_key or None,
        model=model,
        timeout=_GEMINI_TIMEOUT,
        log_cb=log_cb,
    )


def _run_ollama(prompt: str, system: str, url: str, model: str, log_cb) -> str | None:
    """Ollama'ya istek atar; başarılı metin veya None döndürür."""
    return _llm._ollama_generate(
        prompt,
        system=system,
        ollama_url=url,
        model=model,
        timeout=_OLLAMA_TIMEOUT,
        log_cb=log_cb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compare_summaries(
    transcript_text: str,
    *,
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.5-flash",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5:7b",
    log_cb=None,
) -> dict:
    """Aynı transcript'i Gemini ve local Ollama ile paralel özetler.

    Args:
        transcript_text: Ham transcript metni.
        gemini_api_key:  Gemini API anahtarı. Boşsa GEMINI_API_KEY env okunur.
        gemini_model:    Gemini model adı.
        ollama_url:      Ollama sunucu adresi.
        ollama_model:    Ollama model adı.
        log_cb:          İsteğe bağlı log callback fonksiyonu.

    Returns:
        {
            "gemini":       str | None,   # Gemini özeti (None = hata/API key yok)
            "local":        str | None,   # Ollama özeti (None = bağlantı hatası)
            "gemini_model": str,
            "local_model":  str,
        }
    """
    if not transcript_text or not transcript_text.strip():
        return {"gemini": None, "local": None, "gemini_model": gemini_model, "local_model": ollama_model}

    snippet = transcript_text.strip()[:_MAX_CHARS]
    prompt = _FEW_SHOT + f"Transcript:\n{snippet}"

    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")

    results: dict = {
        "gemini": None,
        "local": None,
        "gemini_model": gemini_model,
        "local_model": ollama_model,
    }

    if log_cb:
        log_cb(
            f"  [Karşılaştırma] Gemini ({gemini_model}) ve "
            f"local ({ollama_model}) paralel başlatılıyor..."
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_run_gemini, prompt, _SYSTEM_PROMPT, api_key, gemini_model, log_cb): "gemini",
            pool.submit(_run_ollama, prompt, _SYSTEM_PROMPT, ollama_url, ollama_model, log_cb): "local",
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                results[key] = fut.result()
            except Exception as exc:
                if log_cb:
                    log_cb(f"  [Karşılaştırma] {key} hatası: {exc}")
                results[key] = None

    if log_cb:
        g_len = len(results["gemini"]) if results["gemini"] else 0
        l_len = len(results["local"]) if results["local"] else 0
        log_cb(f"  [Karşılaştırma] Gemini: {g_len} karakter | Local: {l_len} karakter")

    return results


def format_comparison(result: dict) -> str:
    """Karşılaştırma sonucunu okunabilir metin bloğu olarak formatlar.

    Args:
        result: ``compare_summaries()`` dönüşü dict.

    Returns:
        Karşılaştırmalı metin bloğu.
    """
    sep = "=" * 72
    dash = "-" * 72
    g_model = result.get("gemini_model", "gemini")
    l_model = result.get("local_model", "local")
    g_text = result.get("gemini") or "[boş yanıt / hata]"
    l_text = result.get("local") or "[boş yanıt / hata]"

    lines = [
        sep,
        "  ÖZET KARŞILAŞTIRMASI — Gemini vs Local Model",
        sep,
        "",
        f"▌ GEMİNİ  ({g_model})",
        dash,
        g_text,
        "",
        f"▌ LOCAL   ({l_model})",
        dash,
        l_text,
        "",
        sep,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcript'i Gemini ve local Ollama ile paralel özetler, karşılaştırır.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  python compare_summarizers.py transcript.txt\n"
            "  python compare_summarizers.py transcript.txt "
            "--gemini-model gemini-2.5-flash --ollama-model qwen2.5:7b\n"
        ),
    )
    parser.add_argument("transcript_file", help="Transcript metin dosyası (.txt)")
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Gemini model adı (varsayılan: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2.5:7b",
        help="Ollama model adı (varsayılan: qwen2.5:7b)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama sunucu adresi (varsayılan: http://localhost:11434)",
    )
    parser.add_argument(
        "--gemini-key",
        default="",
        help="Gemini API anahtarı (yoksa GEMINI_API_KEY env okunur)",
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

    print(f"Gemini model : {args.gemini_model}")
    print(f"Local model  : {args.ollama_model}")
    print(f"Ollama URL   : {args.ollama_url}")
    print(f"Transcript   : {transcript_path} ({len(transcript)} karakter)")
    print()

    def log(msg: str) -> None:
        print(msg)

    result = compare_summaries(
        transcript,
        gemini_api_key=args.gemini_key,
        gemini_model=args.gemini_model,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        log_cb=log,
    )
    print()
    print(format_comparison(result))


if __name__ == "__main__":
    main()
