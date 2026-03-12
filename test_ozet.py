"""
TEST: Aynı transcript'i Gemini + Local model ile özetle, karşılaştır.

KULLANIM:
    python test_ozet.py

AYARLAR (aşağıdaki 3 satırı doldur):
"""

GEMINI_API_KEY  = ""                      # ← Gemini API keyini buraya yaz
OLLAMA_MODEL    = "qwen2.5:7b"            # ← Local model adın
OLLAMA_URL      = "http://localhost:11434" # ← Ollama adresi (genelde bu)

# ─────────────────────────────────────────────────────────────────────────────

import os, sys, pathlib, json, re, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# API key env'den de okunur (GEMINI_API_KEY=xxx python test_ozet.py)
GEMINI_API_KEY = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")

TRANSCRIPT_FILE = pathlib.Path(__file__).parent / "test_transcript.txt"

SYSTEM_PROMPT = (
    "You are an analytical summarizer working for a video archive database. "
    "You will be given a transcript (ASR) of a film or video.\n"
    "Your task: analyze the story in the transcript and produce a concise, "
    "direct summary in narrative prose format.\n\n"
    "STRICT RULES:\n"
    "LENGTH: Target 80 words, maximum 100 words.\n"
    "ZERO SUSPENSE: No teaser language. No 'faces challenges', 'what happens next' etc.\n"
    "SPOILER MANDATORY: Include final outcome and each main character's fate.\n"
    "FORMAT: Single paragraph, no bullet points.\n"
    "FOCUS: Main character's key turning point and final decision only.\n"
    "NO INTRODUCTIONS: Don't introduce characters by name/age/job at the start.\n"
    "Output language: Turkish. No title. Start directly with the story."
)

FEW_SHOT = (
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


def call_gemini(transcript: str) -> str:
    if not GEMINI_API_KEY:
        return "[HATA: GEMINI_API_KEY tanımlı değil]"
    prompt = FEW_SHOT + f"Transcript:\n{transcript}"
    contents = [
        {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
        {"role": "model", "parts": [{"text": "Understood."}]},
        {"role": "user", "parts": [{"text": prompt}]},
    ]
    payload = json.dumps({"contents": contents}).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read())
            text = "".join(p.get("text","") for p in data["candidates"][0]["content"]["parts"]).strip()
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or "[boş yanıt]"
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        return f"[HATA: HTTP {e.code} — {body}]"
    except Exception as e:
        return f"[HATA: {e}]"


def call_ollama(transcript: str) -> str:
    prompt = FEW_SHOT + f"Transcript:\n{transcript}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    payload = json.dumps({
        "model": OLLAMA_MODEL, "messages": messages, "stream": False,
        "options": {"temperature": 0.4, "top_p": 0.9, "num_predict": 512, "repeat_penalty": 1.1},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
            text = data.get("message", {}).get("content", "").strip()
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or "[boş yanıt]"
    except urllib.error.URLError as e:
        return f"[HATA: Ollama bağlantı hatası — {e.reason}]"
    except Exception as e:
        return f"[HATA: {e}]"


def main():
    if not TRANSCRIPT_FILE.exists():
        print(f"[HATA] {TRANSCRIPT_FILE} bulunamadı"); sys.exit(1)

    transcript = TRANSCRIPT_FILE.read_text(encoding="utf-8").strip()
    print(f"Transcript: {len(transcript)} karakter")
    print(f"Gemini key: {'VAR' if GEMINI_API_KEY else 'YOK — script tepesine yaz'}")
    print(f"Ollama    : {OLLAMA_URL}  model={OLLAMA_MODEL}")
    print("\nİki model paralel çalışıyor...\n")

    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(call_gemini, transcript): "gemini",
            pool.submit(call_ollama, transcript): "local",
        }
        for fut in as_completed(futures):
            key = futures[fut]
            results[key] = fut.result()
            print(f"  ✓ {key} tamamlandı")

    sep  = "=" * 68
    dash = "-" * 68
    print(f"\n{sep}")
    print("  GEMİNİ  (gemini-2.5-flash)")
    print(dash)
    print(results.get("gemini", "[yok]"))
    print(f"\n{sep}")
    print(f"  LOCAL   ({OLLAMA_MODEL})")
    print(dash)
    print(results.get("local", "[yok]"))
    print(sep)


if __name__ == "__main__":
    main()
