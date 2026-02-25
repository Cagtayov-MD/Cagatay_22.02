import argparse, json, os, sys, time, importlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    wav = args.wav

    # Burada “gerçek ASR çalıştırma” için projendeki entrypoint’e bağlanman lazım.
    # Şimdilik: ASR modülünün import + temel varlık kontrolü.
    # İstersen burayı sonra doğrudan transcribe fonksiyonuna bağlarız.
    mod = importlib.import_module("core.asr_engine")

    result = {
        "wav": wav,
        "module": getattr(mod, "__file__", "unknown"),
        "ok": True,
        "elapsed_s": round(time.time() - t0, 3),
        "note": "Şu an sadece import/varlık testi. ASR fonksiyon çağrısına bağlanacak."
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()