import argparse, json, os, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()

    # Şimdilik: dosya var mı + hızlı metadata placeholder.
    # OCR/pipeline mini-run’a bağlamak için PipelineRunner API’nıza bağlayacağız.
    result = {
        "video": args.video,
        "ok": True,
        "elapsed_s": round(time.time() - t0, 3),
        "note": "Şu an placeholder. Pipeline mini-run’a bağlanacak."
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()