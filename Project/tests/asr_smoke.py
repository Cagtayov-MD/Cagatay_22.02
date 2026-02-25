import argparse, json, os, wave, hashlib

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wav_path = args.wav
    if not os.path.exists(wav_path):
        raise SystemExit(f"WAV not found: {wav_path}")

    with wave.open(wav_path, "rb") as w:
        channels = w.getnchannels()
        sr = w.getframerate()
        frames = w.getnframes()
        dur = frames / float(sr) if sr else 0.0
        sampwidth = w.getsampwidth()

    result = {
        "input": wav_path,
        "ok": True,
        "channels": channels,
        "sample_rate": sr,
        "sample_width": sampwidth,
        "frames": frames,
        "duration_sec": round(dur, 3),
        "sha256": sha256_file(wav_path),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()