import argparse, json, os, subprocess, shutil

def ffprobe_json(video_path: str):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"ffprobe": "missing"}

    cmd = [
        ffprobe, "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {"ffprobe": "error", "stderr": p.stderr[-500:]}
    return {"ffprobe": "ok", "raw": p.stdout}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vp = args.video
    if not os.path.exists(vp):
        raise SystemExit(f"VIDEO not found: {vp}")

    meta = {
        "input": vp,
        "ok": True,
        "size_bytes": os.path.getsize(vp),
        "probe": ffprobe_json(vp),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()