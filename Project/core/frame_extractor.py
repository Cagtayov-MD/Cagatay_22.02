"""
frame_extractor.py — FFmpeg: ingest + split_av + frame extraction.
FIX: subprocess returncode + stderr logging.
FIX: Kısa video koruması (%40 → %15 cap).
"""
import subprocess, json, os
from pathlib import Path
from datetime import timedelta


class FrameExtractor:
    def __init__(self, ffmpeg_path, ffprobe_path, log_cb=None):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
        self._log = log_cb or (lambda m: None)

    def _run(self, cmd, timeout=300, label="cmd"):
        """subprocess çalıştır + returncode/stderr kontrol et."""
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                encoding="utf-8", errors="replace"
            )
            if r.returncode != 0:
                err = r.stderr.strip()[-300:] if r.stderr else "bilinmeyen hata"
                self._log(f"  ⚠️ {label} returncode={r.returncode}: {err}")
            return r
        except subprocess.TimeoutExpired:
            self._log(f"  ❌ {label} timeout ({timeout}s aşıldı)")
            return None
        except Exception as e:
            self._log(f"  ❌ {label} hata: {e}")
            return None

    def probe_video(self, video_path: str) -> dict:
        r = self._run(
            [self.ffprobe, "-v","quiet", "-print_format","json",
             "-show_format", "-show_streams", video_path],
            timeout=30, label="ffprobe"
        )
        if r is None or r.returncode != 0:
            raise RuntimeError(f"FFprobe hatası: {video_path}")

        info = json.loads(r.stdout)
        vs = next((s for s in info.get("streams",[]) if s.get("codec_type")=="video"), None)
        aus = next((s for s in info.get("streams",[]) if s.get("codec_type")=="audio"), None)
        if not vs and not aus:
            raise RuntimeError("Video veya ses stream bulunamadı!")

        dur = float(info["format"].get("duration", 0))

        # Saf ses dosyası (audio_only) — video stream yok, normal
        if not vs:
            return {
                "filename": Path(video_path).name, "filepath": str(video_path),
                "filesize_bytes": int(info["format"].get("size", 0)),
                "duration_seconds": dur,
                "duration_human": str(timedelta(seconds=int(dur))),
                "resolution": "0x0",
                "width": 0, "height": 0, "fps": 0.0,
                "audio_channels": int(aus.get("channels", 0)),
                "audio_sample_rate": int(aus.get("sample_rate", 0)),
                "has_audio": True, "has_video": False,
                "codec_video": "none",
                "codec_audio": aus.get("codec_name", "?"),
            }

        fp = vs.get("r_frame_rate","24/1").split("/")
        fps = float(fp[0])/float(fp[1]) if len(fp)==2 else 24.0

        return {
            "filename": Path(video_path).name, "filepath": str(video_path),
            "filesize_bytes": int(info["format"].get("size",0)),
            "duration_seconds": dur,
            "duration_human": str(timedelta(seconds=int(dur))),
            "resolution": f"{vs.get('width',0)}x{vs.get('height',0)}",
            "width": int(vs.get("width",0)), "height": int(vs.get("height",0)),
            "fps": round(fps,2),
            "audio_channels": int(aus.get("channels",0)) if aus else 0,
            "audio_sample_rate": int(aus.get("sample_rate",0)) if aus else 0,
            "has_audio": aus is not None, "has_video": True,
            "codec_video": vs.get("codec_name","?"),
            "codec_audio": aus.get("codec_name","?") if aus else "none",
        }

    def extract_audio(self, video_path, output_dir) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, "audio.wav")
        self._run(
            [self.ffmpeg, "-y", "-i", video_path, "-vn",
             "-acodec","pcm_s16le", "-ar","16000", "-ac","1", out],
            timeout=600, label="audio_extract"
        )
        return out if os.path.exists(out) and os.path.getsize(out) > 0 else None

    def extract_segment_frames(self, video_path, output_dir,
                                start_sec, duration_sec, fps=1.0, prefix="frame"):
        os.makedirs(output_dir, exist_ok=True)
        pattern = os.path.join(output_dir, f"{prefix}_%06d.png")

        r = self._run(
            [self.ffmpeg, "-y", "-ss", str(start_sec), "-i", video_path,
             "-t", str(duration_sec), "-vf", f"fps={fps}", "-q:v","2", pattern],
            timeout=600, label=f"frame_extract_{prefix}"
        )

        if r is None:
            return []

        frames = []
        for i, fp in enumerate(sorted(Path(output_dir).glob(f"{prefix}_*.png"))):
            frames.append({"path":str(fp), "timecode_sec": round(start_sec+i/fps, 3),
                           "index":i, "segment":prefix})

        if not frames:
            self._log(f"  ⚠️ {prefix}: 0 frame çıkarıldı (FFmpeg başarısız olmuş olabilir)")

        return frames

    def extract_credits_frames(self, video_path, output_dir, video_info,
                                first_min=4.0, last_min=8.0, fps=1.0):
        dur = video_info["duration_seconds"]

        # ── KISA VİDEO KORUMASI ──
        # %40 yerine %15 cap: 2dk videoda max 18sn (klip/teaser koruması)
        # Normal film (120dk) için sınır: min(4dk=240s, 120*0.15=1080s) = 240s — etkilenmez
        ratio_cap = 0.15
        first_sec = min(first_min * 60, dur * ratio_cap)
        last_sec = min(last_min * 60, dur * ratio_cap)

        # En az 10 saniye olsun
        first_sec = max(first_sec, min(10, dur * 0.1))
        last_sec = max(last_sec, min(10, dur * 0.1))

        exit_start = max(0, dur - last_sec)

        # Giriş ve çıkış çakışmasını önle
        if exit_start < first_sec:
            # Video çok kısa, tek geçiş yap
            self._log(f"  ⚠️ Video çok kısa ({dur:.0f}s), tek geçiş yapılıyor")
            first_sec = dur * 0.3
            last_sec = dur * 0.5
            exit_start = max(0, dur - last_sec)

        self._log(f"  📽️ Giriş: 0:00 → {timedelta(seconds=int(first_sec))}")
        self._log(f"  📽️ Çıkış: {timedelta(seconds=int(exit_start))} → {timedelta(seconds=int(dur))}")

        entry = self.extract_segment_frames(video_path,
            os.path.join(output_dir, "entry_frames"), 0, first_sec, fps, "entry")
        exit_ = self.extract_segment_frames(video_path,
            os.path.join(output_dir, "exit_frames"), exit_start, last_sec, fps, "exit")

        return {"entry": entry, "exit": exit_, "total": len(entry)+len(exit_),
                "first_sec": first_sec, "last_sec": last_sec, "exit_start": exit_start}

    def cut_segment(self, video_path, output_path, start_sec, duration_sec):
        self._run(
            [self.ffmpeg, "-y", "-ss", str(start_sec), "-i", video_path,
             "-t", str(duration_sec), "-c","copy", output_path],
            timeout=120, label="cut_segment"
        )
        return output_path if os.path.exists(output_path) else None
