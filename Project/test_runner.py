"""
test_runner.py — Batch test runner with progress tracking.

Usage:
    python test_runner.py video1.mp4 video2.mp4
    python test_runner.py --config test_config.json F:\test\test.mp4
    python test_runner.py --folder F:\test_videos\
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Silence warnings
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddleocr").setLevel(logging.ERROR)

from config.runtime_paths import FFMPEG_BIN_DIR
from core.pipeline_runner import PipelineRunner


class TestRunner:
    """Batch video test runner with progress tracking."""

    def __init__(self, config_path: str = None, output_root: str = ""):
        self.config = self._load_config(config_path)
        self.output_root = output_root or self.config.get("output_root", "")
        self.results = []
        self.start_time = None

        ffmpeg = str(Path(FFMPEG_BIN_DIR) / "ffmpeg.exe")
        ffprobe = str(Path(FFMPEG_BIN_DIR) / "ffprobe.exe")

        self.runner = PipelineRunner(
            ffmpeg, ffprobe,
            config=self.config,
            output_root=self.output_root
        )
        self.runner.set_log_callback(self._log_callback)

    def _load_config(self, config_path: str) -> dict:
        default_config = {
            "scope": "video+audio",
            "first_min": 1.0,
            "last_min": 1.0,
            "difficulty": "medium",
            "use_gpu": True,
            "ocr_fps": 1.0,
            "program_type": "film_dizi",
            "audio_stages": ["extract", "transcribe"],
            "output_root": "",
        }

        if config_path and Path(config_path).is_file():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def _log_callback(self, msg: str):
        try:
            if "✓" in msg or "OK:" in msg or "TAMAMLANDI" in msg:
                print(f"\033[92m{msg}\033[0m")
            elif "!!" in msg or "HATA" in msg or "ERROR" in msg:
                print(f"\033[91m{msg}\033[0m")
            elif "→" in msg or "[Audio]" in msg:
                print(f"\033[94m{msg}\033[0m")
            else:
                print(msg)
        except Exception:
            print(msg)

    def run_batch(self, video_paths: List[str]) -> Dict:
        self.start_time = time.time()
        total_videos = len(video_paths)

        print("\n" + "="*70)
        print(f"  🎬 BATCH TEST RUNNER")
        print("="*70)
        print(f"  Videos: {total_videos}")
        print(f"  Config: {self.config.get('scope')} | "
              f"{self.config.get('first_min')}min + {self.config.get('last_min')}min")
        print(f"  Mode: {self.config.get('difficulty')}")
        if self.output_root:
            print(f"  Output: {self.output_root}")
        print("="*70 + "\n")

        for idx, video_path in enumerate(video_paths, 1):
            self._run_single_video(video_path, idx, total_videos)

        return self._generate_summary()

    def _run_single_video(self, video_path: str, idx: int, total: int):
        video_name = Path(video_path).name

        print("\n" + "─"*70)
        print(f"📹 [{idx}/{total}] {video_name}")
        print("─"*70)

        if not Path(video_path).is_file():
            print(f"❌ File not found: {video_path}")
            self.results.append({
                "video": video_name,
                "status": "error",
                "error": "File not found",
                "duration": 0
            })
            return

        t0 = time.time()

        try:
            result = self.runner.run(
                video_path=video_path,
                scope=self.config.get("scope", "video+audio"),
                first_min=self.config.get("first_min", 1.0),
                last_min=self.config.get("last_min", 1.0),
            )

            elapsed = time.time() - t0

            self.results.append({
                "video": video_name,
                "status": "success",
                "duration": round(elapsed, 2),
                "work_dir": result.get("work_dir"),
                "report_json": result.get("report_json"),
                "report_txt": result.get("report_txt"),
                "ocr_lines": result.get("ocr_lines"),
                "actors": result.get("credits", {}).get("total_actors", 0),
                "crew": result.get("credits", {}).get("total_crew", 0),
                "audio_segments": len(result.get("audio_result", {}).get("transcript", [])),
            })

            print(f"\n✓ Başarılı ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n✗ Hata: {e}")

            import traceback
            traceback.print_exc()

            self.results.append({
                "video": video_name,
                "status": "error",
                "error": str(e),
                "duration": round(elapsed, 2)
            })

    def _generate_summary(self) -> Dict:
        total_elapsed = time.time() - self.start_time
        success_count = sum(1 for r in self.results if r["status"] == "success")
        error_count = len(self.results) - success_count

        print("\n" + "="*70)
        print("  📊 SUMMARY REPORT")
        print("="*70)
        print(f"  Total Videos : {len(self.results)}")
        print(f"  ✓ Success    : {success_count}")
        if error_count > 0:
            print(f"  ✗ Failed     : {error_count}")
        print(f"  Total Time   : {total_elapsed:.1f}s")
        if len(self.results) > 0:
            print(f"  Avg Time     : {total_elapsed/len(self.results):.1f}s per video")
        print("="*70 + "\n")

        if success_count > 0:
            print("📋 Successful Tests:")
            for r in self.results:
                if r["status"] == "success":
                    print(f"  ✓ {r['video']} ({r['duration']}s)")
                    print(f"    → Actors: {r['actors']} | Crew: {r['crew']} | "
                          f"Audio: {r['audio_segments']} segments")

        if error_count > 0:
            print("\n❌ Failed Tests:")
            for r in self.results:
                if r["status"] == "error":
                    print(f"  ✗ {r['video']}: {r['error']}")

        log_dir = self.output_root if self.output_root else "."
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_videos": len(self.results),
                "success": success_count,
                "failed": error_count,
                "total_time": round(total_elapsed, 2),
                "config": self.config,
                "results": self.results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Detailed log saved: {log_path}\n")

        return {
            "success": success_count,
            "failed": error_count,
            "results": self.results
        }


def main():
    # Windows console UTF-8 fix
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    import argparse

    parser = argparse.ArgumentParser(description="Batch video test runner")
    parser.add_argument('videos', nargs='*', help='Video files to process')
    parser.add_argument('--folder', help='Process all videos in folder')
    parser.add_argument('--config', help='Configuration JSON file')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--pattern', default='*.mp4', help='File pattern for folder mode')

    args = parser.parse_args()

    video_files = []

    if args.folder:
        folder = Path(args.folder)
        if folder.is_dir():
            video_files = list(folder.glob(args.pattern))
        else:
            print(f"Error: Folder not found: {args.folder}")
            sys.exit(1)
    elif args.videos:
        video_files = [Path(v) for v in args.videos]
    else:
        print("Error: No videos specified. Use --help for usage.")
        sys.exit(1)

    if not video_files:
        print("Error: No video files found.")
        sys.exit(1)

    output_root = ""
    if args.output:
        output_root = args.output
    elif args.folder:
        output_root = args.folder
    elif len(video_files) > 0:
        output_root = str(Path(video_files[0]).parent)

    runner = TestRunner(config_path=args.config, output_root=output_root)
    summary = runner.run_batch([str(v) for v in video_files])

    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
