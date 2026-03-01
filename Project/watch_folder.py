"""
watch_folder.py — Automatic video test runner with folder monitoring.

Usage:
    python watch_folder.py <YOUR_TEST_DIR>
    python watch_folder.py <YOUR_TEST_DIR> --config test_config.json --interval 10
    python watch_folder.py <YOUR_TEST_DIR> --once (test existing then exit)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Set

from test_runner import TestRunner

VIDEO_EXTENSIONS = ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.ts")


class FolderWatcher:
    """Watches a folder for new video files and automatically tests them."""

    def __init__(self, watch_dir: str, config_path: str = None,
                 interval: int = 30, auto_test_existing: bool = False):
        self.watch_dir = Path(watch_dir)
        self.config_path = config_path
        self.interval = interval
        self.processed_files: Set[str] = set()
        self.runner = TestRunner(config_path=config_path, output_root=str(self.watch_dir))

        if not self.watch_dir.is_dir():
            raise ValueError(f"Watch directory not found: {watch_dir}")

        # Load processed files log
        self.log_file = self.watch_dir / ".processed_videos.json"
        self._load_processed_log()

        # Optionally add existing files to processed set
        if not auto_test_existing:
            for ext in VIDEO_EXTENSIONS:
                for video in self.watch_dir.glob(ext):
                    if video.is_file():
                        self.processed_files.add(str(video.absolute()))

    def _load_processed_log(self):
        """Load list of already processed videos."""
        if self.log_file.is_file():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get("processed", []))
            except (json.JSONDecodeError, OSError):
                pass

    def _save_processed_log(self):
        """Save list of processed videos."""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "last_updated": datetime.now().isoformat(),
                    "processed": list(self.processed_files)
                }, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def scan_for_new_videos(self) -> list:
        """Scan folder for new unprocessed video files."""
        new_videos = []

        for ext in VIDEO_EXTENSIONS:
            for video_path in self.watch_dir.glob(ext):
                if not video_path.is_file():
                    continue

                abs_path = str(video_path.absolute())

                if abs_path not in self.processed_files:
                    new_videos.append(abs_path)

        return new_videos

    def process_video(self, video_path: str):
        """Process a single video file."""
        video_name = Path(video_path).name

        print("\n" + "="*70)
        print(f"NEW VIDEO DETECTED: {video_name}")
        print("="*70)

        try:
            self.runner.run_batch([video_path])
            self.processed_files.add(video_path)
            self._save_processed_log()
            print(f"\nSuccessfully processed: {video_name}")
        except Exception as e:
            print(f"\nError processing {video_name}: {e}")
            import traceback
            traceback.print_exc()

    def watch_once(self):
        """Scan once and process any new videos, then exit."""
        print("\n" + "="*70)
        print("SINGLE SCAN MODE")
        print("="*70)
        print(f"  Folder: {self.watch_dir}")
        print(f"  Config: {self.config_path or 'default'}")
        print("="*70 + "\n")

        new_videos = self.scan_for_new_videos()

        if not new_videos:
            print("No new videos found.")
            return

        print(f"Found {len(new_videos)} new video(s):\n")
        for v in new_videos:
            print(f"  - {Path(v).name}")
        print()

        for video in new_videos:
            self.process_video(video)

    def watch_continuous(self):
        """Continuously watch folder for new videos."""
        print("\n" + "="*70)
        print("FOLDER WATCH MODE - ACTIVE")
        print("="*70)
        print(f"  Folder   : {self.watch_dir}")
        print(f"  Config   : {self.config_path or 'default'}")
        print(f"  Interval : {self.interval}s")
        print(f"  Processed: {len(self.processed_files)} video(s)")
        print("="*70)
        print("\nPress Ctrl+C to stop...\n")

        try:
            while True:
                new_videos = self.scan_for_new_videos()

                if new_videos:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Detected {len(new_videos)} new video(s)")

                    for video in new_videos:
                        self.process_video(video)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Watching... (processed: {len(self.processed_files)})")

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("Watch stopped by user")
            print("="*70)
            print(f"  Total processed: {len(self.processed_files)} video(s)")
            print("="*70 + "\n")


def main():
    # Windows console UTF-8 fix
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description="Watch folder for new videos and automatically test them"
    )
    parser.add_argument('folder', help='Folder to watch for new videos')
    parser.add_argument('--config', help='Test configuration JSON file')
    parser.add_argument('--interval', type=int, default=30,
                        help='Scan interval in seconds (default: 30)')
    parser.add_argument('--once', action='store_true',
                        help="Scan once and exit (don't watch continuously)")
    parser.add_argument('--test-existing', action='store_true',
                        help='Also test existing videos on first run')

    args = parser.parse_args()

    try:
        watcher = FolderWatcher(
            watch_dir=args.folder,
            config_path=args.config,
            interval=args.interval,
            auto_test_existing=args.test_existing
        )

        if args.once:
            watcher.watch_once()
        else:
            watcher.watch_continuous()

    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
