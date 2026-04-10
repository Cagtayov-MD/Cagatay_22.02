import os
import sys
import unittest

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core.frame_extractor import (
    ENTRY_CREDITS_FPS,
    EXIT_CREDITS_FPS,
    FrameExtractor,
)


class _SpyFrameExtractor(FrameExtractor):
    def __init__(self):
        super().__init__("ffmpeg", "ffprobe")
        self.calls = []

    def extract_segment_frames(self, video_path, output_dir,
                               start_sec, duration_sec, fps=1.0, prefix="frame"):
        self.calls.append({
            "video_path": video_path,
            "output_dir": output_dir,
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "fps": fps,
            "prefix": prefix,
        })
        return [{
            "path": os.path.join(output_dir, f"{prefix}_000001.png"),
            "timecode_sec": start_sec,
            "index": 0,
            "segment": prefix,
        }]


class TestFrameExtractorSampling(unittest.TestCase):
    def test_entry_and_exit_use_fixed_sampling_rates(self):
        extractor = _SpyFrameExtractor()

        extractor.extract_credits_frames(
            video_path="film.mp4",
            output_dir="work_dir",
            video_info={"duration_seconds": 3600},
            first_min=6.0,
            last_min=10.0,
        )

        self.assertEqual(len(extractor.calls), 2)
        self.assertEqual(extractor.calls[0]["prefix"], "entry")
        self.assertEqual(extractor.calls[1]["prefix"], "exit")
        self.assertAlmostEqual(extractor.calls[0]["fps"], ENTRY_CREDITS_FPS)
        self.assertAlmostEqual(extractor.calls[1]["fps"], EXIT_CREDITS_FPS)
