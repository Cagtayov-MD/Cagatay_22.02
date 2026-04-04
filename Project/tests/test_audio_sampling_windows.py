import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestAudioSamplingWindows(unittest.TestCase):
    def _make_pipeline(self):
        from core.audio_pipeline import AudioPipeline

        return AudioPipeline(
            config={
                "video_path": "movie.mp4",
                "work_dir": os.path.join(tempfile.gettempdir(), "audio_sampling_windows"),
                "options": {"whisper_language": "tr"},
                "stages": ["detect_language", "extract", "transcribe"],
            },
            log_cb=lambda *args, **kwargs: None,
        )

    def test_builds_expected_windows_for_90_min_video(self):
        pipeline = self._make_pipeline()

        windows = pipeline._build_asr_windows(
            5400.0,
            {"first": 10, "middle": 15, "last": 15},
        )

        self.assertEqual(
            [(w["label"], w["start_sec"], w["end_sec"]) for w in windows],
            [
                ("first", 0.0, 600.0),
                ("middle", 2250.0, 3150.0),
                ("last", 4500.0, 5400.0),
            ],
        )
        self.assertEqual(
            [w["merged_from"] for w in windows],
            [["first"], ["middle"], ["last"]],
        )

    def test_merges_overlapping_windows_for_30_min_video(self):
        pipeline = self._make_pipeline()

        windows = pipeline._build_asr_windows(
            1800.0,
            {"first": 10, "middle": 15, "last": 15},
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["label"], "first+middle+last")
        self.assertEqual(windows[0]["start_sec"], 0.0)
        self.assertEqual(windows[0]["end_sec"], 1800.0)
        self.assertEqual(windows[0]["merged_from"], ["first", "middle", "last"])

    @patch("core.audio_pipeline.TranscribeStage")
    @patch("core.audio_pipeline.ExtractStage")
    @patch("core.audio_pipeline.LanguageDetectionStage")
    def test_sampling_offsets_segments_and_records_windows(
        self,
        mock_lang_stage,
        mock_extract_stage,
        mock_transcribe_stage,
    ):
        mock_lang_stage.return_value.run.return_value = {
            "detected_language": "tr",
            "language_is_turkish": True,
            "confidence": 0.95,
            "samples": [],
            "selected_channel": None,
            "stage_time_sec": 0.1,
            "status": "ok",
        }

        extract_calls = []

        def fake_extract(video_path, audio_dir, **kwargs):
            extract_calls.append(kwargs)
            prefix = kwargs["output_prefix"]
            return {
                "status": "ok",
                "wav_16k": f"/tmp/{prefix}_16k.wav",
                "wav_48k": f"/tmp/{prefix}_48k.wav",
                "duration_sec": kwargs["max_duration_sec"],
                "selected_channel": kwargs.get("selected_channel"),
                "stage_time_sec": 0.2,
            }

        transcribe_calls = {"n": 0}

        def fake_transcribe(audio_path, diarization, options):
            transcribe_calls["n"] += 1
            return {
                "stage_time_sec": 0.3,
                "status": "ok",
                "segments": [
                    {
                        "start": 1.0,
                        "end": 3.0,
                        "text": f"segment-{transcribe_calls['n']}",
                        "speaker": "S1",
                        "words": [{"word": "hello", "start": 1.0, "end": 1.5}],
                    }
                ],
                "total_segments": 1,
                "detected_language": "tr",
            }

        mock_extract_stage.return_value.run.side_effect = fake_extract
        mock_transcribe_stage.return_value.run.side_effect = fake_transcribe

        from core.audio_pipeline import AudioPipeline

        pipeline = AudioPipeline(
            config={
                "video_path": "movie.mp4",
                "work_dir": os.path.join(tempfile.gettempdir(), "audio_sampling_windows"),
                "stages": ["detect_language", "extract", "transcribe"],
                "options": {
                    "whisper_language": "tr",
                    "asr_sampling_mode": "first_middle_last",
                    "asr_window_minutes": {"first": 10, "middle": 15, "last": 15},
                },
                "ffprobe": "ffprobe",
            },
            log_cb=lambda *args, **kwargs: None,
        )

        with patch.object(AudioPipeline, "_probe_video_duration", return_value=5400.0):
            result = pipeline.run()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            [(w["label"], w["start_sec"], w["end_sec"]) for w in result["asr_windows"]],
            [
                ("first", 0.0, 600.0),
                ("middle", 2250.0, 3150.0),
                ("last", 4500.0, 5400.0),
            ],
        )
        self.assertEqual(len(extract_calls), 3)
        self.assertEqual(
            [(c["start_offset_sec"], c["max_duration_sec"]) for c in extract_calls],
            [(0.0, 600.0), (2250.0, 900.0), (4500.0, 900.0)],
        )
        self.assertEqual(
            [seg["start"] for seg in result["transcript"]],
            [1.0, 2251.0, 4501.0],
        )
        self.assertEqual(
            [seg["words"][0]["start"] for seg in result["transcript"]],
            [1.0, 2251.0, 4501.0],
        )
        self.assertEqual(result["stages"]["extract"]["windows"], 3)
        self.assertEqual(result["stages"]["transcribe"]["segments"], 3)

    @patch("core.audio_pipeline.TranscribeStage")
    @patch("core.audio_pipeline.ExtractStage")
    @patch("core.audio_pipeline.LanguageDetectionStage")
    def test_sampling_disabled_keeps_single_full_extract(
        self,
        mock_lang_stage,
        mock_extract_stage,
        mock_transcribe_stage,
    ):
        mock_lang_stage.return_value.run.return_value = {
            "detected_language": "tr",
            "language_is_turkish": True,
            "confidence": 0.95,
            "samples": [],
            "selected_channel": None,
            "stage_time_sec": 0.1,
            "status": "ok",
        }
        mock_extract_stage.return_value.run.return_value = {
            "status": "ok",
            "wav_16k": "/tmp/full_16k.wav",
            "wav_48k": "/tmp/full_48k.wav",
            "duration_sec": 5400.0,
            "selected_channel": None,
            "stage_time_sec": 0.2,
        }
        mock_transcribe_stage.return_value.run.return_value = {
            "stage_time_sec": 0.3,
            "status": "ok",
            "segments": [{"start": 1.0, "end": 2.0, "text": "full", "speaker": "S1"}],
            "total_segments": 1,
            "detected_language": "tr",
        }

        from core.audio_pipeline import AudioPipeline

        pipeline = AudioPipeline(
            config={
                "video_path": "movie.mp4",
                "work_dir": os.path.join(tempfile.gettempdir(), "audio_sampling_windows"),
                "stages": ["detect_language", "extract", "transcribe"],
                "options": {"whisper_language": "tr"},
            },
            log_cb=lambda *args, **kwargs: None,
        )

        result = pipeline.run()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["asr_windows"], [])
        mock_extract_stage.return_value.run.assert_called_once_with(
            "movie.mp4",
            os.path.join(tempfile.gettempdir(), "audio_sampling_windows", "audio_work"),
            selected_channel=None,
            max_duration_sec=None,
        )
        self.assertEqual(result["transcript"][0]["start"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
