"""
test_oneocr_frame_meta.py — OneOCREngine Phase 1 metadata cache testleri.
"""

import os
import sys
from unittest.mock import MagicMock, patch

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core.oneocr_engine import FrameMeta, OneOCREngine


def test_process_frames_populates_last_frame_meta():
    """Phase 1 sonunda frame metadata cache'i dolmalı ve tuple imza korunmalı."""
    engine = object.__new__(OneOCREngine)
    engine.cfg = {}
    engine._log = lambda m: None
    engine._name_db = None
    engine._model = MagicMock()
    engine._last_frame_meta = {}
    engine._extract_layout_pairs = MagicMock(return_value=[])
    engine._estimate_font_type_from_data = MagicMock(return_value="decorative")
    engine._read_frame_with_image = MagicMock(return_value=(
        object(),
        [{
            "text": "KATHLEEN HERBERT",
            "confidence": 0.8,
            "bbox": [0, 0, 120, 30],
            "words": [{"text": "KATHLEEN", "confidence": 0.8, "bbox": [0, 0, 50, 30]}],
        }],
    ))

    frames = [{"path": "/tmp/frame_a.png", "timecode_sec": 1.25}]

    with patch("pathlib.Path.exists", return_value=True):
        ocr_lines, layout_pairs = engine.process_frames(frames, log_callback=lambda m: None)

    assert isinstance(ocr_lines, list)
    assert isinstance(layout_pairs, list)
    assert len(ocr_lines) == 1
    assert "/tmp/frame_a.png" in engine._last_frame_meta

    meta = engine._last_frame_meta["/tmp/frame_a.png"]
    assert isinstance(meta, FrameMeta)
    assert meta.has_text is True
    assert meta.line_count == 1
    assert meta.avg_confidence == 0.8
    assert meta.font_type == "decorative"
    engine._read_frame_with_image.assert_called_once_with("/tmp/frame_a.png")
