import os
import sys
from pathlib import Path

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def test_queue_manager_ensure_videos_requeues_completed_item(tmp_path):
    from core.queue_manager import VideoQueueManager, VideoStatus

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake")

    qm = VideoQueueManager()
    assert qm.add_videos([str(video)]) == 1
    qm.mark_done(str(video.resolve()), 12.5)

    changed = qm.ensure_videos([str(video)])
    item = qm.items[0]

    assert changed == 1
    assert item.status == VideoStatus.PENDING
    assert item.duration_sec is None
    assert item.error_msg == ""


def test_queue_manager_ensure_videos_keeps_processing_item(tmp_path):
    from core.queue_manager import VideoQueueManager, VideoStatus

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake")

    qm = VideoQueueManager()
    assert qm.add_videos([str(video)]) == 1
    qm.mark_processing(str(video.resolve()))

    changed = qm.ensure_videos([str(video)])
    item = qm.items[0]

    assert changed == 0
    assert item.status == VideoStatus.PROCESSING


def test_main_window_source_has_video_selector_and_start_button():
    source = (Path(_project_dir) / "ui" / "main_window.py").read_text(encoding="utf-8")

    assert "self.video_edit = QLineEdit()" in source
    assert 'self.start_single_btn = QPushButton("▶ Seçili Videoyu Başlat")' in source
    assert "self.pick_video_btn.clicked.connect(self._pick_video)" in source
    assert "self.start_single_btn.clicked.connect(self._on_start)" in source


def test_main_window_source_autostarts_selected_startup_video():
    source = (Path(_project_dir) / "ui" / "main_window.py").read_text(encoding="utf-8")
    main_source = (Path(_project_dir) / "main.py").read_text(encoding="utf-8")

    assert "def load_startup_video" in source
    assert "QTimer.singleShot(0, self._on_start)" in source
    assert "window.load_startup_video(vp, autostart=True)" in main_source
