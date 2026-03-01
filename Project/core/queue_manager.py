"""queue_manager.py — Sıralı video kuyruk yönetimi."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".wmv"}


class VideoStatus(Enum):
    PENDING = "pending"       # ⏳ Bekliyor
    PROCESSING = "processing" # 🔄 İşleniyor
    DONE = "done"             # ✅ Bitti
    ERROR = "error"           # ❌ Hata


@dataclass
class VideoItem:
    path: str
    status: VideoStatus = VideoStatus.PENDING
    duration_sec: float | None = None   # tamamlanma süresi
    error_msg: str = ""
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())


class VideoQueueManager:
    def __init__(self):
        self._items: list[VideoItem] = []

    def add_videos(self, paths: list[str]) -> int:
        """Video dosyalarını kuyruğa ekle. Eklenen sayıyı döndür."""
        added = 0
        existing = {item.path for item in self._items}
        for p in paths:
            p = str(Path(p).resolve())
            ext = Path(p).suffix.lower()
            if ext in VIDEO_EXTENSIONS and p not in existing and os.path.isfile(p):
                self._items.append(VideoItem(path=p))
                existing.add(p)
                added += 1
        return added

    def add_folder(self, folder: str) -> int:
        """Klasördeki tüm video dosyalarını kuyruğa ekle (dosya adına göre sıralı)."""
        folder_path = Path(folder)
        if not folder_path.is_dir():
            return 0
        videos = sorted([
            str(f) for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ])
        return self.add_videos(videos)

    def get_next(self) -> VideoItem | None:
        """Sıradaki bekleyen videoyu döndür."""
        for item in self._items:
            if item.status == VideoStatus.PENDING:
                return item
        return None

    def mark_processing(self, path: str):
        item = self._find(path)
        if item:
            item.status = VideoStatus.PROCESSING

    def mark_done(self, path: str, duration_sec: float):
        item = self._find(path)
        if item:
            item.status = VideoStatus.DONE
            item.duration_sec = duration_sec

    def mark_error(self, path: str, reason: str):
        item = self._find(path)
        if item:
            item.status = VideoStatus.ERROR
            item.error_msg = reason

    def remove(self, paths: list[str]):
        """Seçili videoları kuyruktan sil (sadece PENDING olanlar)."""
        remove_set = set(paths)
        self._items = [
            item for item in self._items
            if item.path not in remove_set or item.status == VideoStatus.PROCESSING
        ]

    def clear_completed(self):
        """Tamamlanan ve hatalı videoları listeden sil."""
        self._items = [
            item for item in self._items
            if item.status in (VideoStatus.PENDING, VideoStatus.PROCESSING)
        ]

    def stats(self) -> dict:
        total = len(self._items)
        done = sum(1 for i in self._items if i.status == VideoStatus.DONE)
        error = sum(1 for i in self._items if i.status == VideoStatus.ERROR)
        pending = sum(1 for i in self._items if i.status == VideoStatus.PENDING)
        processing = sum(1 for i in self._items if i.status == VideoStatus.PROCESSING)
        return {"total": total, "done": done, "error": error, "pending": pending, "processing": processing}

    @property
    def items(self) -> list[VideoItem]:
        return list(self._items)

    def _find(self, path: str) -> VideoItem | None:
        for item in self._items:
            if item.path == path:
                return item
        return None
