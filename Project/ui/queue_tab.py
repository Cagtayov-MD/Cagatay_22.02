"""queue_tab.py — Video Kuyruk Sekmesi (PySide6 QTableWidget)."""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QLabel, QProgressBar, QAbstractItemView,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor

from core.queue_manager import VideoQueueManager, VideoStatus

# Durum ikonları ve renkleri
STATUS_DISPLAY = {
    VideoStatus.PENDING:    ("⏳", "Bekliyor",   "#888888"),
    VideoStatus.PROCESSING: ("🔄", "İşleniyor",  "#FFD700"),
    VideoStatus.DONE:       ("✅", "Bitti",       "#22b33a"),
    VideoStatus.ERROR:      ("❌", "Hata",        "#e94560"),
}


class QueueTab(QWidget):
    """Sayfa 2 — Video Kuyruk Yönetimi."""

    # Sinyaller
    start_queue = Signal()    # Başlat butonuna basıldığında
    stop_queue = Signal()     # Durdur butonuna basıldığında
    skip_current = Signal()   # Aktif videoyu atla

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue = VideoQueueManager()
        self._build_ui()

    @property
    def queue_manager(self) -> VideoQueueManager:
        return self._queue

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Üst butonlar ──
        btn_row = QHBoxLayout()

        self.folder_btn = QPushButton("📁 Klasör Seç")
        self.folder_btn.clicked.connect(self._pick_folder)
        btn_row.addWidget(self.folder_btn)

        self.file_btn = QPushButton("📄 Dosya Ekle")
        self.file_btn.clicked.connect(self._pick_files)
        btn_row.addWidget(self.file_btn)

        self.remove_btn = QPushButton("🗑️ Seçili Sil")
        self.remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.remove_btn)

        self.clear_btn = QPushButton("🧹 Temizle")
        self.clear_btn.clicked.connect(self._clear_completed)
        btn_row.addWidget(self.clear_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Tablo ──
        # 4 sütun: İkon | Dosya Adı | Durum | Süre
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "Dosya Adı", "Durum", "Süre"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)        # ikon
        header.setSectionResizeMode(1, QHeaderView.Stretch)      # dosya adı
        header.setSectionResizeMode(2, QHeaderView.Fixed)        # durum
        header.setSectionResizeMode(3, QHeaderView.Fixed)        # süre
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 100)

        layout.addWidget(self.table)

        # ── Progress bar ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%v/%m  (%p%)")
        layout.addWidget(self.progress)

        # ── Alt butonlar + durum ──
        bottom_row = QHBoxLayout()

        self.start_btn = QPushButton("▶ Başlat")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.start_queue.emit)
        bottom_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏸ Durdur")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_queue.emit)
        bottom_row.addWidget(self.stop_btn)

        self.skip_btn = QPushButton("⏭ Geç")
        self.skip_btn.setEnabled(False)
        self.skip_btn.clicked.connect(self.skip_current.emit)
        bottom_row.addWidget(self.skip_btn)

        bottom_row.addStretch()

        self.status_label = QLabel("Toplam: 0  |  Biten: 0  |  Kalan: 0")
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        bottom_row.addWidget(self.status_label)

        layout.addLayout(bottom_row)

    def _pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Video Klasörü Seç")
        if folder:
            self._queue.add_folder(folder)
            self._refresh_table()

    def _pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Video Dosyaları Seç", "",
            "Video (*.mp4 *.avi *.mkv *.mov *.ts *.wmv);;Tüm Dosyalar (*)"
        )
        if files:
            self._queue.add_videos(files)
            self._refresh_table()

    def _remove_selected(self):
        rows = set(index.row() for index in self.table.selectedIndexes())
        items = self._queue.items
        paths = [items[r].path for r in rows if r < len(items)]
        self._queue.remove(paths)
        self._refresh_table()

    def _clear_completed(self):
        self._queue.clear_completed()
        self._refresh_table()

    def _refresh_table(self):
        """Tabloyu queue_manager verisiyle güncelle."""
        items = self._queue.items
        self.table.setRowCount(len(items))

        for row, item in enumerate(items):
            icon, status_text, color = STATUS_DISPLAY[item.status]

            # Sütun 0: İkon
            icon_item = QTableWidgetItem(icon)
            icon_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, icon_item)

            # Sütun 1: Dosya adı (sadece dosya adı, tam path değil)
            name_item = QTableWidgetItem(Path(item.path).name)
            name_item.setToolTip(item.path)  # tam path tooltip'te
            self.table.setItem(row, 1, name_item)

            # Sütun 2: Durum
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QColor(color))
            status_item.setTextAlignment(Qt.AlignCenter)
            if item.status == VideoStatus.ERROR:
                status_item.setToolTip(item.error_msg)
            self.table.setItem(row, 2, status_item)

            # Sütun 3: Süre
            if item.duration_sec is not None:
                duration_str = f"{item.duration_sec:.0f}s"
            else:
                duration_str = "—"
            dur_item = QTableWidgetItem(duration_str)
            dur_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 3, dur_item)

        # Progress bar ve durum güncelle
        stats = self._queue.stats()
        total = stats["total"]
        done = stats["done"]
        pending = stats["pending"]

        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.progress.setFormat(f"{done}/{total}  ({(done * 100 // max(total, 1))}%)")

        self.status_label.setText(
            f"Toplam: {total}  |  Biten: {done}  |  Kalan: {pending}"
        )

    @Slot(str, object, object, str)
    def update_item_status(self, path: str, status: VideoStatus,
                           duration_sec: float | None = None, error_msg: str = ""):
        """Dışarıdan (pipeline worker'dan) bir video durumunu güncelle."""
        if status == VideoStatus.PROCESSING:
            self._queue.mark_processing(path)
        elif status == VideoStatus.DONE:
            self._queue.mark_done(path, duration_sec or 0)
        elif status == VideoStatus.ERROR:
            self._queue.mark_error(path, error_msg)
        self._refresh_table()

    @Slot(bool)
    def set_running(self, running: bool):
        """Kuyruk çalışma durumunu UI'a yansıt."""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.skip_btn.setEnabled(running)
        self.folder_btn.setEnabled(not running)
        self.file_btn.setEnabled(not running)
        self.remove_btn.setEnabled(not running)
