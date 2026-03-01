"""
main_window.py — PySide6 Dark Theme Ana Pencere

Değişiklikler v3.0:
- İçerik profili sistemi: scope/mod/segment slider'lar kaldırıldı, profil JSON'dan okunuyor
- content_combo → Kontrol Kulesi'ne taşındı (FilmDizi, Spor, StudyoProgram, MuzikProgram, KisaHaber)
- Sistem bilgisi widget: CPU %, GPU %, RAM GB, Saat, Tarih
- DAG Widget entegrasyonu: profil bazlı pipeline diyagramı
- QTabWidget: Sekme 1 = Ana UI, Sekme 2 = (ileride doldurulacak)
- Yeni layout: başlık + sistem bilgisi (üst), Splitter sol/sağ (alt)
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QLineEdit,
    QTextEdit, QProgressBar, QFileDialog, QSplitter, QFrame,
    QApplication, QMessageBox, QTabWidget,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QFont

from config.runtime_paths import API_KEYS_JSON, FFMPEG_BIN_DIR, GOOGLE_KEYS_JSON, LOGOLAR_DIR
from ui.queue_tab import QueueTab
from core.queue_manager import VideoStatus


# ═══════════════════════════════════════════════════════════════════
# DARK THEME STYLESHEET
# ═══════════════════════════════════════════════════════════════════
DARK_STYLE = """
QMainWindow { background-color: #1a1a2e; }
QWidget { background-color: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI'; font-size: 13px; }

QGroupBox {
    background-color: #16213e; border: 1px solid #0f3460;
    border-radius: 6px; margin-top: 14px; padding: 12px 8px 8px 8px;
    font-weight: bold; color: #e94560;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
    padding: 2px 10px; color: #e94560; }

QPushButton {
    background-color: #0f3460; color: #e0e0e0; border: 1px solid #533483;
    border-radius: 4px; padding: 8px 16px; font-weight: bold; min-height: 28px;
}
QPushButton:hover { background-color: #533483; border-color: #e94560; }
QPushButton:pressed { background-color: #e94560; }
QPushButton:disabled { background-color: #2a2a3e; color: #555; border-color: #333; }

QPushButton#startBtn {
    background-color: #1b8a2e; border-color: #22b33a; font-size: 15px; min-height: 40px;
}
QPushButton#startBtn:hover { background-color: #22b33a; }
QPushButton#startBtn:disabled { background-color: #2a3a2e; color: #555; }
QPushButton#stopSafeBtn { background-color: #b8860b; border-color: #daa520; }
QPushButton#stopSafeBtn:hover { background-color: #daa520; }
QPushButton#stopHardBtn { background-color: #8b0000; border-color: #e94560; }
QPushButton#stopHardBtn:hover { background-color: #e94560; }

QComboBox {
    background-color: #0f3460; border: 1px solid #533483;
    border-radius: 4px; padding: 4px 8px; min-height: 24px;
}
QComboBox:hover { border-color: #e94560; }
QComboBox QAbstractItemView { background-color: #16213e; color: #e0e0e0;
    selection-background-color: #533483; }

QSlider::groove:horizontal { height: 6px; background: #0f3460; border-radius: 3px; }
QSlider::handle:horizontal { background: #e94560; width: 16px; height: 16px;
    margin: -5px 0; border-radius: 8px; }
QSlider::sub-page:horizontal { background: #533483; border-radius: 3px; }

QLineEdit {
    background-color: #0f3460; border: 1px solid #533483;
    border-radius: 4px; padding: 4px 8px; color: #e0e0e0;
}
QLineEdit:focus { border-color: #e94560; }
QLineEdit:read-only { background-color: #1a1a2e; color: #888; }

QTextEdit {
    background-color: #0d1117; border: 1px solid #0f3460;
    border-radius: 4px; color: #c0c0c0;
    font-family: 'Consolas'; font-size: 12px; padding: 4px;
}

QProgressBar {
    background-color: #0f3460; border: 1px solid #533483;
    border-radius: 4px; text-align: center; color: #e0e0e0; min-height: 22px;
}
QProgressBar::chunk { background-color: #e94560; border-radius: 3px; }

QLabel { color: #e0e0e0; }
QLabel#titleLabel { font-size: 18px; font-weight: bold; color: #e94560; }
QLabel#subtitleLabel { font-size: 11px; color: #888; }
QLabel#statValue { font-size: 16px; font-weight: bold; color: #e94560; }
QLabel#pathInfo { font-size: 11px; color: #5a8a5a; font-style: italic; }
"""


# ═══════════════════════════════════════════════════════════════════
# SİNYAL KÖPRÜSÜ (thread → UI)
# ═══════════════════════════════════════════════════════════════════
class PipelineSignals(QObject):
    log_message    = Signal(str)
    stage_update   = Signal(str, int)
    pipeline_done  = Signal(dict)
    pipeline_error = Signal(str)


# ═══════════════════════════════════════════════════════════════════
# ANA PENCERE
# ═══════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vitos v7")
        self.setMinimumSize(1100, 820)
        self.resize(1200, 900)

        self.video_path = ""
        self.output_dir = ""
        self.running    = False
        self.pipeline_thread = None
        self.stage_bars = {}
        self._queue_running = False
        self._queue_stop_requested = False
        self._queue_thread = None
        self._queue_profile_name = "FilmDizi"

        self.signals = PipelineSignals()
        self.signals.log_message.connect(self._append_log)
        self.signals.stage_update.connect(self._update_stage)
        self.signals.pipeline_done.connect(self._on_done)
        self.signals.pipeline_error.connect(self._on_error)

        self.config = self._load_config()
        self._build_ui()
        self.setStyleSheet(DARK_STYLE)
        self._resolve_and_show_paths()

    # ═══════════════════════════════════════════════════════════════
    # UI İNŞASI
    # ═══════════════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(10, 8, 10, 8)

        # ── Başlık satırı (sol: isim+subtitle, sağ: sistem bilgisi) ──
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title = QLabel("VİTOS")
        title.setObjectName("titleLabel")
        sub = QLabel("Video İşleme ve Tarama Otomasyonu — ASR · OCR · LLM")
        sub.setObjectName("subtitleLabel")
        title_col.addWidget(title)
        title_col.addWidget(sub)
        header.addLayout(title_col)
        header.addStretch()
        header.addWidget(self._build_system_info())
        root.addLayout(header)

        # ── QTabWidget ──────────────────────────────────────────────
        tabs = QTabWidget()

        # Sekme 1: Ana UI
        tab1 = QWidget()
        tab1_lay = QVBoxLayout(tab1)
        tab1_lay.setContentsMargins(0, 4, 0, 0)
        tab1_lay.setSpacing(6)

        # Ana splitter: sol (kontrol + DAG), sağ (canlı log — tam yükseklik)
        mid = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)
        left_lay.addWidget(self._build_control_panel())
        left_lay.addWidget(self._build_dag_panel())
        mid.addWidget(left_widget)
        mid.addWidget(self._build_log_panel())
        mid.setSizes([500, 500])
        tab1_lay.addWidget(mid)

        tabs.addTab(tab1, "Analiz")

        # Sekme 2: Kuyruk
        self.queue_tab = QueueTab()
        self.queue_tab.start_queue.connect(self._on_queue_start)
        self.queue_tab.stop_queue.connect(self._on_queue_stop)
        tabs.addTab(self.queue_tab, "Kuyruk")

        root.addWidget(tabs)

    def _build_control_panel(self):
        grp = QGroupBox("Kontrol Kulesi")
        lay = QVBoxLayout(grp)
        lay.setSpacing(8)

        # İçerik profili seçimi
        lay.addWidget(self._sep())
        icerik_row = QHBoxLayout()
        icerik_row.addWidget(QLabel("İçerik:"))
        self.content_combo = QComboBox()
        self.content_combo.addItems(["FilmDizi", "Spor", "StudyoProgram", "MuzikProgram", "KisaHaber"])
        self.content_combo.currentTextChanged.connect(self._on_profile_changed)
        icerik_row.addWidget(self.content_combo)
        lay.addLayout(icerik_row)
        lay.addWidget(self._sep())

        # Başlat
        self.start_btn = QPushButton("  BASLAT")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._on_start)
        lay.addWidget(self.start_btn)

        # Durdur
        row = QHBoxLayout()
        self.stop_safe_btn = QPushButton("  Guvenli Dur")
        self.stop_safe_btn.setObjectName("stopSafeBtn")
        self.stop_safe_btn.setEnabled(False)
        self.stop_safe_btn.clicked.connect(self._on_stop_safe)
        row.addWidget(self.stop_safe_btn)
        self.stop_hard_btn = QPushButton("  Zorla Dur")
        self.stop_hard_btn.setObjectName("stopHardBtn")
        self.stop_hard_btn.setEnabled(False)
        self.stop_hard_btn.clicked.connect(self._on_stop_hard)
        row.addWidget(self.stop_hard_btn)
        lay.addLayout(row)

        lay.addWidget(self._sep())

        # ── Sadece kullanıcının seçeceği yollar ─────────────────
        # Kaynak Video
        r1 = QHBoxLayout()
        lbl1 = QLabel("Kaynak Video:")
        lbl1.setMinimumWidth(110)
        self.video_edit = QLineEdit()
        self.video_edit.setReadOnly(True)
        self.video_edit.setPlaceholderText("Video dosyası seç...")
        btn1 = QPushButton("...")
        btn1.setMaximumWidth(36)
        btn1.clicked.connect(self._pick_video)
        r1.addWidget(lbl1); r1.addWidget(self.video_edit); r1.addWidget(btn1)
        lay.addLayout(r1)

        # Çıktı Dizini
        r2 = QHBoxLayout()
        lbl2 = QLabel("Cikti Dizini:")
        lbl2.setMinimumWidth(110)
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("(Video ile aynı dizin)")
        btn2 = QPushButton("...")
        btn2.setMaximumWidth(36)
        btn2.clicked.connect(self._pick_output)
        r2.addWidget(lbl2); r2.addWidget(self.output_edit); r2.addWidget(btn2)
        lay.addLayout(r2)

        lay.addWidget(self._sep())

        # ── Sabit yol bilgileri (okunur, düzenlenemez) ──────────
        info_lbl = QLabel("Sabit Sistem Yollari:")
        info_lbl.setObjectName("pathInfo")
        lay.addWidget(info_lbl)

        self._path_info_labels = {}
        for key, default in [
            ("FFmpeg",      str(Path(FFMPEG_BIN_DIR) / "ffmpeg.exe")),
            ("FFprobe",     str(Path(FFMPEG_BIN_DIR) / "ffprobe.exe")),
            ("LOGOLAR",     str(LOGOLAR_DIR)),
            ("Google JSON", str(GOOGLE_KEYS_JSON)),
            ("TMDB Keys",  str(API_KEYS_JSON)),
        ]:
            ri = QHBoxLayout()
            k = QLabel(f"{key}:")
            k.setMinimumWidth(90)
            k.setObjectName("pathInfo")
            v = QLineEdit(default)
            v.setReadOnly(True)
            v.setStyleSheet("color: #5a8a5a; font-size: 11px;")
            self._path_info_labels[key] = v
            ri.addWidget(k); ri.addWidget(v)
            lay.addLayout(ri)

        lay.addWidget(self._sep())

        # TMDB key
        tr = QHBoxLayout()
        tr.addWidget(QLabel("TMDB Key:"))
        self.tmdb_edit = QLineEdit()
        self.tmdb_edit.setReadOnly(True)
        self.tmdb_edit.setPlaceholderText("Otomatik: api_keys.json")
        tr.addWidget(self.tmdb_edit)
        lay.addLayout(tr)

        lay.addStretch()
        return grp

    def _build_log_panel(self):
        """Canlı Log — eski Sound Detail panelinin yerine."""
        grp = QGroupBox("Canli Log")
        lay = QVBoxLayout(grp)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        lay.addWidget(self.log_text)
        return grp

    def _build_stats_panel(self):
        grp = QGroupBox("Istatistik")
        lay = QGridLayout(grp)
        lay.setSpacing(6)
        stats = [
            ("Calisma:",  "stat_elapsed", "00:00:00"),
            ("Video:",    "stat_video",   "—"),
            ("Frame:",    "stat_frames",  "—"),
            ("OCR Satir:", "stat_ocr",   "—"),
            ("Oyuncu:",   "stat_actors",  "—"),
            ("Ekip:",     "stat_crew",    "—"),
        ]
        for i, (label, attr, default) in enumerate(stats):
            lay.addWidget(QLabel(label), i, 0)
            val = QLabel(default)
            val.setObjectName("statValue")
            setattr(self, attr, val)
            lay.addWidget(val, i, 1)
        lay.setRowStretch(len(stats), 1)
        return grp

    def _build_pipeline_panel(self):
        grp = QGroupBox("Pipeline (DAG)")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)
        self.stage_bars = {}
        stages = ["INGEST", "FRAME_EXTRACT", "TEXT_FILTER",
                  "OCR_CREDITS", "CREDITS_PARSE", "EXPORT"]
        for name in stages:
            r = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setMinimumWidth(120)
            font = QFont("Consolas", 10)
            font.setBold(True)
            lbl.setFont(font)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat(f"{name}  %p%")
            self.stage_bars[name] = bar
            r.addWidget(lbl)
            r.addWidget(bar)
            lay.addLayout(r)
        lay.addStretch()
        return grp

    def _build_system_info(self):
        """Tarih, Saat ve CPU/GPU/RAM — 3 satır dikey, sağa yaslanmış."""
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self._sys_date_lbl = QLabel("—")
        self._sys_time_lbl = QLabel("—")
        self._sys_hw_lbl   = QLabel("CPU: —  GPU: —  RAM: —")

        for lbl in (self._sys_date_lbl, self._sys_time_lbl):
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet("font-size: 14px; color: #e0e0e0;")
            lay.addWidget(lbl)

        self._sys_hw_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._sys_hw_lbl.setStyleSheet("font-size: 12px; color: #888;")
        lay.addWidget(self._sys_hw_lbl)

        self._sys_timer = QTimer(widget)
        self._sys_timer.timeout.connect(self._update_system_info)
        self._sys_timer.start(1000)
        self._update_system_info()
        return widget

    @staticmethod
    def _get_gpu_usage() -> str:
        """GPU kullanımını 3 farklı yöntemle dene."""
        # Yöntem 1: GPUtil
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return f"{gpus[0].load * 100:.0f}%"
        except Exception:
            pass
        # Yöntem 2: pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            return f"{util.gpu}%"
        except Exception:
            pass
        # Yöntem 3: nvidia-smi subprocess
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return f"{result.stdout.strip()}%"
        except Exception:
            pass
        return "N/A"

    def _update_system_info(self):
        """Sistem bilgisi etiketlerini güncelle."""
        now = datetime.now()
        self._sys_date_lbl.setText(now.strftime("%d.%m.%Y"))
        self._sys_time_lbl.setText(now.strftime("%H:%M:%S"))

        cpu_txt = "—"
        ram_txt = "—"
        try:
            import psutil
            cpu_txt = f"{psutil.cpu_percent(interval=None):.0f}%"
            ram_txt = f"{psutil.virtual_memory().used / (1024 ** 3):.1f}GB"
        except Exception:
            pass

        gpu_txt = self._get_gpu_usage()
        self._sys_hw_lbl.setText(f"CPU: {cpu_txt}  GPU: {gpu_txt}  RAM: {ram_txt}")

    def _build_dag_panel(self):
        """DAG diagram widget'ını içeren grup kutusu."""
        try:
            from ui.dag_widget import DAGWidget
            grp = QGroupBox("Pipeline DAG")
            lay = QVBoxLayout(grp)
            initial_profile = self.content_combo.currentText() if hasattr(self, "content_combo") else "FilmDizi"
            self.dag_widget = DAGWidget(profile_name=initial_profile)
            self.dag_widget.setMinimumHeight(180)
            lay.addWidget(self.dag_widget)
            return grp
        except Exception:
            grp = QGroupBox("Pipeline DAG")
            grp.setMinimumHeight(120)
            return grp

    # ═══════════════════════════════════════════════════════════════
    # EYLEMLER
    # ═══════════════════════════════════════════════════════════════
    def _on_profile_changed(self, profile_name: str):
        """İçerik profili değiştiğinde DAG widget'ı güncelle."""
        if hasattr(self, "dag_widget"):
            self.dag_widget.set_profile(profile_name)

    def _on_start(self):
        if not self.video_path or not os.path.isfile(self.video_path):
            QMessageBox.warning(self, "Hata", "Lutfen gecerli bir video dosyasi secin!")
            return

        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_safe_btn.setEnabled(True)
        self.stop_hard_btn.setEnabled(True)
        self.log_text.clear()
        for bar in self.stage_bars.values():
            bar.setValue(0)

        # ── FIX: Tüm UI değerleri ANA THREAD'de okunup dict'e kopyalanır ──
        # Worker thread ASLA widget'a dokunmaz.
        profile_name = self.content_combo.currentText()

        # Profil JSON'dan ayarları oku
        try:
            from config.profile_loader import load_profile
            content_profile = load_profile(profile_name)
            if content_profile:
                content_profile["_name"] = profile_name
            else:
                content_profile = None
        except Exception:
            content_profile = None

        # Profil ayarlarını config'e merge et
        base_cfg = dict(self.config.get("WORKSTATION", {}))
        if content_profile:
            for key in ("text_filter_threshold", "text_filter_mser_min_boxes",
                        "text_filter_max_per_segment"):
                if key in content_profile:
                    base_cfg[key] = content_profile[key]

        params = {
            "video_path":      self.video_path,
            "output_dir":      self.output_dir or os.path.dirname(self.video_path),
            "config":          base_cfg,
            "content_profile": content_profile,
        }

        self._log(f"Pipeline baslatiliyor: {Path(self.video_path).name}")
        self._log(f"  Profil: {profile_name}")
        self.pipeline_thread = threading.Thread(
            target=self._run_pipeline, args=(params,), daemon=True)
        self.pipeline_thread.start()

        self._start_time = time.time()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._timer.start(1000)

    def _on_stop_safe(self):
        self._log("Guvenli durdurma istegi gonderildi...")
        self.running = False

    def _on_stop_hard(self):
        self._log("Zorla durduruldu!")
        self.running = False
        self._reset_ui()

    def _on_queue_start(self):
        """Kuyruk Başlat butonuna basıldığında."""
        qm = self.queue_tab.queue_manager
        if qm.stats()["pending"] == 0:
            QMessageBox.warning(self, "Uyarı", "Kuyrukta bekleyen video yok!")
            return

        self._queue_running = True
        self._queue_stop_requested = False
        self._queue_profile_name = self.content_combo.currentText()
        self.queue_tab.set_running(True)

        self._queue_thread = threading.Thread(
            target=self._run_queue, daemon=True)
        self._queue_thread.start()

    def _on_queue_stop(self):
        """Kuyruk Durdur butonuna basıldığında."""
        self._queue_stop_requested = True

    def _run_queue(self):
        """Kuyruktaki videoları sırayla işle (arka plan thread)."""
        from PySide6.QtCore import QMetaObject, Qt as QtConst
        from utils.path_resolver import PathResolver
        from config.profile_loader import load_profile

        qm = self.queue_tab.queue_manager

        while not self._queue_stop_requested:
            item = qm.get_next()
            if item is None:
                break  # kuyruk bitti

            video_path = item.path

            # UI güncelle: İşleniyor (ana thread'de çalıştır)
            self.signals.log_message.emit(f"\n[KUYRUK] İşleniyor: {video_path}")
            QMetaObject.invokeMethod(
                self.queue_tab, "update_item_status",
                QtConst.QueuedConnection,
                video_path,
                VideoStatus.PROCESSING,
                None,
                "",
            )

            t_start = time.time()
            try:
                resolver = PathResolver()
                resolver.resolve_all()
                if not resolver.ffmpeg:
                    raise RuntimeError(f"FFmpeg bulunamadı: {resolver.errors}")

                profile_name = self._queue_profile_name
                try:
                    content_profile = load_profile(profile_name)
                    if content_profile:
                        content_profile["_name"] = profile_name
                    else:
                        content_profile = None
                except Exception:
                    content_profile = None

                base_cfg = dict(self.config.get("WORKSTATION", {}))
                if content_profile:
                    for key in ("text_filter_threshold", "text_filter_mser_min_boxes",
                                "text_filter_max_per_segment"):
                        if key in content_profile:
                            base_cfg[key] = content_profile[key]

                out_dir = str(Path(video_path).parent)

                from core.pipeline_runner import PipelineRunner
                scope = "video+audio"
                first_min = 4.0
                last_min = 8.0
                if content_profile:
                    scope = content_profile.get("scope", scope)
                    try:
                        first_min = float(content_profile.get("first_segment_minutes", first_min))
                        last_min = float(content_profile.get("last_segment_minutes", last_min))
                    except (ValueError, TypeError):
                        pass

                runner = PipelineRunner(
                    ffmpeg=resolver.ffmpeg,
                    ffprobe=resolver.ffprobe,
                    config=base_cfg,
                    output_root=out_dir,
                    google_json=resolver.google_json,
                    logolar_dir=resolver.logolar,
                )
                runner.set_log_callback(
                    lambda msg: self.signals.log_message.emit(msg))

                runner.run(
                    video_path=video_path,
                    scope=scope,
                    first_min=first_min,
                    last_min=last_min,
                    content_profile=content_profile,
                )

                elapsed = time.time() - t_start
                QMetaObject.invokeMethod(
                    self.queue_tab, "update_item_status",
                    QtConst.QueuedConnection,
                    video_path,
                    VideoStatus.DONE,
                    elapsed,
                    "",
                )
                self.signals.log_message.emit(
                    f"  [KUYRUK] ✅ Bitti: {Path(video_path).name} ({elapsed:.0f}s)")

            except Exception as e:
                import traceback
                elapsed = time.time() - t_start
                error_msg = str(e)
                self.signals.log_message.emit(
                    f"  [KUYRUK] ❌ Hata: {Path(video_path).name} — {error_msg}\n"
                    f"{traceback.format_exc()}")
                QMetaObject.invokeMethod(
                    self.queue_tab, "update_item_status",
                    QtConst.QueuedConnection,
                    video_path,
                    VideoStatus.ERROR,
                    None,
                    error_msg,
                )

        # Kuyruk bitti veya durduruldu
        self._queue_running = False
        QMetaObject.invokeMethod(
            self.queue_tab, "set_running",
            QtConst.QueuedConnection,
            False,
        )
        self.signals.log_message.emit("\n[KUYRUK] Tamamlandı.")

    def _run_pipeline(self, params: dict):
        """Pipeline'ı arka plan thread'inde çalıştır."""
        try:
            from core.pipeline_runner import PipelineRunner
            from utils.path_resolver import PathResolver

            resolver = PathResolver()
            resolver.resolve_all()

            if not resolver.ffmpeg:
                self.signals.pipeline_error.emit(
                    f"FFmpeg bulunamadi!\nBeklenen: {FFMPEG_BIN_DIR}\n"
                    f"{chr(10).join(resolver.errors)}"
                )
                return

            cfg = dict(params["config"])
            content_profile = params.get("content_profile")

            # Profil varsa scope ve segment değerlerini profilden al
            scope     = "video+audio"
            first_min = 4.0
            last_min  = 8.0
            if content_profile:
                scope     = content_profile.get("scope", scope)
                first_min = float(content_profile.get("first_segment_minutes", first_min))
                last_min  = float(content_profile.get("last_segment_minutes", last_min))

            runner = PipelineRunner(
                ffmpeg=resolver.ffmpeg,
                ffprobe=resolver.ffprobe,
                config=cfg,
                output_root=params["output_dir"],
                google_json=resolver.google_json,
                logolar_dir=resolver.logolar,
            )
            runner.set_log_callback(
                lambda msg: self.signals.log_message.emit(msg))

            result = runner.run(
                video_path=params["video_path"],
                scope=scope,
                first_min=first_min,
                last_min=last_min,
                content_profile=content_profile,
            )
            self.signals.pipeline_done.emit(result)

        except Exception as e:
            import traceback
            self.signals.pipeline_error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _on_done(self, result: dict):
        self._reset_ui()
        if hasattr(self, 'stat_ocr'):
            self.stat_ocr.setText(str(result.get("ocr_lines", 0)))
        cr = result.get("credits", {})
        if hasattr(self, 'stat_actors'):
            self.stat_actors.setText(str(cr.get("total_actors", 0)))
        if hasattr(self, 'stat_crew'):
            self.stat_crew.setText(str(cr.get("total_crew", 0)))
        self._log("\nTAMAMLANDI!")
        self._log(f"  JSON      : {result.get('report_json', '')}")
        self._log(f"  Rapor     : {result.get('report_txt', '')}")
        self._log(f"  Transcript: {result.get('transcript_txt', '')}")
        for bar in self.stage_bars.values():
            bar.setValue(100)

    def _on_error(self, msg: str):
        self._reset_ui()
        self._log(f"\nHATA: {msg}")
        QMessageBox.critical(self, "Pipeline Hatasi", msg)

    def _reset_ui(self):
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_safe_btn.setEnabled(False)
        self.stop_hard_btn.setEnabled(False)
        if hasattr(self, "_timer"):
            self._timer.stop()

    def _update_elapsed(self):
        if hasattr(self, "_start_time") and hasattr(self, 'stat_elapsed'):
            e = int(time.time() - self._start_time)
            h, r = divmod(e, 3600)
            m, s = divmod(r, 60)
            self.stat_elapsed.setText(f"{h:02d}:{m:02d}:{s:02d}")

    # ═══════════════════════════════════════════════════════════════
    # DOSYA SEÇİCİLER (sadece video ve output)
    # ═══════════════════════════════════════════════════════════════
    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Video Sec", "",
            "Video (*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.ts);;Tum dosyalar (*)")
        if path:
            self.video_edit.setText(path)
            self.video_path = path
            if hasattr(self, 'stat_video'):
                self.stat_video.setText(Path(path).name)
            # Çıktı dizini yoksa videoyla aynı dizini öner
            if not self.output_dir:
                self.output_edit.setText(os.path.dirname(path))

    def _pick_output(self):
        path = QFileDialog.getExistingDirectory(self, "Cikti Dizini Sec")
        if path:
            self.output_edit.setText(path)
            self.output_dir = path

    # ═══════════════════════════════════════════════════════════════
    # LOG + STAGE
    # ═══════════════════════════════════════════════════════════════
    def _sep(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #0f3460;")
        line.setFixedHeight(2)
        return line

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _append_log(self, msg: str):
        self._log(msg)

    def _update_stage(self, name: str, pct: int):
        if name in self.stage_bars:
            self.stage_bars[name].setValue(pct)

    # ═══════════════════════════════════════════════════════════════
    # YARDIMCI
    # ═══════════════════════════════════════════════════════════════
    def _load_config(self) -> dict:
        p = Path(__file__).parent.parent / "config" / "profiles.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _resolve_and_show_paths(self):
        """Sabit yolları PathResolver ile doğrula, bilgi etiketlerine yaz."""
        try:
            from utils.path_resolver import PathResolver
            r = PathResolver()
            r.resolve_all()

            def _show(key, val, ok):
                lbl = self._path_info_labels.get(key)
                if lbl:
                    lbl.setText(val or "— bulunamadi")
                    color = "#5a8a5a" if ok else "#e94560"
                    lbl.setStyleSheet(f"color: {color}; font-size: 11px;")

            _show("FFmpeg",      r.ffmpeg,      bool(r.ffmpeg))
            _show("FFprobe",     r.ffprobe,     bool(r.ffprobe))
            _show("LOGOLAR",     r.logolar,     bool(r.logolar))
            _show("Google JSON", r.google_json, bool(r.google_json))
            _show("TMDB Keys",  str(API_KEYS_JSON), API_KEYS_JSON.is_file())

            if API_KEYS_JSON.is_file():
                self.tmdb_edit.setText("api_keys.json (otomatik)")
            else:
                self.tmdb_edit.setText("api_keys.json bulunamadi")

            if r.errors:
                self._log("⚠️  Eksik araçlar:")
                for e in r.errors:
                    self._log(f"   {e}")
        except Exception:
            pass
