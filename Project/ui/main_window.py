"""
main_window.py — PySide6 Dark Theme Ana Pencere

Değişiklikler v2.0:
- Sound Detail paneli kaldırıldı → yerine Canlı Log taşındı
- FFmpeg, FFprobe, Google API JSON, LOGOLAR yolları sabit (path_resolver)
- Kullanıcı sadece Kaynak Video + Çıktı Dizini seçer
- mode_combo → pipeline_runner'a TextFilter.from_config() ile iletilir
- scope_combo → audio/video/both seçimi korundu
- UI thread fix korundu
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
    QGroupBox, QLabel, QPushButton, QComboBox, QSlider, QLineEdit,
    QTextEdit, QProgressBar, QFileDialog, QSplitter, QFrame,
    QApplication, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QFont


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
        self.setWindowTitle("ARSIV DECODE v2.0")
        self.setMinimumSize(1100, 820)
        self.resize(1200, 900)

        self.video_path = ""
        self.output_dir = ""
        self.running    = False
        self.pipeline_thread = None

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

        # Başlık
        title = QLabel("ARSIV DECODE")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        sub = QLabel("Video Analiz Sistemi — PaddleOCR GPU + WhisperX")
        sub.setObjectName("subtitleLabel")
        sub.setAlignment(Qt.AlignCenter)
        root.addWidget(title)
        root.addWidget(sub)

        # Üst blok — analiz ayarları
        root.addWidget(self._build_top_block())

        # Orta — sol: kontrol + sağ: canlı log
        mid = QSplitter(Qt.Horizontal)
        mid.addWidget(self._build_control_panel())
        mid.addWidget(self._build_log_panel())   # Canlı Log → eski Sound Detail yeri
        mid.setSizes([500, 500])
        root.addWidget(mid)

        # Alt — istatistik + pipeline
        bot = QSplitter(Qt.Horizontal)
        bot.addWidget(self._build_stats_panel())
        bot.addWidget(self._build_pipeline_panel())
        bot.setSizes([350, 650])
        root.addWidget(bot)

    def _build_top_block(self):
        grp = QGroupBox("Analiz Ayarlari")
        lay = QGridLayout(grp)
        lay.setSpacing(8)

        # Scope
        lay.addWidget(QLabel("Scope:"), 0, 0)
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Sadece Video", "Sadece Ses", "Video + Ses"])
        lay.addWidget(self.scope_combo, 0, 1)

        # İçerik
        lay.addWidget(QLabel("Icerik:"), 0, 2)
        self.content_combo = QComboBox()
        self.content_combo.addItems([
            "Film / Dizi", "Spor Maçı", "Haber", "Belgesel", "Magazin"
        ])
        lay.addWidget(self.content_combo, 0, 3)

        # Mod — TextFilter.from_config() ile pipeline'a iletilir
        lay.addWidget(QLabel("Mod:"), 0, 4)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Light", "Medium", "Heavy"])
        self.mode_combo.setCurrentIndex(1)
        lay.addWidget(self.mode_combo, 0, 5)

        # Giriş segmenti
        lay.addWidget(QLabel("Giris segment:"), 1, 0)
        self.entry_slider = QSlider(Qt.Horizontal)
        self.entry_slider.setRange(1, 8)
        self.entry_slider.setValue(4)
        self.entry_slider.setTickPosition(QSlider.TicksBelow)
        self.entry_slider.setTickInterval(1)
        self.entry_label = QLabel("4 dk")
        self.entry_label.setMinimumWidth(50)
        self.entry_slider.valueChanged.connect(
            lambda v: self.entry_label.setText(f"{v} dk"))
        lay.addWidget(self.entry_slider, 1, 1, 1, 3)
        lay.addWidget(self.entry_label, 1, 4)

        # Çıkış segmenti
        lay.addWidget(QLabel("Cikis segment:"), 2, 0)
        self.exit_slider = QSlider(Qt.Horizontal)
        self.exit_slider.setRange(1, 20)
        self.exit_slider.setValue(8)
        self.exit_slider.setTickPosition(QSlider.TicksBelow)
        self.exit_slider.setTickInterval(2)
        self.exit_label = QLabel("8 dk")
        self.exit_label.setMinimumWidth(50)
        self.exit_slider.valueChanged.connect(
            lambda v: self.exit_label.setText(f"{v} dk"))
        lay.addWidget(self.exit_slider, 2, 1, 1, 3)
        lay.addWidget(self.exit_label, 2, 4)

        return grp

    def _build_control_panel(self):
        grp = QGroupBox("Kontrol Kulesi")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

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
            ("FFmpeg",      r"F:\Source\ffmpeg\bin\ffmpeg.exe"),
            ("FFprobe",     r"F:\Source\ffmpeg\bin\ffprobe.exe"),
            ("LOGOLAR",     r"F:\Source\Logo"),
            ("Google JSON", r"F:\docs\keys\google_vid.json"),
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
        self.tmdb_edit.setPlaceholderText("TMDB API anahtari...")
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

    # ═══════════════════════════════════════════════════════════════
    # EYLEMLER
    # ═══════════════════════════════════════════════════════════════
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
        params = {
            "video_path": self.video_path,
            "output_dir": self.output_dir or os.path.dirname(self.video_path),
            "scope_idx":  self.scope_combo.currentIndex(),
            "mode":       self.mode_combo.currentText().lower(),   # light/medium/heavy
            "entry_min":  float(self.entry_slider.value()),
            "exit_min":   float(self.exit_slider.value()),
            "tmdb_key":   self.tmdb_edit.text().strip(),
            "config":     self.config.get("WORKSTATION", {}),
        }

        self._log(f"Pipeline baslatiliyor: {Path(self.video_path).name}")
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

    def _run_pipeline(self, params: dict):
        """Pipeline'ı arka plan thread'inde çalıştır."""
        try:
            from core.pipeline_runner import PipelineRunner
            from utils.path_resolver import PathResolver

            resolver = PathResolver()
            resolver.resolve_all()

            if not resolver.ffmpeg:
                self.signals.pipeline_error.emit(
                    f"FFmpeg bulunamadi!\nBeklenen: F:\\Source\\ffmpeg\\bin\\ffmpeg.exe\n"
                    f"{chr(10).join(resolver.errors)}"
                )
                return

            # mode → pipeline'a geçir (TextFilter.from_config() için)
            cfg = dict(params["config"])
            cfg["difficulty"] = params["mode"]      # light/medium/heavy
            if params["tmdb_key"]:
                cfg["tmdb_api_key"] = params["tmdb_key"]

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

            scope_map = {0: "video_only", 1: "audio_only", 2: "video+audio"}
            scope = scope_map.get(params["scope_idx"], "video_only")

            result = runner.run(
                video_path=params["video_path"],
                scope=scope,
                first_min=params["entry_min"],
                last_min=params["exit_min"],
            )
            self.signals.pipeline_done.emit(result)

        except Exception as e:
            import traceback
            self.signals.pipeline_error.emit(f"{e}\n\n{traceback.format_exc()}")

    def _on_done(self, result: dict):
        self._reset_ui()
        self.stat_ocr.setText(str(result.get("ocr_lines", 0)))
        cr = result.get("credits", {})
        self.stat_actors.setText(str(cr.get("total_actors", 0)))
        self.stat_crew.setText(str(cr.get("total_crew", 0)))
        self._log("\nTAMAMLANDI!")
        self._log(f"  JSON: {result.get('report_json', '')}")
        self._log(f"  TXT : {result.get('report_txt', '')}")
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
        if hasattr(self, "_start_time"):
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

            if r.errors:
                self._log("⚠️  Eksik araçlar:")
                for e in r.errors:
                    self._log(f"   {e}")
        except Exception:
            pass
