# --- HEADLESS (CLI) MODE ---
# Set HEADLESS=1 to run without PySide6/Qt UI (avoids Qt font warnings).
import os, sys
from pathlib import Path

if os.environ.get("HEADLESS", "0") == "1":
    # UTF-8 safe console (best-effort)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # Silence PaddleOCR warnings (ppocr)
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    logging.getLogger("paddleocr").setLevel(logging.ERROR)

    from config.runtime_paths import FFMPEG_BIN_DIR
    from core.pipeline_runner import PipelineRunner

    ffmpeg = str(Path(FFMPEG_BIN_DIR) / "ffmpeg.exe")
    ffprobe = str(Path(FFMPEG_BIN_DIR) / "ffprobe.exe")

    video_path = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
    scope = os.environ.get("SCOPE", "video+audio")
    first_min = float(os.environ.get("FIRST_MIN", "1.0"))
    last_min = float(os.environ.get("LAST_MIN", "1.0"))

    runner = PipelineRunner(ffmpeg, ffprobe)
    runner.run(video_path=video_path, scope=scope, first_min=first_min, last_min=last_min)

    raise SystemExit(0)
# --- END HEADLESS ---

"""
main.py â€” Arsiv Decode giris noktasi.
Her zaman PySide6 UI ile baslar. Lite mod yok.
"""
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()  # .env dosyasÄ±nÄ± yÃ¼kle â€” diÄŸer her ÅŸeyden Ã¶nce

import sys
import os
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
    except ImportError:
        print("PySide6 kurulu degil!  pip install PySide6")
        sys.exit(1)

    try:
        import paddleocr  # noqa: F401
    except ImportError:
        print("PaddleOCR kurulu degil â€” OCR calismayacak!")
        print("  pip install paddlepaddle paddleocr")
        print("  Not: PaddleOCR 3.x (PP-OCRv5) iÃ§in NumPy 1.26.x Ã¶nerilir: pip install numpy==1.26.4")

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("Arsiv Decode")
    app.setOrganizationName("ArsivDecode")

    from PySide6.QtGui import QFont
    app.setFont(QFont("Segoe UI", 10))

    from ui.main_window import MainWindow
    window = MainWindow()

    # Komut satirindan video yolu verilmisse otomatik yukle
    if len(sys.argv) > 1:
        vp = sys.argv[1]
        if os.path.isfile(vp):
            window.video_path = vp
            window.video_edit.setText(vp)
            window.stat_video.setText(Path(vp).name)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()





