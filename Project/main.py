"""
main.py — Arsiv Decode giris noktasi.
Her zaman PySide6 UI ile baslar. Lite mod yok.
"""
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()  # .env dosyasını yükle — diğer her şeyden önce

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
        print("PaddleOCR kurulu degil — OCR calismayacak!")
        print("  pip install paddlepaddle paddleocr")
        print("  Not: PaddleOCR 3.x (PP-OCRv5) için NumPy 1.26.x önerilir: pip install numpy==1.26.4")

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
