"""
main.py — Arsiv Decode giris noktasi.

Varsayılan davranış:
  - Batch / CLI çalıştırmalar headless pipeline moduna düşer.
  - Etkileşimli kullanım PySide6 UI ile açılır.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_args, **_kwargs):
        return False


PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()  # .env dosyasını yükle — diğer her şeyden önce
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Windows charmap hatasını önle: stdout/stderr'ı UTF-8'e zorla
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _should_run_headless(argv: list[str] | None = None, environ: dict | None = None) -> bool:
    argv = list(sys.argv if argv is None else argv)
    environ = os.environ if environ is None else environ

    if _is_truthy(environ.get("HEADLESS")):
        return True

    if any(str(arg).strip().lower() in {"--headless", "-headless"} for arg in argv[1:]):
        return True

    batch_markers = ("SCOPE", "CONTENT_PROFILE", "FIRST_MIN", "LAST_MIN")
    if any(str(environ.get(key, "")).strip() for key in batch_markers):
        return True

    return False


def _configure_utf8_console() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _run_headless(argv: list[str] | None = None, environ: dict | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    environ = os.environ if environ is None else environ

    _configure_utf8_console()

    import logging
    from config.runtime_paths import FFMPEG_BIN_DIR
    from core.pipeline_runner import PipelineRunner

    logging.getLogger("ppocr").setLevel(logging.ERROR)
    logging.getLogger("paddleocr").setLevel(logging.ERROR)

    ffmpeg = str(Path(FFMPEG_BIN_DIR) / "ffmpeg.exe")
    ffprobe = str(Path(FFMPEG_BIN_DIR) / "ffprobe.exe")

    video_path = argv[1] if len(argv) > 1 else "test.mp4"
    scope = environ.get("SCOPE", "video+audio")
    first_min = float(environ.get("FIRST_MIN", "1.0"))
    last_min = float(environ.get("LAST_MIN", "1.0"))

    content_profile = None
    content_profile_name = environ.get("CONTENT_PROFILE", "")
    if content_profile_name:
        from config.profile_loader import load_profile

        content_profile = load_profile(content_profile_name)
        if content_profile:
            content_profile["_name"] = content_profile_name
            scope = content_profile.get("scope", scope)
            first_min = float(content_profile.get("first_segment_minutes", first_min))
            last_min = float(content_profile.get("last_segment_minutes", last_min))

    runner = PipelineRunner(ffmpeg, ffprobe)
    runner.run(
        video_path=video_path,
        scope=scope,
        first_min=first_min,
        last_min=last_min,
        content_profile=content_profile,
    )
    return 0


def _run_ui(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt, qInstallMessageHandler
    except ImportError:
        print("PySide6 kurulu degil!  pip install PySide6")
        return 1

    try:
        import paddleocr  # noqa: F401
    except ImportError:
        print("PaddleOCR kurulu degil — OCR calismayacak!")
        print("  pip install paddlepaddle paddleocr")
        print("  Not: PaddleOCR 3.x (PP-OCRv5) için NumPy 1.26.x önerilir: pip install numpy==1.26.4")

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    previous_handler = None

    def _qt_message_filter(msg_type, context, message):
        if "QFont::setPointSize: Point size <= 0 (-1)" in str(message):
            return
        if previous_handler is not None:
            previous_handler(msg_type, context, message)

    previous_handler = qInstallMessageHandler(_qt_message_filter)

    app = QApplication(argv)
    app.setApplicationName("Arsiv Decode")
    app.setOrganizationName("ArsivDecode")

    from PySide6.QtGui import QFont
    app.setFont(QFont("Segoe UI", 10))

    from ui.main_window import MainWindow

    window = MainWindow()
    positional_args = [arg for arg in argv[1:] if not str(arg).startswith("-")]
    if positional_args:
        vp = positional_args[0]
        if os.path.isfile(vp):
            window.load_startup_video(vp, autostart=True)
    window.show()
    return app.exec()


def main(argv: list[str] | None = None, environ: dict | None = None) -> int:
    if _should_run_headless(argv=argv, environ=environ):
        return _run_headless(argv=argv, environ=environ)
    return _run_ui(argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())





