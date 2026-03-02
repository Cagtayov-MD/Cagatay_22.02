import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.pipeline_runner import PipelineRunner
from config.runtime_paths import FFMPEG_BIN_DIR

# GİRİŞ
video_path = r"F:\test\test.mp4"

# ÇIKTI
output_dir = r"F:\test"

# CONFIG
config = {
    "scope": "video_only",  # ASR KAPALI
    "program_type": "film_dizi",
    "first_min": 2.0,
    "last_min": 6.0,
    "use_gpu": True,
    "output_root": output_dir
}

ffmpeg = str(Path(FFMPEG_BIN_DIR) / "ffmpeg.exe")
ffprobe = str(Path(FFMPEG_BIN_DIR) / "ffprobe.exe")

# ÇALIŞTIR
print("🎬 OCR Test başlıyor...")
print(f"📹 Video: {video_path}")
print(f"📁 Çıktı: {output_dir}\n")

runner = PipelineRunner(
    ffmpeg, ffprobe,
    config=config,
    output_root=output_dir
)
result = runner.run(
    video_path=video_path,
    scope="video_only",
    first_min=config["first_min"],
    last_min=config["last_min"],
)

print(f"\n✅ Tamamlandı!")
print(f"📁 Sonuçlar: {result.get('work_dir')}")