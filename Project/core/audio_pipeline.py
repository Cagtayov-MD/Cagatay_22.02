"""
audio_pipeline.py — Ses pipeline orkestratör.

Her stage'i sırayla çalıştırır, VRAM yönetimi yapar.
Stage sırası (Film/Dizi):
  [A] EXTRACT → [B] DENOISE → [C] DIARIZE → [D] TRANSCRIBE → [E] POST_PROCESS

Stage'ler arası veri JSON ile taşınır (dict).
Herhangi bir stage patlarsa pipeline durur ama mevcut sonuçlar korunur.
"""

import json
import os
import time
from pathlib import Path

from core.extract import ExtractStage
from core.denoise import DenoiseStage
from core.diarize import DiarizeStage
from core.transcribe import TranscribeStage
from core.post_process import PostProcessStage


class AudioPipeline:
    """
    Ses pipeline orkestratör.
    Film/Dizi: DF3 → PyAnnote → WhisperX → Ollama
    """

    VERSION = "1.0"

    def __init__(self, config: dict, log_cb=None):
        self.config = config
        self.work_dir = config.get("work_dir", "")
        self._log = log_cb or print

    def run(self) -> dict:
        """
        Tüm stage'leri çalıştır, sonuç JSON döndür.

        Returns:
            audio_result.json formatında dict
        """
        t0 = time.time()
        video_path = self.config.get("video_path", "")
        work_dir = self.config.get("work_dir", "")
        options = self.config.get("options", {})
        stages_to_run = self.config.get(
            "stages",
            ["extract", "denoise", "diarize", "transcribe", "post_process"]
        )

        audio_dir = str(Path(work_dir) / "audio_work")
        os.makedirs(audio_dir, exist_ok=True)

        ffmpeg = self.config.get("ffmpeg", "ffmpeg")

        result = {
            "version": self.VERSION,
            "video_path": video_path,
            "duration_sec": 0.0,
            "processing_time_sec": 0.0,
            "speakers": {},
            "transcript": [],
            "summary_tr": "",
            "stages": {},
        }

        self._log(f"\n{'='*60}")
        self._log(f"  SES PIPELINE — Film/Dizi")
        self._log(f"{'='*60}")
        self._log(f"  Video: {Path(video_path).name}")
        self._log(f"  Çalışma: {audio_dir}")

        # ══════════════════════════════════════════════════════
        # [A] EXTRACT
        # ══════════════════════════════════════════════════════
        wav_16k = ""
        wav_48k = ""

        if "extract" in stages_to_run:
            self._log(f"\n[A] EXTRACT")
            extract = ExtractStage(ffmpeg_path=ffmpeg, log_cb=self._log)
            extract_result = extract.run(video_path, audio_dir)
            result["stages"]["extract"] = {
                "duration_sec": extract_result["stage_time_sec"],
                "status": extract_result["status"],
            }

            if extract_result["status"] != "ok":
                self._log(f"  !! Extract başarısız — pipeline durduruluyor")
                result["processing_time_sec"] = round(time.time() - t0, 2)
                return result

            wav_16k = extract_result["wav_16k"]
            wav_48k = extract_result["wav_48k"]
            result["duration_sec"] = extract_result["duration_sec"]

        # ══════════════════════════════════════════════════════
        # [B] DENOISE
        # ══════════════════════════════════════════════════════
        clean_wav = wav_16k  # default: denoise yapılmazsa 16k kullan

        if "denoise" in stages_to_run and wav_48k:
            self._log(f"\n[B] DENOISE")
            denoise = DenoiseStage(log_cb=self._log)
            clean_48k = str(Path(audio_dir) / "audio_clean_48k.wav")
            denoise_result = denoise.run(
                wav_48k, clean_48k,
                denoise_enabled=options.get("denoise_enabled", True),
            )
            result["stages"]["denoise"] = {
                "duration_sec": denoise_result["stage_time_sec"],
                "status": denoise_result["status"],
            }

            # DF3 48kHz çıktı → 16kHz'e resample (WhisperX/PyAnnote için)
            if denoise_result["status"] == "ok":
                clean_wav = self._resample_to_16k(
                    denoise_result["output_path"], audio_dir, ffmpeg
                )
            # fallback veya skip → 16k raw kullan
            if not clean_wav or not Path(clean_wav).is_file():
                clean_wav = wav_16k

        # ══════════════════════════════════════════════════════
        # [C] DIARIZE
        # ══════════════════════════════════════════════════════
        diarize_result = {"segments": []}

        if "diarize" in stages_to_run and clean_wav:
            self._log(f"\n[C] DIARIZE")
            diarize = DiarizeStage(log_cb=self._log)
            diarize_result = diarize.run(
                clean_wav,
                hf_token=self.config.get("hf_token", ""),
                max_speakers=options.get("max_speakers"),
            )
            result["stages"]["diarize"] = {
                "duration_sec": diarize_result["stage_time_sec"],
                "status": diarize_result["status"],
                "speakers_found": diarize_result.get("speakers_found", 0),
            }

        # ══════════════════════════════════════════════════════
        # [D] TRANSCRIBE
        # ══════════════════════════════════════════════════════
        transcribe_result = {"segments": []}

        if "transcribe" in stages_to_run and clean_wav:
            self._log(f"\n[D] TRANSCRIBE")
            transcribe = TranscribeStage(log_cb=self._log)
            # BUG-01 FIX: hf_token transcribe.run() çağrısından kaldırıldı.
            # DiarizeStage [C] zaten PyAnnote'u çalıştırdı ve diarize_result
            # içinde segmentler var. Eğer hf_token geçilirse TranscribeStage
            # yeniden DiarizationPipeline oluşturur → VRAM ve zaman ikiye katlanır.
            # Çözüm: diarization= parametresi yeterli; hf_token YOK.
            transcribe_result = transcribe.run(
                clean_wav,
                diarization=diarize_result,
                whisper_model=options.get("whisper_model", "large-v3"),
                whisper_language=options.get("whisper_language", "tr"),
                compute_type=options.get("compute_type", "float16"),
                batch_size=options.get("batch_size", 16),
            )
            result["stages"]["transcribe"] = {
                "duration_sec": transcribe_result["stage_time_sec"],
                "status": transcribe_result["status"],
                "segments": transcribe_result.get("total_segments", 0),
            }

        # ══════════════════════════════════════════════════════
        # [E] POST_PROCESS
        # ══════════════════════════════════════════════════════
        if "post_process" in stages_to_run and transcribe_result.get("segments"):
            self._log(f"\n[E] POST_PROCESS")
            post = PostProcessStage(log_cb=self._log)
            post_result = post.run(
                transcribe_result["segments"],
                ollama_url=self.config.get("ollama_url", "http://localhost:11434"),
                ollama_model=options.get("ollama_model", "llama3.1:8b"),
                tmdb_cast=self.config.get("tmdb_cast", []),
            )
            result["stages"]["post_process"] = {
                "duration_sec": post_result["stage_time_sec"],
                "status": post_result["status"],
                "corrections": post_result.get("corrections", 0),
            }
            result["transcript"] = post_result["segments"]
            result["summary_tr"] = post_result.get("summary_tr", "")
        else:
            # PostProcess atlandıysa ham transcript'i kullan
            result["transcript"] = transcribe_result.get("segments", [])

        # ══════════════════════════════════════════════════════
        # Konuşmacı istatistikleri
        # ══════════════════════════════════════════════════════
        speakers = {}
        for seg in result["transcript"]:
            spk = seg.get("speaker", "UNKNOWN")
            if spk not in speakers:
                speakers[spk] = {
                    "total_time_sec": 0.0,
                    "segment_count": 0,
                    "label": seg.get("speaker_label"),
                }
            speakers[spk]["total_time_sec"] += seg.get("end", 0) - seg.get("start", 0)
            speakers[spk]["segment_count"] += 1
        # Round
        for spk in speakers:
            speakers[spk]["total_time_sec"] = round(speakers[spk]["total_time_sec"], 2)
        result["speakers"] = speakers

        # ══════════════════════════════════════════════════════
        # Finalize
        # ══════════════════════════════════════════════════════
        result["status"] = "ok"                                    # BUG-K1 FIX
        result["processing_time_sec"] = round(time.time() - t0, 2)

        self._log(f"\n{'='*60}")
        self._log(
            f"  SES PIPELINE TAMAMLANDI — {result['processing_time_sec']:.1f}s | "
            f"{len(result['transcript'])} segment | "
            f"{len(speakers)} konuşmacı"
        )
        self._log(f"{'='*60}")

        return result

    def _resample_to_16k(self, input_48k: str, audio_dir: str,
                          ffmpeg: str) -> str:
        """DF3 çıktısı (48kHz) → 16kHz mono WAV (WhisperX/PyAnnote için)."""
        import subprocess

        out_path = str(Path(audio_dir) / "audio_clean_16k.wav")
        try:
            cmd = [
                ffmpeg, '-y',
                '-i', input_48k,
                '-ar', '16000',
                '-ac', '1',
                out_path,
            ]
            r = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding='utf-8', errors='replace',
                timeout=120,
            )
            if r.returncode == 0 and Path(out_path).is_file():
                self._log(f"  [Resample] 48kHz → 16kHz: {Path(out_path).name}")
                return out_path
        except Exception as e:
            self._log(f"  [Resample] Hata: {e}")

        return ""
