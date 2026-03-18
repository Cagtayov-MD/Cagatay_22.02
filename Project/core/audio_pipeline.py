"""
audio_pipeline.py — Ses pipeline orkestratör.

Her stage'i sırayla çalıştırır, VRAM yönetimi yapar.

Stage sırası (Film/Dizi) — v2:
  [B] DİL TESPİTİ → [A] EXTRACT → [C] DENOISE → [D] DIARIZE → [E] TRANSCRIBE → [F] POST_PROCESS

  ÖNEMLİ DEĞİŞİKLİK (v2):
    Dil tespiti artık extract'tan ÖNCE çalışır.
    Çünkü arşiv materyalde stereo kanallar farklı dillerde olabiliyor
    (örn: sol=İngilizce+Türkçe karışık, sağ=sadece Türkçe).
    detect_language hangi kanalın en iyi olduğunu tespit eder,
    extract o kanalla ses çıkarır.

Stage'ler arası veri JSON ile taşınır (dict).
Herhangi bir stage patlarsa pipeline durur ama mevcut sonuçlar korunur.
"""

import json
import os
import time
from pathlib import Path

from audio.stages.extract import ExtractStage
from audio.stages.denoise import DenoiseStage
from audio.stages.diarize import DiarizeStage
from audio.stages.transcribe import TranscribeStage
from audio.stages.post_process import PostProcessStage
from audio.stages.detect_language import LanguageDetectionStage


class AudioPipeline:
    """
    Ses pipeline orkestratör.
    Film/Dizi: DF3 → PyAnnote → faster-whisper → Ollama
    """

    VERSION = "1.2"

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
            ["extract", "detect_language", "denoise", "diarize", "transcribe", "post_process"]
        )

        audio_dir = str(Path(work_dir) / "audio_work")
        os.makedirs(audio_dir, exist_ok=True)

        ffmpeg = self.config.get("ffmpeg", "ffmpeg")

        result = {
            "version": self.VERSION,
            "video_path": video_path,
            "duration_sec": 0.0,
            "processing_time_sec": 0.0,
            "asr_engine": "faster-whisper",
            "whisper_model": options.get("whisper_model", "large-v3"),
            "speakers": {},
            "transcript": [],
            "summary_tr": "",
            "detected_language": "unknown",
            "language_is_turkish": False,
            "detected_language_samples": [],
            "selected_channel": None,
            "stages": {},
        }

        self._log(f"\n{'='*60}")
        self._log(f"  SES PIPELINE v{self.VERSION} — Film/Dizi")
        self._log(f"{'='*60}")
        self._log(f"  Video: {Path(video_path).name}")
        self._log(f"  Çalışma: {audio_dir}")

        # ══════════════════════════════════════════════════════
        # [B] DİL TESPİTİ  (artık EXTRACT'tan ÖNCE çalışır)
        # ══════════════════════════════════════════════════════
        language_is_turkish = True  # default: Türkçe varsay (mevcut davranışı koru)
        selected_channel = None     # None = karışık mono (eski davranış)

        if "detect_language" in stages_to_run:
            self._log(f"\n[B] DİL TESPİTİ (pre-extract)")
            lang_detector = LanguageDetectionStage(
                ffmpeg_path=ffmpeg, log_cb=self._log
            )
            lang_result = lang_detector.run(
                video_path, audio_dir,
                duration_sec=0.0,  # ffprobe ile kendisi alacak
            )
            result["stages"]["detect_language"] = {
                "duration_sec": lang_result["stage_time_sec"],
                "status": lang_result["status"],
                "detected_language": lang_result["detected_language"],
                "confidence": lang_result["confidence"],
                "decision_logic": lang_result.get("decision_logic", ""),
                "selected_channel": lang_result.get("selected_channel"),
                "channel_trials": lang_result.get("channel_trials", []),
            }
            result["detected_language"] = lang_result["detected_language"]
            result["language_is_turkish"] = lang_result["language_is_turkish"]
            result["detected_language_samples"] = lang_result.get("samples", [])
            result["selected_channel"] = lang_result.get("selected_channel")
            language_is_turkish = lang_result["language_is_turkish"]
            selected_channel = lang_result.get("selected_channel")

            if not language_is_turkish:
                self._log(
                    f"  [DİL TESPİTİ] Dil: {lang_result['detected_language'].upper()} "
                    f"— yabancı dilde transcribe yapılacak"
                )
            if selected_channel is not None:
                self._log(
                    f"  [DİL TESPİTİ] Kanal seçimi: {selected_channel} "
                    f"→ extract bu kanalla çalışacak"
                )

        # ══════════════════════════════════════════════════════
        # [A] EXTRACT  (selected_channel bilgisiyle)
        # ══════════════════════════════════════════════════════
        wav_16k = ""
        wav_48k = ""

        if "extract" in stages_to_run:
            self._log(f"\n[A] EXTRACT")
            extract = ExtractStage(ffmpeg_path=ffmpeg, log_cb=self._log)
            extract_result = extract.run(
                video_path, audio_dir,
                selected_channel=selected_channel,
            )
            result["stages"]["extract"] = {
                "duration_sec": extract_result["stage_time_sec"],
                "status": extract_result["status"],
                "selected_channel": extract_result.get("selected_channel"),
            }

            if extract_result["status"] != "ok":
                self._log(f"  !! Extract başarısız — pipeline durduruluyor")
                result["status"] = "error"
                result["error"] = extract_result.get("error", "extract stage failed")
                result["processing_time_sec"] = round(time.time() - t0, 2)
                return result

            wav_16k = extract_result["wav_16k"]
            wav_48k = extract_result["wav_48k"]
            result["duration_sec"] = extract_result["duration_sec"]

        # ══════════════════════════════════════════════════════
        # [C] DENOISE
        # ══════════════════════════════════════════════════════
        clean_wav = wav_16k  # default: denoise yapılmazsa 16k kullan

        if "denoise" in stages_to_run and wav_48k:
            self._log(f"\n[C] DENOISE")
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

            # DF3 48kHz çıktı → 16kHz'e resample (faster-whisper/PyAnnote için)
            if denoise_result["status"] == "ok":
                clean_wav = self._resample_to_16k(
                    denoise_result["output_path"], audio_dir, ffmpeg
                )
            # fallback veya skip → 16k raw kullan
            if not clean_wav or not Path(clean_wav).is_file():
                clean_wav = wav_16k

        # ══════════════════════════════════════════════════════
        # [D] DIARIZE
        # ══════════════════════════════════════════════════════
        diarize_result = {"segments": []}

        if "diarize" in stages_to_run and clean_wav:
            self._log(f"\n[D] DIARIZE")
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
        # [E] TRANSCRIBE
        # ══════════════════════════════════════════════════════
        transcribe_result = {"segments": []}

        # Tespit edilen dili Whisper'a aktar
        # Dil tespiti başarısız (unknown/hata) olursa config'deki whisper_language'ı kullan.
        # None geçilmesi Whisper'ın her segment için kendi dil tespiti yapmasına yol açar
        # ve Türkçe içeriği yanlış dilde transcribe etme riskini doğurur.
        _whisper_lang = result.get("detected_language", options.get("whisper_language", "tr"))
        if _whisper_lang in ("unknown", None, ""):
            _whisper_lang = options.get("whisper_language", "tr")

        if "transcribe" in stages_to_run and clean_wav:
            self._log(f"\n[E] TRANSCRIBE")
            transcribe = TranscribeStage(log_cb=self._log)
            transcribe_result = transcribe.run(
                clean_wav,
                diarization=diarize_result.get("segments", []),
                options={
                    "whisper_model": options.get("whisper_model", "large-v3"),
                    "whisper_language": _whisper_lang,
                    "compute_type": options.get("compute_type", "float16"),
                    "beam_size": options.get("beam_size", 3),
                    "tmdb_cast": self.config.get("tmdb_cast", []),
                },
            )
            result["stages"]["transcribe"] = {
                "duration_sec": transcribe_result["stage_time_sec"],
                "status": transcribe_result["status"],
                "segments": transcribe_result.get("total_segments", 0),
            }
            # Dil tespiti başarısız olmuşsa Whisper'ın bulduğu dili kullan.
            # Örnek: detect_language network path'te "unknown" döndürdü ama
            # Whisper tr tespit etti → post_process çalışmalı.
            if result.get("detected_language") in ("unknown", None, ""):
                whisper_lang = transcribe_result.get("detected_language", "")
                if whisper_lang:
                    result["detected_language"] = whisper_lang
                    result["language_is_turkish"] = (whisper_lang == "tr")
                    language_is_turkish = (whisper_lang == "tr")
                    self._log(f"  [TRANSCRIBE] Dil Whisper'dan güncellendi: {whisper_lang.upper()}")

        # ══════════════════════════════════════════════════════
        # [F] POST_PROCESS
        # ══════════════════════════════════════════════════════
        if "post_process" in stages_to_run and transcribe_result.get("segments") and language_is_turkish:
            self._log(f"\n[F] POST_PROCESS")
            post = PostProcessStage(log_cb=self._log)
            # Pre-check Ollama availability to fail fast
            ollama_url = self.config.get("ollama_url", "http://localhost:11434")
            if not post._check_ollama(ollama_url):
                self._log(f"  [POST_PROCESS] Uyarı: Ollama erişilemiyor ({ollama_url}), stage atlanıyor")
                result["transcript"] = transcribe_result.get("segments", [])
                result["stages"]["post_process"] = {
                    "status": "skipped",
                    "reason": f"Ollama unreachable: {ollama_url}",
                    "duration_sec": 0.0,
                }
            else:
                post_result = post.run(
                    transcribe_result["segments"],
                    ollama_url=ollama_url,
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
        if transcribe_result.get("status") == "error":
            result["status"] = "error"
            result["error"] = transcribe_result.get("error", "transcribe stage failed")
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
        """DF3 çıktısı (48kHz) → 16kHz mono WAV (faster-whisper/PyAnnote için)."""
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
