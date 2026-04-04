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
import subprocess
import time
from pathlib import Path

from core.denoise import DenoiseStage
from core.diarize import DiarizeStage
from core.extract import ExtractStage
from core.post_process import PostProcessStage
from core.whisper_model import normalize_whisper_model_name
from utils.audio.stages.detect_language import LanguageDetectionStage
from utils.audio.stages.transcribe import TranscribeStage


class AudioPipeline:
    """
    Ses pipeline orkestratör.
    Film/Dizi: DF3 → PyAnnote → faster-whisper → Ollama
    """

    VERSION = "1.3"

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
        options = dict(self.config.get("options", {}))
        stages_to_run = self.config.get(
            "stages",
            ["extract", "detect_language", "denoise", "diarize", "transcribe", "post_process"]
        )
        raw_whisper_model = options.get("whisper_model")
        whisper_model = normalize_whisper_model_name(raw_whisper_model)
        if raw_whisper_model not in (None, "") and raw_whisper_model != whisper_model:
            self._log(
                f"  [AudioPipeline] Whisper model normalize edildi: "
                f"{raw_whisper_model} -> {whisper_model}"
            )
        options["whisper_model"] = whisper_model

        audio_dir = str(Path(work_dir) / "audio_work")
        os.makedirs(audio_dir, exist_ok=True)

        ffmpeg = self.config.get("ffmpeg", "ffmpeg")

        result = {
            "version": self.VERSION,
            "video_path": video_path,
            "duration_sec": 0.0,
            "processing_time_sec": 0.0,
            "asr_engine": "faster-whisper",
            "whisper_model": whisper_model,
            "speakers": {},
            "transcript": [],
            "summary_tr": "",
            "detected_language": "unknown",
            "language_is_turkish": False,
            "detected_language_samples": [],
            "selected_channel": None,
            "asr_windows": [],
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
        transcribe_result = {"segments": []}

        # Tespit edilen dili Whisper'a aktar
        # Dil tespiti başarısız (unknown/hata) olursa config'deki whisper_language'ı kullan.
        # None geçilmesi Whisper'ın her segment için kendi dil tespiti yapmasına yol açar
        # ve Türkçe içeriği yanlış dilde transcribe etme riskini doğurur.
        _whisper_lang = result.get("detected_language")
        if _whisper_lang in (None, ""):
            _whisper_lang = options.get("whisper_language", "tr")
        elif _whisper_lang == "unknown":
            _whisper_lang = None  # Whisper kendi dil tespitini yapsın

        sampled_windows = []
        sampling_mode = str(options.get("asr_sampling_mode") or "").strip().lower()
        if sampling_mode == "first_middle_last":
            video_duration = self._probe_video_duration(
                video_path,
                self.config.get("ffprobe", "ffprobe"),
            )
            if video_duration > 0:
                sampled_windows = self._build_asr_windows(
                    video_duration,
                    options.get("asr_window_minutes"),
                )
                if sampled_windows:
                    result["duration_sec"] = round(video_duration, 3)
                    result["asr_windows"] = sampled_windows
                    self._log(
                        f"  [ASR] Örnekleme aktif: "
                        f"{', '.join(w['label'] for w in sampled_windows)}"
                    )
            else:
                self._log("  [ASR] Video süresi alınamadı — tam ASR akışına dönülüyor")

        if sampled_windows:
            sampled_result = self._run_sampled_asr_windows(
                video_path=video_path,
                audio_dir=audio_dir,
                ffmpeg=ffmpeg,
                stages_to_run=stages_to_run,
                selected_channel=selected_channel,
                whisper_model=whisper_model,
                whisper_language=_whisper_lang,
                options=options,
                windows=sampled_windows,
            )
            if sampled_result["status"] != "ok":
                result["status"] = "error"
                result["error"] = sampled_result.get("error", "sampled asr failed")
                result["processing_time_sec"] = round(time.time() - t0, 2)
                result["stages"].update(sampled_result.get("stages", {}))
                return result

            result["stages"].update(sampled_result.get("stages", {}))
            transcribe_result = sampled_result["transcribe_result"]
        else:
            if "extract" in stages_to_run:
                self._log(f"\n[A] EXTRACT")
                extract = ExtractStage(ffmpeg_path=ffmpeg, log_cb=self._log)
                extract_result = extract.run(
                    video_path, audio_dir,
                    selected_channel=selected_channel,
                    max_duration_sec=options.get("audio_max_sec"),
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
            if "transcribe" in stages_to_run and clean_wav:
                self._log(f"\n[E] TRANSCRIBE")
                transcribe = TranscribeStage(log_cb=self._log)
                transcribe_result = transcribe.run(
                    clean_wav,
                    diarization=diarize_result.get("segments", []),
                    options={
                        "whisper_model": whisper_model,
                        "whisper_language": _whisper_lang,
                        "compute_type": options.get("compute_type", "float16"),
                        "beam_size": options.get("beam_size", 1),
                        "tmdb_cast": self.config.get("tmdb_cast", []),
                        "live_transcript_path": options.get("live_transcript_path"),
                        "initial_prompt": options.get("initial_prompt", ""),
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

    def _run_sampled_asr_windows(
        self,
        *,
        video_path: str,
        audio_dir: str,
        ffmpeg: str,
        stages_to_run: list[str],
        selected_channel: int | None,
        whisper_model: str,
        whisper_language,
        options: dict,
        windows: list[dict],
    ) -> dict:
        extract = ExtractStage(ffmpeg_path=ffmpeg, log_cb=self._log)
        transcribe = TranscribeStage(log_cb=self._log)
        denoise = DenoiseStage(log_cb=self._log) if "denoise" in stages_to_run else None
        diarize = DiarizeStage(log_cb=self._log) if "diarize" in stages_to_run else None

        extract_total = 0.0
        denoise_total = 0.0
        diarize_total = 0.0
        transcribe_total = 0.0
        denoise_statuses: list[str] = []
        diarize_statuses: list[str] = []
        combined_segments: list[dict] = []
        live_path = options.get("live_transcript_path")
        detected_language = ""
        diarize_speakers_found = 0

        for window in windows:
            label = window["label"]
            start_sec = float(window["start_sec"])
            end_sec = float(window["end_sec"])
            duration_sec = max(0.0, end_sec - start_sec)
            prefix = f"audio_{label.replace('+', '_')}"

            self._log(
                f"\n[A] EXTRACT ({label}) "
                f"{start_sec:.1f}s → {end_sec:.1f}s"
            )
            extract_result = extract.run(
                video_path,
                audio_dir,
                selected_channel=selected_channel,
                start_offset_sec=start_sec,
                max_duration_sec=duration_sec,
                output_prefix=prefix,
            )
            extract_total += extract_result.get("stage_time_sec", 0.0)
            if extract_result.get("status") != "ok":
                return {
                    "status": "error",
                    "error": extract_result.get("error", f"extract failed ({label})"),
                    "stages": {
                        "extract": {
                            "duration_sec": round(extract_total, 2),
                            "status": extract_result.get("status", "error"),
                            "selected_channel": selected_channel,
                            "windows": len(windows),
                        }
                    },
                }

            clean_wav = extract_result["wav_16k"]
            diarize_result = {"segments": []}

            if denoise and extract_result.get("wav_48k"):
                self._log(f"\n[C] DENOISE ({label})")
                clean_48k = str(Path(audio_dir) / f"{prefix}_clean_48k.wav")
                denoise_result = denoise.run(
                    extract_result["wav_48k"],
                    clean_48k,
                    denoise_enabled=options.get("denoise_enabled", True),
                )
                denoise_total += denoise_result.get("stage_time_sec", 0.0)
                denoise_statuses.append(denoise_result.get("status", "unknown"))
                if denoise_result.get("status") == "ok":
                    clean_wav = self._resample_to_16k(
                        denoise_result["output_path"],
                        audio_dir,
                        ffmpeg,
                        output_name=f"{prefix}_clean_16k.wav",
                    )
                if not clean_wav or not Path(clean_wav).is_file():
                    clean_wav = extract_result["wav_16k"]

            if diarize and clean_wav:
                self._log(f"\n[D] DIARIZE ({label})")
                diarize_result = diarize.run(
                    clean_wav,
                    hf_token=self.config.get("hf_token", ""),
                    max_speakers=options.get("max_speakers"),
                )
                diarize_total += diarize_result.get("stage_time_sec", 0.0)
                diarize_statuses.append(diarize_result.get("status", "unknown"))
                diarize_speakers_found += int(diarize_result.get("speakers_found", 0) or 0)

            if "transcribe" in stages_to_run and clean_wav:
                self._log(f"\n[E] TRANSCRIBE ({label})")
                window_transcribe_result = transcribe.run(
                    clean_wav,
                    diarization=diarize_result.get("segments", []),
                    options={
                        "whisper_model": whisper_model,
                        "whisper_language": whisper_language,
                        "compute_type": options.get("compute_type", "float16"),
                        "beam_size": options.get("beam_size", 1),
                        "tmdb_cast": self.config.get("tmdb_cast", []),
                        "live_transcript_path": None,
                        "initial_prompt": options.get("initial_prompt", ""),
                    },
                )
                transcribe_total += window_transcribe_result.get("stage_time_sec", 0.0)
                if window_transcribe_result.get("status") == "error":
                    return {
                        "status": "error",
                        "error": window_transcribe_result.get("error", f"transcribe failed ({label})"),
                        "stages": {
                            "extract": {
                                "duration_sec": round(extract_total, 2),
                                "status": "ok",
                                "selected_channel": selected_channel,
                                "windows": len(windows),
                            },
                            "transcribe": {
                                "duration_sec": round(transcribe_total, 2),
                                "status": "error",
                                "segments": len(combined_segments),
                                "windows": len(windows),
                            },
                        },
                    }

                if not detected_language:
                    detected_language = window_transcribe_result.get("detected_language", "")

                offset_segments = self._offset_transcript_segments(
                    window_transcribe_result.get("segments", []),
                    start_sec,
                )
                combined_segments.extend(offset_segments)
                self._append_live_transcript(live_path, offset_segments)

        combined_segments.sort(
            key=lambda seg: (
                float(seg.get("start", 0.0) or 0.0),
                float(seg.get("end", 0.0) or 0.0),
            )
        )
        transcribe_result = {
            "status": "ok",
            "segments": combined_segments,
            "total_segments": len(combined_segments),
            "stage_time_sec": round(transcribe_total, 2),
            "detected_language": detected_language,
        }

        stages = {
            "extract": {
                "duration_sec": round(extract_total, 2),
                "status": "ok",
                "selected_channel": selected_channel,
                "windows": len(windows),
            },
            "transcribe": {
                "duration_sec": round(transcribe_total, 2),
                "status": "ok",
                "segments": len(combined_segments),
                "windows": len(windows),
            },
        }
        if denoise:
            stages["denoise"] = {
                "duration_sec": round(denoise_total, 2),
                "status": self._combine_stage_statuses(denoise_statuses, default="skipped"),
                "windows": len(windows),
            }
        if diarize:
            stages["diarize"] = {
                "duration_sec": round(diarize_total, 2),
                "status": self._combine_stage_statuses(diarize_statuses, default="skipped"),
                "speakers_found": diarize_speakers_found,
                "windows": len(windows),
            }

        return {
            "status": "ok",
            "transcribe_result": transcribe_result,
            "stages": stages,
        }

    def _build_asr_windows(self, duration_sec: float, window_minutes: dict | None) -> list[dict]:
        if duration_sec <= 0:
            return []

        cfg = window_minutes if isinstance(window_minutes, dict) else {}
        first_min = self._safe_window_minutes(cfg.get("first"), 10.0)
        middle_min = self._safe_window_minutes(cfg.get("middle"), 15.0)
        last_min = self._safe_window_minutes(cfg.get("last"), 15.0)
        middle_half_sec = middle_min * 30.0

        requested = [
            {
                "label": "first",
                "start_sec": 0.0,
                "end_sec": min(duration_sec, first_min * 60.0),
                "merged_from": ["first"],
            },
            {
                "label": "middle",
                "start_sec": max(0.0, (duration_sec / 2.0) - middle_half_sec),
                "end_sec": min(duration_sec, (duration_sec / 2.0) + middle_half_sec),
                "merged_from": ["middle"],
            },
            {
                "label": "last",
                "start_sec": max(0.0, duration_sec - (last_min * 60.0)),
                "end_sec": duration_sec,
                "merged_from": ["last"],
            },
        ]

        normalized = []
        for window in requested:
            start_sec = max(0.0, float(window["start_sec"]))
            end_sec = min(duration_sec, float(window["end_sec"]))
            if end_sec <= start_sec:
                continue
            normalized.append({
                "label": window["label"],
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "merged_from": list(window["merged_from"]),
            })

        return self._merge_asr_windows(normalized)

    def _merge_asr_windows(self, windows: list[dict]) -> list[dict]:
        if not windows:
            return []

        merged: list[dict] = []
        for window in sorted(windows, key=lambda item: (item["start_sec"], item["end_sec"])):
            if not merged:
                merged.append({
                    "label": window["label"],
                    "start_sec": window["start_sec"],
                    "end_sec": window["end_sec"],
                    "merged_from": list(window.get("merged_from", [])),
                })
                continue

            current = merged[-1]
            if window["start_sec"] <= current["end_sec"]:
                current["end_sec"] = round(max(current["end_sec"], window["end_sec"]), 3)
                current["merged_from"] = self._merge_window_labels(
                    current.get("merged_from", []),
                    window.get("merged_from", []),
                )
                current["label"] = "+".join(current["merged_from"])
                continue

            merged.append({
                "label": window["label"],
                "start_sec": window["start_sec"],
                "end_sec": window["end_sec"],
                "merged_from": list(window.get("merged_from", [])),
            })

        for window in merged:
            if window.get("merged_from"):
                window["label"] = "+".join(window["merged_from"])
        return merged

    def _offset_transcript_segments(self, segments: list[dict], offset_sec: float) -> list[dict]:
        adjusted = []
        for seg in segments or []:
            if not isinstance(seg, dict):
                continue
            shifted = dict(seg)
            shifted["start"] = round(float(seg.get("start", 0.0) or 0.0) + offset_sec, 3)
            shifted["end"] = round(float(seg.get("end", 0.0) or 0.0) + offset_sec, 3)

            words = []
            for word in seg.get("words") or []:
                if not isinstance(word, dict):
                    continue
                shifted_word = dict(word)
                if "start" in shifted_word:
                    shifted_word["start"] = round(
                        float(shifted_word.get("start", 0.0) or 0.0) + offset_sec,
                        3,
                    )
                if "end" in shifted_word:
                    shifted_word["end"] = round(
                        float(shifted_word.get("end", 0.0) or 0.0) + offset_sec,
                        3,
                    )
                words.append(shifted_word)
            if words:
                shifted["words"] = words

            adjusted.append(shifted)
        return adjusted

    def _append_live_transcript(self, live_path: str | None, segments: list[dict]) -> None:
        if not live_path or not segments:
            return

        try:
            with open(live_path, "a", encoding="utf-8") as handle:
                for seg in segments:
                    text = str(seg.get("text") or "").strip()
                    if not text:
                        continue
                    start_sec = float(seg.get("start", 0.0) or 0.0)
                    m = int(start_sec) // 60
                    s = start_sec % 60
                    handle.write(f"[{m:02d}:{s:05.2f}] {text}\n")
        except Exception:
            pass

    def _probe_video_duration(self, video_path: str, ffprobe: str) -> float:
        try:
            cmd = [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.DEVNULL,
                timeout=30,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).strip()
            return float(out)
        except Exception:
            return 0.0

    @staticmethod
    def _combine_stage_statuses(statuses: list[str], default: str = "skipped") -> str:
        if not statuses:
            return default
        if any(status == "error" for status in statuses):
            return "error"
        if any(status == "fallback" for status in statuses):
            return "fallback"
        if any(status == "ok" for status in statuses):
            return "ok"
        if all(status == "skipped" for status in statuses):
            return "skipped"
        return statuses[-1]

    @staticmethod
    def _merge_window_labels(existing: list[str], incoming: list[str]) -> list[str]:
        ordered = []
        for label in list(existing or []) + list(incoming or []):
            if label and label not in ordered:
                ordered.append(label)
        return ordered

    @staticmethod
    def _safe_window_minutes(value, default: float) -> float:
        try:
            number = float(value)
            if number <= 0:
                raise ValueError("window minutes must be positive")
            return number
        except (TypeError, ValueError):
            return default

    def _resample_to_16k(self, input_48k: str, audio_dir: str,
                          ffmpeg: str,
                          output_name: str = "audio_clean_16k.wav") -> str:
        """DF3 çıktısı (48kHz) → 16kHz mono WAV (faster-whisper/PyAnnote için)."""
        out_path = str(Path(audio_dir) / output_name)
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
