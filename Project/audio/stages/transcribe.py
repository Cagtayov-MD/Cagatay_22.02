# Compat TranscribeStage: supports BOTH calling conventions.
# 1) run(context: dict, config: dict)
# 2) run(audio_path: str, ..., diarization=..., **kwargs)

import time
from audio.utils.vram_manager import VRAMManager

# Modul seviyesi model cache.
# UYARI: Bu dict thread-safe degildir. Cok-thread ortaminda race condition
# olusabilir. Simdilik tek worker calistigi icin sorun yok.
_model_cache = {"key": None, "model": None}


class TranscribeStage:
    def __init__(self, log_cb=None):
        self._log = log_cb or print

    def run(self, *args, **kwargs) -> dict:
        # Dispatch by signature
        # If called as run(context, config):
        if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            context, config = args[0], args[1]
            return self._run_from_context(context, config)

        # Else assume legacy call: run(audio_path, ...)
        return self._run_legacy(*args, **kwargs)

    def _run_from_context(self, context: dict, config: dict) -> dict:
        opts = (config.get("options") or {})

        audio_path = None
        if isinstance(context.get("denoise"), dict):
            audio_path = context["denoise"].get("clean_wav")
        if not audio_path and isinstance(context.get("extract"), dict):
            audio_path = context["extract"].get("wav_16k") or context["extract"].get("wav_48k")

        diar_segments = (context.get("diarize") or {}).get("segments", [])  # optional
        return self._transcribe(
            audio_path=audio_path,
            opts=opts,
            diarization=diar_segments,
        )

    def _run_legacy(self, audio_path=None, *args, **kwargs) -> dict:
        # audio_pipeline.py sometimes passes diarization=...
        diarization = kwargs.get("diarization", None) or kwargs.get("diarize", None)
        # Normalise: if the full diarize-result dict is passed, extract the list
        if isinstance(diarization, dict):
            diarization = diarization.get("segments", [])
        # It may also pass options directly or as config/options
        config = kwargs.get("config", {}) or {}
        opts = kwargs.get("options", None)
        if opts is None and isinstance(config, dict):
            opts = config.get("options") or {}
        if opts is None:
            opts = {}
        # Extract direct kwargs that mirror option keys (legacy calling convention)
        for key in ("whisper_model", "whisper_language", "compute_type", "batch_size"):
            if key in kwargs and key not in opts:
                opts[key] = kwargs[key]

        return self._transcribe(
            audio_path=audio_path,
            opts=opts,
            diarization=diarization,
        )

    def _transcribe(self, audio_path: str, opts: dict, diarization=None) -> dict:
        t0 = time.time()

        model_name = opts.get("whisper_model", "large-v3")
        language = opts.get("whisper_language", "tr")
        # beam_size is a decoding parameter; batch_size kept as legacy fallback only
        try:
            _raw_beam = opts.get("beam_size") if opts.get("beam_size") is not None else opts.get("batch_size", 5)
            beam_size = min(int(_raw_beam), 10)
        except (ValueError, TypeError):
            self._log("  [Whisper] Geçersiz beam_size — varsayılan 5 kullanılıyor")
            beam_size = 5

        device = VRAMManager.get_device()
        compute_type = self._resolve_compute_type(opts.get("compute_type", None), device)

        self._log(f"  [Whisper] Input WAV: {audio_path}")
        if not audio_path:
            return {
                "status": "error",
                "segments": [],
                "total_segments": 0,
                "stage_time_sec": round(time.time() - t0, 2),
                "error": "missing_audio_path",
            }

        fw_model = None
        cached = False
        cache_key = f"{model_name}|{device}|{compute_type}"
        use_cache = opts.get("cache_model", True)
        try:
            from faster_whisper import WhisperModel

            if use_cache and _model_cache["key"] == cache_key and _model_cache["model"] is not None:
                fw_model = _model_cache["model"]
                self._log(f"  [Whisper] Model cache'den kullaniliyor: {model_name}")
                cached = True
            else:
                # Eski model varsa temizle
                if _model_cache["model"] is not None:
                    del _model_cache["model"]
                    _model_cache["model"] = None
                    _model_cache["key"] = None
                    VRAMManager.release()

                self._log(f"  [Whisper] {model_name} yukleniyor ({device}, {compute_type})...")
                fw_model = WhisperModel(model_name, device=device, compute_type=compute_type)

                if use_cache:
                    _model_cache["key"] = cache_key
                    _model_cache["model"] = fw_model

            raw_segments, info = fw_model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            )

            self._log(f"  [Whisper] Dil: {info.language} (olasilik: {info.language_probability:.2f})")

            segments = []
            for seg in raw_segments:
                words = []
                scores = []
                for w in (seg.words or []):
                    words.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "score": round(w.probability, 3),
                    })
                    scores.append(w.probability)

                avg_conf = round(sum(scores) / len(scores), 3) if scores else 0.0
                segments.append({
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "speaker": "",
                    "confidence": avg_conf,
                    "words": words,
                })

            # diarization can be list[dict] with start/end/speaker
            if diarization:
                self._assign_speakers(segments, diarization)

            elapsed = round(time.time() - t0, 2)
            self._log(f"  [Whisper] {len(segments)} segment ({elapsed:.1f}s)")

            return {
                "status": "ok",
                "segments": segments,
                "total_segments": len(segments),
                "stage_time_sec": elapsed,
                "detected_language": info.language,
                "language_probability": round(info.language_probability, 3),
            }

        except ImportError:
            self._log("  [Whisper] Kurulu degil - pip install faster-whisper")
            return {
                "status": "error",
                "segments": [],
                "total_segments": 0,
                "stage_time_sec": round(time.time() - t0, 2),
                "error": "faster-whisper not installed",
            }

        except Exception as e:
            self._log(f"  [Whisper] Hata: {e}")
            import traceback; traceback.print_exc()
            return {
                "status": "error",
                "segments": [],
                "total_segments": 0,
                "stage_time_sec": round(time.time() - t0, 2),
                "error": str(e),
            }

        finally:
            # Cache'deyse modeli silme; cache disindaysa sil ve VRAM'i serbest birak
            if not cached and fw_model is not None and not use_cache:
                del fw_model
            if not cached and not use_cache:
                VRAMManager.release()

    def _resolve_compute_type(self, raw_compute_type, device):
        default_type = "float16" if device == "cuda" else "int8"
        if raw_compute_type is None:
            return default_type
        normalized = str(raw_compute_type).strip().lower()
        if normalized in ("", "none", "null", "auto"):
            return default_type
        if device != "cuda" and normalized == "float16":
            return "int8"
        return normalized

    def _assign_speakers(self, segments, diar_segments):
        # diar_segments expected: list[dict] with keys start/end/speaker
        # Be tolerant: sometimes pipeline passes a string/None or full result dict
        if isinstance(diar_segments, dict):
            diar_segments = diar_segments.get("segments", [])
        if not diar_segments or isinstance(diar_segments, (str, bytes)):
            return
        if not isinstance(diar_segments, (list, tuple)):
            return

        for seg in segments:
            best_speaker = ""
            best_overlap = 0.0
            for ds in diar_segments:
                if not isinstance(ds, dict):
                    continue
                ds_start = float(ds.get("start", 0) or 0)
                ds_end = float(ds.get("end", 0) or 0)
                overlap_start = max(seg["start"], ds_start)
                overlap_end = min(seg["end"], ds_end)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = (ds.get("speaker", "") or "").strip()
            if best_speaker:
                seg["speaker"] = best_speaker
