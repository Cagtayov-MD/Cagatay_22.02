"""
[D] TRANSCRIBE - Re-export wrapper.
Gercek implementasyon: audio/stages/transcribe.py
Bu dosya geriye donuk uyumluluk icin korunmustur.
"""

try:
    from audio.stages.transcribe import TranscribeStage  # noqa: F401
except ImportError:
    # Fallback: eger audio paketi erisilemiyorsa, eski implementasyonu kullan
    import time
    from core.vram_manager import VRAMManager

    class TranscribeStage:
        def __init__(self, log_cb=None):
            self._log = log_cb or print

        def run(self, context: dict, config: dict) -> dict:
            t0 = time.time()

            opts = (config.get('options') or {})
            model_name = opts.get('whisper_model', 'large-v3')
            language = opts.get('whisper_language', 'tr')
            # beam_size is a decoding parameter; batch_size kept as legacy fallback only
            try:
                _raw_beam = opts.get('beam_size') if opts.get('beam_size') is not None else opts.get('batch_size', 3)
                beam_size = min(int(_raw_beam), 10)
            except (ValueError, TypeError):
                self._log('  [Whisper] Gecersiz beam_size — varsayilan 3 kullaniliyor')
                beam_size = 3

            device = VRAMManager.get_device()
            compute_type = self._resolve_compute_type(opts.get('compute_type', 'float16'), device)

            # WAV secimi (pipeline context'inden)
            audio_path = None
            if isinstance(context.get('denoise'), dict):
                audio_path = context['denoise'].get('clean_wav')
            if not audio_path and isinstance(context.get('extract'), dict):
                audio_path = context['extract'].get('wav_16k') or context['extract'].get('wav_48k')

            self._log(f"  [Whisper] Input WAV: {audio_path}")
            if not audio_path:
                return {
                    'status': 'error',
                    'segments': [],
                    'total_segments': 0,
                    'stage_time_sec': round(time.time() - t0, 2),
                    'error': "missing_audio_path (expected context['denoise'].clean_wav or context['extract'].wav_16k)",
                }

            fw_model = None
            try:
                from faster_whisper import WhisperModel

                self._log(f"  [Whisper] {model_name} yukleniyor ({device}, {compute_type})...")
                fw_model = WhisperModel(model_name, device=device, compute_type=compute_type)
                self._log(f"  [Whisper] Yuklendi (VRAM: {VRAMManager.get_usage()})")
                self._log('  [Whisper] Transkripsiyon basliyor...')

                initial_prompt = self._build_initial_prompt(opts)
                if initial_prompt:
                    self._log(f"  [Whisper] initial_prompt: {initial_prompt[:80]}...")

                raw_segments, info = fw_model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=beam_size,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                    initial_prompt=initial_prompt if initial_prompt else None,
                )

                self._log(f"  [Whisper] Dil: {info.language} (olasilik: {info.language_probability:.2f})")
                self._log(f"  [Whisper] Transkripsiyon başlıyor (beam={beam_size}, word_ts=True)...")

                segments = []
                seg_count = 0
                last_progress = time.time()
                try:
                    for seg in raw_segments:
                        seg_count += 1
                        now = time.time()
                        if seg_count % 10 == 0 or (now - last_progress) > 30:
                            self._log(f"  [Whisper] ... {seg_count} segment işlendi ({seg.end:.0f}s/{now - t0:.0f}s)")
                            last_progress = now

                        words = []
                        scores = []
                        for w in (seg.words or []):
                            words.append({
                                'word': w.word.strip(),
                                'start': round(w.start, 3),
                                'end': round(w.end, 3),
                                'score': round(w.probability, 3),
                            })
                            scores.append(w.probability)

                        avg_conf = round(sum(scores) / len(scores), 3) if scores else 0.0
                        segments.append({
                            'start': round(seg.start, 3),
                            'end': round(seg.end, 3),
                            'text': seg.text.strip(),
                            'speaker': '',
                            'confidence': avg_conf,
                            'words': words,
                        })
                except Exception as gen_err:
                    self._log(f"  [Whisper] Generator hatası ({seg_count} segment sonra): {gen_err}")
                    self._log(f"  [Whisper] word_timestamps=False ile yeniden deneniyor...")
                    raw2, info2 = fw_model.transcribe(
                        audio_path, language=language, beam_size=beam_size,
                        word_timestamps=False, vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                        initial_prompt=initial_prompt if initial_prompt else None,
                    )
                    segments = []
                    for seg in raw2:
                        segments.append({
                            'start': round(seg.start, 3), 'end': round(seg.end, 3),
                            'text': seg.text.strip(), 'speaker': '', 'confidence': 0.0, 'words': [],
                        })

                # 0 segment → fallback
                if not segments:
                    self._log(f"  [Whisper] 0 segment — word_timestamps=False ile yeniden deneniyor...")
                    raw2, info2 = fw_model.transcribe(
                        audio_path, language=language, beam_size=beam_size,
                        word_timestamps=False, vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                        initial_prompt=initial_prompt if initial_prompt else None,
                    )
                    for seg in raw2:
                        segments.append({
                            'start': round(seg.start, 3), 'end': round(seg.end, 3),
                            'text': seg.text.strip(), 'speaker': '', 'confidence': 0.0, 'words': [],
                        })

                # diarize stage varsa konusmaci ata
                diar_segments = (context.get('diarize') or {}).get('segments', [])
                if diar_segments:
                    self._assign_speakers(segments, diar_segments)

                elapsed = round(time.time() - t0, 2)
                self._log(f"  [Whisper] {len(segments)} segment ({elapsed:.1f}s)")

                return {
                    'status': 'ok',
                    'segments': segments,
                    'total_segments': len(segments),
                    'stage_time_sec': elapsed,
                    'detected_language': info.language,
                    'language_probability': round(info.language_probability, 3),
                }

            except ImportError:
                self._log('  [Whisper] Kurulu degil - pip install faster-whisper')
                return {
                    'status': 'error',
                    'segments': [],
                    'total_segments': 0,
                    'stage_time_sec': round(time.time() - t0, 2),
                    'error': 'faster-whisper not installed',
                }

            except Exception as e:
                self._log(f"  [Whisper] Hata: {e}")
                import traceback
                tb_str = traceback.format_exc()
                self._log(f"  [Whisper] Traceback: {tb_str[-500:]}")
                return {
                    'status': 'error',
                    'segments': [],
                    'total_segments': 0,
                    'stage_time_sec': round(time.time() - t0, 2),
                    'error': str(e),
                }

            finally:
                if fw_model is not None:
                    del fw_model
                VRAMManager.release()
                self._log(f"  [Whisper] Model bosaltildi (VRAM: {VRAMManager.get_usage()})")

        def _build_initial_prompt(self, opts: dict) -> str:
            """
            TMDB cast listesinden Whisper initial_prompt oluştur.
            Yabancı isimlerin fonetik bozulmasını önlemek için kullanılır.
            """
            cast = opts.get('tmdb_cast') or []
            if not cast:
                return ''
            names = []
            for entry in cast:
                actor = (entry.get('actor_name') or entry.get('name') or '').strip()
                char = (entry.get('character_name') or entry.get('character') or '').strip()
                if actor:
                    names.append(actor)
                if char:
                    names.append(char)
            seen = set()
            unique = []
            for n in names:
                if n and n not in seen:
                    seen.add(n)
                    unique.append(n)
                if len(unique) >= 20:
                    break
            if not unique:
                return ''
            return 'Bu filmde şu isimler geçmektedir: ' + ', '.join(unique) + '.'

        def _resolve_compute_type(self, raw_compute_type, device):
            default_type = 'float16' if device == 'cuda' else 'int8'
            if raw_compute_type is None:
                return default_type
            normalized = str(raw_compute_type).strip().lower()
            if normalized in ('', 'none', 'null', 'auto'):
                return default_type
            if device != 'cuda' and normalized == 'float16':
                return 'int8'
            return normalized

        def _assign_speakers(self, segments, diar_segments):
            """Assign speakers from diarization to transcription segments."""
            def safe_float(val, default=0.0):
                """Safely convert to float, handling None and invalid values."""
                if val is None:
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            for seg in segments:
                best_speaker = ''
                best_overlap = 0.0
                for ds in diar_segments:
                    # Safe float conversion for timestamps
                    ds_start = safe_float(ds.get('start', 0))
                    ds_end = safe_float(ds.get('end', 0))
                    seg_start = safe_float(seg.get('start', 0))
                    seg_end = safe_float(seg.get('end', 0))

                    overlap_start = max(seg_start, ds_start)
                    overlap_end = min(seg_end, ds_end)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = ds.get('speaker', '')
                if best_speaker:
                    seg['speaker'] = best_speaker
