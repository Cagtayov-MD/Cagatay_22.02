"""
[D] TRANSCRIBE - faster-whisper large-v3 Turkce transcript + kelime timestamp.

Bu stage, AudioPipeline standardina uyar:
  run(context: dict, config: dict) -> dict
"""

import time
from audio.utils.vram_manager import VRAMManager


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
            _raw_beam = opts.get('beam_size') if opts.get('beam_size') is not None else opts.get('batch_size', 5)
            beam_size = min(int(_raw_beam), 10)
        except (ValueError, TypeError):
            self._log('  [Whisper] Geçersiz beam_size — varsayılan 5 kullanılıyor')
            beam_size = 5

        device = VRAMManager.get_device()
        compute_type = self._resolve_compute_type(opts.get('compute_type', None), device)

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
            traceback.print_exc()
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
        for seg in segments:
            best_speaker = ''
            best_overlap = 0.0
            for ds in diar_segments:
                overlap_start = max(seg['start'], ds['start'])
                overlap_end = min(seg['end'], ds['end'])
                overlap = max(0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = ds['speaker']
            if best_speaker:
                seg['speaker'] = best_speaker
