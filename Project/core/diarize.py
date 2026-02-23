"""
diarize.py — [C] PyAnnote speaker-diarization-3.1 konuşmacı tespiti.

Girdi:  audio_clean.wav (16kHz mono)
Çıktı:  segments[ {start, end, speaker_id} ]

"Kim ne zaman konuştu?" → SPEAKER_00, SPEAKER_01...

Gereksinim:
  - HuggingFace token (gated model)
  - pip install pyannote.audio

VRAM: ~1-2 GB → stage sonrası boşaltılır.
"""

import time

from core.vram_manager import VRAMManager


class DiarizeStage:
    """PyAnnote 3.1 konuşmacı diarizasyonu."""

    MODEL_ID = "pyannote/speaker-diarization-3.1"

    def __init__(self, log_cb=None):
        self._log = log_cb or print

    def run(self, audio_path: str, **opts) -> dict:
        """
        Konuşmacı diarizasyonu.

        Args:
            audio_path: 16kHz mono WAV
            opts:
                hf_token: HuggingFace token (zorunlu)
                max_speakers: Maksimum konuşmacı sayısı (opsiyonel)

        Returns:
            {
                "status": "ok",
                "segments": [{"start": 0.5, "end": 3.2, "speaker": "SPEAKER_00"}],
                "speakers_found": 5,
                "stage_time_sec": 60.3
            }
        """
        t0 = time.time()
        hf_token = opts.get("hf_token", "")
        max_speakers = opts.get("max_speakers")

        if not hf_token:
            self._log("  [PyAnnote] HF_TOKEN yok — diarizasyon atlanıyor")
            return {
                "status": "skipped",
                "segments": [],
                "speakers_found": 0,
                "stage_time_sec": 0.0,
                "error": "hf_token missing",
            }

        pipeline = None
        try:
            from pyannote.audio import Pipeline
            import torch

            device = VRAMManager.get_device()
            self._log(f"  [PyAnnote] Model yükleniyor ({device})...")

            pipeline = Pipeline.from_pretrained(
                self.MODEL_ID,
                use_auth_token=hf_token
            )
            pipeline = pipeline.to(torch.device(device))
            self._log(f"  [PyAnnote] Yüklendi (VRAM: {VRAMManager.get_usage()})")

            # Diarizasyon
            self._log("  [PyAnnote] Diarizasyon başlıyor...")
            kwargs = {}
            if max_speakers and int(max_speakers) > 0:
                kwargs["max_speakers"] = int(max_speakers)

            diarization = pipeline(audio_path, **kwargs)

            # Sonuçları parse et
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker,
                })

            speakers_found = len(set(s["speaker"] for s in segments))
            elapsed = round(time.time() - t0, 2)

            self._log(
                f"  [PyAnnote] {speakers_found} konuşmacı, "
                f"{len(segments)} segment ({elapsed:.1f}s)"
            )

            return {
                "status": "ok",
                "segments": segments,
                "speakers_found": speakers_found,
                "stage_time_sec": elapsed,
            }

        except ImportError:
            self._log("  [PyAnnote] Kurulu değil — pip install pyannote.audio")
            return {
                "status": "error",
                "segments": [],
                "speakers_found": 0,
                "stage_time_sec": round(time.time() - t0, 2),
                "error": "pyannote.audio not installed",
            }

        except Exception as e:
            self._log(f"  [PyAnnote] Hata: {e}")
            return {
                "status": "error",
                "segments": [],
                "speakers_found": 0,
                "stage_time_sec": round(time.time() - t0, 2),
                "error": str(e),
            }

        finally:
            # VRAM boşalt
            if pipeline is not None:
                del pipeline
            VRAMManager.release()
            self._log(f"  [PyAnnote] Model boşaltıldı (VRAM: {VRAMManager.get_usage()})")
