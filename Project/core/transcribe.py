"""
transcribe.py — [D] WhisperX large-v3 Türkçe transcript + kelime hizalama.

Girdi:  audio_clean.wav (16kHz mono) + diarization segments
Çıktı:  segments[ {start, end, speaker_id, text, confidence, words} ]

Kelime bazlı timestamp + konuşmacı ataması.

VRAM: ~3-4 GB → stage sonrası boşaltılır.
"""

import time

from audio.utils.vram_manager import VRAMManager


class TranscribeStage:
    """WhisperX large-v3 transkripsiyon + hizalama."""

    def __init__(self, log_cb=None):
        self._log = log_cb or print

    def run(self, audio_path: str, diarization: dict = None, **opts) -> dict:
        """
        Transcript üret + kelime hizalama + konuşmacı atama.

        Args:
            audio_path: 16kHz mono WAV
            diarization: DiarizeStage çıktısı (segments listesi)
            opts:
                whisper_model: "large-v3" (default)
                whisper_language: "tr" (default)
                compute_type: "float16" (default, GPU) / "int8" (CPU)
                batch_size: 16 (default)

        Returns:
            {
                "status": "ok",
                "segments": [{
                    "start": 0.5, "end": 3.2,
                    "speaker": "SPEAKER_00",
                    "text": "Merhaba",
                    "confidence": 0.95,
                    "words": [{"word": "Merhaba", "start": 0.5, "end": 0.9, "score": 0.95}]
                }],
                "total_segments": 245,
                "stage_time_sec": 180.0
            }
        """
        t0 = time.time()
        model_name = opts.get("whisper_model", "large-v3")
        language = opts.get("whisper_language", "tr")
        batch_size = int(opts.get("batch_size", 16))
        device = VRAMManager.get_device()
        compute_type = self._resolve_compute_type(opts.get("compute_type"), device)

        # CPU'da float16 hesaplama desteklenmediği için güvenli fallback.
        if device != "cuda" and str(compute_type).lower() == "float16":
            self._log("  [WhisperX] CPU'da float16 desteklenmiyor — int8'e düşülüyor")
            compute_type = "int8"

        whisperx_model = None
        align_model = None

        try:
            import whisperx

            # ── Model yükleme ──
            self._log(f"  [WhisperX] {model_name} yükleniyor ({device}, {compute_type})...")
            whisperx_model = whisperx.load_model(
                model_name, device,
                compute_type=compute_type,
                language=language,
            )
            self._log(f"  [WhisperX] Yüklendi (VRAM: {VRAMManager.get_usage()})")

            # ── Transkripsiyon ──
            self._log("  [WhisperX] Transkripsiyon başlıyor...")
            result = whisperx_model.transcribe(
                audio_path,
                batch_size=batch_size,
                language=language,
            )

            # ── Kelime hizalama ──
            try:
                self._log("  [WhisperX] Kelime hizalama...")
                align_model, metadata = whisperx.load_align_model(
                    language_code=language, device=device
                )
                result = whisperx.align(
                    result["segments"], align_model, metadata,
                    audio_path, device,
                )
            except Exception as e:
                self._log(f"  [WhisperX] Hizalama hatası: {e} — devam ediliyor")

            # ── Konuşmacı ataması (diarizasyon varsa) ──
            # BUG-K3 FIX: DiarizationPipeline burada TEKRAR oluşturulmaz.
            # DiarizeStage [C] zaten çalıştırdı. Sadece zaman bazlı eşleştirme yap.
            diar_segments = (diarization or {}).get("segments", [])
            if diar_segments:
                self._assign_speakers_by_time(result, diar_segments)

            # ── Sonuçları formatla ──
            segments = []
            for seg in result.get("segments", []):
                words = seg.get("words", [])
                scores = [w.get("score", 0) for w in words if "score" in w]
                avg_conf = round(sum(scores) / len(scores), 3) if scores else 0.0

                segments.append({
                    "start": round(seg.get("start", 0), 3),
                    "end": round(seg.get("end", 0), 3),
                    "text": seg.get("text", "").strip(),
                    "speaker": seg.get("speaker", ""),
                    "confidence": avg_conf,
                    "words": words,
                })

            elapsed = round(time.time() - t0, 2)
            self._log(f"  [WhisperX] {len(segments)} segment ({elapsed:.1f}s)")

            return {
                "status": "ok",
                "segments": segments,
                "total_segments": len(segments),
                "stage_time_sec": elapsed,
            }

        except ImportError:
            self._log("  [WhisperX] Kurulu değil — pip install whisperx")
            return {
                "status": "error",
                "segments": [],
                "total_segments": 0,
                "stage_time_sec": round(time.time() - t0, 2),
                "error": "whisperx not installed",
            }

        except Exception as e:
            self._log(f"  [WhisperX] Hata: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "segments": [],
                "total_segments": 0,
                "stage_time_sec": round(time.time() - t0, 2),
                "error": str(e),
            }

        finally:
            # VRAM boşalt
            if whisperx_model is not None:
                del whisperx_model
            if align_model is not None:
                del align_model
            VRAMManager.release()
            self._log(f"  [WhisperX] Model boşaltıldı (VRAM: {VRAMManager.get_usage()})")

    def _resolve_compute_type(self, raw_compute_type, device: str) -> str:
        """compute_type değerini normalize et ve cihaza göre güvenli hale getir."""
        default_type = "float16" if device == "cuda" else "int8"

        if raw_compute_type is None:
            return default_type

        normalized = str(raw_compute_type).strip().lower()
        if normalized in ("", "none", "null", "auto"):
            return default_type

        # CPU'da float16 hesaplama desteklenmediği için güvenli fallback.
        if device != "cuda" and normalized == "float16":
            self._log("  [WhisperX] CPU'da float16 desteklenmiyor — int8'e düşülüyor")
            return "int8"

        return normalized

    def _assign_speakers_by_time(self, result: dict,
                                  diar_segments: list):
        """
        WhisperX sonuçlarına zaman bazlı konuşmacı ataması.
        Her transcript segment için en fazla örtüşen diarizasyon segment'ini bul.
        """
        for seg in result.get("segments", []):
            best_speaker = ""
            best_overlap = 0.0

            for ds in diar_segments:
                # Overlap hesabı
                overlap_start = max(seg.get("start", 0), ds["start"])
                overlap_end = min(seg.get("end", 0), ds["end"])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = ds["speaker"]

            if best_speaker:
                seg["speaker"] = best_speaker
