"""
denoise.py — [B] DeepFilterNet3 gürültü temizleme.

Girdi:  audio_raw_48k.wav (48kHz — DF3 48kHz istiyor)
Çıktı:  audio_clean.wav (48kHz → sonra 16kHz'e resample edilecek)

Konuşma netleşir → WhisperX doğruluğu artar.
Hata durumunda orijinal WAV döner (fallback — pipeline durmaz).

VRAM: ~200 MB → stage sonrası boşaltılır.
"""

import time
from pathlib import Path

from core.vram_manager import VRAMManager


class DenoiseStage:
    """DeepFilterNet3 ile arka plan gürültü temizleme."""

    def __init__(self, log_cb=None):
        self._log = log_cb or print

    def run(self, input_path: str, output_path: str, **opts) -> dict:
        """
        Gürültü azaltılmış WAV üret.

        Args:
            input_path: 48kHz WAV
            output_path: Temizlenmiş WAV çıktı yolu

        Returns:
            {
                "status": "ok" | "fallback",
                "output_path": "path/audio_clean.wav",
                "stage_time_sec": 45.0
            }
        """
        t0 = time.time()
        enabled = opts.get("denoise_enabled", True)

        if not enabled:
            self._log("  [DF3] Devre dışı — orijinal WAV kullanılıyor")
            return {
                "status": "skipped",
                "output_path": input_path,
                "stage_time_sec": 0.0,
            }

        try:
            self._log("  [DF3] Model yükleniyor...")
            from df import enhance, init_df
            import torchaudio

            model, df_state, _ = init_df()
            self._log(f"  [DF3] Model yüklendi (VRAM: {VRAMManager.get_usage()})")

            audio, sr = torchaudio.load(input_path)

            # DF3 48kHz bekliyor — sample rate uyumsuzluğu varsa resample
            target_sr = df_state.sr()
            if sr != target_sr:
                self._log(f"  [DF3] Resample: {sr}Hz → {target_sr}Hz")
                audio = torchaudio.functional.resample(audio, sr, target_sr)

            self._log("  [DF3] Gürültü azaltma başlıyor...")
            enhanced = enhance(model, df_state, audio)

            # Kaydet
            # BUG-K6 FIX: DF3 enhance() 2D tensor dönebilir, unsqueeze sadece 1D için
            save_tensor = enhanced if enhanced.dim() >= 2 else enhanced.unsqueeze(0)
            torchaudio.save(output_path, save_tensor, target_sr)
            self._log(f"  [DF3] Gürültü azaltıldı → {Path(output_path).name}")

            # VRAM boşalt
            del model, df_state, audio, enhanced
            VRAMManager.release()
            self._log(f"  [DF3] Model boşaltıldı (VRAM: {VRAMManager.get_usage()})")

            elapsed = round(time.time() - t0, 2)
            return {
                "status": "ok",
                "output_path": output_path,
                "stage_time_sec": elapsed,
            }

        except ImportError:
            self._log("  [DF3] DeepFilterNet kurulu değil — pip install deepfilternet")
            self._log("  [DF3] Orijinal WAV kullanılıyor (fallback)")
            return {
                "status": "fallback",
                "output_path": input_path,
                "stage_time_sec": round(time.time() - t0, 2),
                "error": "deepfilternet not installed",
            }

        except Exception as e:
            self._log(f"  [DF3] Hata: {e} — orijinal WAV kullanılıyor (fallback)")
            VRAMManager.release()
            return {
                "status": "fallback",
                "output_path": input_path,
                "stage_time_sec": round(time.time() - t0, 2),
                "error": str(e),
            }
