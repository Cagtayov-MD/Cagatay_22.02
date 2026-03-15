"""
detect_language.py — ASR öncesi dil tespiti (faster-whisper tiny model)

Strateji:
  • Video >= 4 dakika: 2 örnek (0-30s, 240-270s)
  • Video < 4 dakika: 1 örnek (0-30s)

Karar mantığı:
  • Her iki örnek aynı dil → o dil seçilir (both_matched)
  • Örnekler çelişiyor → yüksek confidence olan seçilir (conflicting_use_second)
  • Tek örnek → doğrudan kullanılır (single_sample)
  • Hata → "unknown" döner (error)
"""

import os
import subprocess
import time
from pathlib import Path


class LanguageDetectionStage:
    """
    faster-whisper tiny model ile dil tespiti yapar.
    Pipeline'daki diğer stage'lerle aynı arayüzü kullanır.
    """

    SAMPLE_DURATION = 30   # saniye
    MID_OFFSET = 240       # 4. dakika (240s)
    MIN_DURATION_FOR_TWO = 240  # iki örnek için minimum video süresi

    def __init__(self, ffmpeg_path: str = "ffmpeg", log_cb=None):
        self.ffmpeg = ffmpeg_path
        self._log = log_cb or print

    def run(self, video_path: str, audio_dir: str, duration_sec: float = 0.0) -> dict:
        """
        Dil tespiti yap.

        Args:
            video_path:   Video dosyası yolu
            audio_dir:    Geçici ses dosyaları için dizin
            duration_sec: Video süresi (saniye). 0 ise ffprobe ile tespit edilir.

        Returns:
            {
                "status": "ok" | "error",
                "stage_time_sec": float,
                "detected_language": "tr" | "en" | ... | "unknown",
                "language_is_turkish": bool,
                "confidence": float,
                "samples": [{"sample_name": str, "language": str, "confidence": float}],
                "decision_logic": str,
                "error": str  # sadece status=="error" ise
            }
        """
        t0 = time.time()
        result = {
            "status": "ok",
            "stage_time_sec": 0.0,
            "detected_language": "unknown",
            "language_is_turkish": False,
            "confidence": 0.0,
            "samples": [],
            "decision_logic": "unknown",
        }

        try:
            # Video süresini al
            if duration_sec <= 0:
                duration_sec = self._get_duration(video_path)

            # Örnekleme noktalarını belirle
            sample_points = [("initial", 0)]
            if duration_sec >= self.MIN_DURATION_FOR_TWO:
                sample_points.append(("mid_4min", self.MID_OFFSET))

            # Her örnek için dil tespiti
            samples = []
            for sample_name, offset in sample_points:
                seg_end = min(offset + self.SAMPLE_DURATION, duration_sec)
                if seg_end <= offset:
                    continue
                seg_path = os.path.join(audio_dir, f"lang_sample_{sample_name}.wav")
                ok = self._extract_segment(video_path, offset, seg_end, seg_path)
                if not ok:
                    self._log(f"  [DİL TESPİTİ] Segment çıkartılamadı: {sample_name}")
                    continue
                lang, conf = self._detect_from_file(seg_path)
                self._log(
                    f"  [DİL TESPİTİ] {sample_name:12s} ({offset}-{int(seg_end)}s): "
                    f"{lang.upper()} ({conf*100:.1f}%)"
                )
                samples.append({
                    "sample_name": sample_name,
                    "language": lang,
                    "confidence": round(conf, 4),
                    "start_sec": offset,
                    "end_sec": int(seg_end),
                })
                # Geçici dosyayı temizle
                try:
                    os.remove(seg_path)
                except OSError:
                    pass

            result["samples"] = samples

            if not samples:
                result["status"] = "error"
                result["error"] = "Hiçbir örnek işlenemedi"
                result["detected_language"] = "unknown"
                result["decision_logic"] = "no_samples"
            elif len(samples) == 1:
                result["detected_language"] = samples[0]["language"]
                result["confidence"] = samples[0]["confidence"]
                result["decision_logic"] = "single_sample"
            else:
                # İki örnek var — karar mantığı
                s0, s1 = samples[0], samples[1]
                if s0["language"] == s1["language"]:
                    # Her ikisi de aynı dil
                    result["detected_language"] = s0["language"]
                    result["confidence"] = round((s0["confidence"] + s1["confidence"]) / 2, 4)
                    result["decision_logic"] = "both_matched"
                else:
                    # Çelişki — yüksek confidence olan kazanır
                    winner = s1 if s1["confidence"] >= s0["confidence"] else s0
                    result["detected_language"] = winner["language"]
                    result["confidence"] = winner["confidence"]
                    result["decision_logic"] = "conflicting_use_second"
                    self._log(
                        f"  [DİL TESPİTİ] Çelişki: {s0['language']} vs {s1['language']} "
                        f"→ {winner['language'].upper()} seçildi (yüksek conf)"
                    )

            result["language_is_turkish"] = (result["detected_language"] == "tr")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["detected_language"] = "unknown"
            result["decision_logic"] = "exception"
            self._log(f"  [DİL TESPİTİ] HATA: {e}")

        result["stage_time_sec"] = round(time.time() - t0, 2)
        lang_display = result["detected_language"].upper()
        conf_pct = result["confidence"] * 100
        self._log(
            f"  [DİL TESPİTİ] Sonuç: {lang_display} "
            f"({conf_pct:.1f}%) — {result['decision_logic']} "
            f"[{result['stage_time_sec']}s]"
        )
        return result

    # ─────────────────────────────────────────────────────────────────
    # Yardımcı metodlar
    # ─────────────────────────────────────────────────────────────────

    def _get_duration(self, video_path: str) -> float:
        """ffprobe ile video süresini saniye olarak al."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]
            out = subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, timeout=30
            ).strip()
            return float(out)
        except Exception:
            return 0.0

    def _extract_segment(
        self, video_path: str, start: float, end: float, out_path: str
    ) -> bool:
        """ffmpeg ile ses segmenti çıkart (16kHz mono WAV)."""
        try:
            cmd = [
                self.ffmpeg, "-y",
                "-ss", str(int(start)),
                "-i", video_path,
                "-t", str(int(end - start)),
                "-vn",
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                out_path,
            ]
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
                check=True,
            )
            return Path(out_path).is_file() and Path(out_path).stat().st_size > 0
        except Exception:
            return False

    def _detect_from_file(self, wav_path: str) -> tuple[str, float]:
        """
        faster-whisper tiny model ile dil tespiti.
        Model sadece bu çağrı sırasında yüklenir, sonra bellekten temizlenir.

        Returns:
            (language_code, confidence)  örn: ("tr", 0.923)
        """
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                "tiny",
                device="cuda",
                compute_type="int8",  # tiny için int8 yeterli, hızlı
            )
            segments, info = model.transcribe(
                wav_path,
                beam_size=1,
                language=None,        # otomatik dil tespiti
                task="transcribe",
                without_timestamps=True,
            )
            # Segmentleri tüket (generator)
            list(segments)

            lang = info.language or "unknown"
            conf = float(info.language_probability) if info.language_probability else 0.0

            # Modeli bellekten temizle
            del model
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            return lang, conf

        except Exception as e:
            self._log(f"  [DİL TESPİTİ] faster-whisper hatası: {e}")
            return "unknown", 0.0
