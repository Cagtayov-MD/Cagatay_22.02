"""
detect_language.py — ASR öncesi dil tespiti (faster-whisper tiny model)

Strateji (v2 — kanal deneme destekli):
  1. Önce karışık mono (mix) ile dene
  2. Güven < CHANNEL_RETRY_THRESHOLD ise → kanal 0 dene → kanal 1 dene
  3. Hâlâ düşükse ve video >= 8dk → 8. dakikadan ek sample al
  4. En yüksek güveni veren kanalı seç → selected_channel olarak döndür

Örnekleme:
  • Video >= 4 dakika: 2 zaman noktası (0-30s, 240-270s)
  • Video < 4 dakika: 1 zaman noktası (0-30s)

Karar mantığı (zaman noktası bazında — mevcut davranış korundu):
  • Her iki örnek aynı dil → o dil seçilir (both_matched)
  • Örnekler çelişiyor → yüksek confidence olan seçilir (conflicting_use_higher)
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

    v2: Stereo arşiv materyalde kanal deneme desteği.
    """

    SAMPLE_DURATION = 30   # saniye
    MID_OFFSET = 240       # 4. dakika (240s)
    MIN_DURATION_FOR_TWO = 240  # iki örnek için minimum video süresi

    # ── Kanal deneme sabitleri ──
    CHANNEL_RETRY_THRESHOLD = 0.75   # bu güvenin altındaysa kanal dene
    FALLBACK_OFFSET = 480            # 8. dakika — ek sample noktası
    MIN_DURATION_FOR_FALLBACK = 510  # 8.5dk — fallback sample için minimum süre

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
                "samples": [...],
                "decision_logic": str,
                "selected_channel": None | 0 | 1,   # ← YENİ
                "channel_trials": [...],              # ← YENİ (debug)
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
            "selected_channel": None,       # None = karışık mono (eski davranış)
            "channel_trials": [],           # debug bilgisi
        }

        try:
            # Video süresini al
            if duration_sec <= 0:
                duration_sec = self._get_duration(video_path)

            # Kaç ses kanalı var?
            n_channels = self._get_audio_channels(video_path)
            self._log(f"  [DİL TESPİTİ] Kaynak ses kanalı: {n_channels}")

            # ── Aşama 1: Karışık mono ile dene (mevcut davranış) ──
            lang, conf, samples, logic = self._detect_with_channel(
                video_path, audio_dir, duration_sec, channel=None, tag="mix"
            )
            result["samples"] = samples
            result["channel_trials"].append({
                "channel": "mix", "language": lang, "confidence": conf,
            })

            best_lang, best_conf, best_channel = lang, conf, None
            best_logic = logic

            # ── Aşama 2: Güven düşükse ve stereo ise → kanal dene ──
            if conf < self.CHANNEL_RETRY_THRESHOLD and n_channels >= 2:
                self._log(
                    f"  [DİL TESPİTİ] Mix güven düşük ({conf*100:.1f}% < "
                    f"{self.CHANNEL_RETRY_THRESHOLD*100:.0f}%) — kanal deneme başlıyor"
                )

                for ch in (0, 1):
                    ch_lang, ch_conf, ch_samples, ch_logic = self._detect_with_channel(
                        video_path, audio_dir, duration_sec,
                        channel=ch, tag=f"ch{ch}"
                    )
                    result["channel_trials"].append({
                        "channel": ch, "language": ch_lang, "confidence": ch_conf,
                    })
                    self._log(
                        f"  [DİL TESPİTİ] Kanal {ch}: {ch_lang.upper()} ({ch_conf*100:.1f}%)"
                    )
                    if ch_conf > best_conf:
                        best_lang, best_conf, best_channel = ch_lang, ch_conf, ch
                        best_logic = f"channel_{ch}_selected"
                        result["samples"] = ch_samples  # en iyi kanalın sample'ları

            # ── Aşama 3: Hâlâ düşükse ve video yeterince uzunsa → 8. dk'dan dene ──
            if (best_conf < self.CHANNEL_RETRY_THRESHOLD
                    and duration_sec >= self.MIN_DURATION_FOR_FALLBACK):
                self._log(
                    f"  [DİL TESPİTİ] Güven hâlâ düşük ({best_conf*100:.1f}%) "
                    f"— 8. dakikadan ek sample alınıyor"
                )
                # Ek sample: en iyi kanal (veya None) ile 8. dk
                fb_lang, fb_conf = self._detect_single_sample(
                    video_path, audio_dir,
                    offset=self.FALLBACK_OFFSET,
                    channel=best_channel,
                    tag="fallback_8min"
                )
                result["channel_trials"].append({
                    "channel": best_channel if best_channel is not None else "mix",
                    "offset": self.FALLBACK_OFFSET,
                    "language": fb_lang,
                    "confidence": fb_conf,
                    "tag": "fallback_8min",
                })
                if fb_conf > best_conf:
                    best_lang, best_conf = fb_lang, fb_conf
                    best_logic = f"fallback_8min_ch{best_channel if best_channel is not None else 'mix'}"

            # ── Sonuç ──
            result["detected_language"] = best_lang
            result["confidence"] = best_conf
            result["decision_logic"] = best_logic
            result["selected_channel"] = best_channel
            result["language_is_turkish"] = (best_lang == "tr")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["detected_language"] = "unknown"
            result["decision_logic"] = "exception"
            self._log(f"  [DİL TESPİTİ] HATA: {e}")

        result["stage_time_sec"] = round(time.time() - t0, 2)
        lang_display = result["detected_language"].upper()
        conf_pct = result["confidence"] * 100
        ch_info = (f"kanal={result['selected_channel']}"
                   if result["selected_channel"] is not None else "mix")
        self._log(
            f"  [DİL TESPİTİ] Sonuç: {lang_display} "
            f"({conf_pct:.1f}%) — {result['decision_logic']} "
            f"[{ch_info}] [{result['stage_time_sec']}s]"
        )
        return result

    # ─────────────────────────────────────────────────────────────────
    # Kanal deneme mantığı
    # ─────────────────────────────────────────────────────────────────

    def _detect_with_channel(
        self, video_path: str, audio_dir: str, duration_sec: float,
        channel, tag: str
    ):
        """
        Belirli bir kanal (veya karışık mono) ile dil tespiti yap.
        Mevcut çoklu-sample + karar mantığını aynen korur.

        Args:
            channel: None=karışık mono, 0=sol, 1=sağ

        Returns:
            (language, confidence, samples_list, decision_logic)
        """
        sample_points = [("initial", 0)]
        if duration_sec >= self.MIN_DURATION_FOR_TWO:
            sample_points.append(("mid_4min", self.MID_OFFSET))

        samples = []
        for sample_name, offset in sample_points:
            seg_end = min(offset + self.SAMPLE_DURATION, duration_sec)
            if seg_end <= offset:
                continue
            seg_path = os.path.join(
                audio_dir, f"lang_sample_{tag}_{sample_name}.wav"
            )
            ok = self._extract_segment(
                video_path, offset, seg_end, seg_path, channel=channel
            )
            if not ok:
                self._log(
                    f"  [DİL TESPİTİ] Segment çıkartılamadı: {tag}_{sample_name}"
                )
                continue
            lang, conf = self._detect_from_file(seg_path)
            self._log(
                f"  [DİL TESPİTİ] {tag}_{sample_name:12s} "
                f"({offset}-{int(seg_end)}s): "
                f"{lang.upper()} ({conf*100:.1f}%)"
            )
            samples.append({
                "sample_name": f"{tag}_{sample_name}",
                "language": lang,
                "confidence": round(conf, 4),
                "start_sec": offset,
                "end_sec": int(seg_end),
                "channel": channel,
            })
            # Geçici dosyayı temizle
            try:
                os.remove(seg_path)
            except OSError:
                pass

        # Karar mantığı (mevcut davranış aynen)
        if not samples:
            return "unknown", 0.0, samples, "no_samples"
        elif len(samples) == 1:
            return samples[0]["language"], samples[0]["confidence"], samples, "single_sample"
        else:
            s0, s1 = samples[0], samples[1]
            if s0["language"] == s1["language"]:
                avg_conf = round((s0["confidence"] + s1["confidence"]) / 2, 4)
                return s0["language"], avg_conf, samples, "both_matched"
            else:
                winner = s1 if s1["confidence"] >= s0["confidence"] else s0
                self._log(
                    f"  [DİL TESPİTİ] Çelişki ({tag}): "
                    f"{s0['language']} vs {s1['language']} "
                    f"→ {winner['language'].upper()} seçildi (yüksek conf)"
                )
                return (winner["language"], winner["confidence"],
                        samples, "conflicting_use_higher")

    def _detect_single_sample(
        self, video_path: str, audio_dir: str,
        offset: float, channel, tag: str
    ):
        """Tek bir noktadan dil tespiti (fallback sample için)."""
        duration_sec = self._get_duration(video_path)
        seg_end = min(offset + self.SAMPLE_DURATION, duration_sec)
        if seg_end <= offset:
            return "unknown", 0.0

        seg_path = os.path.join(audio_dir, f"lang_sample_{tag}.wav")
        ok = self._extract_segment(
            video_path, offset, seg_end, seg_path, channel=channel
        )
        if not ok:
            return "unknown", 0.0
        lang, conf = self._detect_from_file(seg_path)
        ch_display = channel if channel is not None else "mix"
        self._log(
            f"  [DİL TESPİTİ] {tag} ({int(offset)}-{int(seg_end)}s, "
            f"ch={ch_display}): {lang.upper()} ({conf*100:.1f}%)"
        )
        try:
            os.remove(seg_path)
        except OSError:
            pass
        return lang, conf

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

    def _get_audio_channels(self, video_path: str) -> int:
        """ffprobe ile ses kanalı sayısını al."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=channels",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]
            out = subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, timeout=30
            ).strip()
            return int(out)
        except Exception:
            return 1  # bilinmiyorsa mono varsay

    def _extract_segment(
        self, video_path: str, start: float, end: float, out_path: str,
        channel=None,
    ) -> bool:
        """
        ffmpeg ile ses segmenti çıkart (16kHz mono WAV).

        Args:
            channel: None → karışık mono (-ac 1)
                     0    → sol kanal  (-af "pan=mono|c0=c0")
                     1    → sağ kanal  (-af "pan=mono|c0=c1")
        """
        try:
            cmd = [
                self.ffmpeg, "-y",
                "-ss", str(int(start)),
                "-i", video_path,
                "-t", str(int(end - start)),
                "-vn",
                "-ar", "16000",
            ]

            if channel is not None:
                # Belirli kanalı al → mono
                cmd += ["-af", f"pan=mono|c0=c{channel}"]
            else:
                # Karışık mono (eski davranış)
                cmd += ["-ac", "1"]

            cmd += ["-f", "wav", out_path]

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

    def _detect_from_file(self, wav_path: str) -> tuple:
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
                device="cpu",
                compute_type="int8",  # CPU + int8: VRAM çakışması önlenir, tiny için hız farkı yok
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
