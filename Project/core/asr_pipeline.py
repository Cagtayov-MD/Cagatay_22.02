"""
asr_pipeline.py — Arsiv Decode Ses Analiz Sistemi

Sabit kural: Güvenilirlik > Hız (tüm sistem için geçerli)

Desteklenen Program Türleri:
    'film_dizi'     → DF3 → PyAnnote → WhisperX → Ollama
    'muzik_programi'→ Demucs htdemucs_6s → InaSpeech → PyAnnote → WhisperX + MusicBrainz
    'kisa_haber'    → DF3 → PyAnnote → WhisperX → Ollama
    'mac'           → DF3 → PyAnnote → WhisperX + OCR skor bandı + Essentia

Model Atamaları (session_log 21.02.2026 kararları):
    Demucs htdemucs_6s  → SADECE Müzik Programı
    DF3 (DeepFilterNet3)→ Film/Dizi, Maçlar, Kısa Haber
    Ollama llama3.1:8b  → tüm başlıklar, local
    PyAnnote 3.1        → tüm başlıklar
    WhisperX large-v3   → tüm başlıklar
    Qwen3-VL            → görsel analiz (pipeline_runner'dan çağrılır)

Kimlik Sistemi:
    Yüz vektörü (512 float) + ses vektörü → tek kimlik nesnesi
    Eşleşme yoksa: KARAKTER_N / KONUŞMACI_N numaralandırma

Confidence Engine:
    İki kaynak örtüşüyor → HIGH (>= 0.85)
    Bir kaynak           → MEDIUM (0.60-0.84)
    Çelişiyor            → low_confidence flag + her iki değer kaydedilir
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Veri Yapıları
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioSegment:
    """Tek ses segmenti — tüm program türlerinde ortak yapı."""
    start: float                    # saniye
    end: float                      # saniye
    type: str                       # 'speech' | 'music' | 'noise' | 'silence'
    speaker_id: str = ''            # 'SPEAKER_00' | 'TMDB:Ayşe Kaya' | 'KONUŞMACI_1'
    speaker_label: str = ''         # Çözümlenmiş görünen ad
    transcript: Optional[str] = None
    confidence: float = 0.0         # WhisperX ortalama kelime confidence
    language: str = 'tr'
    words: list = field(default_factory=list)  # [{word, start, end, score}]
    # Müzik programı ekleri
    song_title: Optional[str] = None
    song_artist: Optional[str] = None
    song_composer: Optional[str] = None
    song_lyricist: Optional[str] = None
    song_year: Optional[int] = None
    musicbrainz_id: Optional[str] = None
    # Maç ekleri
    match_event: Optional[str] = None  # 'goal' | 'card' | 'substitution' | 'pause'
    match_minute: Optional[int] = None
    # Kalite işaretçileri
    low_confidence: bool = False
    confidence_sources: list = field(default_factory=list)  # hangi kaynaklar çelişti

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SpeakerIdentity:
    """Tek konuşmacı kimliği — ses + yüz vektörü birleşimi."""
    speaker_id: str                         # İç ID: SPEAKER_00, SPEAKER_01 ...
    display_name: str = ''                  # Çözümlendi: 'Ayşe Kaya' veya 'KONUŞMACI_1'
    tmdb_person_id: Optional[int] = None
    face_vector: Optional[list] = None     # 512 float — InsightFace/ArcFace
    voice_vector: Optional[list] = None    # PyAnnote speaker embedding
    # Foto/ses kayıt tanıtma sonrası silinir (privacy)
    temp_reference: bool = False

    @property
    def is_identified(self) -> bool:
        return bool(self.display_name and not self.display_name.startswith('KONUŞMACI'))


@dataclass
class ConfidenceResult:
    """İki kaynak karşılaştırma sonucu."""
    value: str                      # Seçilen değer
    level: str                      # 'HIGH' | 'MEDIUM' | 'LOW'
    score: float                    # 0.0 - 1.0
    sources: list[tuple] = field(default_factory=list)  # [(source, value, score), ...]
    conflict: bool = False
    alternatives: list = field(default_factory=list)


@dataclass
class ASRResult:
    """Tam ASR çıktısı — JSON'a serialize edilir."""
    video_path: str
    program_type: str
    duration_sec: float
    segments: list[AudioSegment] = field(default_factory=list)
    speakers: list[SpeakerIdentity] = field(default_factory=list)
    # Film/Dizi
    summary_tr: str = ''            # Ollama — 5-8 cümle Türkçe özet
    # Maç
    match_events: list = field(default_factory=list)
    final_score: str = ''
    # Meta
    pipeline_elapsed_sec: float = 0.0
    models_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['segments'] = [asdict(s) for s in self.segments]
        d['speakers'] = [asdict(sp) for sp in self.speakers]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Engine
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceEngine:
    """
    İki veya daha fazla kaynak değerini karşılaştırır.
    Oturum kararı: örtüşme → HIGH, çelişme → LOW + flag
    """

    HIGH_THRESHOLD = 0.85
    MEDIUM_THRESHOLD = 0.60

    def merge(
        self,
        sources: list[tuple[str, str, float]],   # (kaynak_adı, değer, skor)
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.85
    ) -> ConfidenceResult:
        """
        Kaynakları birleştir.

        Args:
            sources: [("transcript", "Sönmez", 0.92), ("ocr", "Sonmez", 0.80)]
            fuzzy: Benzer ama tam eşit olmayan değerleri aynı say
            fuzzy_threshold: Ne kadar benzerlik gerekli

        Returns:
            ConfidenceResult
        """
        if not sources:
            return ConfidenceResult('', 'LOW', 0.0, conflict=True)

        if len(sources) == 1:
            name, value, score = sources[0]
            level = self._level(score)
            return ConfidenceResult(value, level, score, sources=list(sources))

        # Değerleri normalize et ve grupla
        groups: dict[str, list] = {}
        for src_name, value, score in sources:
            matched = False
            if fuzzy:
                for key in groups:
                    if self._similar(value, key, fuzzy_threshold):
                        groups[key].append((src_name, value, score))
                        matched = True
                        break
            if not matched:
                groups[value] = [(src_name, value, score)]

        if len(groups) == 1:
            # Tüm kaynaklar aynı fikirde
            all_scores = [s for _, _, s in sources]
            avg_score = sum(all_scores) / len(all_scores)
            best_value = max(sources, key=lambda x: x[2])[1]
            return ConfidenceResult(
                best_value, 'HIGH', avg_score,
                sources=list(sources), conflict=False
            )
        else:
            # Çelişme — en yüksek skorlu grubu seç, flag'le
            best_group = max(groups.values(), key=lambda g: sum(s for _, _, s in g))
            best_value = max(best_group, key=lambda x: x[2])[1]
            avg_score = sum(s for _, _, s in best_group) / len(best_group)
            alternatives = [
                v for v in groups if not self._similar(v, best_value, fuzzy_threshold)
            ]
            return ConfidenceResult(
                best_value, 'LOW', avg_score * 0.7,
                sources=list(sources), conflict=True, alternatives=alternatives
            )

    def _level(self, score: float) -> str:
        if score >= self.HIGH_THRESHOLD:
            return 'HIGH'
        if score >= self.MEDIUM_THRESHOLD:
            return 'MEDIUM'
        return 'LOW'

    def _similar(self, a: str, b: str, threshold: float) -> bool:
        """Basit karakter benzerlik kontrolü — RapidFuzz olmadan da çalışır."""
        if a == b:
            return True
        try:
            from rapidfuzz import fuzz
            return fuzz.ratio(a, b) / 100.0 >= threshold
        except ImportError:
            # Basit fallback
            a_norm = a.upper().replace(' ', '')
            b_norm = b.upper().replace(' ', '')
            if not a_norm or not b_norm:
                return False
            matches = sum(c1 == c2 for c1, c2 in zip(a_norm, b_norm))
            return matches / max(len(a_norm), len(b_norm)) >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# Model Sarmalayıcılar (Lazy Load)
# ─────────────────────────────────────────────────────────────────────────────

class _AudioExtractor:
    """FFmpeg ile ses çıkarma — 16kHz mono WAV."""

    def __init__(self, ffmpeg_path: str, log_cb=None):
        self._ffmpeg = ffmpeg_path
        self._log = log_cb or print

    def extract(self, video_path: str, out_path: str) -> str:
        """
        16kHz mono WAV çıkar.
        Güvenilirlik: returncode + stderr her zaman kontrol edilir.
        """
        cmd = [
            self._ffmpeg, '-y',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            out_path
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg ses çıkarma hatası (rc={result.returncode}):\n{result.stderr[-500:]}"
            )
        if not Path(out_path).is_file():
            raise FileNotFoundError(f"FFmpeg çıktı dosyası oluşmadı: {out_path}")
        self._log(f"  [Audio] WAV çıkarıldı: {Path(out_path).name}")
        return out_path


class _DF3Denoiser:
    """
    DeepFilterNet3 gürültü azaltma.
    Film/Dizi, Maçlar, Kısa Haber'de kullanılır.
    Demucs'un yerine DEĞİL — sadece gürültü azaltma.
    """

    def __init__(self, log_cb=None):
        self._log = log_cb or print
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from df import enhance, init_df
            model, df_state, _ = init_df()
            self._model = (model, df_state)
            self._log("  [DF3] Model yüklendi")
        except ImportError:
            raise ImportError(
                "DeepFilterNet kurulu değil: pip install deepfilternet"
            )

    def enhance(self, wav_path: str, out_path: str) -> str:
        """Gürültü azaltılmış WAV döndür."""
        self._load()
        try:
            import torchaudio
            from df import enhance, init_df

            model, df_state = self._model
            audio, sr = torchaudio.load(wav_path)
            if sr != df_state.sr():
                import torch
                audio = torchaudio.functional.resample(audio, sr, df_state.sr())
            enhanced = enhance(model, df_state, audio)
            torchaudio.save(out_path, enhanced.unsqueeze(0), df_state.sr())
            self._log(f"  [DF3] Gürültü azaltıldı → {Path(out_path).name}")
            return out_path
        except Exception as e:
            self._log(f"  [DF3] Hata: {e} — orijinal WAV kullanılıyor")
            return wav_path


class _DemucsSeperator:
    """
    Demucs htdemucs_6s — SADECE Müzik Programı.
    6 kaynak: drums, bass, other, vocals, piano, guitar
    """

    def __init__(self, log_cb=None):
        self._log = log_cb or print

    def separate(self, wav_path: str, out_dir: str) -> dict[str, str]:
        """
        Returns:
            {
                'vocals': '/path/vocals.wav',
                'accompaniment': '/path/no_vocals.wav',
                'drums': ..., 'bass': ..., 'other': ..., 'piano': ..., 'guitar': ...
            }
        """
        try:
            import demucs.api
        except ImportError:
            raise ImportError("Demucs kurulu değil: pip install demucs")

        self._log("  [Demucs] htdemucs_6s ayrıştırma başlıyor...")
        t0 = time.time()

        separator = demucs.api.Separator(model='htdemucs_6s')
        _, stems = separator.separate_audio_file(wav_path)

        paths = {}
        for stem_name, tensor in stems.items():
            out_path = str(Path(out_dir) / f"{stem_name}.wav")
            demucs.api.save_audio(tensor, out_path, samplerate=separator.samplerate)
            paths[stem_name] = out_path

        # Accompaniment = vocals hariç her şey (InaSpeech için değil, referans için)
        self._log(
            f"  [Demucs] Tamamlandı ({time.time()-t0:.1f}s) — "
            f"stems: {list(paths.keys())}"
        )
        return paths


class _InaSpeechDetector:
    """
    InaSpeech ile konuşma/müzik/gürültü segmentleri tespit et.
    Müzik Programı'nda şarkı/konuşma bloğu ayrımı için (±3sn tolerans).
    """

    TOLERANCE_SEC = 3.0  # ±3sn birleştirme penceresi

    def __init__(self, log_cb=None):
        self._log = log_cb or print

    def detect(self, wav_path: str) -> list[dict]:
        """
        Returns:
            [{'start': 0.0, 'end': 15.3, 'type': 'music'}, ...]
        """
        try:
            from inaSpeechSegmenter import Segmenter
        except ImportError:
            raise ImportError("InaSpeech kurulu değil: pip install inaSpeechSegmenter")

        self._log("  [InaSpeech] Ses sınıflandırma başlıyor...")
        seg = Segmenter()
        segmentation = seg(wav_path)

        segments = []
        for label, start, end in segmentation:
            seg_type = self._map_label(label)
            segments.append({'start': start, 'end': end, 'type': seg_type})

        # ±3sn toleransla aynı tip ardışık segmentleri birleştir
        segments = self._merge_adjacent(segments)
        self._log(f"  [InaSpeech] {len(segments)} segment bulundu")
        return segments

    def _map_label(self, label: str) -> str:
        label = label.lower()
        if 'music' in label:
            return 'music'
        if 'speech' in label or 'male' in label or 'female' in label:
            return 'speech'
        return 'noise'

    def _merge_adjacent(self, segments: list[dict]) -> list[dict]:
        if not segments:
            return segments
        merged = [segments[0].copy()]
        for seg in segments[1:]:
            last = merged[-1]
            gap = seg['start'] - last['end']
            if seg['type'] == last['type'] and gap <= self.TOLERANCE_SEC:
                last['end'] = seg['end']
            else:
                merged.append(seg.copy())
        return merged


class _PyAnnoteDiarizer:
    """
    PyAnnote 3.1 konuşmacı diarizasyonu.
    HuggingFace token gerektirir: pyannote/speaker-diarization-3.1 (gated model)
    """

    def __init__(self, hf_token: str, log_cb=None):
        self._token = hf_token
        self._log = log_cb or print
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline
            import torch
        except ImportError:
            raise ImportError("PyAnnote kurulu değil: pip install pyannote.audio")

        self._log("  [PyAnnote] Model yükleniyor (pyannote/speaker-diarization-3.1)...")
        import torch
        from pyannote.audio import Pipeline

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self._token
        )
        device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
        self._pipeline = self._pipeline.to(__import__('torch').device(device))
        self._log(f"  [PyAnnote] Yüklendi ({device})")

    def diarize(
        self, wav_path: str,
        speech_segments: Optional[list[dict]] = None
    ) -> list[dict]:
        """
        Konuşmacı diarizasyonu — sadece speech segmentlerini gönder.

        Returns:
            [{'start': 0.0, 'end': 5.2, 'speaker': 'SPEAKER_00'}, ...]
        """
        self._load()

        # Eğer speech_segments verilmişse sadece o bölümleri analiz et
        # (InaSpeech sonuçlarıyla entegrasyon)
        if speech_segments:
            from pyannote.core import Annotation, Segment

            regions = Annotation()
            for s in speech_segments:
                if s.get('type') == 'speech':
                    regions[Segment(s['start'], s['end'])] = 'speech'

            diarization = self._pipeline(
                wav_path, regions=regions
            )
        else:
            diarization = self._pipeline(wav_path)

        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                'start': round(turn.start, 3),
                'end': round(turn.end, 3),
                'speaker': speaker
            })

        self._log(
            f"  [PyAnnote] {len(set(r['speaker'] for r in results))} "
            f"konuşmacı, {len(results)} segment"
        )
        return results

    def release(self):
        """
        ISSUE-04 FIX: VRAM serbest bırak.
        diarize() sonrası çağrılmalı.
        """
        if self._pipeline is not None:
            try:
                del self._pipeline
                self._pipeline = None
                import gc; gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                self._log("  [PyAnnote] Model boşaltıldı (VRAM serbest)")
            except Exception as e:
                self._log(f"  [PyAnnote] Boşaltma hatası: {e}")


class _WhisperXTranscriber:
    """
    WhisperX large-v3 — Türkçe transcript + kelime bazlı zaman damgası.
    """

    def __init__(self, device: str = 'auto', log_cb=None):
        self._device = device if device != 'auto' else (
            'cuda' if self._has_cuda() else 'cpu'
        )
        self._log = log_cb or print
        self._model = None
        self._align_model = None

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load(self):
        if self._model is not None:
            return
        try:
            import whisperx
        except ImportError:
            raise ImportError("WhisperX kurulu değil: pip install whisperx")

        self._log(f"  [WhisperX] large-v3 yükleniyor ({self._device})...")
        import whisperx
        self._model = whisperx.load_model(
            'large-v3', self._device,
            compute_type='float16' if self._device == 'cuda' else 'int8',
            language='tr'
        )
        self._log("  [WhisperX] Yüklendi")

    def transcribe(
        self,
        wav_path: str,
        diarization: Optional[list[dict]] = None,
        hf_token: str = ''
    ) -> list[dict]:
        """
        Transcript + kelime hizalama.

        Returns:
            [
                {
                    'start': 0.0, 'end': 5.2,
                    'text': '...', 'speaker': 'SPEAKER_00',
                    'words': [{'word': '...', 'start': ..., 'end': ..., 'score': ...}],
                    'avg_confidence': 0.91
                }, ...
            ]
        """
        self._load()
        import whisperx

        self._log("  [WhisperX] Transkripsiyon başlıyor...")
        t0 = time.time()

        result = self._model.transcribe(wav_path, batch_size=16, language='tr')

        # Kelime hizalama
        try:
            if self._align_model is None:
                self._align_model, metadata = whisperx.load_align_model(
                    language_code='tr', device=self._device
                )
                self._align_meta = metadata
            result = whisperx.align(
                result['segments'], self._align_model, self._align_meta,
                wav_path, self._device
            )
        except Exception as e:
            self._log(f"  [WhisperX] Hizalama hatası: {e} — devam ediliyor")

        # BUG-01 FIX (asr_pipeline içi):
        # Diarizasyon segmentleri zaten geçilmişse yeni DiarizationPipeline AÇMA.
        # Sadece geçilen segmentleri kullanarak speaker ataması yap.
        # hf_token olsa bile tekrar PyAnnote çalıştırmak VRAM + zaman kaybı.
        if diarization:
            try:
                # Varolan diarizasyon → assign_word_speakers
                from pyannote.core import Annotation, Segment
                diarize_annotation = Annotation()
                for seg in diarization:
                    diarize_annotation[Segment(seg['start'], seg['end'])] = seg.get('speaker', 'UNK')
                result = whisperx.assign_word_speakers(diarize_annotation, result)
            except Exception as e:
                self._log(f"  [WhisperX] Diarizasyon atama hatası: {e}")
        elif hf_token:
            # Diarizasyon verilmedi ama token var → bir kez çalıştır
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token, device=self._device
                )
                diarize_segments = diarize_model(wav_path)
                result = whisperx.assign_word_speakers(
                    diarize_segments, result
                )
            except Exception as e:
                self._log(f"  [WhisperX] Diarizasyon atama hatası: {e}")

        segments = []
        for seg in result.get('segments', []):
            words = seg.get('words', [])
            scores = [w.get('score', 0) for w in words if 'score' in w]
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            segments.append({
                'start': round(seg.get('start', 0), 3),
                'end': round(seg.get('end', 0), 3),
                'text': seg.get('text', '').strip(),
                'speaker': seg.get('speaker', ''),
                'words': words,
                'avg_confidence': round(avg_conf, 3)
            })

        elapsed = time.time() - t0
        self._log(f"  [WhisperX] {len(segments)} segment ({elapsed:.1f}s)")
        return segments


class _OllamaProcessor:
    """
    Ollama llama3.1:8b — local, tüm başlıklar için post-process.
    Türkçe cümle bütünlüğü + özet üretimi.
    """

    def __init__(self, model: str = 'llama3.1:8b', base_url: str = 'http://localhost:11434', log_cb=None):
        self._model = model
        self._base_url = base_url
        self._log = log_cb or print

    def _chat(self, prompt: str, system: str = '') -> str:
        import urllib.request

        payload = {
            'model': self._model,
            'messages': [],
            'stream': False,
            'options': {'temperature': 0.1}
        }
        if system:
            payload['messages'].append({'role': 'system', 'content': system})
        payload['messages'].append({'role': 'user', 'content': prompt})

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result['message']['content'].strip()
        except Exception as e:
            self._log(f"  [Ollama] İstek hatası: {e}")
            return ''

    def fix_turkish(self, text: str) -> str:
        """OCR/ASR çıktısında Türkçe cümle bütünlüğünü onar."""
        if not text or len(text) < 10:
            return text
        system = (
            "Sen bir Türkçe metin editörüsün. Sana verilen metni dilbilgisi ve "
            "noktalama açısından düzelt. Anlamı değiştirme. Sadece düzeltilmiş "
            "metni yaz, açıklama ekleme."
        )
        return self._chat(text, system) or text

    def summarize(self, transcript: str, program_type: str) -> str:
        """5-8 cümle Türkçe özet üret."""
        if not transcript or len(transcript) < 50:
            return ''
        type_hint = {
            'film_dizi': 'film veya dizi',
            'muzik_programi': 'müzik programı',
            'kisa_haber': 'haber bülteni',
            'mac': 'spor maçı'
        }.get(program_type, 'program')
        system = (
            f"Sen bir {type_hint} analistisın. "
            "Sana verilen transcript'ten 5-8 cümlelik Türkçe özet çıkar. "
            "Önemli kişileri, olayları ve konuları vurgula. "
            "Sadece özeti yaz."
        )
        return self._chat(f"Transcript:\n{transcript[:4000]}", system)

    def detect_program_type(self, transcript_sample: str) -> str:
        """
        İlk 2 dakikalık transcript'ten program türünü tahmin et.
        Returns: 'film_dizi' | 'muzik_programi' | 'kisa_haber' | 'mac' | 'bilinmiyor'
        """
        if not transcript_sample:
            return 'bilinmiyor'
        system = (
            "Sana bir videonun ses transkripsiyonundan alınan kısa bir örnek verilecek. "
            "Bu videonun türünü belirle. Sadece şu değerlerden birini yaz:\n"
            "film_dizi | muzik_programi | kisa_haber | mac | bilinmiyor\n"
            "Başka hiçbir şey yazma."
        )
        result = self._chat(transcript_sample[:1000], system)
        valid = {'film_dizi', 'muzik_programi', 'kisa_haber', 'mac', 'bilinmiyor'}
        result = result.strip().lower()
        return result if result in valid else 'bilinmiyor'


class _MusicBrainzResolver:
    """
    MusicBrainz API — şarkı metadata + sözler.
    Güvenilirlik: iki kaynak (MusicBrainz + WhisperX vocals) örtüşüyorsa HIGH.
    """

    BASE_URL = "https://musicbrainz.org/ws/2"
    USER_AGENT = "ArsivDecode/1.0 (contact@example.com)"

    def __init__(self, log_cb=None):
        self._log = log_cb or print

    def search_recording(
        self, title: str, artist: str = ''
    ) -> Optional[dict]:
        """
        Şarkı adı + sanatçı ile MusicBrainz'de ara.

        Returns:
            {
                'mbid': '...', 'title': '...', 'artist': '...',
                'composer': '...', 'lyricist': '...', 'year': 2003,
                'score': 0.95
            }
        """
        import urllib.request
        import urllib.parse

        query_parts = [f'recording:"{title}"']
        if artist:
            query_parts.append(f'artist:"{artist}"')
        query = ' AND '.join(query_parts)

        # ISSUE-05 FIX: &inc=artist-credits+recording-rels olmadan
        # MusicBrainz /recording endpoint'i 'relations' alanını döndürmüyor.
        # Besteci (composer) ve söz yazarı (lyricist) hep boş kalıyordu.
        url = (
            f"{self.BASE_URL}/recording?"
            f"query={urllib.parse.quote(query)}&fmt=json&limit=5"
            f"&inc=artist-credits+recording-rels"
        )
        try:
            req = urllib.request.Request(url, headers={'User-Agent': self.USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            self._log(f"  [MusicBrainz] Arama hatası: {e}")
            return None

        recordings = data.get('recordings', [])
        if not recordings:
            return None

        best = recordings[0]
        # Yıl çıkar
        year = None
        first_release = best.get('first-release-date', '')
        if first_release and len(first_release) >= 4:
            try:
                year = int(first_release[:4])
            except ValueError:
                pass

        # Besteci/söz yazarı — relations alanında
        composer = lyricist = ''
        for rel in best.get('relations', []):
            rel_type = rel.get('type', '').lower()
            person = rel.get('artist', {}).get('name', '')
            if 'composer' in rel_type and person:
                composer = person
            elif 'lyricist' in rel_type and person:
                lyricist = person

        return {
            'mbid': best.get('id', ''),
            'title': best.get('title', ''),
            'artist': best.get('artist-credit', [{}])[0].get('name', '') if best.get('artist-credit') else '',
            'composer': composer,
            'lyricist': lyricist,
            'year': year,
            'score': best.get('score', 0) / 100.0
        }

    def get_lyrics_whisperx_fallback(
        self, vocals_wav: str, whisperx_model
    ) -> str:
        """MusicBrainz'den söz bulunamazsa vocals WhisperX ile transcript al."""
        try:
            segments = whisperx_model.transcribe(vocals_wav)
            return ' '.join(s.get('text', '') for s in segments)
        except Exception as e:
            self._log(f"  [MusicBrainz] Fallback lyrics hatası: {e}")
            return ''


class _EssentiaAnalyzer:
    """
    Essentia ses pattern analizi — Maç pipeline'ı için.
    Maç duraklaması tespiti: ±30sn pencere içinde Ollama sınıflandırma.
    """

    def __init__(self, log_cb=None):
        self._log = log_cb or print

    def detect_pause_patterns(self, wav_path: str) -> list[dict]:
        """
        Maç duraklama anları (düdük, kalabalık sessizliği, vs).

        Returns:
            [{'start': 45.0, 'end': 47.2, 'pattern': 'whistle', 'confidence': 0.82}, ...]
        """
        try:
            import essentia.standard as es
        except ImportError:
            self._log("  [Essentia] Kurulu değil — pip install essentia")
            return []

        self._log("  [Essentia] Ses pattern analizi başlıyor...")
        loader = es.MonoLoader(filename=wav_path, sampleRate=44100)
        audio = loader()

        # Enerji ve sıfır geçiş oranı ile sessizlik/ani değişim tespiti
        frame_size = 2048
        hop_size = 512
        energy_algo = es.Energy()
        zcr_algo = es.ZeroCrossingRate()

        events = []
        timestamps = []
        energies = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy = energy_algo(frame)
            energies.append(energy)
            timestamps.append(i / 44100.0)

        if not energies:
            return events

        # Ani enerji düşüşü = muhtemel düdük/durak
        avg_energy = sum(energies) / len(energies)
        threshold = avg_energy * 0.1  # %10 altına düşüş

        in_pause = False
        pause_start = 0.0
        for ts, energy in zip(timestamps, energies):
            if energy < threshold and not in_pause:
                in_pause = True
                pause_start = ts
            elif energy >= threshold and in_pause:
                duration = ts - pause_start
                if duration >= 1.0:  # Minimum 1sn durak
                    events.append({
                        'start': round(pause_start, 2),
                        'end': round(ts, 2),
                        'pattern': 'low_energy',
                        'confidence': 0.65  # Essentia skoru — Ollama doğrulayacak
                    })
                in_pause = False

        self._log(f"  [Essentia] {len(events)} potansiyel duraklama bulundu")
        return events


class _SpeakerIdentifier:
    """
    Konuşmacı kimlik çözümleme.
    TMDB cast listesi → PyAnnote SPEAKER_ID → kanonik isim
    Eşleşme yoksa: KONUŞMACI_N / KARAKTER_N
    """

    def __init__(self, tmdb_cast: Optional[list] = None, log_cb=None):
        self._tmdb_cast = tmdb_cast or []
        self._log = log_cb or print
        # {speaker_id: SpeakerIdentity}
        self._identities: dict[str, SpeakerIdentity] = {}
        self._counter = 0

    def resolve(
        self,
        speaker_id: str,
        context_words: list[str] = None,
        prefix: str = 'KONUŞMACI'
    ) -> SpeakerIdentity:
        """
        Konuşmacı ID'sini çöz.
        Öncelik sırası:
          1. Önceden çözümlenmişse cache'den döndür
          2. TMDB cast listesinde bağlam kelimeleriyle eşleştir
          3. Numaralandır (KONUŞMACI_1, KARAKTER_1 vb.)
        """
        if speaker_id in self._identities:
            return self._identities[speaker_id]

        # TMDB eşleştirme — bağlam kelimelerinde isim geçiyor mu?
        if context_words and self._tmdb_cast:
            identity = self._match_tmdb(speaker_id, context_words)
            if identity:
                self._identities[speaker_id] = identity
                return identity

        # Numaralandır
        self._counter += 1
        identity = SpeakerIdentity(
            speaker_id=speaker_id,
            display_name=f'{prefix}_{self._counter}'
        )
        self._identities[speaker_id] = identity
        return identity

    def _match_tmdb(
        self, speaker_id: str, context_words: list[str]
    ) -> Optional[SpeakerIdentity]:
        """
        TMDB cast listesinde bağlam kelimelerine göre eşleştir.

        ISSUE-08 FIX: Eski bag-of-words yaklaşımı yanlış pozitif üretiyordu.
        Örnek: "Ali Kaya" → "ali dedi ki kaya gibi" ifadesinde her iki kelime
        ayrı ayrı geçtiği için eşleşiyordu.

        Çözüm: Bigram sliding-window — isim kelimelerinin art arda (veya 1 kelime
        arayla) geçmesi gerekiyor.
        """
        context_lower = [w.lower() for w in context_words]

        for person in self._tmdb_cast:
            name = person.get('name', '') or person.get('character', '')
            if not name:
                continue
            name_parts = name.lower().split()
            if not name_parts:
                continue

            matched = False

            if len(name_parts) == 1:
                # Tek kelimeli isim: tam kelime eşleşmesi
                matched = name_parts[0] in context_lower
            else:
                # Çok kelimeli isim: sliding window ile ardışık eşleşme
                # "Ali Kaya" → context içinde "ali ... kaya" bitişik veya 1 ara kelimeyle
                window_size = len(name_parts) + 1  # +1 ara kelimeye izin ver
                for i in range(len(context_lower) - len(name_parts) + 1):
                    window = context_lower[i:i + window_size]
                    # Tüm isim parçaları bu pencerede mi ve sıralı mı?
                    positions = []
                    for part in name_parts:
                        for j, w in enumerate(window):
                            if w == part and j not in positions:
                                positions.append(j)
                                break
                    if (len(positions) == len(name_parts) and
                            positions == sorted(positions) and
                            positions[-1] - positions[0] <= len(name_parts)):
                        matched = True
                        break

            if matched:
                return SpeakerIdentity(
                    speaker_id=speaker_id,
                    display_name=name,
                    tmdb_person_id=person.get('id')
                )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Ana Pipeline Orkestratörü
# ─────────────────────────────────────────────────────────────────────────────

class ASRPipeline:
    """
    Arsiv Decode ASR (Otomatik Ses Tanıma) Pipeline.

    Kullanım:
        config = {
            'ffmpeg': 'F:/Source/ffmpeg/bin/ffmpeg.exe',
            'hf_token': 'hf_xxx',
            'ollama_model': 'llama3.1:8b',
            'program_type': 'film_dizi',   # veya 'auto'
            'tmdb_cast': [...],
        }
        asr = ASRPipeline(config)
        result = asr.run('F:/input/film.mp4')
        print(result.to_json())
    """

    SUPPORTED_TYPES = {'film_dizi', 'muzik_programi', 'kisa_haber', 'mac'}

    def __init__(self, config: dict, log_cb=None):
        self._cfg = config
        self._log = log_cb or print
        self._confidence = ConfidenceEngine()

        ffmpeg = config.get('ffmpeg', 'ffmpeg')
        hf_token = config.get('hf_token', '')
        ollama_model = config.get('ollama_model', 'llama3.1:8b')
        ollama_url = config.get('ollama_url', 'http://localhost:11434')
        device = config.get('device', 'auto')

        # Lazy init — sadece kullanıldığında yüklenir
        self._extractor = _AudioExtractor(ffmpeg, log_cb)
        self._df3 = _DF3Denoiser(log_cb)
        self._demucs = _DemucsSeperator(log_cb)
        self._ina = _InaSpeechDetector(log_cb)
        self._pyannote = _PyAnnoteDiarizer(hf_token, log_cb)
        self._whisperx = _WhisperXTranscriber(device, log_cb)
        self._ollama = _OllamaProcessor(ollama_model, ollama_url, log_cb)
        self._musicbrainz = _MusicBrainzResolver(log_cb)
        self._essentia = _EssentiaAnalyzer(log_cb)

    def run(
        self,
        video_path: str,
        program_type: str = 'auto',
        work_dir: str = '',
        tmdb_cast: Optional[list] = None
    ) -> ASRResult:
        """
        Ana giriş noktası.

        Args:
            video_path: Video dosyası (F:/input/...)
            program_type: 'film_dizi' | 'muzik_programi' | 'kisa_haber' | 'mac' | 'auto'
            work_dir: Geçici dosyalar için klasör
            tmdb_cast: TMDB cast listesi (konuşmacı çözümleme için)
        """
        t0 = time.time()
        video_path = str(Path(video_path).resolve())
        vname = Path(video_path).stem

        if not work_dir:
            work_dir = str(Path(video_path).parent / f"asr_{vname}")
        Path(work_dir).mkdir(parents=True, exist_ok=True)

        result = ASRResult(
            video_path=video_path,
            program_type=program_type,
            duration_sec=0.0
        )

        self._log(f"\n{'='*60}")
        self._log(f"  ASR PIPELINE — {vname}")
        self._log(f"{'='*60}")

        try:
            # ── Ses Çıkar ─────────────────────────────────────────────
            wav_path = str(Path(work_dir) / f"{vname}_raw.wav")
            self._extractor.extract(video_path, wav_path)

            # Süre bilgisi
            result.duration_sec = self._get_duration(wav_path)

            # ── Program Türü Tespiti ───────────────────────────────────
            if program_type == 'auto':
                program_type = self._detect_type(wav_path)
                self._log(f"  [Auto] Program türü: {program_type}")
            result.program_type = program_type

            if program_type not in self.SUPPORTED_TYPES:
                raise ValueError(f"Desteklenmeyen program türü: {program_type}")

            # ── Program Tipine Göre Route ──────────────────────────────
            if program_type == 'muzik_programi':
                result = self._run_muzik(wav_path, work_dir, vname, result, tmdb_cast)
            elif program_type == 'mac':
                result = self._run_mac(wav_path, work_dir, vname, result, tmdb_cast)
            elif program_type in ('film_dizi', 'kisa_haber'):
                result = self._run_film_haber(
                    wav_path, work_dir, vname, result, tmdb_cast, program_type
                )

        except Exception as e:
            self._log(f"\n!! ASR HATA: {e}")
            import traceback
            traceback.print_exc()
            result.warnings.append(f"Pipeline hatası: {str(e)}")

        result.pipeline_elapsed_sec = round(time.time() - t0, 2)
        self._log(
            f"\n  ASR TAMAMLANDI — {result.pipeline_elapsed_sec:.1f}s | "
            f"{len(result.segments)} segment | {len(result.speakers)} konuşmacı"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Film/Dizi + Kısa Haber Pipeline
    # ─────────────────────────────────────────────────────────────────────

    def _run_film_haber(
        self, wav_path, work_dir, vname, result, tmdb_cast, program_type
    ) -> ASRResult:
        """
        DF3 → PyAnnote → WhisperX → Ollama
        Film/Dizi: Konuşmacı → TMDB → KARAKTER_N
        Kısa Haber: Konuşmacı → KJ → yüz tanıma → bağlam → KONUŞMACI_N
        """
        models = ['DF3', 'PyAnnote 3.1', 'WhisperX large-v3', 'Ollama llama3.1:8b']
        result.models_used = models

        # [1] DF3 Gürültü Azaltma
        self._log(f"\n[1] DF3 Gürültü Azaltma")
        clean_wav = str(Path(work_dir) / f"{vname}_clean.wav")
        clean_wav = self._df3.enhance(wav_path, clean_wav)

        # [2] PyAnnote Diarizasyon
        self._log(f"\n[2] PyAnnote Diarizasyon")
        diarization = self._pyannote.diarize(clean_wav)

        # [3] WhisperX Transkripsiyon
        self._log(f"\n[3] WhisperX Transkripsiyon")
        hf_token = self._cfg.get('hf_token', '')
        transcripts = self._whisperx.transcribe(clean_wav, diarization, hf_token)

        # [4] Konuşmacı Kimlik Çözümleme
        prefix = 'KARAKTER' if program_type == 'film_dizi' else 'KONUŞMACI'
        identifier = _SpeakerIdentifier(tmdb_cast, self._log)
        full_transcript = ''

        for seg_data in transcripts:
            words = [w.get('word', '') for w in seg_data.get('words', [])]
            speaker_id = seg_data.get('speaker', 'SPEAKER_00')
            identity = identifier.resolve(speaker_id, words, prefix)

            seg = AudioSegment(
                start=seg_data['start'],
                end=seg_data['end'],
                type='speech',
                speaker_id=speaker_id,
                speaker_label=identity.display_name,
                transcript=seg_data.get('text', ''),
                confidence=seg_data.get('avg_confidence', 0.0),
                words=seg_data.get('words', [])
            )
            # Confidence flag
            if seg.confidence < 0.60:
                seg.low_confidence = True
                seg.confidence_sources = ['whisperx_low']

            result.segments.append(seg)
            full_transcript += f"{identity.display_name}: {seg.transcript}\n"

        result.speakers = list(identifier._identities.values())

        # [5] Ollama Post-Process
        self._log(f"\n[5] Ollama Post-Process")
        if full_transcript:
            result.summary_tr = self._ollama.summarize(full_transcript, program_type)
            if result.summary_tr:
                self._log(f"  [Ollama] Özet üretildi ({len(result.summary_tr)} karakter)")

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Müzik Programı Pipeline
    # ─────────────────────────────────────────────────────────────────────

    def _run_muzik(
        self, wav_path, work_dir, vname, result, tmdb_cast
    ) -> ASRResult:
        """
        Demucs htdemucs_6s → InaSpeech → PyAnnote → WhisperX + MusicBrainz

        Çıktı: speech segmentleri + şarkı metadata'sı
        Confidence: iki kaynak örtüşüyor → HIGH, çelişiyor → low_confidence
        """
        models = [
            'Demucs htdemucs_6s', 'InaSpeech', 'PyAnnote 3.1',
            'WhisperX large-v3', 'MusicBrainz API', 'Ollama llama3.1:8b'
        ]
        result.models_used = models

        # [1] Demucs — Ses Kaynağı Ayrıştırma
        self._log(f"\n[1] Demucs htdemucs_6s Ayrıştırma")
        stems = self._demucs.separate(wav_path, work_dir)
        vocals_wav = stems.get('vocals', wav_path)

        # [2] InaSpeech — Konuşma/Müzik Bloğu Tespiti
        self._log(f"\n[2] InaSpeech Sınıflandırma")
        ina_segments = self._ina.detect(vocals_wav)

        speech_segs = [s for s in ina_segments if s['type'] == 'speech']
        music_segs = [s for s in ina_segments if s['type'] == 'music']
        self._log(f"  konuşma:{len(speech_segs)} | müzik:{len(music_segs)}")

        # [3] PyAnnote — Sadece Speech Bloklarında
        self._log(f"\n[3] PyAnnote Diarizasyon (speech bloğu)")
        diarization = self._pyannote.diarize(vocals_wav, speech_segs)

        # [4] WhisperX — Vocals Kanalı
        self._log(f"\n[4] WhisperX Transkripsiyon (vocals)")
        hf_token = self._cfg.get('hf_token', '')
        transcripts = self._whisperx.transcribe(vocals_wav, diarization, hf_token)

        identifier = _SpeakerIdentifier(tmdb_cast, self._log)

        # Speech segmentlerini işle
        transcript_by_time: dict[float, dict] = {
            t['start']: t for t in transcripts
        }

        for ina_seg in ina_segments:
            if ina_seg['type'] == 'speech':
                # En yakın WhisperX segmentini bul
                matching = [
                    t for t in transcripts
                    if t['start'] >= ina_seg['start'] - 1.0
                    and t['end'] <= ina_seg['end'] + 1.0
                ]
                for seg_data in matching:
                    speaker_id = seg_data.get('speaker', 'SPEAKER_00')
                    identity = identifier.resolve(speaker_id, prefix='KONUŞMACI')
                    result.segments.append(AudioSegment(
                        start=seg_data['start'],
                        end=seg_data['end'],
                        type='speech',
                        speaker_id=speaker_id,
                        speaker_label=identity.display_name,
                        transcript=seg_data.get('text', ''),
                        confidence=seg_data.get('avg_confidence', 0.0),
                        words=seg_data.get('words', [])
                    ))

            elif ina_seg['type'] == 'music':
                # [5] Şarkı Tespiti — Hibrit: ses + WhisperX transcript duyurusu
                self._log(
                    f"\n[5] Müzik Segmenti → MusicBrainz "
                    f"({ina_seg['start']:.1f}s - {ina_seg['end']:.1f}s)"
                )
                song_seg = self._resolve_song(ina_seg, transcripts, vocals_wav)
                result.segments.append(song_seg)

        result.speakers = list(identifier._identities.values())
        result.segments.sort(key=lambda s: s.start)
        return result

    def _resolve_song(
        self,
        ina_seg: dict,
        all_transcripts: list,
        vocals_wav: str
    ) -> AudioSegment:
        """
        Şarkı kimliğini belirle.
        Kim söylüyor öncelik: transcript duyurusu → KJ → yüz tanıma → ses tanıma → bilinmiyor
        Confidence: ses tespiti + transcript örtüşüyorsa HIGH
        """
        seg = AudioSegment(
            start=ina_seg['start'],
            end=ina_seg['end'],
            type='music'
        )

        # Spiker duyurusu: müzik segmentinden ±10sn önceki transcript'e bak
        announced_title = ''
        announced_artist = ''
        window_start = ina_seg['start'] - 10.0
        window_end = ina_seg['start'] + 3.0

        for t in all_transcripts:
            if window_start <= t['start'] <= window_end:
                text = t.get('text', '')
                # Basit kural tabanlı duyuru tespiti
                # Gerçek implementasyonda Ollama NER kullanılabilir
                if text:
                    announced_title = text  # Placeholder — NER ile geliştirilecek
                break

        # MusicBrainz arama
        mb_result = None
        if announced_title or announced_artist:
            mb_result = self._musicbrainz.search_recording(
                announced_title, announced_artist
            )

        # Confidence Engine: transcript duyurusu + MusicBrainz örtüşüyor mu?
        sources = []
        if announced_title:
            sources.append(('transcript_announcement', announced_title, 0.70))
        if mb_result:
            sources.append(('musicbrainz', mb_result.get('title', ''), mb_result.get('score', 0.5)))

        if sources:
            confidence_result = self._confidence.merge(sources)
            seg.song_title = confidence_result.value
            seg.confidence = confidence_result.score
            seg.low_confidence = confidence_result.conflict
            if confidence_result.conflict:
                seg.confidence_sources = [f"{s[0]}:{s[1]}" for s in sources]

        if mb_result:
            seg.song_artist = mb_result.get('artist', '')
            seg.song_composer = mb_result.get('composer', '')
            seg.song_lyricist = mb_result.get('lyricist', '')
            seg.song_year = mb_result.get('year')
            seg.musicbrainz_id = mb_result.get('mbid', '')

        return seg

    # ─────────────────────────────────────────────────────────────────────
    # Maç Pipeline
    # ─────────────────────────────────────────────────────────────────────

    def _run_mac(
        self, wav_path, work_dir, vname, result, tmdb_cast
    ) -> ASRResult:
        """
        DF3 → PyAnnote → WhisperX + Essentia maç duraklama tespiti

        Çıktı:
            - Tam transcript (spiker yorumu)
            - Yapılandırılmış maç olayları: gol, kart, skor, dakika
            - Maç duraklama: Essentia → ±30sn pencere → Ollama sınıflandırma
        """
        models = [
            'DF3', 'Essentia', 'PyAnnote 3.1',
            'WhisperX large-v3', 'Ollama llama3.1:8b'
        ]
        result.models_used = models

        # [1] DF3
        self._log(f"\n[1] DF3 Gürültü Azaltma")
        clean_wav = str(Path(work_dir) / f"{vname}_clean.wav")
        clean_wav = self._df3.enhance(wav_path, clean_wav)

        # [2] Essentia — Maç Duraklama Pattern Tespiti (PyAnnote'dan önce)
        self._log(f"\n[2] Essentia Maç Duraklama Analizi")
        pause_patterns = self._essentia.detect_pause_patterns(clean_wav)

        # [3] PyAnnote
        self._log(f"\n[3] PyAnnote Diarizasyon")
        diarization = self._pyannote.diarize(clean_wav)

        # [4] WhisperX
        self._log(f"\n[4] WhisperX Transkripsiyon")
        hf_token = self._cfg.get('hf_token', '')
        transcripts = self._whisperx.transcribe(clean_wav, diarization, hf_token)

        identifier = _SpeakerIdentifier(None, self._log)
        full_transcript = ''

        for seg_data in transcripts:
            words = [w.get('word', '') for w in seg_data.get('words', [])]
            speaker_id = seg_data.get('speaker', 'SPEAKER_00')
            identity = identifier.resolve(speaker_id, words, 'KONUŞMACI')

            seg = AudioSegment(
                start=seg_data['start'],
                end=seg_data['end'],
                type='speech',
                speaker_id=speaker_id,
                speaker_label=identity.display_name,
                transcript=seg_data.get('text', ''),
                confidence=seg_data.get('avg_confidence', 0.0),
                words=seg_data.get('words', [])
            )
            result.segments.append(seg)
            full_transcript += seg_data.get('text', '') + ' '

        # [5] Essentia Duraklama + Ollama Sınıflandırma
        self._log(f"\n[5] Maç Olayları — Ollama Sınıflandırma")
        match_events = self._classify_match_events(
            pause_patterns, full_transcript, transcripts
        )
        result.match_events = match_events

        result.speakers = list(identifier._identities.values())
        return result

    def _classify_match_events(
        self,
        pause_patterns: list[dict],
        full_transcript: str,
        transcripts: list[dict]
    ) -> list[dict]:
        """
        Essentia duraklama patternları → ±30sn transcript window → Ollama sınıflandır.
        Returns: [{'time': 45.0, 'event': 'goal', 'minute': 45, 'detail': '...'}]
        """
        WINDOW_SEC = 30.0
        events = []

        for pause in pause_patterns:
            if pause.get('confidence', 0) < 0.50:
                continue

            pause_time = (pause['start'] + pause['end']) / 2.0
            # ±30sn transcript penceresi
            window_segs = [
                t for t in transcripts
                if abs((t['start'] + t['end']) / 2.0 - pause_time) <= WINDOW_SEC
            ]
            window_text = ' '.join(s.get('text', '') for s in window_segs)

            if not window_text.strip():
                continue

            # Ollama ile sınıflandır
            prompt = (
                f"Bir futbol maçı yorumcusunun sözleri:\n{window_text}\n\n"
                f"Bu an ({pause_time:.0f}. saniye civarında) ne oldu? "
                f"Şu formatta yanıt ver (başka hiçbir şey yazma):\n"
                f"olay: gol|kart|penalti|uzatma|durak|diger\n"
                f"dakika: [sayı veya bilinmiyor]\n"
                f"detay: [kısa açıklama]"
            )
            ollama_response = self._ollama._chat(prompt)

            # Response parse
            event_type = 'bilinmiyor'
            minute = None
            detail = ''
            for line in ollama_response.split('\n'):
                if line.startswith('olay:'):
                    event_type = line.split(':', 1)[1].strip()
                elif line.startswith('dakika:'):
                    try:
                        minute = int(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith('detay:'):
                    detail = line.split(':', 1)[1].strip()

            events.append({
                'time_sec': round(pause_time, 1),
                'event': event_type,
                'minute': minute,
                'detail': detail,
                'essentia_confidence': pause.get('confidence', 0.0),
                'ollama_classified': True
            })
            self._log(f"  [Maç] {pause_time:.0f}s → {event_type} (dk:{minute})")

        return events

    # ─────────────────────────────────────────────────────────────────────
    # Yardımcılar
    # ─────────────────────────────────────────────────────────────────────

    def _detect_type(self, wav_path: str) -> str:
        """İlk 2 dakika WhisperX ile transcript al → Ollama türü tahmin et."""
        try:
            # Sadece ilk 120sn
            import torchaudio
            audio, sr = torchaudio.load(wav_path)
            sample_len = min(audio.shape[-1], sr * 120)
            sample = audio[..., :sample_len]

            tmp = wav_path.replace('.wav', '_sample.wav')
            torchaudio.save(tmp, sample, sr)

            segs = self._whisperx.transcribe(tmp)
            sample_text = ' '.join(s.get('text', '') for s in segs[:10])

            if Path(tmp).exists():
                Path(tmp).unlink()

            return self._ollama.detect_program_type(sample_text)
        except Exception as e:
            self._log(f"  [AutoDetect] Hata: {e} — 'film_dizi' varsayıldı")
            return 'film_dizi'

    def _get_duration(self, wav_path: str) -> float:
        """WAV dosyasının süresini saniye cinsinden döndür."""
        try:
            import torchaudio
            info = torchaudio.info(wav_path)
            return info.num_frames / info.sample_rate
        except Exception:
            return 0.0

    def save_result(self, result: ASRResult, out_dir: str) -> tuple[str, str]:
        """
        ASR sonucunu JSON + TXT olarak kaydet.
        Returns: (json_path, txt_path)
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        vname = Path(result.video_path).stem

        json_path = str(Path(out_dir) / f"{vname}_asr.json")
        txt_path = str(Path(out_dir) / f"{vname}_asr.txt")

        # JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(result.to_json())

        # TXT — insan okunabilir
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"ARSIV DECODE — ASR Raporu\n")
            f.write(f"Video: {Path(result.video_path).name}\n")
            f.write(f"Program Türü: {result.program_type}\n")
            f.write(f"Süre: {result.duration_sec:.1f}s\n")
            f.write(f"Segment Sayısı: {len(result.segments)}\n")
            f.write(f"Konuşmacı Sayısı: {len(result.speakers)}\n")
            f.write('=' * 60 + '\n\n')

            if result.summary_tr:
                f.write(f"ÖZET:\n{result.summary_tr}\n\n")
                f.write('=' * 60 + '\n\n')

            if result.match_events:
                f.write("MAÇ OLAYLARI:\n")
                for ev in result.match_events:
                    f.write(
                        f"  {ev.get('time_sec', 0):.0f}s | "
                        f"{ev.get('event', '')} | "
                        f"Dk:{ev.get('minute', '?')} | "
                        f"{ev.get('detail', '')}\n"
                    )
                f.write('\n' + '=' * 60 + '\n\n')

            f.write("TRANSCRIPT:\n")
            for seg in result.segments:
                if seg.transcript:
                    low = ' [⚠]' if seg.low_confidence else ''
                    label = seg.speaker_label or seg.speaker_id
                    f.write(
                        f"[{seg.start:.1f}s-{seg.end:.1f}s] "
                        f"{label}{low}: {seg.transcript}\n"
                    )
                elif seg.type == 'music':
                    song_info = seg.song_title or 'Bilinmeyen Şarkı'
                    if seg.song_artist:
                        song_info += f" — {seg.song_artist}"
                    f.write(
                        f"[{seg.start:.1f}s-{seg.end:.1f}s] "
                        f"🎵 {song_info}\n"
                    )

        self._log(f"  ASR JSON: {json_path}")
        self._log(f"  ASR TXT : {txt_path}")
        return json_path, txt_path
