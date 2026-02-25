"""
asr_pipeline.py — Ortak veri yapıları ve Confidence Engine.

Profil bazlı pipeline'lar (film_dizi, müzik, maç, haber) gelecekte
bu yapıları kullanacak. Şu an aktif pipeline: audio_pipeline.py

Dataclass'lar:
    AudioSegment, SpeakerIdentity, ConfidenceResult, ASRResult

Motorlar:
    ConfidenceEngine — çoklu kaynak birleştirme
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
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

