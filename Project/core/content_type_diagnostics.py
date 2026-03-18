"""
content_type_diagnostics.py — İçerik türü güç/zayıflık analizi.

Her içerik kategorisi için pipeline'ın hangi aşamalarda güçlü, hangilerinde
zayıf olduğunu raporlar.  load_profile() ile gelen profil verilerindeki
_strength / _weakness alanlarından türetilir; ek olarak yapısal analiz de
üretir.

Kullanım:
    from core.content_type_diagnostics import ContentTypeDiagnostics
    diag = ContentTypeDiagnostics()
    report = diag.build_report()
    print(diag.format_report(report))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from config.profile_loader import load_profile, list_profiles


# ─── Kategori tanımları ──────────────────────────────────────────────────────

CONTENT_TYPES: List[str] = [
    "EskiFilm",
    "YeniFilm",
    "OrijinalDilFilm",
    "YeniDizi",
    "EskiDizi",
]

# Pipeline aşamaları — güç/zayıflık skorlaması için
PIPELINE_STAGES = [
    "ocr_credits",
    "tmdb_verify",
    "audio_transcribe",
    "language_detect",
    "guest_actor",
    "post_process_summary",
]

# Her içerik türü için aşama skorları (1=zayıf, 2=orta, 3=güçlü)
_STAGE_SCORES: Dict[str, Dict[str, int]] = {
    "EskiFilm": {
        "ocr_credits":          2,  # VLM OCR kısmen çalışır ama font sorunları var
        "tmdb_verify":          1,  # TMDB'de eski filmler az/eksik veri
        "audio_transcribe":     2,  # Transkript çalışır
        "language_detect":      2,  # Dil genellikle Türkçe
        "guest_actor":          2,  # Credits var ama güçlük var
        "post_process_summary": 2,  # Özet alınabilir
    },
    "YeniFilm": {
        "ocr_credits":          3,  # Credits temiz, hibrit OCR mükemmel okur
        "tmdb_verify":          3,  # TMDB'de tam veri
        "audio_transcribe":     3,  # Transkript sorunsuz
        "language_detect":      3,  # Dil Türkçe, sorun yok
        "guest_actor":          3,  # Tüm oyuncular credits'te
        "post_process_summary": 3,  # Özet mükemmel
    },
    "OrijinalDilFilm": {
        "ocr_credits":          2,  # Eski/yeni bağımlı
        "tmdb_verify":          2,  # TMDB verisi var ama dil uyumu değişken
        "audio_transcribe":     3,  # Whisper orijinal dilde transkript üretir
        "language_detect":      3,  # Dil tespiti güçlü
        "guest_actor":          2,  # Eski/yeni bağımlı
        "post_process_summary": 2,  # Özet kalitesi dile bağlı değişken
    },
    "YeniDizi": {
        "ocr_credits":          1,  # Credits neredeyse yok, oyuncu okunamaz
        "tmdb_verify":          1,  # TMDB TV serisi verisi az
        "audio_transcribe":     3,  # Transkript sorunsuz
        "language_detect":      3,  # Dil Türkçe, sorun yok
        "guest_actor":          1,  # Credits yoksa konuk oyuncu da yakalanamaz
        "post_process_summary": 2,  # Özet alınabilir ama cast doğrulaması zayıf
    },
    "EskiDizi": {
        "ocr_credits":          3,  # Credits mevcut ve okunabilir
        "tmdb_verify":          2,  # TMDB TV verisi genellikle yeterli
        "audio_transcribe":     2,  # Transkript çalışır
        "language_detect":      2,  # Dil genellikle Türkçe
        "guest_actor":          2,  # guest_actor_enabled ile kısmen yakalanır
        "post_process_summary": 2,  # Özet alınabilir
    },
}

_STAGE_LABELS_TR = {
    "ocr_credits":          "Credits OCR (oyuncu/ekip okuma)",
    "tmdb_verify":          "TMDB Doğrulama",
    "audio_transcribe":     "Ses Transkripti",
    "language_detect":      "Dil Tespiti",
    "guest_actor":          "Konuk Oyuncu Algılama",
    "post_process_summary": "Özet (Post-Process)",
}

_SCORE_LABELS = {1: "❌ Zayıf", 2: "⚠️ Orta", 3: "✅ Güçlü"}


@dataclass
class StageScore:
    stage: str
    label: str
    score: int
    verdict: str


@dataclass
class ContentTypeDiagnosticResult:
    profile_name: str
    description: str
    strength_note: str
    weakness_note: str
    stage_scores: List[StageScore] = field(default_factory=list)
    overall_score: float = 0.0

    def overall_verdict(self) -> str:
        if self.overall_score >= 2.7:
            return "✅ Güçlü"
        if self.overall_score >= 2.0:
            return "⚠️ Orta"
        return "❌ Zayıf"


class ContentTypeDiagnostics:
    """İçerik türleri için güç/zayıflık analizi üretir."""

    def build_report(self) -> List[ContentTypeDiagnosticResult]:
        """Tüm içerik türleri için tanı sonuçlarını döndür."""
        results = []
        for ct in CONTENT_TYPES:
            profile = load_profile(ct)
            scores_map = _STAGE_SCORES.get(ct, {})
            stage_scores = []
            for stage_key in PIPELINE_STAGES:
                score = scores_map.get(stage_key, 2)
                stage_scores.append(StageScore(
                    stage=stage_key,
                    label=_STAGE_LABELS_TR.get(stage_key, stage_key),
                    score=score,
                    verdict=_SCORE_LABELS[score],
                ))
            overall = sum(s.score for s in stage_scores) / len(stage_scores) if stage_scores else 0.0
            results.append(ContentTypeDiagnosticResult(
                profile_name=ct,
                description=profile.get("_description", ""),
                strength_note=profile.get("_strength", ""),
                weakness_note=profile.get("_weakness", ""),
                stage_scores=stage_scores,
                overall_score=round(overall, 2),
            ))
        return results

    def format_report(self, results: List[ContentTypeDiagnosticResult]) -> str:
        """İnsan okunabilir rapor metni üret."""
        lines = []
        lines.append("=" * 70)
        lines.append("  VİTOS — İçerik Türü Güç/Zayıflık Analizi")
        lines.append("=" * 70)
        lines.append("")

        for r in results:
            lines.append(f"▶ {r.profile_name}  [{r.overall_verdict()}  {r.overall_score:.1f}/3.0]")
            if r.description:
                # Kısa açıklama (ilk cümle yeterli)
                short_desc = r.description.split(".")[0] + "."
                lines.append(f"  {short_desc}")
            lines.append("")
            for ss in r.stage_scores:
                lines.append(f"    {ss.verdict:<18} {ss.label}")
            if r.strength_note:
                lines.append(f"  + GÜÇLÜ  : {r.strength_note}")
            if r.weakness_note:
                lines.append(f"  - ZAYIF  : {r.weakness_note}")
            lines.append("")

        lines.append("-" * 70)
        lines.append("  Özet Sıralama (en güçlüden en zayıfa):")
        sorted_results = sorted(results, key=lambda x: x.overall_score, reverse=True)
        for i, r in enumerate(sorted_results, 1):
            lines.append(f"  {i}. {r.profile_name:<20} {r.overall_verdict()} ({r.overall_score:.1f}/3.0)")
        lines.append("=" * 70)
        return "\n".join(lines)

    def get_weakest_stages(
        self, content_type: str, threshold: int = 1
    ) -> List[StageScore]:
        """Belirli içerik türü için zayıf (score <= threshold) aşamaları döndür."""
        scores_map = _STAGE_SCORES.get(content_type, {})
        return [
            StageScore(
                stage=k,
                label=_STAGE_LABELS_TR.get(k, k),
                score=v,
                verdict=_SCORE_LABELS[v],
            )
            for k, v in scores_map.items()
            if v <= threshold
        ]

    def get_strongest_stages(
        self, content_type: str, threshold: int = 3
    ) -> List[StageScore]:
        """Belirli içerik türü için güçlü (score >= threshold) aşamaları döndür."""
        scores_map = _STAGE_SCORES.get(content_type, {})
        return [
            StageScore(
                stage=k,
                label=_STAGE_LABELS_TR.get(k, k),
                score=v,
                verdict=_SCORE_LABELS[v],
            )
            for k, v in scores_map.items()
            if v >= threshold
        ]
