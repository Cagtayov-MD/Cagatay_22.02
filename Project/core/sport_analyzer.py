"""
sport_analyzer.py — Spor Maçı Analiz Motoru
=============================================
Videonun ilk 15dk + son 15dk ASR transcript'i ve son 15dk OCR frame'lerinden
maç bilgilerini çıkarır, yapılandırır ve çapraz doğrulama yapar.

Pipeline akışı:
  1. Segment Extract  → ilk 15dk + son 15dk ses/görüntü ayır
  2. ASR Transcribe   → her iki segmenti Whisper/ASR ile dinle
  3. Frame Extract    → son 15dk'dan 10sn'de bir frame çek
  4. OCR Read         → frame'lerden skor tabelası / istatistik grafiği oku
  5. Sport Analyze    → Gemini ile birleştir, yapılandır, çapraz doğrula
  6. Export           → rapor oluştur (D:\DATABASE + F:\Sonuclar)

Katmanlar:
  - ASR ilk 15dk  → maç meta bilgileri (takımlar, şehir, lig, tarih, hava, hakem)
  - ASR son 15dk  → skor, goller, kartlar, spiker özeti
  - OCR son 15dk  → skor tabelası, istatistik grafiği
  - Gemini        → yapılandırma + çapraz doğrulama
  - Spor dalı     → otomatik tespit (futbol/basketbol/voleybol/diğer)

Çağatay Mert — Mart 2026
"""

import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("VITOS.sport_analyzer")


# ============================================================
#  SPOR DALI TESPİT KURALLARI
# ============================================================

SPORT_KEYWORDS = {
    "futbol": {
        "keywords": [
            "gol", "penaltı", "korner", "ofsayt", "faul",
            "kaleye", "kaleci", "defans", "forvet", "orta saha",
            "sarı kart", "kırmızı kart", "serbest vuruş",
            "taç atışı", "devre arası", "uzatma dakikaları",
            "aut", "elle oynama", "top çizgiyi geçti",
            "goal", "penalty", "corner", "offside",
            "halftime", "free kick", "yellow card", "red card",
        ],
        "weight": 1.0,
    },
    "basketbol": {
        "keywords": [
            "faul", "serbest atış", "üç sayı", "çeyrek",
            "ribaund", "asist", "blok", "top çalma",
            "periyot", "uzatma", "bonus", "sayı",
            "slam dunk", "alley oop", "fast break",
            "three pointer", "free throw", "rebound",
            "quarter", "overtime",
        ],
        "weight": 1.0,
    },
    "voleybol": {
        "keywords": [
            "set", "servis", "smaç", "blok", "pas",
            "manşet", "parmak pas", "çapraz", "paralel",
            "deuce", "sayı", "libero", "setter",
            "ace", "spike", "dig", "rally",
        ],
        "weight": 1.0,
    },
}

# Futbol ve basketbol ortak kelimeler — belirsizlik çözücü
DISAMBIGUATORS = {
    "futbol_only": ["gol", "goal", "korner", "corner", "ofsayt", "offside", "kaleci", "taç atışı"],
    "basketbol_only": ["serbest atış", "free throw", "üç sayı", "three pointer", "ribaund", "rebound", "çeyrek", "quarter", "periyot"],
    "voleybol_only": ["set", "smaç", "spike", "manşet", "libero", "ace", "servis"],
}


# ============================================================
#  FUTBOL — GOL VE KART ÇIKARMA PATERNLERİ
# ============================================================

# Spiker genellikle şöyle söyler:
# "34. dakikada Metin Oktay golü attı"
# "82'de Fatih Terim'den müthiş bir gol"
# "Penaltıdan gol! Hagi!"

GOL_PATTERNS = [
    # "34. dakikada Metin Oktay" / "34'de Metin Oktay"
    r"(\d{1,3})[\.\']?\s*(?:dakika(?:da|sında)?|dk)\s+(\w[\w\s]{2,30}?)(?:\s+gol)",
    # "golü attı ... Metin Oktay ... 34. dakika"
    r"gol[üu]?\s+(?:att[ıi]|bul\w*|geldi)\s*[,.:!]?\s*(\w[\w\s]{2,30}?)\s*[,.:!]?\s*(\d{1,3})",
    # "Metin Oktay ... gol ... 34'"
    r"(\w[\w\s]{2,30}?)\s+gol[üu]?\s*[,.:!]?\s*(\d{1,3})",
    # "penaltıdan gol ... isim"
    r"penalt[ıi]\w*\s+gol\w*\s*[,.:!]?\s*(\w[\w\s]{2,30})",
]

KART_PATTERNS = [
    # "sarı kart ... 23. dakika ... Alpaslan"
    r"(sar[ıi]\s+kart|yellow\s+card)\s*[,.:!]?\s*(\d{1,3})[\.\']?\s*(?:dakika\w*)?\s*[,.:!]?\s*(\w[\w\s]{2,30})",
    # "kırmızı kart ... isim ... dakika"
    r"(k[ıi]rm[ıi]z[ıi]\s+kart|red\s+card)\s*[,.:!]?\s*(\w[\w\s]{2,30}?)\s*[,.:!]?\s*(\d{1,3})",
    # "isim ... sarı kart gördü ... dakika"
    r"(\w[\w\s]{2,30}?)\s+(sar[ıi]\s+kart|k[ıi]rm[ıi]z[ıi]\s+kart)\s*\w*\s*(\d{1,3})",
]

NO_GOAL_PATTERNS = [
    r"\bgol yok\b",
    r"gol izleyemedik",
    r"gol sesi \w* gelmedi",
]

SUMMARY_CARD_PATTERNS = [
    r"ma[çc][ıi]n\s+(\d{1,3})[\.\']?\s*dakikas[ıi]nda\s+"
    r"([A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+){0,3})\s+"
    r"(sar[ıi]\s+kart|k[ıi]rm[ıi]z[ıi]\s+kart)\s+g[öo]rd[üu]",
    r"(\d{1,3})[\.\']?\s*dakikas[ıi]nda\s+"
    r"([A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+){0,3})\s+"
    r"(sar[ıi]\s+kart|k[ıi]rm[ıi]z[ıi]\s+kart)\s+g[öo]rd[üu]",
]

# OCR skor tabelası desenleri
SKOR_PATTERNS = [
    # "GS 2 - FB 1" / "Galatasaray 2 - 1 Fenerbahçe"
    r"([A-ZÇĞİÖŞÜa-zçğıöşü\s\.]{2,25})\s*(\d{1,3})\s*[-–—:]\s*(\d{1,3})\s*([A-ZÇĞİÖŞÜa-zçğıöşü\s\.]{2,25})",
    # "2:1" / "87-74" (sadece sayılar)
    r"(\d{1,3})\s*[-–—:]\s*(\d{1,3})",
]

EVIDENCE_HINTS = {
    "match_competition": [
        r"\btak[ıi]m\w*\b",
        r"\b(?:ptt|tff|bank asya|s[üu]per)\b",
        r"\blig\w*\b",
        r"\bhafta\w*\b",
        r"\btarih\w*\b",
        r"\bstadyum\w*\b",
        r"\barena\b",
        r"\bhava\w*\b",
        r"\bg[üu]neşli\b",
        r"\byağmurlu\b",
        r"\br[üu]zgarl[ıi]\b",
        r"\b(?:spor|idman yurdu|belediyespor|birli[ğg]i|g[üu]c[üu])\b",
    ],
    "match_officials": [
        r"\bhakem\w*\b",
        r"\byard[ıi]mc[ıi]\w*\b",
        r"\b(?:d[öo]rd[üu]nc[üu]|4\.)\s+hakem\b",
        r"\byan hakem\w*\b",
        r"\bçizgi hakem\w*\b",
    ],
    "score": [
        r"\bskor\w*\b",
        r"\bsonu[çc]\w*\b",
        r"\bma[çc][ıi]n\s+sonu[çc]\w*\b",
        r"\bberabere\b",
        r"\beşitlik\b",
        r"\bma[çc]\s+bitti\b",
        r"\b\d{1,3}\s*[-–—:]\s*\d{1,3}\b",
    ],
    "goals": [
        r"\bgol\w*\b",
        r"\bpenalt[ıi]\w*\b",
        r"\bağlar\w*\b",
        r"\bfile\w*\b",
        r"\böne geçti\b",
        r"\bberaberli[ğg]i getirdi\b",
        r"\bfark[ıi]\s+açt[ıi]\b",
    ],
    "cards": [
        r"\bkart\w*\b",
        r"\bsar[ıi]\w*\b",
        r"\bk[ıi]rm[ıi]z[ıi]\w*\b",
        r"\boyundan at[ıi]ld[ıi]\b",
        r"\bihra[çc]\w*\b",
        r"\b10 kişi\b",
        r"\beksik kald[ıi]\b",
    ],
}

MATCH_INFO_GEMINI_QUERIES = (
    {
        "name": "competition",
        "fields": ("takimlar", "lig", "hafta", "tarih", "sehir", "stadyum", "hava"),
        "hint_key": "match_competition",
        "source": "first",
        "max_chars": 1800,
        "prompt": """Bu bir spor maçı yayınının transcript'inden seçilmiş kısa parçalardır.
Sadece bu parçaya bak. Uydurma yapma. Parçada açıkça geçmeyen alanları boş bırak.

Sadece JSON formatında cevap ver:
{{
    "takimlar": "Takım1 — Takım2",
    "lig": "Lig adı",
    "hafta": "Kaçıncı hafta",
    "tarih": "Tarih bilgisi",
    "sehir": "Şehir",
    "stadyum": "Stadyum adı",
    "hava": "Hava durumu",
    "evidence": ["kanıt cümlesi 1", "kanıt cümlesi 2"]
}}

Parça:
{evidence}""",
    },
    {
        "name": "officials",
        "fields": ("hakem", "yardimci_hakemler", "dorduncu_hakem"),
        "hint_key": "match_officials",
        "source": "first",
        "max_chars": 1400,
        "prompt": """Bu bir spor maçı yayınının transcript'inden seçilmiş kısa parçalardır.
Sadece bu parçaya bak. Uydurma yapma. Parçada açıkça geçmeyen alanları boş bırak.

Sadece JSON formatında cevap ver:
{{
    "hakem": "Baş hakem adı",
    "yardimci_hakemler": "1. ve 2. yardımcı hakem",
    "dorduncu_hakem": "Dördüncü hakem",
    "evidence": ["kanıt cümlesi 1", "kanıt cümlesi 2"]
}}

Parça:
{evidence}""",
    },
)


# ============================================================
#  ANA SINIF: SportAnalyzer
# ============================================================

class SportAnalyzer:
    """Spor maçı analiz motoru."""

    def __init__(self, config: Dict[str, Any]):
        """
        config beklenen alanlar:
            segment_minutes: int (default 15) — ilk/son kaç dakika
            frame_interval_sec: int (default 10) — frame çekme aralığı
            gemini_enabled: bool
            gemini_model: str
            asr_engine: str (whisper / gemini_audio)
            ocr_engine: str (paddleocr / oneocr)
        """
        self.segment_minutes = config.get("segment_minutes", 15)
        self.frame_interval_sec = config.get("frame_interval_sec", 10)
        self.gemini_enabled = config.get("gemini_enabled", True)
        self.gemini_model = config.get("gemini_model", "gemini-2.0-flash")
        self.asr_engine = config.get("asr_engine", "whisper")
        self.ocr_engine = config.get("ocr_engine", "paddleocr")
        self.ffmpeg = config.get("ffmpeg", "ffmpeg")
        self.ffprobe = config.get("ffprobe", "ffprobe")
        self.speech_separation = config.get("speech_separation", True)
        self.selected_channel = config.get("selected_channel", None)
        self.selected_channel_confidence = float(
            config.get("selected_channel_confidence", 0.0) or 0.0
        )
        self.include_mix_fallback = bool(config.get("include_mix_fallback", True))
        self.venv_audio_python = config.get(
            "venv_audio_python",
            r"F:\Root\venv_audio\Scripts\python.exe",
        )

        # Analiz sonuçları
        self.match_info = {}          # Maç bilgileri (takımlar, şehir, lig...)
        self.score_info = {}          # Skor bilgisi
        self.goals = []               # Goller (futbol)
        self.cards = []               # Kartlar (futbol)
        self.sport_type = "bilinmiyor"  # Tespit edilen spor dalı
        self.speaker_notes = []       # Spiker notları
        self.verification_log = []    # Doğrulama logu

        # Ham veriler
        self._asr_first_segment = ""
        self._asr_last_segment = ""
        self._ocr_frames = []         # [(frame_no, ocr_text), ...]
        self._video_filename = ""     # Dosya adı fallback tespiti için
        self._segment_audio_candidates = {"first": [], "last": []}

    # ----------------------------------------------------------
    #  1. SEGMENT EXTRACT — Video'dan ilk/son N dakika ayır
    # ----------------------------------------------------------

    def extract_segments(self, video_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Video'dan ilk ve son N dakika ses segmentlerini çıkarır.
        FFmpeg kullanır.

        Returns:
            (first_segment_path, last_segment_path)
        """
        import subprocess

        duration_cmd = [
            self.ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]

        try:
            result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=30)
            total_seconds = float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Video süresi alınamadı: {e}")
            raise

        n_channels = self._get_audio_channels(video_path)

        seg_seconds = self.segment_minutes * 60
        last_start = max(0, total_seconds - seg_seconds)
        self._segment_audio_candidates = {"first": [], "last": []}
        primary_mode = self._resolve_primary_channel_mode(n_channels)
        use_mix_fallback = self.include_mix_fallback and n_channels >= 2 and primary_mode != "mix"

        primary_label = "mix" if primary_mode == "mix" else f"kanal {primary_mode[-1]}"
        self._log("segment_extract", f"Video: {total_seconds:.0f}sn toplam")
        self._log(
            "segment_extract",
            f"İlk segment: 0 — {seg_seconds}sn | Son segment: {last_start:.0f} — {total_seconds:.0f}sn"
        )
        self._log(
            "segment_extract",
            f"ASR ana kaynak seçimi: {primary_label} "
            f"(dil tespiti güveni={self.selected_channel_confidence*100:.1f}%)"
        )

        first_path = self._extract_segment_variant(
            video_path, 0, seg_seconds, output_dir, "first", primary_mode
        )
        last_path = self._extract_segment_variant(
            video_path, last_start, seg_seconds, output_dir, "last", primary_mode
        )
        self._register_audio_candidate("first", first_path, "raw", primary_mode)
        self._register_audio_candidate("last", last_path, "raw", primary_mode)

        if use_mix_fallback:
            first_mix = self._extract_segment_variant(
                video_path, 0, seg_seconds, output_dir, "first", "mix"
            )
            last_mix = self._extract_segment_variant(
                video_path, last_start, seg_seconds, output_dir, "last", "mix"
            )
            self._register_audio_candidate("first", first_mix, "raw", "mix")
            self._register_audio_candidate("last", last_mix, "raw", "mix")
            self._log("segment_extract", "Mix fallback adayı üretildi (raw)")

        # ── Demucs ile spiker sesini efekten ayır ──────────────────────
        if self.speech_separation:
            try:
                from audio.stages.separate_speech import SpeechSeparationStage
                sep = SpeechSeparationStage(
                    venv_python=self.venv_audio_python,
                    ffmpeg=self.ffmpeg,
                    log_cb=lambda m: self._log("segment_extract", m),
                )
                sep_first = sep.run(first_path, output_dir)
                sep_last = sep.run(last_path, output_dir)
                if sep_first:
                    self._register_audio_candidate("first", sep_first, "vocals", primary_mode)
                    self._log("segment_extract", "✓ İlk segment → Demucs vocals adayı eklendi")
                else:
                    self._log("segment_extract", "⚠ İlk segment ayırma başarısız — sadece raw kullanılacak")
                if sep_last:
                    self._register_audio_candidate("last", sep_last, "vocals", primary_mode)
                    self._log("segment_extract", "✓ Son segment → Demucs vocals adayı eklendi")
                else:
                    self._log("segment_extract", "⚠ Son segment ayırma başarısız — sadece raw kullanılacak")
            except Exception as e:
                self._log("segment_extract", f"⚠ Ses ayırma modülü yüklenemedi: {e} — raw ile devam")

        return first_path, last_path

    def _resolve_primary_channel_mode(self, n_channels: int) -> str:
        """Dil tespiti sonucuna göre hangi kanalın ana ASR kaynağı olacağını belirle."""
        if (
            isinstance(self.selected_channel, int)
            and 0 <= self.selected_channel < max(1, n_channels)
        ):
            return f"ch{self.selected_channel}"
        return "mix"

    def _build_speech_filter(self, channel_mode: str) -> str:
        """ASR öncesi hafif konuşma odaklı filtre zinciri."""
        stages = [
            "afftdn=nf=-25",
            "highpass=f=180",
            "lowpass=f=4200",
            "dynaudnorm=f=350:g=21",
        ]
        if channel_mode.startswith("ch"):
            channel_index = channel_mode[2:]
            stages.insert(0, f"pan=mono|c0=c{channel_index}")
        return ",".join(stages)

    def _extract_segment_variant(
        self,
        video_path: str,
        start_sec: float,
        duration_sec: int,
        output_dir: str,
        segment_key: str,
        channel_mode: str,
    ) -> str:
        """Belirli kanal/mix modu için segment WAV üret."""
        import subprocess

        out_name = f"segment_{segment_key}_{channel_mode}.wav"
        out_path = os.path.join(output_dir, out_name)
        cmd = [
            self.ffmpeg, "-y", "-i", video_path,
            "-ss", str(start_sec), "-t", str(duration_sec), "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000",
        ]
        if channel_mode == "mix":
            cmd += ["-ac", "1"]
        cmd += ["-af", self._build_speech_filter(channel_mode), out_path]
        subprocess.run(cmd, capture_output=True, timeout=max(300, int(duration_sec * 2)))
        return out_path

    def _register_audio_candidate(
        self,
        segment_key: str,
        path: str,
        candidate_kind: str,
        source_mode: str,
    ) -> None:
        """ASR seçiminde değerlendirilecek aday sesi kaydet."""
        if not path:
            return
        candidates = self._segment_audio_candidates.setdefault(segment_key, [])
        if any(existing.get("path") == path for existing in candidates):
            return
        candidates.append({
            "path": path,
            "candidate_kind": candidate_kind,
            "source_mode": source_mode,
            "label": f"{source_mode}_{candidate_kind}",
        })

    def get_segment_audio_candidates(self, segment_key: str) -> List[Dict[str, Any]]:
        """Belirli segment için üretilen ses adaylarını döndür."""
        return list(self._segment_audio_candidates.get(segment_key, []))

    def _get_audio_channels(self, video_path: str) -> int:
        """ffprobe ile ses kanalı sayısını al."""
        import subprocess

        try:
            result = subprocess.run(
                [
                    self.ffprobe, "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=channels",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return int((result.stdout or "1").strip() or "1")
        except Exception:
            return 1

    # ----------------------------------------------------------
    #  2. FRAME EXTRACT — Son segment'ten frame çıkar
    # ----------------------------------------------------------

    def extract_frames(self, video_path: str, output_dir: str) -> List[str]:
        """
        Video'nun son N dakikasından her X saniyede bir frame çıkarır.

        Returns:
            Liste of frame dosya yolları
        """
        import subprocess

        duration_cmd = [
            self.ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]

        result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=30)
        total_seconds = float(result.stdout.strip())
        seg_seconds = self.segment_minutes * 60
        start_time = max(0, total_seconds - seg_seconds)

        frames_dir = os.path.join(output_dir, "sport_frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Her frame_interval_sec saniyede bir frame
        fps_value = 1.0 / self.frame_interval_sec

        subprocess.run([
            self.ffmpeg, "-y", "-i", video_path,
            "-ss", str(start_time),
            "-vf", f"fps={fps_value}",
            os.path.join(frames_dir, "frame_%04d.png")
        ], capture_output=True, timeout=300)

        frame_files = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.endswith(".png")
        ])

        self._log("frame_extract", f"Son {self.segment_minutes}dk'dan {len(frame_files)} frame çıkarıldı ({self.frame_interval_sec}sn aralıkla)")

        return frame_files

    # ----------------------------------------------------------
    #  3. ASR — Ses segmentlerini yazıya çevir
    # ----------------------------------------------------------

    def transcribe_segment(self, audio_path: str, segment_name: str) -> str:
        """
        Ses dosyasını ASR ile yazıya çevirir.
        Şimdilik placeholder — gerçek ASR motoru pipeline_runner'dan gelecek.
        """
        # TODO: Gerçek ASR entegrasyonu
        # self.asr_engine == "whisper" → Whisper
        # self.asr_engine == "gemini_audio" → Gemini Audio API

        transcript = ""

        # Pipeline runner bu fonksiyonu override edecek
        # veya dışarıdan transcript verilecek
        self._log("asr_transcribe", f"{segment_name}: {len(transcript)} karakter transcript")

        return transcript

    def set_video_name(self, filename: str):
        """Dosya adını ayarla (ASR boş kalınca spor dalı tespiti için fallback)."""
        self._video_filename = filename or ""

    def set_transcripts(self, first_transcript: str, last_transcript: str):
        """Pipeline runner'dan gelen transcript'leri ayarla."""
        self._asr_first_segment = first_transcript
        self._asr_last_segment = last_transcript

        self._log("asr_transcribe", f"İlk 15dk transcript: {len(first_transcript)} karakter")
        self._log("asr_transcribe", f"Son 15dk transcript: {len(last_transcript)} karakter")

    # ----------------------------------------------------------
    #  4. OCR — Frame'lerden metin oku
    # ----------------------------------------------------------

    def set_ocr_results(self, ocr_results: List[Tuple[int, str]]):
        """
        Pipeline runner'dan gelen OCR sonuçlarını ayarla.
        ocr_results: [(frame_no, ocr_text), ...]
        """
        self._ocr_frames = ocr_results
        non_empty = [(fno, txt) for fno, txt in ocr_results if txt.strip()]
        self._log("ocr_read", f"{len(ocr_results)} frame okundu, {len(non_empty)} frame'de metin bulundu")

    # ----------------------------------------------------------
    #  5. SPORT ANALYZE — Ana analiz
    # ----------------------------------------------------------

    def analyze(self) -> Dict[str, Any]:
        """
        Tüm verileri birleştirir ve analiz eder.

        Sıra:
        1. Spor dalı tespit et (ASR ilk 15dk'dan)
        2. Maç bilgilerini çıkar (ASR ilk 15dk → Gemini)
        3. Skor bilgisi çıkar (ASR son 15dk + OCR)
        4. Futbolsa: goller + kartlar çıkar
        5. Çapraz doğrulama yap
        6. Rapor oluştur

        Returns:
            Yapılandırılmış maç verisi dict
        """
        self._log("analyze", "═" * 50)
        self._log("analyze", "ANALİZ BAŞLADI")
        self._log("analyze", "═" * 50)

        # 5.1 — Spor dalı tespiti
        self.sport_type = self._detect_sport_type()

        # 5.2 — Maç bilgileri (Gemini ile)
        self.match_info = self._extract_match_info()

        # 5.3 — Skor bilgisi (ASR + OCR)
        self.score_info = self._extract_score()

        # 5.4 — Futbolsa detaylı bilgi
        if self.sport_type == "futbol":
            self.goals = self._extract_goals()
            self.cards = self._extract_cards()

        # 5.5 — Spiker notları
        self.speaker_notes = self._extract_speaker_notes()

        # 5.6 — Çapraz doğrulama
        self._cross_validate()

        # 5.7 — Sonuç oluştur
        result = self._build_result()

        self._log("analyze", "═" * 50)
        self._log("analyze", "ANALİZ TAMAMLANDI")
        self._log("analyze", "═" * 50)

        return result

    # ----------------------------------------------------------
    #  5.1 — Spor dalı tespiti
    # ----------------------------------------------------------

    def _detect_sport_type(self) -> str:
        """ASR transcript'teki anahtar kelimelere göre spor dalını tespit eder."""

        combined_text = (self._asr_first_segment + " " + self._asr_last_segment).lower()
        scores = {}

        for sport, data in SPORT_KEYWORDS.items():
            count = 0
            found_keywords = []
            for kw in data["keywords"]:
                occurrences = combined_text.count(kw.lower())
                if occurrences > 0:
                    count += occurrences * data["weight"]
                    found_keywords.append(f"{kw}({occurrences}x)")
            scores[sport] = (count, found_keywords)

        # Belirsizlik çözücü — sadece o spor dalına özgü kelimeler
        for sport, exclusive_kws in DISAMBIGUATORS.items():
            sport_name = sport.replace("_only", "")
            exclusive_count = sum(
                1 for kw in exclusive_kws
                if kw.lower() in combined_text
            )
            if sport_name in scores:
                old_score = scores[sport_name][0]
                scores[sport_name] = (
                    old_score + exclusive_count * 2,  # Özel kelimeler 2x ağırlık
                    scores[sport_name][1]
                )

        # En yüksek skoru bul
        best_sport = max(scores, key=lambda s: scores[s][0])
        best_score = scores[best_sport][0]

        if best_score == 0:
            # Dosya adı fallback: ASR boşsa dosya adındaki spor etiketine bak
            fname_upper = self._video_filename.upper()
            _FILENAME_SPORT_HINTS = {
                "futbol": ["FUTBOL"],
                "basketbol": ["BASKETBOL"],
                "voleybol": ["VOLEYBOL"],
            }
            for sport, hints in _FILENAME_SPORT_HINTS.items():
                if any(h in fname_upper for h in hints):
                    self._log(
                        "spor_tespit",
                        f"✓ Spor dalı dosya adından tespit edildi: {sport.upper()} "
                        f"(ASR boş — '{self._video_filename}')"
                    )
                    return sport
            self._log("spor_tespit", "⚠ Spor dalı tespit edilemedi — yeterli anahtar kelime bulunamadı")
            return "bilinmiyor"

        self._log("spor_tespit", f"✓ Spor dalı: {best_sport.upper()}")
        for sport, (score, keywords) in scores.items():
            if score > 0:
                kw_str = ", ".join(keywords[:10])
                self._log("spor_tespit", f"  {sport}: skor={score:.1f} — [{kw_str}]")

        return best_sport

    # ----------------------------------------------------------
    #  5.2 — Maç bilgileri çıkarma (Gemini)
    # ----------------------------------------------------------

    def _extract_match_info(self) -> Dict[str, str]:
        """
        İlk 15dk ASR transcript'inden maç meta bilgilerini çıkarır.
        Gemini açıksa API ile, kapalıysa regex ile.
        """
        info = {
            "spor_dali": self.sport_type,
            "takimlar": "",
            "lig": "",
            "hafta": "",
            "tarih": "",
            "sehir": "",
            "stadyum": "",
            "hava": "",
            "hakem": "",
            "yardimci_hakemler": "",
            "dorduncu_hakem": "",
        }

        if not self._asr_first_segment.strip():
            self._log("mac_bilgileri", "⚠ İlk 15dk transcript boş — Gemini/regex atlandı, fallback deneniyor")
        else:
            if self.gemini_enabled:
                info = self._extract_match_info_gemini(info)
            else:
                info = self._extract_match_info_regex(info)

        info = self._fill_match_info_fallbacks(info)

        # Log
        for key, val in info.items():
            if val and key != "spor_dali":
                self._log("mac_bilgileri", f"✓ {key}: {val}")

        empty_fields = [k for k, v in info.items() if not v and k != "spor_dali"]
        if empty_fields:
            self._log("mac_bilgileri", f"⚠ Bulunamayan: {', '.join(empty_fields)}")

        return info

    def _get_evidence_source_text(self, source: str) -> str:
        """Alan bazlı arama için hangi transcript bloğunun kullanılacağını seç."""
        if source == "first":
            return self._asr_first_segment
        if source == "last":
            return self._asr_last_segment
        if source == "combined":
            return "\n".join(
                part for part in (self._asr_first_segment, self._asr_last_segment) if part
            )
        return ""

    def _split_into_sentences(self, text: str) -> List[str]:
        """Transcript'i kaba cümle parçalarına ayır."""
        protected = re.sub(
            r"(\d)\.\s+(L[İI]G\b)",
            r"\1.__NOSPLIT__\2",
            text or "",
            flags=re.IGNORECASE,
        )
        raw_parts = re.split(
            r"(?<=[!?])\s+|(?<=\.)\s+(?=[A-ZÇĞİÖŞÜ])|[\r\n]+",
            protected,
        )
        sentences = []
        for part in raw_parts:
            cleaned = self._normalise_text(part.replace(".__NOSPLIT__", ". "))
            if cleaned:
                sentences.append(cleaned)
        return sentences

    def _sentence_matches_hints(self, sentence: str, hints: List[str]) -> bool:
        """Cümle verilen kök/kalıp ailesinden en az birine uyuyor mu?"""
        lowered = self._normalise_text(sentence).lower()
        return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in hints)

    def _collect_evidence_text(
        self,
        text: str,
        hint_key: str,
        *,
        window_radius: int = 1,
        max_matches: int = 6,
        max_chars: int = 1600,
    ) -> str:
        """İlgili cümleleri ve komşularını küçük bir kanıt parçası olarak topla."""
        hints = EVIDENCE_HINTS.get(hint_key, [])
        if not hints:
            return ""

        sentences = self._split_into_sentences(text)
        if not sentences:
            return ""

        matched_indices = [
            idx for idx, sentence in enumerate(sentences)
            if self._sentence_matches_hints(sentence, hints)
        ]
        if not matched_indices:
            return ""

        selected = []
        seen = set()
        total_chars = 0
        reached_limit = False

        for idx in matched_indices[:max_matches]:
            start = max(0, idx - window_radius)
            end = min(len(sentences), idx + window_radius + 1)
            for pos in range(start, end):
                sentence = sentences[pos]
                token = sentence.lower()
                if token in seen:
                    continue
                if total_chars and total_chars + len(sentence) + 1 > max_chars:
                    reached_limit = True
                    break
                seen.add(token)
                selected.append((sentence, self._sentence_matches_hints(sentence, hints)))
                total_chars += len(sentence) + 1
            if reached_limit:
                break

        if not selected:
            return ""

        first_match_idx = next(
            (idx for idx, (_, is_match) in enumerate(selected) if is_match),
            0,
        )
        last_match_idx = max(
            idx for idx, (_, is_match) in enumerate(selected) if is_match
        )
        trimmed = [sentence for sentence, _ in selected[first_match_idx:last_match_idx + 1]]
        return "\n".join(trimmed)

    def _prepare_json_candidate_text(self, response: str) -> str:
        """LLM cevabından parse denemesi için JSON adayı metni hazırla."""
        if not response:
            return ""
        text = response.strip().lstrip("\ufeff")
        if "```" in text:
            fence_match = re.search(
                r"```(?:json)?\s*(.*?)```",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if fence_match:
                text = fence_match.group(1).strip()
        return text

    def _parse_first_json_value(self, response: str, expected_type: type) -> Any | None:
        """Yanıttaki ilk geçerli JSON obje/listesini bulmaya çalış."""
        text = self._prepare_json_candidate_text(response)
        if not text:
            return None

        start_char = "{" if expected_type is dict else "["
        decoder = json.JSONDecoder()

        for idx, char in enumerate(text):
            if char != start_char:
                continue
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, expected_type):
                return parsed
        return None

    def _format_llm_response_preview(self, response: str, max_chars: int = 240) -> str:
        """Log için ham LLM cevabını tek satırlık kısa önizlemeye indir."""
        text = self._prepare_json_candidate_text(response)
        if not text:
            return "<boş cevap>"
        compact = self._normalise_text(text)
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    def _log_gemini_parse_failure(
        self,
        stage: str,
        label: str,
        response: str,
        *,
        expected: str,
    ) -> None:
        """Gemini cevabı parse edilemezse tanı amaçlı ham önizlemeyi logla."""
        preview = self._format_llm_response_preview(response)
        self._log(
            stage,
            f"⚠ Gemini {label} cevabı parse edilemedi ({expected}) | ham cevap: {preview}",
        )

    def _parse_json_object(self, response: str) -> Dict[str, Any] | None:
        """LLM cevabından ilk JSON objesini ayıkla."""
        parsed = self._parse_first_json_value(response, dict)
        return parsed if isinstance(parsed, dict) else None

    def _parse_json_list(self, response: str) -> List[Dict[str, Any]] | None:
        """LLM cevabından ilk JSON listesini ayıkla."""
        parsed = self._parse_first_json_value(response, list)
        return parsed if isinstance(parsed, list) else None

    def _extract_match_info_gemini(self, info: Dict) -> Dict:
        """Gemini API ile maç bilgilerini çıkar."""
        try:
            from core.gemini_client import GeminiClient
            client = GeminiClient(model=self.gemini_model)
            any_success = False
            had_parse_failure = False

            for query in MATCH_INFO_GEMINI_QUERIES:
                source_text = self._get_evidence_source_text(query["source"])
                evidence = self._collect_evidence_text(
                    source_text,
                    query["hint_key"],
                    max_chars=query["max_chars"],
                )
                if not evidence:
                    self._log(
                        "mac_bilgileri",
                        f"⚠ Gemini için {query['name']} kanıtı bulunamadı",
                    )
                    continue

                prompt = query["prompt"].format(evidence=evidence)
                response = client.generate(prompt)
                parsed = self._parse_json_object(response)
                if not parsed:
                    had_parse_failure = True
                    self._log_gemini_parse_failure(
                        "mac_bilgileri",
                        query["name"],
                        response,
                        expected="json object",
                    )
                    continue

                updated_fields = []
                for field in query["fields"]:
                    value = parsed.get(field)
                    if value:
                        info[field] = str(value).strip()
                        updated_fields.append(field)

                if updated_fields:
                    any_success = True
                    self._log(
                        "mac_bilgileri",
                        f"✓ Gemini {query['name']} alanları: {', '.join(updated_fields)}",
                    )
                else:
                    self._log(
                        "mac_bilgileri",
                        f"⚠ Gemini {query['name']} alan bulamadı",
                    )

            if any_success:
                self._log("mac_bilgileri", "✓ Gemini alan bazlı kanıt parçalarıyla çalıştı")
            else:
                self._log("mac_bilgileri", "⚠ Gemini anlamlı alan döndüremedi — fallback devam ediyor")

            if had_parse_failure:
                self._log("mac_bilgileri", "⚠ Gemini parse sorunu görüldü — regex fallback ile eksikler taranıyor")
                info = self._extract_match_info_regex(info)

        except Exception as e:
            logger.warning(f"Gemini maç bilgisi çıkarma hatası: {e}")
            self._log("mac_bilgileri", f"⚠ Gemini hatası: {e} — regex'e düşülüyor")
            info = self._extract_match_info_regex(info)

        return info

    def _extract_match_info_regex(self, info: Dict) -> Dict:
        """Regex ile basit maç bilgisi çıkarma (Gemini kapalıysa fallback)."""
        text = self._normalise_text(self._asr_first_segment)

        # Takım tespiti — en sık geçen büyük harfli kelime çiftleri
        # Bu basit bir heuristik, Gemini çok daha iyi
        self._log("mac_bilgileri", "⚠ Gemini kapalı — sadece regex ile çıkarılıyor (sınırlı)")

        referee_info = self._extract_referee_info(text)
        for key, value in referee_info.items():
            if value and not info.get(key):
                info[key] = value

        if not info.get("lig"):
            match = re.search(
                r"\b((?:PTT|TFF|BANK ASYA)\s*1\.\s*L[İI]G(?:DE|D[EI])?|S[ÜU]PER L[İI]G)\b",
                text,
                re.IGNORECASE,
            )
            if match:
                info["lig"] = match.group(1).strip()

        if not info.get("sehir"):
            city = self._extract_city_from_text(text)
            if city:
                info["sehir"] = city

        return info

    def _fill_match_info_fallbacks(self, info: Dict[str, str]) -> Dict[str, str]:
        """Eksik maç bilgisini transcript ve dosya adından tamamla."""
        combined = self._normalise_text(
            " ".join(
                part for part in (
                    self._asr_first_segment,
                    self._asr_last_segment,
                    self._video_filename.replace("_", " "),
                ) if part
            )
        )

        if not info.get("takimlar"):
            teams = self._extract_teams_from_filename()
            if not teams:
                teams = self._extract_teams_from_summary(combined)
            if teams:
                info["takimlar"] = f"{teams[0]} — {teams[1]}"

        if not info.get("lig"):
            match = re.search(
                r"\b((?:PTT|TFF|BANK ASYA)\s*1\.\s*L[İI]G(?:DE|D[EI])?|S[ÜU]PER L[İI]G)\b",
                combined,
                re.IGNORECASE,
            )
            if match:
                info["lig"] = match.group(1).strip()

        referee_info = self._extract_referee_info(combined)
        for key, value in referee_info.items():
            if value and not info.get(key):
                info[key] = value

        if not info.get("sehir"):
            city = self._extract_city_from_text(combined)
            if city:
                info["sehir"] = city

        return info

    # ----------------------------------------------------------
    #  5.3 — Skor bilgisi
    # ----------------------------------------------------------

    def _extract_score(self) -> Dict[str, Any]:
        """ASR son 15dk + OCR frame'lerden skor çıkarır."""

        score = {
            "asr_score": None,
            "ocr_score": None,
            "final_score": None,
            "source": "",
        }

        # ASR'den skor çıkar
        asr_scores = self._extract_score_from_text(self._asr_last_segment)
        if asr_scores:
            score["asr_score"] = asr_scores[-1]  # Son bulunan skor (maç sonu)
            self._log("skor", f"✓ ASR skor: {asr_scores[-1]}")
        elif self.gemini_enabled:
            gemini_score = self._extract_score_gemini()
            if gemini_score:
                score["asr_score"] = gemini_score
                self._log("skor", f"✓ Gemini skor fallback: {gemini_score}")

        # OCR'den skor çıkar
        for frame_no, ocr_text in self._ocr_frames:
            if not ocr_text.strip():
                continue
            ocr_scores = self._extract_score_from_text(ocr_text)
            if ocr_scores:
                score["ocr_score"] = ocr_scores[0]
                self._log("skor", f"✓ OCR skor (frame {frame_no}): {ocr_scores[0]}")
                break  # İlk bulunan yeterli

        if not score["ocr_score"]:
            self._log("skor", "⚠ OCR'den skor bulunamadı")

        # Final skor belirle
        if score["asr_score"] and score["ocr_score"]:
            # İkisi de varsa çapraz kontrol
            if score["asr_score"] == score["ocr_score"]:
                score["final_score"] = score["asr_score"]
                score["source"] = "ASR + OCR (doğrulandı)"
            else:
                # Uyuşmuyorsa OCR'ye öncelik ver (görsel daha güvenilir)
                score["final_score"] = score["ocr_score"]
                score["source"] = "OCR (ASR ile uyuşmadı)"
                self._log("skor", f"⚠ ASR ({score['asr_score']}) ≠ OCR ({score['ocr_score']}) — OCR tercih edildi")
        elif score["asr_score"]:
            score["final_score"] = score["asr_score"]
            score["source"] = "sadece ASR"
        elif score["ocr_score"]:
            score["final_score"] = score["ocr_score"]
            score["source"] = "sadece OCR"
        else:
            self._log("skor", "✗ Skor bulunamadı — ne ASR ne OCR")

        return score

    def _extract_score_from_text(self, text: str) -> List[str]:
        """Metinden skor paternlerini çıkarır."""
        scores = []
        summary_pattern = re.compile(
            r"([A-ZÇĞİÖŞÜa-zçğıöşü\s\.]{3,40}?)\s+(\d{1,3})[\.\-–—:]\s*"
            r"([A-ZÇĞİÖŞÜa-zçğıöşü\s\.]{3,40}?)\s+(\d{1,3})(?:\s+ma[çc][ıi]n sonucu)?",
            re.IGNORECASE,
        )
        for m in summary_pattern.finditer(text):
            scores.append(f"{m.group(2)}-{m.group(4)}")
        for pattern in SKOR_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for m in matches:
                scores.append(m.group(0).strip())
        seen = []
        for score in scores:
            if score not in seen:
                seen.append(score)
        return seen

    def _extract_score_gemini(self) -> Optional[str]:
        """Regex skor bulamazsa ilgili cümleleri Gemini'ye sor."""
        evidence = self._collect_evidence_text(
            self._asr_last_segment,
            "score",
            max_chars=1400,
        )
        if not evidence:
            self._log("skor", "⚠ Gemini skor için kanıt parçası bulunamadı")
            return None

        try:
            from core.gemini_client import GeminiClient

            client = GeminiClient(model=self.gemini_model)
            prompt = f"""Bu bir spor maçı yayınının transcript'inden seçilmiş kısa parçalardır.
Sadece bu parçaya bak. Uydurma yapma. Parçada açıkça geçmeyen skoru boş bırak.

Sadece JSON formatında cevap ver:
{{
    "final_score": "0-0",
    "evidence": ["kanıt cümlesi 1", "kanıt cümlesi 2"]
}}

Parça:
{evidence}"""

            response = client.generate(prompt)
            parsed = self._parse_json_object(response)
            if not parsed:
                self._log_gemini_parse_failure("skor", "skor", response, expected="json object")
                return None

            final_score = str(parsed.get("final_score", "")).strip()
            if re.fullmatch(r"\d{1,3}\s*[-–—:]\s*\d{1,3}", final_score):
                left, right = re.split(r"\s*[-–—:]\s*", final_score)
                return f"{left}-{right}"
        except Exception as e:
            logger.warning(f"Gemini skor çıkarma hatası: {e}")

        return None

    # ----------------------------------------------------------
    #  5.4 — Gol ve kart çıkarma (sadece futbol)
    # ----------------------------------------------------------

    def _extract_goals(self) -> List[Dict]:
        """ASR son 15dk'dan gol bilgisi çıkarır."""
        if self._has_no_goal_summary(self._asr_last_segment):
            self._log("goller", "✓ Özet cümlesi 'gol yok' diyor — gol listesi boş bırakıldı")
            return []

        goals = []

        if self.gemini_enabled:
            goals = self._extract_goals_gemini()
        else:
            goals = self._extract_goals_regex()

        for g in goals:
            self._log("goller", f"✓ {g.get('dakika', '?')}' — {g.get('oyuncu', '?')} ({g.get('takim', '?')})")

        if not goals:
            self._log("goller", "⚠ Gol bilgisi bulunamadı")

        return goals

    def _extract_goals_gemini(self) -> List[Dict]:
        """Gemini ile gol bilgisi çıkar."""
        try:
            from core.gemini_client import GeminiClient
            client = GeminiClient(model=self.gemini_model)
            evidence = self._collect_evidence_text(
                self._asr_last_segment,
                "goals",
                max_chars=1600,
            )
            if not evidence:
                self._log("goller", "⚠ Gemini gol için kanıt parçası bulunamadı")
                return []

            prompt = f"""Bu bir futbol maçı yayınının transcript'inden seçilmiş kısa parçalardır.
Sadece bu parçaya bak. Gol olmayan pozisyonları gol sayma.
"gol yok" deniyorsa boş liste döndür.

Sadece JSON listesi döndür:
[
    {{"dakika": 34, "oyuncu": "İsim Soyisim", "takim": "Takım Adı"}},
    ...
]

Gol bulamadıysan boş liste döndür: []

Parça:
{evidence}"""

            response = client.generate(prompt)
            parsed = self._parse_json_list(response)
            if parsed is not None:
                return parsed
            self._log_gemini_parse_failure("goller", "gol", response, expected="json listesi")

        except Exception as e:
            logger.warning(f"Gemini gol çıkarma hatası: {e}")
            self._log("goller", f"⚠ Gemini hatası — regex'e düşülüyor")

        return self._extract_goals_regex()

    def _extract_goals_regex(self) -> List[Dict]:
        """Regex ile gol çıkarma (fallback)."""
        goals = []
        for pattern in GOL_PATTERNS:
            matches = re.finditer(pattern, self._asr_last_segment, re.IGNORECASE)
            for m in matches:
                groups = m.groups()
                if len(groups) >= 2:
                    # Hangi group dakika, hangi isim — pattern'a göre değişir
                    goal = {"dakika": "?", "oyuncu": "?", "takim": "?"}
                    for g in groups:
                        if g and g.strip().isdigit():
                            goal["dakika"] = int(g.strip())
                        elif g and not g.strip().isdigit():
                            goal["oyuncu"] = g.strip()
                    if not self._looks_like_false_goal_context(m.group(0)):
                        goals.append(goal)
        return self._dedupe_event_dicts(goals, ("dakika", "oyuncu", "takim"))

    def _extract_cards(self) -> List[Dict]:
        """ASR son 15dk'dan kart bilgisi çıkarır."""
        cards = []

        if self.gemini_enabled:
            cards = self._extract_cards_gemini()
        else:
            cards = self._extract_cards_regex()

        summary_cards = self._extract_cards_from_summary(self._asr_last_segment)
        if summary_cards:
            cards.extend(summary_cards)
            cards = self._dedupe_event_dicts(cards, ("tip", "dakika", "oyuncu", "takim"))

        for c in cards:
            tip = c.get("tip", "?")
            emoji = "🟨" if "sarı" in tip.lower() or "yellow" in tip.lower() else "🟥"
            self._log("kartlar", f"{emoji} {c.get('dakika', '?')}' — {c.get('oyuncu', '?')} ({c.get('takim', '?')})")

        if not cards:
            self._log("kartlar", "⚠ Kart bilgisi bulunamadı")

        return cards

    def _extract_cards_gemini(self) -> List[Dict]:
        """Gemini ile kart bilgisi çıkar."""
        try:
            from core.gemini_client import GeminiClient
            client = GeminiClient(model=self.gemini_model)
            evidence = self._collect_evidence_text(
                self._asr_last_segment,
                "cards",
                max_chars=1600,
            )
            if not evidence:
                self._log("kartlar", "⚠ Gemini kart için kanıt parçası bulunamadı")
                return []

            prompt = f"""Bu bir futbol maçı yayınının transcript'inden seçilmiş kısa parçalardır.
Sadece bu parçaya bak. Uydurma yapma.
"10 kişi" veya "eksik kaldı" ifadesini, parçada kart/ihraç bilgisi yoksa olay sayma.

Kart bilgilerini çıkar. Sadece JSON listesi döndür:
[
    {{"tip": "sarı kart", "dakika": 23, "oyuncu": "İsim", "takim": "Takım"}},
    ...
]

Kart bulamadıysan boş liste döndür: []

Parça:
{evidence}"""

            response = client.generate(prompt)
            parsed = self._parse_json_list(response)
            if parsed is not None:
                return parsed
            self._log_gemini_parse_failure("kartlar", "kart", response, expected="json listesi")

        except Exception as e:
            logger.warning(f"Gemini kart çıkarma hatası: {e}")

        return self._extract_cards_regex()

    def _extract_cards_regex(self) -> List[Dict]:
        """Regex ile kart çıkarma (fallback)."""
        cards = []
        for pattern in KART_PATTERNS:
            matches = re.finditer(pattern, self._asr_last_segment, re.IGNORECASE)
            for m in matches:
                groups = m.groups()
                card = {"tip": "?", "dakika": "?", "oyuncu": "?", "takim": "?"}
                for g in groups:
                    if not g:
                        continue
                    g = g.strip()
                    if g.isdigit():
                        card["dakika"] = int(g)
                    elif "sarı" in g.lower() or "yellow" in g.lower():
                        card["tip"] = "sarı kart"
                    elif "kırmızı" in g.lower() or "red" in g.lower():
                        card["tip"] = "kırmızı kart"
                    else:
                        card["oyuncu"] = g
                cards.append(card)
        return self._dedupe_event_dicts(cards, ("tip", "dakika", "oyuncu", "takim"))

    def _extract_cards_from_summary(self, text: str) -> List[Dict]:
        """Spiker özet cümlelerinden kart olaylarını çıkar."""
        text = self._normalise_text(text)
        cards = []
        for pattern in SUMMARY_CARD_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                dakika = int(match.group(1))
                oyuncu = match.group(2).strip()
                tip = match.group(3).lower()
                cards.append({
                    "tip": "kırmızı kart" if "kırmızı" in tip else "sarı kart",
                    "dakika": dakika,
                    "oyuncu": oyuncu,
                    "takim": "?",
                })
        return self._dedupe_event_dicts(cards, ("tip", "dakika", "oyuncu", "takim"))

    def _extract_teams_from_filename(self) -> Tuple[str, str] | None:
        """Dosya adından takım çiftini çıkarmayı dene."""
        if not self._video_filename:
            return None
        cleaned = self._video_filename.replace("_", " ").strip()
        match = re.search(r"(.+?)\s*[-–—]\s*(.+)", cleaned)
        if not match:
            return None
        left, right = match.group(1), match.group(2)
        left = re.sub(r"^(?:PTT|TFF|BANK ASYA)\s*1\.\s*L[İI]G\s*", "", left, flags=re.IGNORECASE).strip(" .-_")
        right = right.strip(" .-_")
        if left and right:
            return left, right
        return None

    def _extract_teams_from_summary(self, text: str) -> Tuple[str, str] | None:
        """Skor özetinden takım isimlerini çekmeye çalış."""
        match = re.search(
            r"([A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü\s]{3,40}?)\s+\d{1,3}[\.\-–—:]\s*"
            r"([A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü\s]{3,40}?)\s+\d{1,3}\s+ma[çc][ıi]n sonucu",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).strip(), match.group(2).strip()

    def _extract_referee_info(self, text: str) -> Dict[str, str]:
        """Baş hakem, yardımcılar ve dördüncü hakemi transcript'ten çıkar."""
        text = self._normalise_text(text)
        info = {
            "hakem": "",
            "yardimci_hakemler": "",
            "dorduncu_hakem": "",
        }

        head_match = re.search(
            r"ma[çc][ıi]n hakemi\s+([A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+)+)",
            text,
            re.IGNORECASE,
        )
        if head_match:
            info["hakem"] = head_match.group(1).strip()

        assistant_match = re.search(
            r"([A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+)*)\s+ve\s+"
            r"([A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+)*)\s+birinci yard[ıi]mc[ıi]lar[ıi]",
            text,
            re.IGNORECASE,
        )
        if assistant_match:
            info["yardimci_hakemler"] = (
                f"{assistant_match.group(1).strip()}, {assistant_match.group(2).strip()}"
            )

        fourth_match = re.search(
            r"([A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü]+)+)\s+da\s+d[öo]rd[üu]nc[üu]\s+hakem",
            text,
            re.IGNORECASE,
        )
        if fourth_match:
            info["dorduncu_hakem"] = fourth_match.group(1).strip()

        return info

    def _extract_city_from_text(self, text: str) -> str:
        """'Mersin'de' gibi şehir adaylarını yakala, lig adı gibi false-positive'leri ele."""
        false_positives = {"lig", "maç", "dakika", "skor", "gol", "set", "periyot"}
        for match in re.finditer(r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)'de\b", text):
            candidate = match.group(1).strip()
            if candidate.lower() not in false_positives:
                return candidate
        return ""

    def _has_no_goal_summary(self, text: str) -> bool:
        """Özet cümlesi açıkça gol olmadığını söylüyor mu?"""
        lowered = self._normalise_text(text).lower()
        return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in NO_GOAL_PATTERNS)

    def _looks_like_false_goal_context(self, text: str) -> bool:
        """'gole yaklaştı' gibi pozisyonları gol olarak sayma."""
        lowered = self._normalise_text(text).lower()
        false_fragments = (
            "gole çok yaklaşt",
            "gol pozisyonu",
            "gol sesi",
            "gol yok",
            "gol izleyemedik",
        )
        return any(fragment in lowered for fragment in false_fragments)

    def _dedupe_event_dicts(self, items: List[Dict], keys: Tuple[str, ...]) -> List[Dict]:
        """Aynı olayı birden fazla regex yakalarsa tekilleştir."""
        seen = set()
        unique = []
        for item in items:
            token = tuple(str(item.get(key, "")).strip().lower() for key in keys)
            if token in seen:
                continue
            seen.add(token)
            unique.append(item)
        return unique

    def _normalise_text(self, text: str) -> str:
        """Regex için whitespace'i sadeleştir."""
        return re.sub(r"\s+", " ", text or "").strip()

    # ----------------------------------------------------------
    #  5.5 — Spiker notları
    # ----------------------------------------------------------

    def _extract_speaker_notes(self) -> List[str]:
        """ASR transcript'ten anlamlı spiker yorumları çıkarır."""
        notes = []

        if self.gemini_enabled and (self._asr_first_segment or self._asr_last_segment):
            try:
                from core.gemini_client import GeminiClient
                client = GeminiClient(model=self.gemini_model)

                combined = self._asr_first_segment[-2000:] + "\n" + self._asr_last_segment[-2000:]
                prompt = f"""Bu bir spor maçı yayınının transcript'idir.
Spikerin en anlamlı 2-3 yorumunu kısa birer cümle olarak çıkar.
Sadece JSON listesi döndür: ["yorum1", "yorum2"]

Transcript:
{combined}"""

                response = client.generate(prompt)
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    notes = json.loads(json_match.group())

            except Exception as e:
                logger.warning(f"Spiker notları hatası: {e}")

        if notes:
            self._log("spiker_notlari", f"✓ {len(notes)} spiker notu çıkarıldı")
        else:
            self._log("spiker_notlari", "⚠ Spiker notu çıkarılamadı")

        return notes

    # ----------------------------------------------------------
    #  5.6 — Çapraz doğrulama
    # ----------------------------------------------------------

    def _cross_validate(self):
        """ASR ve OCR verilerini çapraz doğrula."""
        self._log("capraz_dogrulama", "────────────── ÇAPRAZ DOĞRULAMA ──────────────")

        # Skor doğrulama
        asr_s = self.score_info.get("asr_score")
        ocr_s = self.score_info.get("ocr_score")

        if asr_s and ocr_s:
            if asr_s == ocr_s:
                self._log("capraz_dogrulama", f"✓ Skor: ASR \"{asr_s}\" = OCR \"{ocr_s}\" → DOĞRULANDI")
            else:
                self._log("capraz_dogrulama", f"⚠ Skor: ASR \"{asr_s}\" ≠ OCR \"{ocr_s}\" → OCR tercih edildi")
        elif asr_s:
            self._log("capraz_dogrulama", f"? Skor: sadece ASR \"{asr_s}\" — OCR doğrulaması yok")
        elif ocr_s:
            self._log("capraz_dogrulama", f"? Skor: sadece OCR \"{ocr_s}\" — ASR doğrulaması yok")
        else:
            self._log("capraz_dogrulama", "✗ Skor doğrulanamadı — her iki kaynakta da bulunamadı")

        # Gol doğrulama (futbol)
        if self.sport_type == "futbol" and self.goals:
            self._log("capraz_dogrulama", f"✓ Goller: ASR'den {len(self.goals)} gol çıkarıldı")
            # OCR'den gol listesi varsa karşılaştır
            ocr_goal_count = self._count_goals_in_ocr()
            if ocr_goal_count > 0:
                if ocr_goal_count == len(self.goals):
                    self._log("capraz_dogrulama", f"✓ Gol sayısı: ASR {len(self.goals)} = OCR {ocr_goal_count} → DOĞRULANDI")
                else:
                    self._log("capraz_dogrulama", f"⚠ Gol sayısı: ASR {len(self.goals)} ≠ OCR {ocr_goal_count}")

        # Kart doğrulama
        if self.sport_type == "futbol" and self.cards:
            self._log("capraz_dogrulama", f"? Kartlar: ASR'den {len(self.cards)} kart — OCR'da kart bilgisi kontrol edilemedi")

    def _count_goals_in_ocr(self) -> int:
        """OCR frame'lerinde gol listesi var mı kontrol et."""
        for frame_no, text in self._ocr_frames:
            # Basit heuristik: rakam + isim deseni sayısı
            goal_like = re.findall(r"\d{1,3}['\.]\s*\w+", text)
            if len(goal_like) >= 1:
                return len(goal_like)
        return 0

    # ----------------------------------------------------------
    #  6. EXPORT — Rapor oluştur
    # ----------------------------------------------------------

    def _build_result(self) -> Dict[str, Any]:
        """Yapılandırılmış sonuç dict'i oluşturur."""
        return {
            "spor_dali": self.sport_type,
            "mac_bilgileri": self.match_info,
            "skor": self.score_info,
            "goller": self.goals if self.sport_type == "futbol" else [],
            "kartlar": self.cards if self.sport_type == "futbol" else [],
            "spiker_notlari": self.speaker_notes,
            "verification_log": self.verification_log,
        }

    def build_report_text(self, result: Optional[Dict] = None) -> str:
        """
        Okunabilir TXT rapor oluşturur.
        Tüm metin: Türkçe kelimeler Türkçe büyük harf kuralıyla,
        yabancı isimler/kelimeler ASCII büyük harfle yazılır.
        """
        from core.export_engine import _to_upper_tr
        U = _to_upper_tr  # kısayol

        if result is None:
            result = self._build_result()

        lines = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines.append("=" * 60)
        lines.append("SPOR MAÇI RAPORU")
        lines.append(f"PROFİL: SPOR | OLUŞTURULMA: {now}")
        lines.append("=" * 60)
        lines.append("")

        # MAÇ BİLGİLERİ
        lines.append("MAÇ BİLGİLERİ")
        lines.append("-" * 40)
        info = result.get("mac_bilgileri", {})
        lines.append(f"  SPOR DALI  : {U(info.get('spor_dali', 'BİLİNMİYOR'))}")
        if info.get("takimlar"):
            lines.append(f"  TAKIMLAR   : {U(info['takimlar'])}")
        if info.get("lig"):
            lines.append(f"  LİG        : {U(info['lig'])}")
        if info.get("hafta"):
            lines.append(f"  HAFTA      : {U(str(info['hafta']))}")
        if info.get("tarih"):
            lines.append(f"  TARİH      : {U(str(info['tarih']))}")
        if info.get("sehir"):
            sehir_str = info["sehir"]
            if info.get("stadyum"):
                sehir_str += f" — {info['stadyum']}"
            lines.append(f"  ŞEHİR      : {U(sehir_str)}")
        if info.get("hava"):
            lines.append(f"  HAVA       : {U(info['hava'])}")
        if info.get("hakem"):
            lines.append(f"  HAKEM      : {U(info['hakem'])}")
        if info.get("yardimci_hakemler"):
            lines.append(f"  YARD. HAK. : {U(info['yardimci_hakemler'])}")
        if info.get("dorduncu_hakem"):
            lines.append(f"  4. HAKEM   : {U(info['dorduncu_hakem'])}")
        lines.append("")

        # SKOR
        lines.append("SKOR")
        lines.append("-" * 40)
        score = result.get("skor", {})
        final = score.get("final_score", "BİLİNMİYOR")
        source = score.get("source", "")
        lines.append(f"  {U(str(final))}")
        if source:
            lines.append(f"  (KAYNAK: {U(source)})")
        lines.append("")

        # GOLLER (futbol)
        if result.get("goller"):
            lines.append("GOLLER")
            lines.append("-" * 40)
            for g in result["goller"]:
                dk = g.get("dakika", "?")
                oyuncu = g.get("oyuncu", "?")
                takim = g.get("takim", "")
                takim_str = f" ({U(takim)})" if takim and takim != "?" else ""
                lines.append(f"  {dk}' — {U(oyuncu)}{takim_str}")
            lines.append("")

        # KARTLAR (futbol)
        if result.get("kartlar"):
            lines.append("KARTLAR")
            lines.append("-" * 40)
            for c in result["kartlar"]:
                tip = c.get("tip", "?")
                dk = c.get("dakika", "?")
                oyuncu = c.get("oyuncu", "?")
                takim = c.get("takim", "")
                etiket = "[SARI]" if "sarı" in str(tip).lower() else "[KIRMIZI]"
                takim_str = f" ({U(takim)})" if takim and takim != "?" else ""
                lines.append(f"  {etiket} {dk}' — {U(oyuncu)}{takim_str}")
            lines.append("")

        # SPİKER NOTLARI
        if result.get("spiker_notlari"):
            lines.append("SPİKER NOTLARI")
            lines.append("-" * 40)
            for note in result["spiker_notlari"]:
                lines.append(f"  \"{U(note)}\"")
            lines.append("")

        # DOĞRULAMA LOGU
        lines.append("DOĞRULAMA LOGU")
        lines.append("=" * 60)
        for log_entry in result.get("verification_log", []):
            stage = U(log_entry.get("stage", ""))
            msg   = U(log_entry.get("message", ""))
            lines.append(f"  [{stage}] {msg}")
        lines.append("=" * 60)

        return "\n".join(lines)

    # ----------------------------------------------------------
    #  LOGGING
    # ----------------------------------------------------------

    def _log(self, stage: str, message: str):
        """Doğrulama loguna entry ekle."""
        self.verification_log.append({
            "stage": stage,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })
        logger.info(f"[{stage}] {message}")

    # ----------------------------------------------------------
    #  HUMAN-READABLE LOG
    # ----------------------------------------------------------

    def get_readable_log(self) -> str:
        """Okunabilir log metni döndürür."""
        lines = []
        current_stage = ""

        for entry in self.verification_log:
            stage = entry.get("stage", "")
            msg = entry.get("message", "")

            if stage != current_stage:
                current_stage = stage
                header = self._stage_header(stage)
                if header:
                    lines.append("")
                    lines.append(header)

            lines.append(f"  {msg}")

        return "\n".join(lines)

    def _stage_header(self, stage: str) -> str:
        """Log aşaması için başlık döndürür."""
        headers = {
            "segment_extract": "──────────── SEGMENT EXTRACT ────────────",
            "asr_transcribe": "──────────── ASR TRANSCRİBE ────────────",
            "ocr_read": "──────────── OCR READ ────────────",
            "frame_extract": "──────────── FRAME EXTRACT ────────────",
            "spor_tespit": "──────────── SPOR DALI TESPİT ────────────",
            "mac_bilgileri": "──────────── MAÇ BİLGİLERİ (Gemini) ────────────",
            "skor": "──────────── SKOR ────────────",
            "goller": "──────────── GOLLER ────────────",
            "kartlar": "──────────── KARTLAR ────────────",
            "spiker_notlari": "──────────── SPİKER NOTLARI ────────────",
            "capraz_dogrulama": "",  # Kendi header'ı var
            "analyze": "",  # Kendi header'ı var
        }
        return headers.get(stage, f"──────────── {stage.upper()} ────────────")
