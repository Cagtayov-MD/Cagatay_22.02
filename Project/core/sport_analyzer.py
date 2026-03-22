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

# OCR skor tabelası desenleri
SKOR_PATTERNS = [
    # "GS 2 - FB 1" / "Galatasaray 2 - 1 Fenerbahçe"
    r"([A-ZÇĞİÖŞÜa-zçğıöşü\s\.]{2,25})\s*(\d{1,3})\s*[-–—:]\s*(\d{1,3})\s*([A-ZÇĞİÖŞÜa-zçğıöşü\s\.]{2,25})",
    # "2:1" / "87-74" (sadece sayılar)
    r"(\d{1,3})\s*[-–—:]\s*(\d{1,3})",
]


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

        seg_seconds = self.segment_minutes * 60
        last_start = max(0, total_seconds - seg_seconds)

        first_path = os.path.join(output_dir, "segment_first.wav")
        last_path = os.path.join(output_dir, "segment_last.wav")

        # İlk segment
        subprocess.run([
            self.ffmpeg, "-y", "-i", video_path,
            "-ss", "0", "-t", str(seg_seconds),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            first_path
        ], capture_output=True, timeout=120)

        # Son segment
        subprocess.run([
            self.ffmpeg, "-y", "-i", video_path,
            "-ss", str(last_start), "-t", str(seg_seconds),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            last_path
        ], capture_output=True, timeout=120)

        self._log("segment_extract", f"Video: {total_seconds:.0f}sn toplam")
        self._log("segment_extract", f"İlk segment: 0 — {seg_seconds}sn")
        self._log("segment_extract", f"Son segment: {last_start:.0f} — {total_seconds:.0f}sn")

        return first_path, last_path

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
        }

        if not self._asr_first_segment.strip():
            self._log("mac_bilgileri", "⚠ İlk 15dk transcript boş — maç bilgileri çıkarılamadı")
            return info

        if self.gemini_enabled:
            info = self._extract_match_info_gemini(info)
        else:
            info = self._extract_match_info_regex(info)

        # Log
        for key, val in info.items():
            if val and key != "spor_dali":
                self._log("mac_bilgileri", f"✓ {key}: {val}")

        empty_fields = [k for k, v in info.items() if not v and k != "spor_dali"]
        if empty_fields:
            self._log("mac_bilgileri", f"⚠ Bulunamayan: {', '.join(empty_fields)}")

        return info

    def _extract_match_info_gemini(self, info: Dict) -> Dict:
        """Gemini API ile maç bilgilerini çıkar."""
        try:
            from core.gemini_client import GeminiClient
            client = GeminiClient(model=self.gemini_model)

            prompt = f"""Bu bir spor maçı yayınının ilk 15 dakikasının transcript'idir.
Bu metinden aşağıdaki bilgileri çıkar. Bulamadığın alanları boş bırak.

Sadece JSON formatında cevap ver, başka hiçbir şey yazma:
{{
    "takimlar": "Takım1 — Takım2",
    "lig": "Lig adı",
    "hafta": "Kaçıncı hafta",
    "tarih": "Tarih bilgisi",
    "sehir": "Şehir",
    "stadyum": "Stadyum adı",
    "hava": "Hava durumu",
    "hakem": "Hakem adı"
}}

Transcript:
{self._asr_first_segment[:8000]}"""

            response = client.generate(prompt)
            # JSON parse
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                for key in info:
                    if key in parsed and parsed[key]:
                        info[key] = str(parsed[key]).strip()

            self._log("mac_bilgileri", "✓ Gemini API ile bilgiler çıkarıldı")

        except Exception as e:
            logger.warning(f"Gemini maç bilgisi çıkarma hatası: {e}")
            self._log("mac_bilgileri", f"⚠ Gemini hatası: {e} — regex'e düşülüyor")
            info = self._extract_match_info_regex(info)

        return info

    def _extract_match_info_regex(self, info: Dict) -> Dict:
        """Regex ile basit maç bilgisi çıkarma (Gemini kapalıysa fallback)."""
        text = self._asr_first_segment

        # Takım tespiti — en sık geçen büyük harfli kelime çiftleri
        # Bu basit bir heuristik, Gemini çok daha iyi
        self._log("mac_bilgileri", "⚠ Gemini kapalı — sadece regex ile çıkarılıyor (sınırlı)")

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
        for pattern in SKOR_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for m in matches:
                scores.append(m.group(0).strip())
        return scores

    # ----------------------------------------------------------
    #  5.4 — Gol ve kart çıkarma (sadece futbol)
    # ----------------------------------------------------------

    def _extract_goals(self) -> List[Dict]:
        """ASR son 15dk'dan gol bilgisi çıkarır."""
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

            prompt = f"""Bu bir futbol maçı yayınının son 15 dakikasının transcript'idir.
Spiker maçın gollerini özetliyor olabilir.

Bu metinden gol bilgilerini çıkar. Sadece JSON listesi döndür:
[
    {{"dakika": 34, "oyuncu": "İsim Soyisim", "takim": "Takım Adı"}},
    ...
]

Gol bulamadıysan boş liste döndür: []

Transcript:
{self._asr_last_segment[:8000]}"""

            response = client.generate(prompt)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

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
                    goals.append(goal)
        return goals

    def _extract_cards(self) -> List[Dict]:
        """ASR son 15dk'dan kart bilgisi çıkarır."""
        cards = []

        if self.gemini_enabled:
            cards = self._extract_cards_gemini()
        else:
            cards = self._extract_cards_regex()

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

            prompt = f"""Bu bir futbol maçı yayınının son 15 dakikasının transcript'idir.

Kart bilgilerini çıkar. Sadece JSON listesi döndür:
[
    {{"tip": "sarı kart", "dakika": 23, "oyuncu": "İsim", "takim": "Takım"}},
    ...
]

Kart bulamadıysan boş liste döndür: []

Transcript:
{self._asr_last_segment[:8000]}"""

            response = client.generate(prompt)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

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
        return cards

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
        Örnek çıktıdaki formata uygun.
        """
        if result is None:
            result = self._build_result()

        lines = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines.append("=" * 60)
        lines.append("SPOR MAÇI RAPORU")
        lines.append(f"Profil: Spor | Oluşturulma: {now}")
        lines.append("=" * 60)
        lines.append("")

        # MAÇ BİLGİLERİ
        lines.append("MAÇ BİLGİLERİ")
        lines.append("-" * 40)
        info = result.get("mac_bilgileri", {})
        lines.append(f"  Spor dalı  : {info.get('spor_dali', 'bilinmiyor').upper()}")
        if info.get("takimlar"):
            lines.append(f"  Takımlar   : {info['takimlar']}")
        if info.get("lig"):
            lines.append(f"  Lig        : {info['lig']}")
        if info.get("hafta"):
            lines.append(f"  Hafta      : {info['hafta']}")
        if info.get("tarih"):
            lines.append(f"  Tarih      : {info['tarih']}")
        if info.get("sehir"):
            sehir_str = info["sehir"]
            if info.get("stadyum"):
                sehir_str += f" — {info['stadyum']}"
            lines.append(f"  Şehir      : {sehir_str}")
        if info.get("hava"):
            lines.append(f"  Hava       : {info['hava']}")
        if info.get("hakem"):
            lines.append(f"  Hakem      : {info['hakem']}")
        lines.append("")

        # SKOR
        lines.append("SKOR")
        lines.append("-" * 40)
        score = result.get("skor", {})
        final = score.get("final_score", "BİLİNMİYOR")
        source = score.get("source", "")
        lines.append(f"  {final}")
        if source:
            lines.append(f"  (Kaynak: {source})")
        lines.append("")

        # GOLLER (futbol)
        if result.get("goller"):
            lines.append("GOLLER")
            lines.append("-" * 40)
            for g in result["goller"]:
                dk = g.get("dakika", "?")
                oyuncu = g.get("oyuncu", "?")
                takim = g.get("takim", "")
                takim_str = f" ({takim})" if takim and takim != "?" else ""
                lines.append(f"  {dk}' — {oyuncu}{takim_str}")
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
                emoji = "[SARI]" if "sarı" in str(tip).lower() else "[KIRMIZI]"
                takim_str = f" ({takim})" if takim and takim != "?" else ""
                lines.append(f"  {emoji} {dk}' — {oyuncu}{takim_str}")
            lines.append("")

        # SPİKER NOTLARI
        if result.get("spiker_notlari"):
            lines.append("SPİKER NOTLARI")
            lines.append("-" * 40)
            for note in result["spiker_notlari"]:
                lines.append(f"  \"{note}\"")
            lines.append("")

        # DOĞRULAMA LOGU
        lines.append("DOĞRULAMA LOGU")
        lines.append("=" * 60)
        for log_entry in result.get("verification_log", []):
            stage = log_entry.get("stage", "")
            msg = log_entry.get("message", "")
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
