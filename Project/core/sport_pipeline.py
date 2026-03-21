"""
PIPELINE RUNNER — Spor Maçı Akışı
====================================
Bu kodu mevcut pipeline_runner.py'ye ekle.

Mevcut run_pipeline() fonksiyonunda profil kontrolü yap:
  if profile == "SporMaci":
      return run_sport_pipeline(video_path, config, cdata)
  else:
      # mevcut FilmDizi akışı devam eder

Ya da ayrı bir fonksiyon olarak çağır.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("VITOS.pipeline_runner")


def run_sport_pipeline(video_path: str, config: Dict[str, Any], cdata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Spor maçı pipeline'ı — 6 adımlı akış.

    Args:
        video_path: Video dosyasının tam yolu
        config: SporMaci profil konfigürasyonu
        cdata: Pipeline boyunca taşınan ortak veri dict'i

    Returns:
        cdata dict'i (tüm sonuçlarla birlikte)
    """
    from core.sport_analyzer import SportAnalyzer

    # Config'den ayarları al
    segment_minutes = config.get("segment_minutes", 15)
    frame_interval_sec = config.get("frame_interval_sec", 10)
    gemini_enabled = config.get("gemini_enabled", True)
    gemini_model = config.get("gemini_model", "gemini-2.0-flash")
    asr_engine = config.get("asr_engine", "whisper")
    ocr_engine = config.get("ocr_engine", "paddleocr")
    output_root = config.get("output_root", r"F:\Sonuclar")
    database_root = config.get("database_root", r"D:\DATABASE")

    # Çalışma dizini oluştur
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    work_dir = os.path.join(database_root, "SporMaci", f"spor_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    os.makedirs(work_dir, exist_ok=True)

    # SportAnalyzer oluştur
    analyzer = SportAnalyzer({
        "segment_minutes": segment_minutes,
        "frame_interval_sec": frame_interval_sec,
        "gemini_enabled": gemini_enabled,
        "gemini_model": gemini_model,
        "asr_engine": asr_engine,
        "ocr_engine": ocr_engine,
    })

    # ─────────────────────────────────────────────
    # [1/6] SEGMENT EXTRACT
    # ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[1/6] SEGMENT EXTRACT — İlk ve son %d dk ayırılıyor", segment_minutes)
    logger.info("=" * 60)

    first_audio, last_audio = analyzer.extract_segments(video_path, work_dir)
    cdata["first_audio_path"] = first_audio
    cdata["last_audio_path"] = last_audio

    # ─────────────────────────────────────────────
    # [2/6] ASR TRANSCRIBE
    # ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[2/6] ASR TRANSCRIBE — Ses segmentleri yazıya çevriliyor")
    logger.info("=" * 60)

    first_transcript = _run_asr(first_audio, asr_engine, config)
    last_transcript = _run_asr(last_audio, asr_engine, config)

    # Transcript'leri kaydet
    _save_text(os.path.join(work_dir, "transcript_first.txt"), first_transcript)
    _save_text(os.path.join(work_dir, "transcript_last.txt"), last_transcript)

    analyzer.set_transcripts(first_transcript, last_transcript)
    cdata["first_transcript"] = first_transcript
    cdata["last_transcript"] = last_transcript

    # ─────────────────────────────────────────────
    # [3/6] FRAME EXTRACT
    # ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[3/6] FRAME EXTRACT — Son %d dk'dan frame çıkarılıyor (%d sn aralık)", segment_minutes, frame_interval_sec)
    logger.info("=" * 60)

    frame_paths = analyzer.extract_frames(video_path, work_dir)
    cdata["frame_paths"] = frame_paths

    # ─────────────────────────────────────────────
    # [4/6] OCR READ
    # ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[4/6] OCR READ — Frame'lerden metin okunuyor")
    logger.info("=" * 60)

    ocr_results = _run_ocr_on_frames(frame_paths, ocr_engine, config)
    analyzer.set_ocr_results(ocr_results)
    cdata["ocr_results"] = ocr_results

    # OCR sonuçlarını kaydet
    ocr_log_path = os.path.join(work_dir, "ocr_frames.txt")
    with open(ocr_log_path, "w", encoding="utf-8") as f:
        for frame_no, text in ocr_results:
            f.write(f"--- Frame {frame_no} ---\n{text}\n\n")

    # ─────────────────────────────────────────────
    # [5/6] SPORT ANALYZE
    # ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[5/6] SPORT ANALYZE — Verileri birleştir, analiz et")
    logger.info("=" * 60)

    result = analyzer.analyze()
    cdata["match_result"] = result

    # Sonuçları JSON olarak kaydet
    result_json_path = os.path.join(work_dir, "match_result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ─────────────────────────────────────────────
    # [6/6] EXPORT
    # ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[6/6] EXPORT — Rapor oluşturuluyor")
    logger.info("=" * 60)

    # TXT rapor oluştur
    report_text = analyzer.build_report_text(result)

    # DATABASE'e yaz
    report_db_path = os.path.join(work_dir, f"{base_name}_rapor.txt")
    _save_text(report_db_path, report_text)

    # SONUCLAR'a kopyala
    os.makedirs(output_root, exist_ok=True)
    report_output_path = os.path.join(output_root, f"{base_name}_rapor.txt")
    _save_text(report_output_path, report_text)

    # Verification log'u ayrıca kaydet
    vlog_path = os.path.join(work_dir, "verification_log.txt")
    _save_text(vlog_path, analyzer.get_readable_log())

    cdata["report_path"] = report_output_path
    cdata["database_path"] = work_dir

    logger.info("=" * 60)
    logger.info("SPOR PIPELINE TAMAMLANDI")
    logger.info("  Rapor: %s", report_output_path)
    logger.info("  Database: %s", work_dir)
    logger.info("=" * 60)

    return cdata


# ============================================================
#  YARDIMCI FONKSİYONLAR
# ============================================================

def _run_asr(audio_path: str, engine: str, config: Dict) -> str:
    """
    ASR motorunu çalıştırır.

    Desteklenen motorlar:
    - whisper: OpenAI Whisper (lokal)
    - gemini_audio: Gemini Audio API

    Returns:
        Transcript metni
    """
    if not os.path.exists(audio_path):
        logger.warning("ASR: Ses dosyası bulunamadı: %s", audio_path)
        return ""

    if engine == "whisper":
        return _run_whisper(audio_path, config)
    elif engine == "gemini_audio":
        return _run_gemini_audio(audio_path, config)
    else:
        logger.warning("ASR: Bilinmeyen motor: %s — whisper kullanılıyor", engine)
        return _run_whisper(audio_path, config)


def _run_whisper(audio_path: str, config: Dict) -> str:
    """Whisper ile transcript al."""
    try:
        import whisper

        model_name = config.get("asr_model", "large-v3")
        language = config.get("asr_language", "tr")

        model = whisper.load_model(model_name)
        result = model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
        )

        transcript = result.get("text", "")
        logger.info("Whisper transcript: %d karakter", len(transcript))
        return transcript

    except ImportError:
        logger.error("Whisper yüklü değil — pip install openai-whisper")
        return ""
    except Exception as e:
        logger.error("Whisper hatası: %s", e)
        return ""


def _run_gemini_audio(audio_path: str, config: Dict) -> str:
    """Gemini Audio API ile transcript al."""
    try:
        from core.gemini_client import GeminiClient
        import base64

        client = GeminiClient(model=config.get("gemini_model", "gemini-2.0-flash"))

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        prompt = "Bu ses kaydını Türkçe olarak yazıya çevir. Sadece transcript'i ver, başka bir şey yazma."

        # Gemini audio API çağrısı
        # Not: Gemini client'ın audio desteği varsa kullan
        # Yoksa whisper'a fallback yap
        transcript = client.generate_with_audio(prompt, audio_b64, "audio/wav")

        logger.info("Gemini Audio transcript: %d karakter", len(transcript))
        return transcript

    except Exception as e:
        logger.warning("Gemini Audio hatası: %s — whisper'a düşülüyor", e)
        return _run_whisper(audio_path, config)


def _run_ocr_on_frames(frame_paths: list, engine: str, config: Dict) -> list:
    """
    Frame listesi üzerinde OCR çalıştırır.

    Returns:
        [(frame_index, ocr_text), ...]
    """
    results = []

    if engine == "paddleocr":
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

            for i, fpath in enumerate(frame_paths):
                try:
                    ocr_result = ocr.ocr(fpath, cls=True)
                    texts = []
                    if ocr_result and ocr_result[0]:
                        for line in ocr_result[0]:
                            if line[1] and line[1][0]:
                                texts.append(line[1][0])
                    full_text = "\n".join(texts)
                    results.append((i + 1, full_text))
                except Exception as e:
                    logger.warning("OCR frame %d hatası: %s", i + 1, e)
                    results.append((i + 1, ""))

        except ImportError:
            logger.error("PaddleOCR yüklü değil")

    elif engine == "oneocr":
        try:
            from core.oneocr_engine import OneOCREngine
            ocr = OneOCREngine()

            for i, fpath in enumerate(frame_paths):
                try:
                    text = ocr.read_image(fpath)
                    results.append((i + 1, text))
                except Exception as e:
                    logger.warning("OneOCR frame %d hatası: %s", i + 1, e)
                    results.append((i + 1, ""))

        except ImportError:
            logger.error("OneOCR engine bulunamadı")

    else:
        logger.warning("Bilinmeyen OCR engine: %s", engine)

    non_empty = sum(1 for _, t in results if t.strip())
    logger.info("OCR: %d frame okundu, %d frame'de metin var", len(results), non_empty)

    return results


def _save_text(path: str, text: str):
    """Metin dosyası kaydet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
