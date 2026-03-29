"""
PIPELINE RUNNER — Spor Maçı Akışı
====================================
Bu kodu mevcut pipeline_runner.py'ye ekle.

Mevcut run_pipeline() fonksiyonunda profil kontrolü yap:
  if profile == "Spor":
      return run_sport_pipeline(video_path, config, cdata)
  else:
      # mevcut FilmDizi akışı devam eder

Ya da ayrı bir fonksiyon olarak çağır.
"""

import os
import json
import logging
import locale
import subprocess
from datetime import datetime
from typing import Dict, Any

from core.whisper_model import normalize_whisper_model_name

logger = logging.getLogger("VITOS.pipeline_runner")


_SUBPROCESS_ENCODINGS = (
    "utf-8",
    "utf-8-sig",
    locale.getpreferredencoding(False) or "utf-8",
    "cp1254",
    "cp1252",
    "latin-1",
)


def run_sport_pipeline(
    video_path: str,
    config: Dict[str, Any],
    cdata: Dict[str, Any],
    log_cb=None,
) -> Dict[str, Any]:
    """
    Spor maçı pipeline'ı — 6 adımlı akış.

    Args:
        video_path: Video dosyasının tam yolu
        config: Spor profil konfigürasyonu
        cdata: Pipeline boyunca taşınan ortak veri dict'i
        log_cb: UI log callback'i — None ise sadece standart logging kullanılır

    Returns:
        cdata dict'i (tüm sonuçlarla birlikte)
    """
    from core.sport_analyzer import SportAnalyzer

    def _log(msg: str):
        logger.info(msg)
        if log_cb:
            log_cb(msg)

    # Config'den ayarları al
    segment_minutes = config.get("segment_minutes", 15)
    frame_interval_sec = config.get("frame_interval_sec", 10)
    gemini_enabled = config.get("gemini_enabled", True)
    gemini_model = config.get("gemini_model", "gemini-2.0-flash")
    asr_engine = config.get("asr_engine", "whisper")
    ocr_engine = config.get("ocr_engine", "paddleocr")
    output_root = config.get("output_root", r"F:\Sonuclar")
    database_root = config.get("database_root", r"D:\DATABASE")
    ffmpeg = config.get("ffmpeg", "ffmpeg")
    ffprobe = config.get("ffprobe", "ffprobe")

    # Çalışma dizini oluştur
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    work_dir = os.path.join(database_root, "Spor", f"spor_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    os.makedirs(work_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # [1/7] DİL TESPİTİ
    # ─────────────────────────────────────────────
    _log("=" * 60)
    _log("[1/7] DİL TESPİTİ — İlk 30sn + 4. dk 30sn örnekleniyor")
    _log("=" * 60)

    detected_language = config.get("asr_language", "tr")
    selected_channel = None
    selected_channel_confidence = 0.0
    try:
        import sys as _sys
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _venv_audio_python = os.environ.get(
            "VENV_AUDIO_PYTHON", r"F:\Root\venv_audio\Scripts\python.exe"
        )
        # LanguageDetectionStage'i venv_audio subprocess'te çalıştır
        # (faster_whisper ana venv'de yok, venv_audio'da var)
        _detect_script = f"""
import sys, json
sys.path.insert(0, {_project_root!r})
from audio.stages.detect_language import LanguageDetectionStage
stage = LanguageDetectionStage(ffmpeg_path={ffmpeg!r})
result = stage.run({video_path!r}, {work_dir!r})
# Logları stderr'a yaz
for trial in result.get("channel_trials", []):
    pass  # zaten stage._log ile loglandı
safe = {{k: v for k, v in result.items() if k not in ("channel_trials", "samples")}}
print(json.dumps(safe))
"""
        _proc = subprocess.run(
            [_venv_audio_python, "-c", _detect_script],
            capture_output=True, text=False,
            env=_build_subprocess_env(), timeout=120,
        )
        _stdout = _normalize_subprocess_text(_proc.stdout).strip()
        _stderr = _normalize_subprocess_text(_proc.stderr).strip()

        # stderr'daki DİL TESPİTİ satırlarını UI'ye ilet
        for _line in _stderr.splitlines():
            if "[DİL" in _line or "[DIL" in _line or "DİL TESPİTİ" in _line.upper():
                _log(f"  {_line.strip()}")

        if _proc.returncode == 0 and _stdout:
            import json as _json
            _lang_result = _json.loads(_stdout)
            detected = _lang_result.get("detected_language", "unknown")
            conf_pct  = _lang_result.get("confidence", 0.0) * 100
            if detected and detected != "unknown":
                detected_language = detected
                _log(f"  [DİL TESPİTİ] ✓ {detected_language.upper()} ({conf_pct:.1f}%) — ASR bu dille başlayacak")
            else:
                _log(f"  [DİL TESPİTİ] ⚠ Tespit edilemedi — varsayılan: {detected_language.upper()}")
            selected_channel = _lang_result.get("selected_channel")
            selected_channel_confidence = float(_lang_result.get("confidence", 0.0) or 0.0)
            cdata["detected_language"] = detected_language
            cdata["language_confidence"] = round(_lang_result.get("confidence", 0.0), 4)
            cdata["selected_channel"] = selected_channel
        else:
            _log(f"  [DİL TESPİTİ] ⚠ Subprocess hatası (rc={_proc.returncode}) — varsayılan: {detected_language.upper()}")
    except Exception as e:
        _log(f"  [DİL TESPİTİ] ⚠ Hata: {e} — varsayılan: {detected_language.upper()}")

    # Tespit edilen dili config'e yaz (tüm alt fonksiyonlar bu config'i okur)
    config = {
        **config,
        "asr_language": detected_language,
        "selected_channel": selected_channel,
        "selected_channel_confidence": selected_channel_confidence,
    }

    # SportAnalyzer oluştur
    _venv_python = os.environ.get(
        "VENV_AUDIO_PYTHON", r"F:\Root\venv_audio\Scripts\python.exe"
    )
    analyzer = SportAnalyzer({
        "segment_minutes": segment_minutes,
        "frame_interval_sec": frame_interval_sec,
        "gemini_enabled": gemini_enabled,
        "gemini_model": gemini_model,
        "asr_engine": asr_engine,
        "ocr_engine": ocr_engine,
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "speech_separation": config.get("speech_separation", True),
        "selected_channel": selected_channel,
        "selected_channel_confidence": selected_channel_confidence,
        "include_mix_fallback": config.get("include_mix_fallback", True),
        "venv_audio_python": _venv_python,
    })

    # ─────────────────────────────────────────────
    # [2/7] SEGMENT EXTRACT
    # ─────────────────────────────────────────────
    _log("=" * 60)
    _log(f"[2/7] SEGMENT EXTRACT — İlk ve son {segment_minutes} dk ayırılıyor")
    _log("=" * 60)

    first_audio, last_audio = analyzer.extract_segments(video_path, work_dir)
    first_candidates = analyzer.get_segment_audio_candidates("first")
    last_candidates = analyzer.get_segment_audio_candidates("last")
    cdata["first_audio_path"] = first_audio
    cdata["last_audio_path"] = last_audio
    cdata["first_audio_candidates"] = first_candidates
    cdata["last_audio_candidates"] = last_candidates

    # ─────────────────────────────────────────────
    # [3/7] ASR TRANSCRIBE
    # ─────────────────────────────────────────────
    _log("=" * 60)
    _log(f"[3/7] ASR TRANSCRIBE — Ses segmentleri yazıya çevriliyor ({detected_language.upper()})")
    _log("=" * 60)

    first_asr = _run_asr_bundle(first_candidates, asr_engine, config, "ilk", _log)
    last_asr = _run_asr_bundle(last_candidates, asr_engine, config, "son", _log)
    first_transcript = first_asr.get("transcript", "")
    last_transcript = last_asr.get("transcript", "")
    cdata["first_asr_debug"] = first_asr
    cdata["last_asr_debug"] = last_asr

    # Transcript'leri kaydet
    _save_text(os.path.join(work_dir, "transcript_first.txt"), first_transcript)
    _save_text(os.path.join(work_dir, "transcript_last.txt"), last_transcript)

    analyzer.set_transcripts(first_transcript, last_transcript)
    cdata["first_transcript"] = first_transcript
    cdata["last_transcript"] = last_transcript

    # ─────────────────────────────────────────────
    # [4/7] FRAME EXTRACT
    # ─────────────────────────────────────────────
    _log("=" * 60)
    _log(f"[4/7] FRAME EXTRACT — Son {segment_minutes} dk'dan frame çıkarılıyor ({frame_interval_sec} sn aralık)")
    _log("=" * 60)

    frame_paths = analyzer.extract_frames(video_path, work_dir)
    cdata["frame_paths"] = frame_paths

    # ─────────────────────────────────────────────
    # [5/7] OCR READ
    # ─────────────────────────────────────────────
    _log("=" * 60)
    _log("[5/7] OCR READ — Frame'lerden metin okunuyor")
    _log("=" * 60)

    ocr_results = _run_ocr_on_frames(frame_paths, ocr_engine, config, _log)
    analyzer.set_ocr_results(ocr_results)
    cdata["ocr_results"] = ocr_results

    # OCR sonuçlarını kaydet
    ocr_log_path = os.path.join(work_dir, "ocr_frames.txt")
    with open(ocr_log_path, "w", encoding="utf-8") as f:
        for frame_no, text in ocr_results:
            f.write(f"--- Frame {frame_no} ---\n{text}\n\n")

    # ─────────────────────────────────────────────
    # [6/7] SPORT ANALYZE
    # ─────────────────────────────────────────────
    _log("=" * 60)
    _log("[6/7] SPORT ANALYZE — Verileri birleştir, analiz et")
    _log("=" * 60)

    analyzer.set_video_name(base_name)
    result = analyzer.analyze()
    cdata["match_result"] = result

    # Sonuçları JSON olarak kaydet
    result_json_path = os.path.join(work_dir, "match_result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ─────────────────────────────────────────────
    # [7/7] EXPORT
    # ─────────────────────────────────────────────
    _log("=" * 60)
    _log("[7/7] EXPORT — Rapor oluşturuluyor")
    _log("=" * 60)

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

    _log("=" * 60)
    _log("SPOR PIPELINE TAMAMLANDI")
    _log(f"  Rapor: {report_output_path}")
    _log(f"  Database: {work_dir}")
    _log("=" * 60)

    return cdata


# ============================================================
#  YARDIMCI FONKSİYONLAR
# ============================================================

def _run_asr_bundle(
    audio_candidates: list,
    engine: str,
    config: Dict,
    segment_name: str,
    _log=None,
) -> dict:
    """Bir segment için ham/vocals/mix adaylarını çalıştır, en iyi transcript'i seç."""
    def log(msg):
        logger.info(msg)
        if _log:
            _log(msg)

    usable = [c for c in (audio_candidates or []) if c.get("path")]
    if not usable:
        log(f"  [ASR:{segment_name}] Kullanılabilir ses adayı yok")
        return {"transcript": "", "selected_candidate": None, "candidates": []}

    scored = []
    for candidate in usable:
        label = candidate.get("label") or os.path.basename(candidate.get("path", ""))
        log(f"  [ASR:{segment_name}] Aday deneniyor: {label}")
        detail = _run_asr_detailed(candidate["path"], engine, config, _log)
        detail["candidate_label"] = label
        detail["candidate_kind"] = candidate.get("candidate_kind", "")
        detail["source_mode"] = candidate.get("source_mode", "")
        detail["quality_score"] = _score_asr_detail(detail)
        scored.append(detail)
        log(
            f"  [ASR:{segment_name}] {label}: skor={detail['quality_score']:.2f} "
            f"| transcript={len(detail.get('transcript', ''))} karakter"
        )

    best = _select_best_asr_result(scored)
    if best:
        log(
            f"  [ASR:{segment_name}] Seçilen aday: {best.get('candidate_label')} "
            f"(skor={best.get('quality_score', 0.0):.2f})"
        )
    else:
        log(f"  [ASR:{segment_name}] Uygun aday bulunamadı")

    return {
        "transcript": best.get("transcript", "") if best else "",
        "selected_candidate": best.get("candidate_label") if best else None,
        "selected_source_mode": best.get("source_mode") if best else None,
        "candidates": scored,
    }


def _run_asr(audio_path: str, engine: str, config: Dict, _log=None) -> str:
    """
    ASR motorunu çalıştırır.

    Desteklenen motorlar:
    - whisper: OpenAI Whisper (lokal)
    - gemini_audio: Gemini Audio API

    Returns:
        Transcript metni
    """
    detail = _run_asr_detailed(audio_path, engine, config, _log)
    return detail.get("transcript", "")


def _run_asr_detailed(audio_path: str, engine: str, config: Dict, _log=None) -> dict:
    """ASR sonucunu metin + kalite metrikleriyle döndür."""
    def log(msg):
        logger.warning(msg) if "bulunamadı" in msg or "Bilinmeyen" in msg else logger.info(msg)
        if _log:
            _log(msg)

    if not os.path.exists(audio_path):
        log(f"  [ASR] Ses dosyası bulunamadı: {audio_path}")
        return {
            "transcript": "",
            "segments": [],
            "audio_duration": 0.0,
            "speech_duration": 0.0,
            "avg_logprob": -5.0,
            "avg_no_speech_prob": 1.0,
            "avg_compression_ratio": 99.0,
        }

    log(f"  [ASR] Motor: {engine} | Dosya: {os.path.basename(audio_path)}")

    if engine == "whisper":
        return _run_whisper_detailed(audio_path, config, _log)
    if engine == "gemini_audio":
        return _run_gemini_audio_detailed(audio_path, config, _log)

    log(f"  [ASR] Bilinmeyen motor: {engine} — whisper kullanılıyor")
    return _run_whisper_detailed(audio_path, config, _log)


def _normalize_subprocess_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        for enc in _SUBPROCESS_ENCODINGS:
            try:
                return value.decode(enc)
            except (LookupError, UnicodeDecodeError):
                continue
        return value.decode("utf-8", errors="replace")
    return str(value)


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _get_audio_duration(audio_path: str, config: Dict) -> float:
    """ffprobe ile ses süresini al."""
    try:
        ffprobe = config.get("ffprobe", "ffprobe")
        dur = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return float((dur.stdout or "0").strip() or 0.0)
    except Exception:
        return 0.0


def _resolve_faster_whisper_compute_type(config: Dict, device: str) -> str:
    raw = (
        config.get("asr_compute_type")
        or config.get("compute_type")
        or ("float16" if device == "cuda" else "int8")
    )
    normalized = str(raw).strip().lower()
    if device != "cuda" and normalized == "float16":
        return "int8"
    if not normalized:
        return "float16" if device == "cuda" else "int8"
    return normalized


def _run_whisper_detailed(audio_path: str, config: Dict, _log=None) -> dict:
    """faster-whisper + VAD ile transcript ve kalite metrikleri üret."""

    def log(msg):
        logger.info(msg)
        if _log:
            _log(msg)

    def log_err(msg):
        logger.error(msg)
        if _log:
            _log(msg)

    venv_audio_python = os.environ.get("VENV_AUDIO_PYTHON", r"F:\Root\venv_audio\Scripts\python.exe")
    raw_model_name = config.get("asr_model", "large-v3")
    model_name = normalize_whisper_model_name(raw_model_name)
    language = config.get("asr_language", "tr")
    asr_device = config.get("asr_device", "cuda")
    device = asr_device if (asr_device == "cuda") else "cpu"
    compute_type = _resolve_faster_whisper_compute_type(config, device)
    no_speech_threshold = float(config.get("asr_no_speech_threshold", 0.75))
    min_silence_ms = int(config.get("sport_vad_min_silence_ms", 450))
    speech_pad_ms = int(config.get("sport_vad_speech_pad_ms", 200))
    beam_size = int(config.get("beam_size", config.get("asr_beam_size", 1)) or 1)
    initial_prompt = config.get("asr_initial_prompt", "Türkçe futbol maçı spiker yorumu.")
    audio_duration = _get_audio_duration(audio_path, config)

    if raw_model_name not in (None, "") and raw_model_name != model_name:
        log(f"  [Whisper] Model normalize edildi: {raw_model_name} -> {model_name}")

    log(
        f"  [Whisper] faster-whisper hazırlanıyor: {model_name} "
        f"| dil={language} | device={device} | compute_type={compute_type}"
    )

    script = f"""
import json
import logging
import torch
from faster_whisper import WhisperModel

logging.getLogger("faster_whisper").setLevel(logging.ERROR)
device = {asr_device!r} if ({asr_device!r} == "cuda" and torch.cuda.is_available()) else "cpu"
compute_type = {compute_type!r}
if device != "cuda" and compute_type == "float16":
    compute_type = "int8"
model = WhisperModel({model_name!r}, device=device, compute_type=compute_type)
segments, info = model.transcribe(
    {audio_path!r},
    language={language!r},
    task="transcribe",
    beam_size={beam_size},
    word_timestamps=False,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms={min_silence_ms},
        speech_pad_ms={speech_pad_ms},
    ),
    condition_on_previous_text=False,
    no_speech_threshold={no_speech_threshold},
    initial_prompt={initial_prompt!r} or None,
)
rows = []
texts = []
speech_duration = 0.0
logprob_sum = 0.0
no_speech_sum = 0.0
compression_sum = 0.0
for seg in segments:
    text = (seg.text or "").strip()
    if not text:
        continue
    start = float(seg.start)
    end = float(seg.end)
    speech_duration += max(0.0, end - start)
    texts.append(text)
    rows.append({{
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text,
        "avg_logprob": float(getattr(seg, "avg_logprob", -5.0) or -5.0),
        "no_speech_prob": float(getattr(seg, "no_speech_prob", 1.0) or 1.0),
        "compression_ratio": float(getattr(seg, "compression_ratio", 99.0) or 99.0),
    }})
    logprob_sum += rows[-1]["avg_logprob"]
    no_speech_sum += rows[-1]["no_speech_prob"]
    compression_sum += rows[-1]["compression_ratio"]

count = len(rows)
payload = {{
    "transcript": " ".join(texts).strip(),
    "segments": rows,
    "speech_duration": round(speech_duration, 3),
    "avg_logprob": round(logprob_sum / count, 4) if count else -5.0,
    "avg_no_speech_prob": round(no_speech_sum / count, 4) if count else 1.0,
    "avg_compression_ratio": round(compression_sum / count, 4) if count else 99.0,
    "detected_language": getattr(info, "language", {language!r}) or {language!r},
    "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
}}
print(json.dumps(payload, ensure_ascii=False))
"""

    try:
        proc = subprocess.run(
            [venv_audio_python, "-c", script],
            capture_output=True,
            text=False,
            env=_build_subprocess_env(),
            timeout=1200,
        )
        stdout_text = _normalize_subprocess_text(proc.stdout).strip()
        stderr_text = _normalize_subprocess_text(proc.stderr).strip()
        if proc.returncode != 0:
            log_err(f"  [Whisper] Subprocess hatası: {stderr_text or proc.returncode}")
            return {
                "transcript": "",
                "segments": [],
                "audio_duration": audio_duration,
                "speech_duration": 0.0,
                "avg_logprob": -5.0,
                "avg_no_speech_prob": 1.0,
                "avg_compression_ratio": 99.0,
            }
        payload = json.loads(stdout_text) if stdout_text else {}
        payload["audio_duration"] = audio_duration
        log(
            f"  [Whisper] Tamamlandı — {len(payload.get('transcript', ''))} karakter, "
            f"{len(payload.get('segments', []))} VAD segmenti"
        )
        return payload

    except Exception as e:
        log_err(f"  [Whisper] Hata: {e}")
        return {
            "transcript": "",
            "segments": [],
            "audio_duration": audio_duration,
            "speech_duration": 0.0,
            "avg_logprob": -5.0,
            "avg_no_speech_prob": 1.0,
            "avg_compression_ratio": 99.0,
        }


def _run_whisper(audio_path: str, config: Dict, _log=None) -> str:
    """Uyumluluk için sadece transcript döndüren sarmalayıcı."""
    return _run_whisper_detailed(audio_path, config, _log).get("transcript", "")


def _run_gemini_audio_detailed(audio_path: str, config: Dict, _log=None) -> dict:
    """Gemini Audio API ile transcript al."""
    def log(msg):
        logger.info(msg)
        if _log:
            _log(msg)

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
        transcript = (transcript or "").strip()

        log(f"  [Gemini Audio] Transcript: {len(transcript)} karakter")
        return {
            "transcript": transcript,
            "segments": [],
            "audio_duration": _get_audio_duration(audio_path, config),
            "speech_duration": 0.0,
            "avg_logprob": -1.0 if transcript else -5.0,
            "avg_no_speech_prob": 0.5 if transcript else 1.0,
            "avg_compression_ratio": 1.0,
        }

    except Exception as e:
        log(f"  [Gemini Audio] Hata: {e} — whisper'a düşülüyor")
        return _run_whisper_detailed(audio_path, config, _log)


def _run_gemini_audio(audio_path: str, config: Dict, _log=None) -> str:
    """Uyumluluk için sadece transcript döndüren sarmalayıcı."""
    return _run_gemini_audio_detailed(audio_path, config, _log).get("transcript", "")


def _score_asr_detail(detail: dict) -> float:
    """Transcript kalitesi için kaba ama açıklanabilir bir skor üret."""
    transcript = (detail.get("transcript") or "").strip()
    if not transcript:
        return -1_000.0

    tokens = _tokenize_transcript(transcript)
    token_count = len(tokens)
    unique_ratio = (len(set(tokens)) / token_count) if token_count else 0.0
    keyword_hits = _count_sport_keyword_hits(transcript)
    repetition_penalty = _compute_repetition_penalty(tokens)

    audio_duration = max(float(detail.get("audio_duration") or 0.0), 1.0)
    speech_duration = min(float(detail.get("speech_duration") or 0.0), audio_duration)
    coverage = speech_duration / audio_duration

    avg_logprob = float(detail.get("avg_logprob") or -5.0)
    avg_no_speech = float(detail.get("avg_no_speech_prob") or 1.0)
    avg_compression = float(detail.get("avg_compression_ratio") or 99.0)

    score = 0.0
    score += min(token_count, 180) * 0.22
    score += coverage * 28.0
    score += max(0.0, min(2.0, avg_logprob + 2.0)) * 14.0
    score += max(0.0, 1.0 - min(max(avg_no_speech, 0.0), 1.0)) * 16.0
    score += min(keyword_hits, 12) * 1.6
    score += unique_ratio * 10.0
    score -= repetition_penalty * 14.0
    if avg_compression > 2.4:
        score -= (avg_compression - 2.4) * 10.0
    if detail.get("candidate_kind") == "vocals":
        score += 1.5
    return round(score, 3)


def _select_best_asr_result(results: list[dict]) -> dict | None:
    """En yüksek kalite skoruna sahip ASR sonucunu seç."""
    usable = [item for item in (results or []) if item.get("transcript")]
    if not usable:
        return None
    return max(
        usable,
        key=lambda item: (
            item.get("quality_score", -1_000.0),
            len(item.get("transcript", "")),
            1 if item.get("candidate_kind") == "vocals" else 0,
        ),
    )


def _tokenize_transcript(text: str) -> list[str]:
    return [tok for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if tok]


def _compute_repetition_penalty(tokens: list[str]) -> float:
    if len(tokens) < 4:
        return 0.0
    repeats = 0
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            repeats += 1
    return repeats / max(1, len(tokens) - 1)


def _count_sport_keyword_hits(text: str) -> int:
    try:
        from core.sport_analyzer import SPORT_KEYWORDS
    except Exception:
        return 0
    lowered = (text or "").lower()
    hits = 0
    for meta in SPORT_KEYWORDS.values():
        for keyword in meta.get("keywords", []):
            if keyword.lower() in lowered:
                hits += 1
    return hits


def _run_ocr_on_frames(frame_paths: list, engine: str, config: Dict, _log=None) -> list:
    """
    Frame listesi üzerinde OCR çalıştırır.

    Returns:
        [(frame_index, ocr_text), ...]
    """
    def log(msg):
        logger.info(msg)
        if _log:
            _log(msg)

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
                    log(f"  [OCR] Frame {i + 1}/{len(frame_paths)} okundu")
                except Exception as e:
                    logger.warning("OCR frame %d hatası: %s", i + 1, e)
                    results.append((i + 1, ""))

        except ImportError:
            log("  [OCR] PaddleOCR yüklü değil")

    elif engine == "oneocr":
        try:
            from core.oneocr_engine import OneOCREngine
            ocr = OneOCREngine()

            for i, fpath in enumerate(frame_paths):
                try:
                    text = ocr.read_image(fpath)
                    results.append((i + 1, text))
                    log(f"  [OCR] Frame {i + 1}/{len(frame_paths)} okundu")
                except Exception as e:
                    logger.warning("OneOCR frame %d hatası: %s", i + 1, e)
                    results.append((i + 1, ""))

        except ImportError:
            log("  [OCR] OneOCR engine bulunamadı")

    else:
        log(f"  [OCR] Bilinmeyen engine: {engine}")

    non_empty = sum(1 for _, t in results if t.strip())
    log(f"  [OCR] Tamamlandı: {len(results)} frame, {non_empty} frame'de metin var")

    return results


def _save_text(path: str, text: str):
    """Metin dosyası kaydet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
