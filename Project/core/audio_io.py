"""
audio_io.py — WAV dosya yardımcıları.

Resample, süre hesabı, format doğrulama.
"""

from pathlib import Path


def get_wav_duration(wav_path: str) -> float:
    """WAV dosyasının süresini saniye cinsinden döndür."""
    try:
        import torchaudio
        info = torchaudio.info(wav_path)
        return info.num_frames / info.sample_rate
    except Exception:
        pass
    # Fallback: wave modülü (stdlib)
    try:
        import wave
        with wave.open(wav_path, 'rb') as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def get_wav_info(wav_path: str) -> dict:
    """WAV dosyası hakkında temel bilgi döndür."""
    result = {
        "path": str(wav_path),
        "exists": Path(wav_path).is_file(),
        "size_mb": 0.0,
        "duration_sec": 0.0,
        "sample_rate": 0,
        "channels": 0,
    }
    if not result["exists"]:
        return result

    result["size_mb"] = round(Path(wav_path).stat().st_size / (1024 * 1024), 2)

    try:
        import wave
        with wave.open(wav_path, 'rb') as wf:
            result["sample_rate"] = wf.getframerate()
            result["channels"] = wf.getnchannels()
            result["duration_sec"] = round(wf.getnframes() / wf.getframerate(), 2)
    except Exception:
        result["duration_sec"] = get_wav_duration(wav_path)

    return result


def validate_wav(wav_path: str, min_duration: float = 1.0) -> tuple[bool, str]:
    """
    WAV dosyasını doğrula.
    Returns: (is_valid, error_message)
    """
    p = Path(wav_path)
    if not p.is_file():
        return False, f"Dosya bulunamadı: {wav_path}"
    if p.stat().st_size < 1000:
        return False, f"Dosya çok küçük: {p.stat().st_size} bytes"

    dur = get_wav_duration(wav_path)
    if dur < min_duration:
        return False, f"Süre çok kısa: {dur:.1f}s (min: {min_duration}s)"

    return True, ""
