"""
[D] TRANSCRIBE — re-export from canonical implementation.

Bu modül geriye dönük uyumluluk için korunmuştur.
Gerçek implementasyon: audio/stages/transcribe.py
"""
from audio.stages.transcribe import TranscribeStage  # noqa: F401

__all__ = ["TranscribeStage"]
