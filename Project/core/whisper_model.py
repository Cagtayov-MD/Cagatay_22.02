"""Whisper model adı yardımcıları."""

DEFAULT_WHISPER_MODEL = "large-v3"

_WHISPER_MODEL_ALIASES = {
    "v3": DEFAULT_WHISPER_MODEL,
    "large-v3": DEFAULT_WHISPER_MODEL,
    "large-v3-turbo": DEFAULT_WHISPER_MODEL,
    "large-v3-turbo-en": DEFAULT_WHISPER_MODEL,
    "large-v3 turbo": DEFAULT_WHISPER_MODEL,
    "large-v3 turbo-en": DEFAULT_WHISPER_MODEL,
    "large-v3turbo": DEFAULT_WHISPER_MODEL,
    "large-v3turbo-en": DEFAULT_WHISPER_MODEL,
    "large-v3-en": DEFAULT_WHISPER_MODEL,
}


def normalize_whisper_model_name(model_name, default: str = DEFAULT_WHISPER_MODEL) -> str:
    """Sık kullanılan alias'ları faster-whisper'ın beklediği model adına indirger."""
    raw = "" if model_name is None else str(model_name).strip()
    if not raw:
        return default

    normalized = raw.lower().replace("_", "-")
    normalized = " ".join(normalized.split())
    compact = normalized.replace(" ", "-")
    return _WHISPER_MODEL_ALIASES.get(compact, raw)
