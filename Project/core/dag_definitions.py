"""dag_definitions.py — Profil bazlı DAG tanımları."""

FILM_DIZI_DAG = {
    "VIDEO_INPUT": {
        "label": "VIDEO_INPUT",
        "children": ["OCR_BRANCH", "ASR_BRANCH"],
    },
    "OCR_BRANCH": {
        "label": "OCR",
        "steps": [
            "Frame Extract",
            "Text Filter",
            "OCR Credits",
            "Credits Parse",
            "Name Verify",
            "TMDB Verify",
            "Gemini Fallback",
            "Export",
        ],
    },
    "ASR_BRANCH": {
        "label": "ASR",
        "steps": [
            "Audio Extract",
            "Whisper Transcribe",
            "Gemini Summarize",
        ],
    },
}

SPOR_DAG = {
    "VIDEO_INPUT": {
        "label": "VIDEO_INPUT",
        "children": ["ASR_BRANCH", "OCR_BRANCH"],
    },
    "ASR_BRANCH": {
        "label": "ASR",
        "steps": ["Extract", "Transcribe"],
    },
    "OCR_BRANCH": {
        "label": "OCR (Son 10dk)",
        "steps": ["Frame Extract", "Text Filter", "OCR"],
    },
    "MATCH_PARSE": {
        "label": "Maç Analizi",
        "steps": ["Ollama Parse"],
    },
}

FILM_DIZI_ONEOCR_DAG = {
    "VIDEO_INPUT": {
        "label": "VIDEO_INPUT",
        "children": ["OCR_BRANCH"],
    },
    "OCR_BRANCH": {
        "label": "OCR (OneOCR)",
        "steps": [
            "Frame Extract",
            "Text Filter",
            "OneOCR Read",
            "Credits Parse",
            "Name Verify",
            "Export",
        ],
    },
}

PROFILE_DAGS = {
    "FilmDizi": FILM_DIZI_DAG,
    "FilmDiziONEOCR": FILM_DIZI_ONEOCR_DAG,
    "Spor": SPOR_DAG,
}


def get_dag(profile_name: str) -> dict:
    """Profil adına göre DAG tanımını döndür. Bilinmeyenler için FilmDizi kullan."""
    return PROFILE_DAGS.get(profile_name, FILM_DIZI_DAG)
