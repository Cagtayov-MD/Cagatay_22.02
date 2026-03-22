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
        "children": ["ASR_BRANCH", "FRAME_BRANCH"],
    },
    "ASR_BRANCH": {
        "label": "ASR (İlk+Son 15dk)",
        "steps": [
            "Segment Extract",
            "ASR Transcribe",
        ],
    },
    "FRAME_BRANCH": {
        "label": "OCR (Son 15dk, 10sn aralık)",
        "steps": [
            "Frame Extract",
            "OCR Read",
        ],
    },
    "SPORT_ANALYZE": {
        "label": "Spor Analizi",
        "steps": [
            "Sport Analyze",
            "Export",
        ],
    },
}

PROFILE_DAGS = {
    "FilmDizi-Hybrid": FILM_DIZI_DAG,
    "Spor": SPOR_DAG,
}


def get_dag(profile_name: str) -> dict:
    """Profil adına göre DAG tanımını döndür. Bilinmeyenler için FilmDizi-Hybrid kullan."""
    return PROFILE_DAGS.get(profile_name, FILM_DIZI_DAG)


def is_sport_match_profile(profile_name: str) -> bool:
    """Profilin Spor pipeline'ı kullanıp kullanmadığını döndür."""
    return profile_name == "Spor"
