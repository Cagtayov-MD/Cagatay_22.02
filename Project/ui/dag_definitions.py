"""dag_definitions.py — Profil bazlı DAG tanımları."""

FILM_DIZI_DAG = {
    "VIDEO_INPUT": {"label": "Video", "children": ["ASR_BRANCH", "OCR_BRANCH"]},
    "ASR_BRANCH": {"label": "ASR", "steps": ["Extract", "Transcribe"]},
    "OCR_BRANCH": {"label": "OCR", "steps": ["Frame Extract", "Text Filter", "OCR", "VLM OCR", "Credits Parse", "TMDB Verify"]},
}

SPOR_DAG = {
    "VIDEO_INPUT": {"label": "Video", "children": ["ASR_BRANCH", "OCR_BRANCH"]},
    "ASR_BRANCH": {"label": "ASR", "steps": ["Extract", "Transcribe"]},
    "OCR_BRANCH": {"label": "OCR (Son 10dk)", "steps": ["Frame Extract", "Text Filter", "OCR"]},
    "MATCH_PARSE": {"label": "Maç Analizi", "steps": ["Ollama Parse"]},
}

PROFILE_DAGS = {
    "FilmDizi": FILM_DIZI_DAG,
    "Spor": SPOR_DAG,
}

def get_dag(profile_name: str) -> dict:
    """Profil adına göre DAG tanımını döndür. Bilinmeyenler için FilmDizi kullan."""
    return PROFILE_DAGS.get(profile_name, FILM_DIZI_DAG)
