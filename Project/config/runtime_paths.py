"""Merkezi path + key erişim modülü.

Bu modül import sırasında dosya okumaz.
JSON/anahtarlar sadece çağrıldığında (lazy) yüklenir.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

# Cross-platform paths with environment variable fallbacks
# If not set, use home directory based defaults
_project_root = Path(__file__).resolve().parents[1]
_home_source = Path(os.environ.get("SOURCE_ROOT", r"F:\Source"))

NAME_DB_DIR = Path(os.environ.get("NAME_DB_DIR", str(_home_source / "name_db")))
NAME_DB_ROLE_ALIAS_TR = NAME_DB_DIR / "credits_role_alias_tr.json"
NAME_DB_NAMES_BY_COUNTRY = NAME_DB_DIR / "names_by_country.json"

API_KEYS_JSON = Path(os.environ.get("API_KEYS_JSON", str(_project_root / "config" / "api_keys.json")))
GOOGLE_KEYS_JSON = Path(
    os.environ.get(
        "GOOGLE_KEYS_JSON",
        str(_project_root / "config" / "google_api.json"),
    )
)
LOGOLAR_DIR = Path(os.environ.get("LOGOLAR_DIR", str(_home_source / "Logo")))

FFMPEG_BIN_DIR = Path(os.environ.get("FFMPEG_BIN_DIR", str(_home_source / "ffmpeg" / "bin")))

IMDB_DB_PATH = Path(os.environ.get("IMDB_DB_PATH", r"F:\IMDB\db\imdb.duckdb"))


def get_imdb_db_path() -> str:
    env_path = (os.environ.get("IMDB_DB_PATH") or "").strip()
    if env_path:
        return env_path
    return str(IMDB_DB_PATH)

# İsim DB için olası dosyalar (varsa ilk bulunan kullanılır)
NAME_DB_SQLITE_CANDIDATES = [
    NAME_DB_DIR / "compiled" / "names.db",
    NAME_DB_DIR / "compiled" / "names.sqlite",
    NAME_DB_DIR / "compiled" / "names.sql",
]


@lru_cache(maxsize=1)
def load_api_keys() -> dict[str, Any]:
    if not API_KEYS_JSON.is_file():
        return {}
    with API_KEYS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def load_google_keys() -> dict[str, Any]:
    if not GOOGLE_KEYS_JSON.is_file():
        return {}
    with GOOGLE_KEYS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def resolve_name_db_path() -> str:
    for p in NAME_DB_SQLITE_CANDIDATES:
        if p.is_file():
            return str(p)
    return ""


def get_tmdb_api_key() -> str:
    env_key = (os.environ.get("TMDB_API_KEY") or "").strip()
    if env_key:
        return env_key
    return str(load_api_keys().get("tmdb_api_key", "")).strip()


def get_gemini_api_key() -> str:
    env_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if env_key:
        return env_key
    keys = load_api_keys()
    return str(keys.get("gemini_api_key") or keys.get("google_api_key") or "").strip()


def get_gemini_film_credit_api_key() -> str:
    env_key = (os.environ.get("GEMINI_FILM_CREDIT_API_KEY") or "").strip()
    if env_key:
        return env_key
    return str(load_api_keys().get("gemini_film_credit_api_key", "")).strip()


def is_gemini_film_credit_shadow_enabled() -> bool:
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        return text in {"1", "true", "yes", "on", "enabled"}

    env_value = os.environ.get("GEMINI_FILM_CREDIT_SHADOW_ENABLED")
    if env_value is not None:
        return _as_bool(env_value)
    return _as_bool(load_api_keys().get("gemini_film_credit_shadow_enabled", False))


def get_gemini_film_credit_expo_dir() -> str:
    env = os.environ.get("GEMINI_FILM_CREDIT_EXPO_DIR", "").strip()
    if env:
        return env
    val = str(load_api_keys().get("gemini_film_credit_expo_dir", "")).strip()
    return val or r"D:\expoOnay"


def get_openai_api_key() -> str:
    env_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if env_key:
        return env_key
    return str(load_api_keys().get("openai_api_key", "")).strip()
