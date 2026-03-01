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

# Kullanıcı kararı: sabit yollar
# SECURITY: Hardcoded paths removed - all paths must be set via environment variables
# This prevents accidental credential exposure in the codebase
NAME_DB_DIR = Path(os.environ.get("NAME_DB_DIR", ""))
NAME_DB_ROLE_ALIAS_TR = NAME_DB_DIR / "credits_role_alias_tr.json" if NAME_DB_DIR else Path("")
NAME_DB_NAMES_BY_COUNTRY = NAME_DB_DIR / "names_by_country.json" if NAME_DB_DIR else Path("")

# API keys and credentials must be configured via environment variables
# No default paths to prevent credential exposure
API_KEYS_JSON = Path(os.environ.get("API_KEYS_JSON", ""))
GOOGLE_KEYS_JSON = Path(os.environ.get("GOOGLE_KEYS_JSON", ""))
LOGOLAR_DIR = Path(os.environ.get("LOGOLAR_DIR", ""))

FFMPEG_BIN_DIR = Path(os.environ.get("FFMPEG_BIN_DIR", ""))

# İsim DB için olası dosyalar (varsa ilk bulunan kullanılır)
NAME_DB_SQLITE_CANDIDATES = [
    NAME_DB_DIR / "compiled" / "names.db" if NAME_DB_DIR else Path(""),
    NAME_DB_DIR / "compiled" / "names.sqlite" if NAME_DB_DIR else Path(""),
    NAME_DB_DIR / "compiled" / "names.sql" if NAME_DB_DIR else Path(""),
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
