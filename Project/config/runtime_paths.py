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
NAME_DB_DIR = Path(os.environ.get("NAME_DB_DIR", r"F:\Source\name_db"))
NAME_DB_ROLE_ALIAS_TR = NAME_DB_DIR / "credits_role_alias_tr.json"
NAME_DB_NAMES_BY_COUNTRY = NAME_DB_DIR / "names_by_country.json"

API_KEYS_JSON = Path(os.environ.get("API_KEYS_JSON", r"F:\Project\config\api_keys.json"))
GOOGLE_KEYS_JSON = Path(
    os.environ.get(
        "GOOGLE_KEYS_JSON",
        r"F:\Project\config\video-analiz-sistemi-a59996f04788.json",
    )
)
LOGOLAR_DIR = Path(os.environ.get("LOGOLAR_DIR", r"F:\Source\Logo"))

FFMPEG_BIN_DIR = Path(os.environ.get("FFMPEG_BIN_DIR", r"F:\Source\ffmpeg\bin"))

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
