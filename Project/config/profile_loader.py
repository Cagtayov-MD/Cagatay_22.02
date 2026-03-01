"""İçerik profili yükleyici. content_profiles.json'dan profil okur."""
import json
from pathlib import Path
from functools import lru_cache

CONTENT_PROFILES_PATH = Path(__file__).parent / "content_profiles.json"

@lru_cache(maxsize=1)
def _load_all():
    if CONTENT_PROFILES_PATH.is_file():
        with CONTENT_PROFILES_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_profile(name: str) -> dict:
    """Profil adına göre ayarları döndür. Bulunamazsa boş dict."""
    return dict(_load_all().get(name, {}))

def list_profiles() -> list[str]:
    """Placeholder olmayan profil adlarını döndür."""
    return [k for k, v in _load_all().items() if not v.get("_placeholder")]

def list_all_profiles() -> list[str]:
    """Tüm profil adlarını döndür (placeholder dahil)."""
    return list(_load_all().keys())
