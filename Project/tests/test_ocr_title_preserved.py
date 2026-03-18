"""
OCR başlığının TMDB accept sonrası korunması testleri.

OCR-TITLE-01: TMDB accept sonrası ocr_title korunmalı (verify_credits)
OCR-TITLE-02: film_title zaten TMDB başlığıyla aynıysa ocr_title set edilmemeli
OCR-TITLE-03: TMDBVerifyResult.ocr_title alanı ham OCR başlığını döndürmeli
OCR-TITLE-04: _apply_tmdb_credits ocr_title korunmalı (pipeline_runner mantığı)
OCR-TITLE-05: ocr_title zaten varsa üzerine yazılmamalı
"""
from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest

from core.tmdb_verify import TMDBVerify, TMDBVerifyResult


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def _make_verifier(log_lines=None):
    tmp = tempfile.mkdtemp()
    logs = log_lines if log_lines is not None else []
    verifier = TMDBVerify(work_dir=tmp, api_key="test-key",
                          log_cb=lambda m: logs.append(m))
    return verifier, logs


# Örnek TMDB girişi — "Tatlı Bela New York'ta" filmi
_JANDARMA_ENTRY = {
    "id": 4728,
    "title": "Tatlı Bela New York'ta",
    "name": None,
    "original_title": "Le Gendarme à New-York",
    "release_date": "1965-01-01",
    "media_type": "movie",
}

_JANDARMA_CREDITS = {
    "cast": [
        {"name": "Louis de Funès",  "order": 0},
        {"name": "Michel Galabru",  "order": 1},
        {"name": "Christian Marin", "order": 2},
        {"name": "Guy Grosso",      "order": 3},
        {"name": "Jean Lefebvre",   "order": 4},
    ],
    "crew": [
        {"name": "Jean Girault", "job": "Director"},
    ],
}

_JANDARMA_DETAILS = {
    "genres": [{"name": "Komedi"}],
    "original_title": "Le Gendarme à New-York",
}


# ─────────────────────────────────────────────────────────────────────────────
# OCR-TITLE-01: TMDB accept sonrası ocr_title korunmalı
# ─────────────────────────────────────────────────────────────────────────────

def test_ocr_title_preserved_on_accept():
    """
    OCR-TITLE-01: cdata["film_title"] = "New York'taki Jandarma" (OCR başlığı) ile başla.
    verify_credits sonrası:
      - cdata["ocr_title"] == "New York'taki Jandarma"  (OCR başlığı korundu)
      - result.matched_id == 4728  (film bulundu)
      - result.ocr_title == "New York'taki Jandarma"  (result'ta da var)
    """
    verifier, _ = _make_verifier()

    cdata = {
        "film_title": "New York'taki Jandarma",
        "cast": [
            {"actor_name": "Louis de Funès",  "confidence": 0.9},
            {"actor_name": "Michel Galabru",  "confidence": 0.9},
            {"actor_name": "Christian Marin", "confidence": 0.8},
            {"actor_name": "Guy Grosso",      "confidence": 0.8},
            {"actor_name": "Jean Lefebvre",   "confidence": 0.8},
        ],
        "directors": [{"name": "Jean Girault"}],
    }

    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(_JANDARMA_ENTRY, "movie", "cast_only")), \
         patch.object(verifier, "_fetch_credits",
                      return_value=_JANDARMA_CREDITS), \
         patch.object(verifier.client, "get_movie_details",
                      return_value=_JANDARMA_DETAILS), \
         patch.object(verifier.client, "get_movie_keywords", return_value=[]), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    assert result.matched_id == 4728, (
        f"Film bulunmuş olmalı (matched_id=4728), aldık: {result.matched_id}"
    )
    assert cdata.get("ocr_title") == "New York'taki Jandarma", (
        f"ocr_title korunmalıydı, aldık: {cdata.get('ocr_title')!r}"
    )
    assert result.ocr_title == "New York'taki Jandarma", (
        f"TMDBVerifyResult.ocr_title OCR başlığını taşımalı, aldık: {result.ocr_title!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# OCR-TITLE-02: film_title zaten TMDB başlığıyla aynıysa ocr_title set edilmemeli
# ─────────────────────────────────────────────────────────────────────────────

def test_ocr_title_not_set_if_same_as_tmdb():
    """
    OCR-TITLE-02: cdata["film_title"] zaten TMDB başlığıyla aynıysa ocr_title set edilmemeli.
    """
    verifier, _ = _make_verifier()

    cdata = {
        "film_title": "Tatlı Bela New York'ta",  # zaten TMDB başlığıyla aynı
        "cast": [
            {"actor_name": "Louis de Funès",  "confidence": 0.9},
            {"actor_name": "Michel Galabru",  "confidence": 0.9},
            {"actor_name": "Christian Marin", "confidence": 0.8},
        ],
        "directors": [{"name": "Jean Girault"}],
    }

    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(_JANDARMA_ENTRY, "movie", "cast_only")), \
         patch.object(verifier, "_fetch_credits",
                      return_value=_JANDARMA_CREDITS), \
         patch.object(verifier.client, "get_movie_details",
                      return_value=_JANDARMA_DETAILS), \
         patch.object(verifier.client, "get_movie_keywords", return_value=[]), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    assert result.matched_id == 4728, (
        f"Film bulunmuş olmalı (matched_id=4728), aldık: {result.matched_id}"
    )
    # Aynı başlık olduğundan ocr_title set edilmemeli
    assert not cdata.get("ocr_title"), (
        f"film_title == tmdb_title iken ocr_title boş kalmalı, aldık: {cdata.get('ocr_title')!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# OCR-TITLE-03: TMDBVerifyResult.ocr_title alanı ham OCR başlığını döndürmeli
# ─────────────────────────────────────────────────────────────────────────────

def test_tmdb_verify_result_ocr_title_field():
    """
    OCR-TITLE-03: TMDBVerifyResult.ocr_title, verify_credits çağrısı sırasındaki
    orijinal film_title değerini taşımalı.
    """
    verifier, _ = _make_verifier()

    cdata = {
        "film_title": "New York'taki Jandarma",
        "cast": [
            {"actor_name": "Louis de Funès",  "confidence": 0.9},
            {"actor_name": "Michel Galabru",  "confidence": 0.9},
            {"actor_name": "Christian Marin", "confidence": 0.8},
        ],
        "directors": [{"name": "Jean Girault"}],
    }

    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(_JANDARMA_ENTRY, "movie", "cast_only")), \
         patch.object(verifier, "_fetch_credits",
                      return_value=_JANDARMA_CREDITS), \
         patch.object(verifier.client, "get_movie_details",
                      return_value=_JANDARMA_DETAILS), \
         patch.object(verifier.client, "get_movie_keywords", return_value=[]), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    assert result.ocr_title == "New York'taki Jandarma", (
        f"TMDBVerifyResult.ocr_title OCR başlığını taşımalı, aldık: {result.ocr_title!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# OCR-TITLE-04: _apply_tmdb_credits mantığı — ocr_title korunmalı
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_tmdb_credits_preserves_ocr_title():
    """
    OCR-TITLE-04: _apply_tmdb_credits benzeri mantıkta:
    cdata["film_title"] != tmdb_result.matched_title iken ocr_title set edilmeli
    ve cdata["film_title"] TMDB başlığına güncellenmeli.
    Bu testi `verify_credits` üzerinden validate ediyoruz.
    """
    verifier, _ = _make_verifier()

    cdata = {
        "film_title": "New York'taki Jandarma",
        "cast": [
            {"actor_name": "Louis de Funès",  "confidence": 0.9},
            {"actor_name": "Michel Galabru",  "confidence": 0.9},
            {"actor_name": "Christian Marin", "confidence": 0.8},
            {"actor_name": "Guy Grosso",      "confidence": 0.8},
        ],
        "directors": [{"name": "Jean Girault"}],
    }

    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(_JANDARMA_ENTRY, "movie", "cast_only")), \
         patch.object(verifier, "_fetch_credits",
                      return_value=_JANDARMA_CREDITS), \
         patch.object(verifier.client, "get_movie_details",
                      return_value=_JANDARMA_DETAILS), \
         patch.object(verifier.client, "get_movie_keywords", return_value=[]), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    # verify_credits OCR başlığını korur
    assert cdata.get("ocr_title") == "New York'taki Jandarma", (
        f"ocr_title korunmalıydı, aldık: {cdata.get('ocr_title')!r}"
    )
    assert result.matched_title == "Tatlı Bela New York'ta", (
        f"matched_title TMDB Türkçe başlığı olmalı, aldık: {result.matched_title!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# OCR-TITLE-05: ocr_title zaten varsa üzerine yazılmamalı
# ─────────────────────────────────────────────────────────────────────────────

def test_ocr_title_not_overwritten_if_already_set():
    """
    OCR-TITLE-05: cdata["ocr_title"] zaten set edilmişse üzerine yazılmamalı.
    """
    verifier, _ = _make_verifier()

    cdata = {
        "film_title": "New York'taki Jandarma",
        "ocr_title": "Önceden Set Edilmiş Başlık",  # zaten var
        "cast": [
            {"actor_name": "Louis de Funès",  "confidence": 0.9},
            {"actor_name": "Michel Galabru",  "confidence": 0.9},
            {"actor_name": "Christian Marin", "confidence": 0.8},
        ],
        "directors": [{"name": "Jean Girault"}],
    }

    with patch.object(verifier, "_find_tmdb_entry",
                      return_value=(_JANDARMA_ENTRY, "movie", "cast_only")), \
         patch.object(verifier, "_fetch_credits",
                      return_value=_JANDARMA_CREDITS), \
         patch.object(verifier.client, "get_movie_details",
                      return_value=_JANDARMA_DETAILS), \
         patch.object(verifier.client, "get_movie_keywords", return_value=[]), \
         patch.object(verifier, "_save_cache"), \
         patch.object(verifier, "_load_cache", return_value=None):

        result = verifier.verify_credits(cdata)

    assert result.matched_id == 4728, (
        f"Film bulunmuş olmalı (matched_id=4728), aldık: {result.matched_id}"
    )
    # Mevcut ocr_title üzerine yazılmamalı
    assert cdata.get("ocr_title") == "Önceden Set Edilmiş Başlık", (
        f"Mevcut ocr_title korunmalı, aldık: {cdata.get('ocr_title')!r}"
    )

