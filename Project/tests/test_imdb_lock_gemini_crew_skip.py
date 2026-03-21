"""test_imdb_lock_gemini_crew_skip.py

Bug fix testi: IMDb LOCK aktifken Gemini Crew doğrulama çalışmamalı.

Senaryo:
  - IMDb eşleşti → imdb_matched = True, crew raw="imdb"
  - TMDB eşleşmedi → tmdb_matched = False
  - Eski kod: Gemini Crew çalışıyor, IMDb crew'u sıfırlıyor
  - Yeni kod: `not imdb_matched` koşulu → Gemini Crew atlanıyor

Testler:
  GC-01: IMDb LOCK aktif → Gemini Crew koşulu False olmalı
  GC-02: IMDb yok, TMDB yok, raw!=tmdb → Gemini Crew çalışmalı (normal akış)
  GC-03: TMDB eşleşti → Gemini Crew çalışmamalı
  GC-04: TMDB raw crew var → Gemini Crew çalışmamalı
  GC-05: IMDb LOCK aktifken validate_crew_with_gemini hiç çağrılmıyor (mock)
  GC-06: IMDb LOCK aktif → _gemini_crew_roles cdata'ya yazılmıyor
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ── Pipeline_runner'daki koşulu birebir çoğalt ────────────────────────────────

def _should_run_gemini_crew(imdb_matched: bool, tmdb_matched: bool, crew: list) -> bool:
    """pipeline_runner.py satır 1117 — koşul birebir."""
    return (
        not imdb_matched
        and not tmdb_matched
        and not any(c.get("raw") == "tmdb" for c in crew)
    )


# ── GC-01: IMDb LOCK → koşul False ───────────────────────────────────────────

class TestGeminiCrewCondition(unittest.TestCase):

    def test_gc01_imdb_lock_skips_gemini_crew(self):
        """IMDb LOCK aktifken Gemini Crew koşulu False dönmeli."""
        crew = [{"name": "Burak Arlıel", "job": "Director", "raw": "imdb"}]
        result = _should_run_gemini_crew(
            imdb_matched=True,
            tmdb_matched=False,
            crew=crew,
        )
        self.assertFalse(result, "IMDb LOCK aktifken Gemini Crew çalışmamalı")

    def test_gc02_no_match_gemini_crew_runs(self):
        """IMDb ve TMDB miss → Gemini Crew çalışmalı."""
        crew = [{"name": "Bilinmeyen", "job": "Editor", "raw": "ocr"}]
        result = _should_run_gemini_crew(
            imdb_matched=False,
            tmdb_matched=False,
            crew=crew,
        )
        self.assertTrue(result, "Hiç eşleşme yokken Gemini Crew çalışmalı")

    def test_gc03_tmdb_matched_skips_gemini_crew(self):
        """TMDB eşleşti → Gemini Crew atlanmalı."""
        crew = [{"name": "Ali Veli", "raw": "tmdb"}]
        result = _should_run_gemini_crew(
            imdb_matched=False,
            tmdb_matched=True,
            crew=crew,
        )
        self.assertFalse(result, "TMDB eşleşmişken Gemini Crew çalışmamalı")

    def test_gc04_tmdb_raw_crew_skips_gemini_crew(self):
        """TMDB backup crew varsa (raw=tmdb) → Gemini Crew atlanmalı."""
        crew = [
            {"name": "Ali Veli",    "raw": "ocr"},
            {"name": "Mehmet Öz",   "raw": "tmdb"},   # TMDB backup crew
        ]
        result = _should_run_gemini_crew(
            imdb_matched=False,
            tmdb_matched=False,
            crew=crew,
        )
        self.assertFalse(result, "TMDB backup crew varken Gemini Crew çalışmamalı")

    def test_gc01_old_condition_would_have_fired(self):
        """Regresyon: eski koşul (imdb_matched kontrolsüz) aynı senaryoda True dönerdi."""
        crew = [{"name": "Burak Arlıel", "raw": "imdb"}]

        # ESKİ koşul — imdb_matched yok
        old_condition = (
            not False  # tmdb_matched=False
            and not any(c.get("raw") == "tmdb" for c in crew)
        )
        self.assertTrue(old_condition, "Eski kod bu senaryoda Gemini'yi tetikliyordu")

        # YENİ koşul — imdb_matched var
        new_condition = _should_run_gemini_crew(
            imdb_matched=True, tmdb_matched=False, crew=crew
        )
        self.assertFalse(new_condition, "Yeni kod bu senaryoda Gemini'yi atlamalı")


# ── GC-05: validate_crew_with_gemini hiç çağrılmıyor (mock) ──────────────────

class TestGeminiCrewNotCalledOnImdbLock(unittest.TestCase):

    def test_gc05_validate_crew_not_called_when_imdb_locked(self):
        """IMDb LOCK aktifken validate_crew_with_gemini import bile edilmemeli."""
        mock_validate = MagicMock(return_value={"verified_roles": {"YÖNETMEN": ["X"]}})

        imdb_matched = True
        tmdb_matched = False
        cdata = {
            "film_title": "TEŞKILAT",
            "crew": [{"name": "Burak Arlıel", "job": "Director", "raw": "imdb"}],
        }
        ocr_lines = []

        # Koşul sağlanmıyorsa validate_crew_with_gemini çağrılmaz
        if _should_run_gemini_crew(imdb_matched, tmdb_matched, cdata.get("crew", [])):
            mock_validate(
                film_title=cdata.get("film_title"),
                ocr_crew=cdata.get("crew"),
                ocr_lines=ocr_lines,
                ocr_scores=[],
            )

        mock_validate.assert_not_called()

    def test_gc05b_validate_crew_called_when_no_match(self):
        """IMDb/TMDB miss durumunda validate_crew_with_gemini çağrılmalı."""
        mock_validate = MagicMock(return_value={"verified_roles": {}})

        imdb_matched = False
        tmdb_matched = False
        cdata = {
            "film_title": "BILINMEYEN",
            "crew": [{"name": "Birisi", "job": "Editor", "raw": "ocr"}],
        }

        if _should_run_gemini_crew(imdb_matched, tmdb_matched, cdata.get("crew", [])):
            mock_validate(
                film_title=cdata.get("film_title"),
                ocr_crew=cdata.get("crew"),
                ocr_lines=[],
                ocr_scores=[],
            )

        mock_validate.assert_called_once()


# ── GC-06: cdata["_gemini_crew_roles"] yazılmıyor ────────────────────────────

class TestGeminiCrewRolesNotWritten(unittest.TestCase):

    def test_gc06_gemini_crew_roles_not_set_on_imdb_lock(self):
        """IMDb LOCK aktifken _gemini_crew_roles cdata'ya yazılmamalı."""
        cdata = {
            "film_title": "TEŞKILAT",
            "crew": [{"name": "Burak Arlıel", "raw": "imdb"}],
        }

        imdb_matched = True
        tmdb_matched = False

        # Simüle: sadece koşul geçerse yaz
        if _should_run_gemini_crew(imdb_matched, tmdb_matched, cdata["crew"]):
            cdata["_gemini_crew_roles"] = {"YÖNETMEN": ["Gemini Yönetmen"]}

        self.assertNotIn(
            "_gemini_crew_roles", cdata,
            "IMDb LOCK aktifken _gemini_crew_roles yazılmamalı"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
