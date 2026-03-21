"""test_crew_confidence_annotation.py

_annotate_crew_confidence(), field_from_tmdb(), verify_crew_role() testleri.

Testler:
  CA-01: field_from_tmdb() — bilinen (dept, job) → doğru canonical alan
  CA-02: field_from_tmdb() — bilinmeyen → boş string
  CA-03: field_from_tmdb() — case-insensitive çalışır
  CA-04: IMDb + TMDB aynı alan → source_confidence=high, source_conflict yok
  CA-05: IMDb + TMDB çelişki → source_confidence=medium, flags=[source_conflict]
  CA-06: Sadece IMDb (TMDB'de yok) → source_confidence=medium, flag yok
  CA-07: Sadece TMDB + Gemini YES → source_confidence=medium, tmdb_gemini_confirmed
  CA-08: Sadece TMDB + Gemini NO → source_confidence=low, tmdb_only_unconfirmed
  CA-09: OCR only (raw=ocr_verified) → source_confidence=low, no_external_match
  CA-10: verify_crew_role() mock → YES döner
  CA-11: verify_crew_role() mock → NO döner
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def _make_runner():
    """Minimal PipelineRunner stub — sadece _annotate_crew_confidence için."""
    from core.pipeline_runner import PipelineRunner
    runner = object.__new__(PipelineRunner)
    runner._log = lambda msg: None
    return runner


def _run_annotation(cdata, imdb_matched, tmdb_matched,
                    gemini_answer="YES"):
    """_annotate_crew_confidence çalıştır; verify_crew_role mock'lanır."""
    runner = _make_runner()
    with patch(
        "core.gemini_crew_validator.verify_crew_role",
        return_value=gemini_answer,
    ):
        runner._annotate_crew_confidence(cdata, imdb_matched, tmdb_matched)
    return cdata


# ─────────────────────────────────────────────────────────────────────────────
# CA-01/02/03: field_from_tmdb()
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldFromTmdb(unittest.TestCase):

    def setUp(self):
        from core.gemini_crew_validator import field_from_tmdb
        self.f = field_from_tmdb

    def test_ca01_known_director(self):
        """CA-01: (Directing, Director) → YÖNETMEN"""
        self.assertEqual(self.f("Directing", "Director"), "YÖNETMEN")

    def test_ca01_known_dop(self):
        """CA-01: (Camera, Director of Photography) → GÖRÜNTÜ YÖNETMENİ"""
        self.assertEqual(self.f("Camera", "Director of Photography"), "GÖRÜNTÜ YÖNETMENİ")

    def test_ca01_known_editor(self):
        """CA-01: (Editing, Editor) → KURGU"""
        self.assertEqual(self.f("Editing", "Editor"), "KURGU")

    def test_ca01_known_assistant_director(self):
        """CA-01: (Directing, Assistant Director) → YÖNETMEN YARDIMCISI"""
        self.assertEqual(self.f("Directing", "Assistant Director"), "YÖNETMEN YARDIMCISI")

    def test_ca01_known_producer(self):
        """CA-01: (Production, Producer) → YAPIMCI"""
        self.assertEqual(self.f("Production", "Producer"), "YAPIMCI")

    def test_ca01_known_executive_producer(self):
        """CA-01: (Production, Executive Producer) → YAPIMCI"""
        self.assertEqual(self.f("Production", "Executive Producer"), "YAPIMCI")

    def test_ca01_known_screenplay(self):
        """CA-01: (Writing, Screenplay) → SENARYO"""
        self.assertEqual(self.f("Writing", "Screenplay"), "SENARYO")

    def test_ca02_unknown_dept_job(self):
        """CA-02: bilinmeyen (dept, job) → boş string"""
        self.assertEqual(self.f("Sound", "Sound Designer"), "")

    def test_ca02_empty_strings(self):
        """CA-02: boş argümanlar → boş string"""
        self.assertEqual(self.f("", ""), "")

    def test_ca03_case_insensitive(self):
        """CA-03: küçük harf dept/job da çalışır"""
        self.assertEqual(self.f("directing", "director"), "YÖNETMEN")
        self.assertEqual(self.f("EDITING", "EDITOR"), "KURGU")
        self.assertEqual(self.f("camera", "Director of Photography"), "GÖRÜNTÜ YÖNETMENİ")


# ─────────────────────────────────────────────────────────────────────────────
# CA-04/05/06: IMDb path — Ana Alanlar
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnotateImdbPath(unittest.TestCase):

    def _director_cdata(self, hint_dept="", hint_job=""):
        """YÖNETMEN için minimal cdata."""
        from core.gemini_crew_validator import _crew_norm
        name = "Ali Kaya"  # ASCII-safe — normalizasyon sorunu yok
        cdata = {
            "film_title": "Test Filmi",
            "cast": [],
            "crew": [{
                "name": name,
                "job": "Director",
                "department": "Directing",
                "raw": "imdb",
            }],
        }
        if hint_dept:
            cdata["_tmdb_dept_hints"] = {
                _crew_norm(name): {"department": hint_dept, "job": hint_job}
            }
        return cdata

    def test_ca04_imdb_tmdb_same_field_high(self):
        """CA-04: IMDb + TMDB aynı alanı gösteriyor → high, source_conflict yok."""
        cdata = self._director_cdata(hint_dept="directing", hint_job="Director")
        _run_annotation(cdata, imdb_matched=True, tmdb_matched=False)
        crew = cdata["crew"][0]
        self.assertEqual(crew["source_confidence"], "high")
        self.assertNotIn("source_conflict", crew.get("flags", []))

    def test_ca05_imdb_tmdb_conflict_medium_source_conflict(self):
        """CA-05: IMDb director, TMDB writing diyor → medium + source_conflict."""
        cdata = self._director_cdata(hint_dept="writing", hint_job="Screenplay")
        _run_annotation(cdata, imdb_matched=True, tmdb_matched=False)
        crew = cdata["crew"][0]
        self.assertEqual(crew["source_confidence"], "medium")
        self.assertIn("source_conflict", crew.get("flags", []))

    def test_ca06_imdb_only_medium_no_flag(self):
        """CA-06: Sadece IMDb (TMDB'de yok) → medium, source_conflict yok."""
        cdata = self._director_cdata()  # _tmdb_dept_hints yok
        _run_annotation(cdata, imdb_matched=True, tmdb_matched=False)
        crew = cdata["crew"][0]
        self.assertEqual(crew["source_confidence"], "medium")
        self.assertNotIn("source_conflict", crew.get("flags", []))


# ─────────────────────────────────────────────────────────────────────────────
# CA-07/08: TMDB-only path — Ana Alanlar + Gemini
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnotateTmdbOnlyPath(unittest.TestCase):

    def _tmdb_director_cdata(self):
        return {
            "film_title": "Test Filmi",
            "cast": [],
            "crew": [{
                "name": "Mehmet Kaya",
                "job": "Director",
                "department": "Directing",
                "raw": "tmdb",
            }],
        }

    def test_ca07_tmdb_gemini_yes_confirmed(self):
        """CA-07: Sadece TMDB, Gemini YES → medium + tmdb_gemini_confirmed."""
        cdata = self._tmdb_director_cdata()
        _run_annotation(cdata, imdb_matched=False, tmdb_matched=True,
                        gemini_answer="YES")
        crew = cdata["crew"][0]
        self.assertEqual(crew["source_confidence"], "medium")
        self.assertIn("tmdb_gemini_confirmed", crew["flags"])
        self.assertNotIn("tmdb_only_unconfirmed", crew["flags"])

    def test_ca08_tmdb_gemini_no_unconfirmed(self):
        """CA-08: Sadece TMDB, Gemini NO → low + tmdb_only_unconfirmed."""
        cdata = self._tmdb_director_cdata()
        _run_annotation(cdata, imdb_matched=False, tmdb_matched=True,
                        gemini_answer="NO")
        crew = cdata["crew"][0]
        self.assertEqual(crew["source_confidence"], "low")
        self.assertIn("tmdb_only_unconfirmed", crew["flags"])
        self.assertNotIn("tmdb_gemini_confirmed", crew["flags"])


# ─────────────────────────────────────────────────────────────────────────────
# CA-09: neither-match path
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnotateNeitherMatchPath(unittest.TestCase):

    def test_ca09_ocr_only_low_no_external_match(self):
        """CA-09: OCR only crew → low + no_external_match."""
        cdata = {
            "film_title": "Test Filmi",
            "cast": [],
            "crew": [{
                "name": "Zeynep Demir",
                "job": "GÖRÜNTÜ YÖNETMENİ",
                "department": "",
                "raw": "ocr_verified",
            }],
        }
        _run_annotation(cdata, imdb_matched=False, tmdb_matched=False)
        crew = cdata["crew"][0]
        self.assertEqual(crew["source_confidence"], "low")
        self.assertIn("no_external_match", crew["flags"])


# ─────────────────────────────────────────────────────────────────────────────
# CA-10/11: verify_crew_role() mock testi
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyCrewRole(unittest.TestCase):

    def test_ca10_gemini_yes(self):
        """CA-10: verify_crew_role — Gemini YES döner."""
        from core.gemini_crew_validator import verify_crew_role

        with patch("core.llm_provider.generate", return_value="YES"):
            result = verify_crew_role("Test Filmi", "Ali Kaya", "YÖNETMEN")

        self.assertEqual(result, "YES")

    def test_ca11_gemini_no(self):
        """CA-11: verify_crew_role — Gemini NO döner."""
        from core.gemini_crew_validator import verify_crew_role

        with patch("core.llm_provider.generate", return_value="NO"):
            result = verify_crew_role("Test Filmi", "Ali Kaya", "YÖNETMEN")

        self.assertEqual(result, "NO")


if __name__ == "__main__":
    unittest.main()
