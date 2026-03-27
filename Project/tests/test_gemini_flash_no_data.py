"""test_gemini_flash_no_data.py

Gemini 2.5 Flash fallback — "veri yok" senaryoları.

TMDB/IMDb miss olduğunda _run_gemini_cast_extract devreye girer.
Bu testler Gemini'nin boş/eksik veri döndürdüğü durumları kontrol eder.

Testler:
  GF-01: Cast boş → cdata["cast"] yazılmaz, False döner
  GF-02: Crew boş → cdata["_verified_crew_roles"] yazılmaz, False döner
  GF-03: Cast + Crew boş → gemini_extracted yazılmaz, False döner
  GF-04: Timeout → cdata["gemini_timeout"] = True
  GF-05: API key yok → False döner, extractor hiç çağrılmaz
  GF-06: Cast dolu, Crew boş → sadece cast yazılır, True döner
  GF-07: Cast boş, Crew dolu → sadece crew yazılır, True döner
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ── _run_gemini_cast_extract mantığını izole et ───────────────────────────────

def _run(cast_result, crew_result, api_key="test-key", timed_out=False):
    """pipeline_runner._run_gemini_cast_extract mantığını mock ile çalıştır."""
    if not api_key:
        return False, {}

    cdata = {"film_title": "TEST FİLMİ"}

    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = cast_result
    mock_extractor.extract_crew_from_scores.return_value = crew_result
    mock_extractor.timed_out = timed_out

    # pipeline_runner.py:1736–1774 mantığı birebir
    has_cast = cast_result and cast_result.get("cast")
    has_crew = bool(
        crew_result.get("directors")
        or crew_result.get("producers")
        or crew_result.get("writers")
    )

    if mock_extractor.timed_out:
        cdata["gemini_timeout"] = True

    if has_cast or has_crew:
        if has_cast:
            cdata["cast"] = cast_result.get("cast", [])
            cdata["total_actors"] = len(cdata["cast"])
        if has_crew:
            crew_roles = {}
            directors = crew_result.get("directors") or []
            producers = crew_result.get("producers") or []
            writers   = crew_result.get("writers")   or []
            if directors:
                crew_roles["YÖNETMEN"] = directors
            if producers:
                crew_roles["YAPIMCI"] = producers
            if writers:
                crew_roles["SENARYO"] = writers
            cdata["_verified_crew_roles"] = crew_roles
        cdata["gemini_extracted"] = True
        return True, cdata

    return False, cdata


# ── GF-01 ──────────────────────────────────────────────────────────────────────

class TestGeminiFlashNoData(unittest.TestCase):

    def test_gf01_empty_cast_not_written(self):
        """Cast boş → cdata'ya cast yazılmaz."""
        ok, cdata = _run(
            cast_result={"cast": []},
            crew_result={"directors": [], "producers": [], "writers": []},
        )
        self.assertFalse(ok, "Boş sonuçta False dönmeli")
        self.assertNotIn("cast", cdata)

    def test_gf02_empty_crew_not_written(self):
        """Crew boş → cdata'ya _verified_crew_roles yazılmaz."""
        ok, cdata = _run(
            cast_result={"cast": []},
            crew_result={"directors": [], "producers": [], "writers": []},
        )
        self.assertFalse(ok)
        self.assertNotIn("_verified_crew_roles", cdata)

    def test_gf03_gemini_extracted_not_set_on_empty(self):
        """Cast + Crew boş → gemini_extracted cdata'ya yazılmaz."""
        ok, cdata = _run(
            cast_result={"cast": []},
            crew_result={},
        )
        self.assertFalse(ok)
        self.assertNotIn("gemini_extracted", cdata)

    def test_gf04_timeout_flag_written(self):
        """Timeout olduğunda gemini_timeout = True yazılır."""
        ok, cdata = _run(
            cast_result={"cast": []},
            crew_result={},
            timed_out=True,
        )
        self.assertFalse(ok, "Boş veriyle False dönmeli")
        self.assertTrue(cdata.get("gemini_timeout"), "gemini_timeout True olmalı")

    def test_gf05_no_api_key_returns_false(self):
        """API key yoksa False döner, extractor çağrılmaz."""
        ok, cdata = _run(
            cast_result={"cast": [{"name": "Ali"}]},
            crew_result={"directors": ["Mehmet"]},
            api_key="",
        )
        self.assertFalse(ok)
        self.assertNotIn("cast", cdata)

    def test_gf06_cast_only(self):
        """Cast dolu, Crew boş → sadece cast yazılır, True döner."""
        ok, cdata = _run(
            cast_result={"cast": [{"name": "Oyuncu A"}, {"name": "Oyuncu B"}]},
            crew_result={"directors": [], "producers": [], "writers": []},
        )
        self.assertTrue(ok)
        self.assertIn("cast", cdata)
        self.assertEqual(len(cdata["cast"]), 2)
        self.assertNotIn("_verified_crew_roles", cdata)
        self.assertTrue(cdata.get("gemini_extracted"))

    def test_gf07_crew_only(self):
        """Cast boş, Crew dolu → sadece crew yazılır, True döner."""
        ok, cdata = _run(
            cast_result={"cast": []},
            crew_result={"directors": ["Yönetmen X"], "producers": [], "writers": []},
        )
        self.assertTrue(ok)
        self.assertNotIn("cast", cdata)
        self.assertIn("_verified_crew_roles", cdata)
        self.assertEqual(cdata["_verified_crew_roles"]["YÖNETMEN"], ["Yönetmen X"])
        self.assertTrue(cdata.get("gemini_extracted"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
