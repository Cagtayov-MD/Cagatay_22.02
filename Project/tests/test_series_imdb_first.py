"""Dizi/Film ayrımı ve IMDb-önce akışı testleri.

SI-01: Dosya adından dizi bayrağı (flag=0) doğru okunuyor mu?
SI-02: Film bayrağı (flag=1) doğru okunuyor mu?
SI-03: Regex eşleşmezse varsayılan film kabul ediliyor mu?
SI-04: Dizi modunda IMDb eşleşirse verify_as_series hiç çağrılmıyor mu?
SI-05: Dizi modunda IMDb miss → verify_as_series çağrılıyor mu?
SI-06: Her ikisi de miss → Gemini fallback çağrılıyor mu?
"""

import re
import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı: dosya adından flag çıkarımı (pipeline_runner mantığı)
# ─────────────────────────────────────────────────────────────────────────────

_FLAG_RE = re.compile(r'\d{4}-\d{2,4}-(\d+)-\d{4}')

def _detect_flag(filename: str):
    m = _FLAG_RE.search(filename)
    flag = m.group(1) if m else None
    return flag, (flag == "0")


# ─────────────────────────────────────────────────────────────────────────────
# SI-01 .. SI-03: Regex / flag tespiti
# ─────────────────────────────────────────────────────────────────────────────

class TestContentFlagDetection(unittest.TestCase):

    def test_SI01_series_flag(self):
        """2020-0054-0-0191 → flag='0' → dizi"""
        fname = "2020-0054-0-0191-90-1-TÜRKAN_HANIMIN_KONAĞI"
        flag, is_series = _detect_flag(fname)
        self.assertEqual(flag, "0")
        self.assertTrue(is_series)

    def test_SI02_film_flag(self):
        """2020-0054-1-0191 → flag='1' → film"""
        fname = "2020-0054-1-0191-90-1-ÖRNEK_FİLM"
        flag, is_series = _detect_flag(fname)
        self.assertEqual(flag, "1")
        self.assertFalse(is_series)

    def test_SI03_no_match_defaults_to_film(self):
        """Regex eşleşmezse flag=None → film kabul edilir"""
        fname = "DOSYA_ADI_OLMAYAN_FORMAT"
        flag, is_series = _detect_flag(fname)
        self.assertIsNone(flag)
        self.assertFalse(is_series)


# ─────────────────────────────────────────────────────────────────────────────
# SI-04 .. SI-06: Dizi modunda IMDb-önce akışı
# Doğrudan pipeline_runner çağırmak yerine akış mantığını izole test ederiz.
# ─────────────────────────────────────────────────────────────────────────────

class TestSeriesImdbFirstFlow(unittest.TestCase):
    """pipeline_runner'daki yeni dizi akışının mantık testi.

    PipelineRunner yerine akışın kendisini simüle ederek bağımlılıkları
    minimize ediyoruz.
    """

    def _run_series_flow(self, imdb_matched: bool, tmdb_matched: bool):
        """
        Dizi akışını simüle eder.
        Döner: (imdb_called, verify_as_series_called, gemini_called, final_matched)
        """
        calls = {"imdb": False, "tmdb": False, "gemini": False}

        # --- IMDb adımı ---
        result_imdb_matched = False
        if True:  # _imdb_enabled
            calls["imdb"] = True
            if imdb_matched:
                result_imdb_matched = True

        # --- verify_as_series fallback ---
        result_tmdb_matched = False
        _series_nv_pending = True
        if not result_imdb_matched and _series_nv_pending:
            calls["tmdb"] = True
            if tmdb_matched:
                result_tmdb_matched = True
            else:
                # her ikisi de miss → Gemini
                calls["gemini"] = True

        final_matched = result_imdb_matched or result_tmdb_matched
        return calls, final_matched

    def test_SI04_imdb_match_skips_tmdb(self):
        """IMDb eşleşirse verify_as_series çağrılmamalı."""
        calls, matched = self._run_series_flow(imdb_matched=True, tmdb_matched=False)
        self.assertTrue(calls["imdb"], "IMDb çağrılmalı")
        self.assertFalse(calls["tmdb"], "IMDb eşleşti — TMDB çağrılmamalı")
        self.assertFalse(calls["gemini"], "Gemini çağrılmamalı")
        self.assertTrue(matched)

    def test_SI05_imdb_miss_triggers_tmdb(self):
        """IMDb miss → verify_as_series çağrılmalı."""
        calls, matched = self._run_series_flow(imdb_matched=False, tmdb_matched=True)
        self.assertTrue(calls["imdb"], "IMDb çağrılmalı")
        self.assertTrue(calls["tmdb"], "IMDb miss → TMDB çağrılmalı")
        self.assertFalse(calls["gemini"], "TMDB eşleşti — Gemini çağrılmamalı")
        self.assertTrue(matched)

    def test_SI06_both_miss_triggers_gemini(self):
        """IMDb ve TMDB her ikisi de miss → Gemini fallback."""
        calls, matched = self._run_series_flow(imdb_matched=False, tmdb_matched=False)
        self.assertTrue(calls["imdb"])
        self.assertTrue(calls["tmdb"])
        self.assertTrue(calls["gemini"], "Her ikisi de miss — Gemini çağrılmalı")
        self.assertFalse(matched)


if __name__ == "__main__":
    unittest.main(verbosity=2)
