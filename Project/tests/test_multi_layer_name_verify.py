"""Tests for multi-layer name verification enhancements.

Covers:
  FT-01 .. FT-04  : TurkishNameDB._fuzzy_find_top2() contract + backward compat
  VS-01 .. VS-08  : verify_single_name() YES/NO parsing + fail-closed error paths
  GP-01 .. GP-10  : NameVerifier._gemini_pass2() gate / candidate branching
  VC-01 .. VC-04  : verify_crew() regression (existing behaviour preserved)
"""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_name_db(top2_return=None, is_name_return=False,
                       find_with_method_return=(None, 0.0, "")):
    """Return a mock TurkishNameDB-like object."""
    db = MagicMock()
    db._fuzzy_find_top2.return_value = top2_return or []
    db.is_name.return_value = is_name_return
    db.find_with_method.return_value = find_with_method_return
    # __bool__ must return True so `self._name_db is None` check works correctly
    db.__bool__ = lambda self: True
    db.__len__ = lambda self: 10
    return db


# ─────────────────────────────────────────────────────────────────────────────
# FT: _fuzzy_find_top2 contract
# ─────────────────────────────────────────────────────────────────────────────

class TestFuzzyFindTop2(unittest.TestCase):

    def _make_db_with_names(self, names):
        """Build a real TurkishNameDB stub with _all_names populated."""
        try:
            from rapidfuzz import fuzz, process as rf_process
        except ImportError:
            self.skipTest("rapidfuzz not installed")

        from core.turkish_name_db import TurkishNameDB
        db = TurkishNameDB.__new__(TurkishNameDB)
        db._all_names = names
        db._db_first = {}
        db._db_surname = {}
        db._all_keys = set()
        db._hardcoded = {}
        db._phonetic_index = {}
        return db

    def test_ft01_returns_list_of_tuples(self):
        """FT-01: _fuzzy_find_top2() returns list[tuple[str, float]]."""
        db = self._make_db_with_names(["Ahmet Yilmaz", "Mehmet Yilmaz"])
        result = db._fuzzy_find_top2("Ahmet Yilmaz", threshold=0)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], float)

    def test_ft02_returns_at_most_two(self):
        """FT-02: _fuzzy_find_top2() returns at most 2 candidates."""
        names = ["Ali Can", "Ali Demir", "Ali Kurt", "Ali Sen", "Ali Yıldız"]
        db = self._make_db_with_names(names)
        result = db._fuzzy_find_top2("Ali Can", threshold=0)
        self.assertLessEqual(len(result), 2)

    def test_ft03_scores_between_0_and_1(self):
        """FT-03: scores returned by _fuzzy_find_top2 are in [0, 1]."""
        db = self._make_db_with_names(["Ahmet Yilmaz", "Mehmet Demir"])
        result = db._fuzzy_find_top2("Ahmet Yilmaz", threshold=0)
        for _, score in result:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_ft04_existing_fuzzy_find_unchanged(self):
        """FT-04: _fuzzy_find() still returns single (str|None, float) tuple."""
        db = self._make_db_with_names(["Ahmet Yilmaz"])
        result = db._fuzzy_find("Ahmet Yilmaz", threshold=50)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_ft05_no_rapidfuzz_returns_empty(self):
        """FT-05: _fuzzy_find_top2() returns [] when rapidfuzz absent."""
        from core.turkish_name_db import TurkishNameDB
        db = TurkishNameDB.__new__(TurkishNameDB)
        db._all_names = ["Ahmet Yilmaz"]
        # Patch HAS_RAPIDFUZZ to False
        import core.turkish_name_db as _m
        old = _m.HAS_RAPIDFUZZ
        try:
            _m.HAS_RAPIDFUZZ = False
            result = db._fuzzy_find_top2("Ahmet Yilmaz")
            self.assertEqual(result, [])
        finally:
            _m.HAS_RAPIDFUZZ = old

    def test_ft06_empty_names_returns_empty(self):
        """FT-06: _fuzzy_find_top2() returns [] when _all_names is empty."""
        from core.turkish_name_db import TurkishNameDB
        db = TurkishNameDB.__new__(TurkishNameDB)
        db._all_names = []
        result = db._fuzzy_find_top2("Ahmet")
        self.assertEqual(result, [])


# ─────────────────────────────────────────────────────────────────────────────
# VS: verify_single_name() parsing + fail-closed paths
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifySingleName(unittest.TestCase):

    def _call(self, mock_response):
        """Call verify_single_name with a mocked llm_provider.generate."""
        fake_llm = types.ModuleType("core.llm_provider")
        fake_llm.generate = MagicMock(return_value=mock_response)
        with patch.dict("sys.modules", {"core.llm_provider": fake_llm}):
            from importlib import reload
            import core.gemini_crew_validator as mod
            reload(mod)
            return mod.verify_single_name("Test Name")

    def test_vs01_yes_response(self):
        """VS-01: 'YES' response → 'YES'."""
        self.assertEqual(self._call("YES"), "YES")

    def test_vs02_no_response(self):
        """VS-02: 'NO' response → 'NO'."""
        self.assertEqual(self._call("NO"), "NO")

    def test_vs03_yes_with_whitespace(self):
        """VS-03: '  yes  ' → 'YES' (strip + upper)."""
        self.assertEqual(self._call("  yes  "), "YES")

    def test_vs04_no_with_whitespace(self):
        """VS-04: '  no\n' → 'NO' (strip + upper)."""
        self.assertEqual(self._call("  no\n"), "NO")

    def test_vs05_invalid_response(self):
        """VS-05: 'MAYBE' → 'invalid_response' (not YES/NO)."""
        self.assertEqual(self._call("MAYBE"), "invalid_response")

    def test_vs06_empty_response(self):
        """VS-06: empty string → 'gemini_parse_error'."""
        self.assertEqual(self._call(""), "gemini_parse_error")

    def test_vs07_none_response(self):
        """VS-07: None response → 'gemini_parse_error'."""
        self.assertEqual(self._call(None), "gemini_parse_error")

    def test_vs08_timeout_error(self):
        """VS-08: TimeoutError → 'gemini_timeout' (fail-closed)."""
        fake_llm = types.ModuleType("core.llm_provider")
        fake_llm.generate = MagicMock(side_effect=TimeoutError("timeout"))
        with patch.dict("sys.modules", {"core.llm_provider": fake_llm}):
            from importlib import reload
            import core.gemini_crew_validator as mod
            reload(mod)
            result = mod.verify_single_name("Test Name")
        self.assertEqual(result, "gemini_timeout")

    def test_vs09_oserror(self):
        """VS-09: OSError → 'gemini_network_error' (fail-closed)."""
        fake_llm = types.ModuleType("core.llm_provider")
        fake_llm.generate = MagicMock(side_effect=OSError("connection refused"))
        with patch.dict("sys.modules", {"core.llm_provider": fake_llm}):
            from importlib import reload
            import core.gemini_crew_validator as mod
            reload(mod)
            result = mod.verify_single_name("Test Name")
        self.assertEqual(result, "gemini_network_error")

    def test_vs10_unexpected_exception(self):
        """VS-10: Unexpected exception → 'gemini_parse_error' (fail-closed)."""
        fake_llm = types.ModuleType("core.llm_provider")
        fake_llm.generate = MagicMock(side_effect=RuntimeError("unexpected"))
        with patch.dict("sys.modules", {"core.llm_provider": fake_llm}):
            from importlib import reload
            import core.gemini_crew_validator as mod
            reload(mod)
            result = mod.verify_single_name("Test Name")
        self.assertEqual(result, "gemini_parse_error")

    def test_vs11_validate_crew_still_works(self):
        """VS-11: validate_crew_with_gemini() is still importable and callable."""
        from core.gemini_crew_validator import validate_crew_with_gemini
        self.assertTrue(callable(validate_crew_with_gemini))

    def test_vs12_yes_only_exact(self):
        """VS-12: 'YES please' is not exact YES → invalid_response."""
        self.assertEqual(self._call("YES please"), "invalid_response")


# ─────────────────────────────────────────────────────────────────────────────
# GP: _gemini_pass2() gate / candidate branching
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiPass2(unittest.TestCase):

    def _make_verifier(self, name_db=None, gemini_response_sequence=None):
        """Create a NameVerifier with mocked internals."""
        from core.name_verify import NameVerifier
        verifier = NameVerifier(name_db=name_db)

        if gemini_response_sequence is not None:
            responses = iter(gemini_response_sequence)

            def _fake_vsn(name, gemini_model="gemini-2.5-flash"):
                return next(responses)

            import core.name_verify as nv_mod
            self._orig_vsn = getattr(nv_mod, "_vsn_override", None)
            # Patch via the import path used inside _gemini_pass2
            self._patcher = patch(
                "core.gemini_crew_validator.verify_single_name",
                side_effect=_fake_vsn,
            )
            self._patcher.start()

        return verifier

    def tearDown(self):
        if hasattr(self, "_patcher"):
            self._patcher.stop()

    def test_gp01_none_name_db_returns_none(self):
        """GP-01: name_db is None → _gemini_pass2 returns None immediately."""
        from core.name_verify import NameVerifier
        verifier = NameVerifier(name_db=None)
        result = verifier._gemini_pass2("YÖNETMEN", "Ahmet Yilmaz")
        self.assertIsNone(result)

    def test_gp02_no_candidates_above_gate(self):
        """GP-02: no fuzzy candidates above gate → returns None."""
        db = _make_fake_name_db(top2_return=[])
        verifier = self._make_verifier(name_db=db)
        result = verifier._gemini_pass2("YÖNETMEN", "xyz123")
        self.assertIsNone(result)

    def test_gp03_cand1_yes_returns_cand1(self):
        """GP-03: cand1 passes gate and Gemini says YES → returns cand1."""
        db = _make_fake_name_db(top2_return=[("Ahmet Yilmaz", 0.85)])
        verifier = self._make_verifier(name_db=db, gemini_response_sequence=["YES"])
        result = verifier._gemini_pass2("YÖNETMEN", "Ahmet Yilmaz")
        self.assertEqual(result, "Ahmet Yilmaz")

    def test_gp04_cand1_no_cand2_yes_returns_cand2(self):
        """GP-04: cand1 NO, cand2 passes gate and YES → returns cand2."""
        db = _make_fake_name_db(
            top2_return=[("Ahmet Yilmaz", 0.85), ("Mehmet Yilmaz", 0.80)]
        )
        verifier = self._make_verifier(name_db=db, gemini_response_sequence=["NO", "YES"])
        result = verifier._gemini_pass2("YÖNETMEN", "Ahmet Yilmz")
        self.assertEqual(result, "Mehmet Yilmaz")

    def test_gp05_both_no_returns_none(self):
        """GP-05: both candidates NO → returns None."""
        db = _make_fake_name_db(
            top2_return=[("Ahmet Yilmaz", 0.85), ("Mehmet Yilmaz", 0.80)]
        )
        verifier = self._make_verifier(name_db=db, gemini_response_sequence=["NO", "NO"])
        result = verifier._gemini_pass2("YÖNETMEN", "Ahmet Yilmz")
        self.assertIsNone(result)

    def test_gp06_cand1_no_cand2_below_gate_returns_none(self):
        """GP-06: cand1 NO, cand2 below gate → cand2 not attempted, returns None."""
        # cand2 score 0.65 < 0.72 gate
        db = _make_fake_name_db(
            top2_return=[("Ahmet Yilmaz", 0.85), ("Ali Demir", 0.65)]
        )
        verifier = self._make_verifier(name_db=db, gemini_response_sequence=["NO"])
        result = verifier._gemini_pass2("YÖNETMEN", "Ahmet Yilmz")
        self.assertIsNone(result)

    def test_gp07_gemini_timeout_fail_closed(self):
        """GP-07: Gemini timeout on cand1 → treated as rejection, returns None."""
        db = _make_fake_name_db(top2_return=[("Ahmet Yilmaz", 0.85)])
        verifier = self._make_verifier(
            name_db=db, gemini_response_sequence=["gemini_timeout"]
        )
        result = verifier._gemini_pass2("YÖNETMEN", "Ahmet Yilmz")
        self.assertIsNone(result)

    def test_gp08_gemini_network_error_fail_closed(self):
        """GP-08: gemini_network_error on cand1 → treated as rejection, returns None."""
        db = _make_fake_name_db(top2_return=[("Ahmet Yilmaz", 0.85)])
        verifier = self._make_verifier(
            name_db=db, gemini_response_sequence=["gemini_network_error"]
        )
        result = verifier._gemini_pass2("YÖNETMEN", "Ahmet Yilmz")
        self.assertIsNone(result)

    def test_gp09_invalid_response_treated_as_rejection(self):
        """GP-09: invalid_response on cand1 → treated as rejection → try cand2."""
        db = _make_fake_name_db(
            top2_return=[("Ahmet Yilmaz", 0.85), ("Mehmet Yilmaz", 0.80)]
        )
        verifier = self._make_verifier(
            name_db=db, gemini_response_sequence=["invalid_response", "YES"]
        )
        result = verifier._gemini_pass2("YÖNETMEN", "Ahmet Yilmz")
        self.assertEqual(result, "Mehmet Yilmaz")

    def test_gp10_only_two_candidates_max(self):
        """GP-10: Even with 2 candidates, exactly 2 Gemini calls at most."""
        call_count = [0]

        db = _make_fake_name_db(
            top2_return=[("Ahmet Yilmaz", 0.85), ("Mehmet Yilmaz", 0.80)]
        )
        from core.name_verify import NameVerifier
        verifier = NameVerifier(name_db=db)

        def counting_vsn(name, gemini_model="gemini-2.5-flash"):
            call_count[0] += 1
            return "NO"

        with patch("core.gemini_crew_validator.verify_single_name", side_effect=counting_vsn):
            verifier._gemini_pass2("YÖNETMEN", "Ahmet Yilmz")

        self.assertLessEqual(call_count[0], 2)

    def test_gp11_name_db_zero_len_not_none_proceeds(self):
        """GP-11: name_db with __len__==0 is NOT None — should proceed (bug-fix check)."""
        # The bug was: `not self._name_db` evaluated True when len==0,
        # short-circuiting _gemini_pass2 even with a valid DB object.
        # After fix: check is `self._name_db is None`.
        db = _make_fake_name_db(top2_return=[("Ahmet Yilmaz", 0.85)])
        db.__len__ = lambda self: 0  # Simulate empty DB
        from core.name_verify import NameVerifier
        verifier = NameVerifier(name_db=db)

        with patch(
            "core.gemini_crew_validator.verify_single_name", return_value="YES"
        ):
            result = verifier._gemini_pass2("YÖNETMEN", "Ahmet Yilmaz")

        # Should NOT have short-circuited — should have attempted Gemini
        self.assertEqual(result, "Ahmet Yilmaz")


# ─────────────────────────────────────────────────────────────────────────────
# VC: verify_crew() regression — existing behaviour preserved
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyCrewRegression(unittest.TestCase):

    def _make_verifier_no_db_no_tmdb(self):
        from core.name_verify import NameVerifier
        return NameVerifier(name_db=None, tmdb_client=None)

    def test_vc01_blacklisted_name_dropped(self):
        """VC-01: Blacklisted name is dropped from crew result."""
        verifier = self._make_verifier_no_db_no_tmdb()
        result = verifier.verify_crew({"YÖNETMEN": ["technicolor"]})
        self.assertEqual(result.get("YÖNETMEN", []), [])

    def test_vc02_structural_failure_dropped(self):
        """VC-02: Structurally invalid name (too short) is dropped."""
        verifier = self._make_verifier_no_db_no_tmdb()
        result = verifier.verify_crew({"YÖNETMEN": ["AB"]})
        self.assertEqual(result.get("YÖNETMEN", []), [])

    def test_vc03_unverified_no_alternative_flagged_in_result(self):
        """VC-03: Unverified name with no alternative is still in result (flagged)."""
        verifier = self._make_verifier_no_db_no_tmdb()
        # No DB, no TMDB, no Gemini → _gemini_pass2 returns None → flag & include
        result = verifier.verify_crew({"YÖNETMEN": ["John Smith"]})
        self.assertIn("John Smith", result.get("YÖNETMEN", []))

    def test_vc04_verified_name_kept(self):
        """VC-04: Name verified by NameDB is kept and returned."""
        db = _make_fake_name_db(is_name_return=True)
        from core.name_verify import NameVerifier
        verifier = NameVerifier(name_db=db, tmdb_client=None)
        result = verifier.verify_crew({"YÖNETMEN": ["Ahmet Yilmaz"]})
        self.assertIn("Ahmet Yilmaz", result.get("YÖNETMEN", []))

    def test_vc05_verified_name_present_unverified_dropped(self):
        """VC-05: When verified name exists in role, unverified candidate is dropped."""
        db = MagicMock()
        # First call (Ahmet Yilmaz) → is_name True; second (Garbage Xyz) → is_name False
        db.is_name.side_effect = lambda n: n == "Ahmet Yilmaz"
        db.find_with_method.return_value = (None, 0.0, "")
        db.__bool__ = lambda self: True
        db.__len__ = lambda self: 10
        db._fuzzy_find_top2.return_value = []

        from core.name_verify import NameVerifier
        verifier = NameVerifier(name_db=db, tmdb_client=None)
        result = verifier.verify_crew({"YÖNETMEN": ["Ahmet Yilmaz", "Garbage Xyz"]})
        names = result.get("YÖNETMEN", [])
        self.assertIn("Ahmet Yilmaz", names)
        self.assertNotIn("Garbage Xyz", names)

    def test_vc06_gemini_pass2_accepts_cand_when_verified(self):
        """VC-06: verify_crew Pass 2 uses _gemini_pass2; accepted name ends up in result."""
        db = _make_fake_name_db(
            is_name_return=False,
            find_with_method_return=(None, 0.0, ""),
            top2_return=[("Ahmet Yilmaz", 0.85)],
        )
        from core.name_verify import NameVerifier
        verifier = NameVerifier(name_db=db, tmdb_client=None)

        with patch(
            "core.gemini_crew_validator.verify_single_name", return_value="YES"
        ):
            result = verifier.verify_crew({"YÖNETMEN": ["Ahmet Yilmz"]})

        self.assertIn("Ahmet Yilmaz", result.get("YÖNETMEN", []))

    def test_vc07_gemini_pass2_rejected_name_flagged_as_unresolved(self):
        """VC-07: verify_crew Pass 2 rejected by Gemini → unresolved (raw OCR in result)."""
        db = _make_fake_name_db(
            is_name_return=False,
            find_with_method_return=(None, 0.0, ""),
            top2_return=[("Ahmet Yilmaz", 0.85)],
        )
        from core.name_verify import NameVerifier
        verifier = NameVerifier(name_db=db, tmdb_client=None)

        with patch(
            "core.gemini_crew_validator.verify_single_name", return_value="NO"
        ):
            result = verifier.verify_crew({"YÖNETMEN": ["Ahmet Yilmz"]})

        # Raw OCR should be preserved in result (flagged as unresolved)
        names = result.get("YÖNETMEN", [])
        self.assertIn("Ahmet Yilmz", names)

    def test_vc08_log_contains_gemini_pass2_entry(self):
        """VC-08: After verify_crew, verification log contains GEMINI_PASS2 entry."""
        db = _make_fake_name_db(
            is_name_return=False,
            find_with_method_return=(None, 0.0, ""),
            top2_return=[("Ahmet Yilmaz", 0.85)],
        )
        from core.name_verify import NameVerifier
        verifier = NameVerifier(name_db=db, tmdb_client=None)

        with patch(
            "core.gemini_crew_validator.verify_single_name", return_value="YES"
        ):
            verifier.verify_crew({"YÖNETMEN": ["Ahmet Yilmz"]})

        log = verifier.get_log()
        layers = [e["layer"] for e in log]
        self.assertIn("GEMINI_PASS2", layers)


if __name__ == "__main__":
    unittest.main()
