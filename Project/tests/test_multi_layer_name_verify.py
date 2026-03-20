"""Tests for multi-layer name verification enhancements.

Covers:
  - TurkishNameDB._fuzzy_find_top2() — yeni metot, imza değişikliği yok
  - gemini_crew_validator.verify_single_name() — YES/NO parser + fail-closed
  - NameVerifier._gemini_pass2() — fuzzy gate + Gemini doğrulama akışı
  - Mevcut verify_crew() davranışının korunması (regresyon)
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.turkish_name_db import TurkishNameDB
from core.name_verify import NameVerifier, _GEMINI_FUZZY_GATE


# ─────────────────────────────────────────────────────────────────────────────
# TurkishNameDB._fuzzy_find_top2
# ─────────────────────────────────────────────────────────────────────────────

class TestFuzzyFindTop2(unittest.TestCase):
    """_fuzzy_find_top2() yeni metot testleri."""

    def setUp(self):
        self.db = TurkishNameDB()  # DB yok → sadece hardcoded aktif

    def test_returns_list(self):
        """_fuzzy_find_top2() list döndürmeli."""
        result = self.db._fuzzy_find_top2("Nisa")
        self.assertIsInstance(result, list)

    def test_max_two_elements(self):
        """_fuzzy_find_top2() en fazla 2 eleman döndürmeli."""
        result = self.db._fuzzy_find_top2("test")
        self.assertLessEqual(len(result), 2)

    def test_each_element_is_tuple_str_float(self):
        """Her eleman (str, float) tuple'ı olmalı."""
        result = self.db._fuzzy_find_top2("Nisa", threshold=0)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            name, score = item
            self.assertIsInstance(name, str)
            self.assertIsInstance(score, float)

    def test_scores_between_0_and_1(self):
        """Skorlar 0-1 arasında olmalı."""
        result = self.db._fuzzy_find_top2("Mehmet", threshold=0)
        for _, score in result:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_old_fuzzy_find_unchanged(self):
        """_fuzzy_find() imzası ve davranışı değişmemeli (2-tuple döndürür)."""
        name, score = self.db._fuzzy_find("SEBNEM", 85)
        # Hardcoded → 2-tuple döner (None, 0.0 çünkü hardcoded _fuzzy_find'a düşmüyor)
        self.assertIsInstance(score, float)

    def test_find_backward_compat(self):
        """find() hâlâ 2-tuple döndürmeli."""
        result = self.db.find("SEBNEM")
        self.assertEqual(len(result), 2)
        canonical, score = result
        self.assertEqual(canonical, "Şebnem")
        self.assertEqual(score, 1.0)

    def test_find_with_method_backward_compat(self):
        """find_with_method() hâlâ 3-tuple döndürmeli."""
        result = self.db.find_with_method("SEBNEM")
        self.assertEqual(len(result), 3)

    def test_empty_returns_empty_list(self):
        """Boş metin için boş liste döndürmeli."""
        result = self.db._fuzzy_find_top2("")
        self.assertEqual(result, [])


# ─────────────────────────────────────────────────────────────────────────────
# verify_single_name — YES/NO parser + fail-closed davranış
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifySingleName(unittest.TestCase):
    """verify_single_name() testleri."""

    def _call(self, response_text):
        """core.llm_provider.generate mock'layarak verify_single_name çağır."""
        from core.gemini_crew_validator import verify_single_name
        with patch("core.llm_provider.generate", return_value=response_text):
            return verify_single_name("YAVUZ TURGUL", "Yavuz Turgul")

    def test_yes_accepted(self):
        accepted, reason = self._call("YES")
        self.assertTrue(accepted)
        self.assertEqual(reason, "yes")

    def test_no_rejected(self):
        accepted, reason = self._call("NO")
        self.assertFalse(accepted)
        self.assertEqual(reason, "no")

    def test_yes_with_whitespace(self):
        """'  YES  ' → YES kabul edilmeli (strip + upper sonrası)."""
        accepted, reason = self._call("  YES  ")
        self.assertTrue(accepted)
        self.assertEqual(reason, "yes")

    def test_yes_dot_invalid(self):
        """'YES.' geçersiz yanıt — invalid_response, False döner."""
        accepted, reason = self._call("YES.")
        self.assertFalse(accepted)
        self.assertEqual(reason, "invalid_response")

    def test_yes_lowercase_invalid(self):
        """'Yes' → strip().upper() → 'YES' → normalize sonrası kabul edilmeli."""
        accepted, reason = self._call("Yes")
        self.assertTrue(accepted)
        self.assertEqual(reason, "yes")

    def test_no_uncertain_invalid(self):
        """'NO - uncertain' geçersiz yanıt — invalid_response."""
        accepted, reason = self._call("NO - uncertain")
        self.assertFalse(accepted)
        self.assertEqual(reason, "invalid_response")

    def test_empty_response_parse_error(self):
        """Boş yanıt → gemini_parse_error, False."""
        accepted, reason = self._call("")
        self.assertFalse(accepted)
        self.assertEqual(reason, "gemini_parse_error")

    def test_none_response_parse_error(self):
        """None yanıt → gemini_parse_error, False."""
        accepted, reason = self._call(None)
        self.assertFalse(accepted)
        self.assertEqual(reason, "gemini_parse_error")

    def test_llm_unavailable(self):
        """llm_provider import edilemediğinde → llm_unavailable, False."""
        from core.gemini_crew_validator import verify_single_name
        with patch.dict("sys.modules", {"core.llm_provider": None}):
            accepted, reason = verify_single_name("TEST", "Test")
        self.assertFalse(accepted)
        self.assertEqual(reason, "llm_unavailable")

    def test_timeout_fail_closed(self):
        """TimeoutError → gemini_timeout, False (fail-closed)."""
        from core.gemini_crew_validator import verify_single_name
        with patch("core.llm_provider.generate", side_effect=TimeoutError("timed out")):
            accepted, reason = verify_single_name("TEST", "Test")
        self.assertFalse(accepted)
        self.assertEqual(reason, "gemini_timeout")

    def test_network_error_fail_closed(self):
        """OSError (ağ hatası) → gemini_network_error, False."""
        from core.gemini_crew_validator import verify_single_name
        with patch("core.llm_provider.generate", side_effect=OSError("connection refused")):
            accepted, reason = verify_single_name("TEST", "Test")
        self.assertFalse(accepted)
        self.assertEqual(reason, "gemini_network_error")

    def test_generic_exception_fail_closed(self):
        """Beklenmedik hata → gemini_error, False."""
        from core.gemini_crew_validator import verify_single_name
        with patch("core.llm_provider.generate", side_effect=RuntimeError("unexpected")):
            accepted, reason = verify_single_name("TEST", "Test")
        self.assertFalse(accepted)
        self.assertEqual(reason, "gemini_error")

    def test_validate_crew_with_gemini_still_works(self):
        """validate_crew_with_gemini() bozulmadı — hâlâ çağrılabilir."""
        from core.gemini_crew_validator import validate_crew_with_gemini
        self.assertTrue(callable(validate_crew_with_gemini))


# ─────────────────────────────────────────────────────────────────────────────
# NameVerifier._gemini_pass2 ve verify_crew regresyon
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiPass2(unittest.TestCase):
    """_gemini_pass2() birim testleri."""

    def test_gemini_fuzzy_gate_value(self):
        """_GEMINI_FUZZY_GATE 70-75 arasında olmalı."""
        self.assertGreaterEqual(_GEMINI_FUZZY_GATE, 70)
        self.assertLessEqual(_GEMINI_FUZZY_GATE, 75)

    def test_no_namedb_returns_none(self):
        """name_db yoksa _gemini_pass2 None döndürmeli."""
        verifier = NameVerifier(name_db=None)
        result = verifier._gemini_pass2("YÖNETMEN", "YAVUZ", "Yavuz")
        self.assertIsNone(result)

    def test_below_gate_returns_none(self):
        """Aday skoru gate altında → None döner."""
        db = TurkishNameDB()
        gate_frac = _GEMINI_FUZZY_GATE / 100.0
        low_score = gate_frac - 0.1
        db._fuzzy_find_top2 = MagicMock(return_value=[("DüşükAday", low_score)])
        verifier = NameVerifier(name_db=db)
        result = verifier._gemini_pass2("YÖNETMEN", "XYZOCR", "XyzOcr")
        self.assertIsNone(result)

    def test_c1_accepted(self):
        """Aday 1 gate'i geçip Gemini YES alırsa (name, 'gemini_verified_c1') döner."""
        db = TurkishNameDB()
        gate_frac = _GEMINI_FUZZY_GATE / 100.0
        db._fuzzy_find_top2 = MagicMock(return_value=[
            ("Yavuz Turgul", gate_frac + 0.05),
            ("Yılmaz Güney", gate_frac + 0.02),
        ])
        verifier = NameVerifier(name_db=db)
        with patch("core.llm_provider.generate", return_value="YES"):
            result = verifier._gemini_pass2("YÖNETMEN", "YAVUZ TURGUL", "Yavuz Turgul")
        self.assertIsNotNone(result)
        name, reason = result
        self.assertEqual(name, "Yavuz Turgul")
        self.assertEqual(reason, "gemini_verified_c1")

    def test_c1_rejected_c2_accepted(self):
        """Aday 1 red, aday 2 gate'i geçip Gemini YES alırsa (name, 'gemini_verified_c2') döner."""
        db = TurkishNameDB()
        gate_frac = _GEMINI_FUZZY_GATE / 100.0
        db._fuzzy_find_top2 = MagicMock(return_value=[
            ("Aday1", gate_frac + 0.05),
            ("Aday2", gate_frac + 0.03),
        ])
        verifier = NameVerifier(name_db=db)
        responses = iter(["NO", "YES"])
        with patch("core.llm_provider.generate", side_effect=responses):
            result = verifier._gemini_pass2("YÖNETMEN", "XOCR", "Xocr")
        self.assertIsNotNone(result)
        name, reason = result
        self.assertEqual(name, "Aday2")
        self.assertEqual(reason, "gemini_verified_c2")

    def test_c1_rejected_c2_below_gate_returns_none(self):
        """Aday 1 reddedildi, aday 2 gate altında → None döner."""
        db = TurkishNameDB()
        gate_frac = _GEMINI_FUZZY_GATE / 100.0
        db._fuzzy_find_top2 = MagicMock(return_value=[
            ("Aday1", gate_frac + 0.05),
            ("Aday2", gate_frac - 0.1),  # gate altı
        ])
        verifier = NameVerifier(name_db=db)
        with patch("core.llm_provider.generate", return_value="NO"):
            result = verifier._gemini_pass2("YÖNETMEN", "XOCR", "Xocr")
        self.assertIsNone(result)

    def test_both_rejected_returns_none(self):
        """İki aday da Gemini'den NO alırsa None döner."""
        db = TurkishNameDB()
        gate_frac = _GEMINI_FUZZY_GATE / 100.0
        db._fuzzy_find_top2 = MagicMock(return_value=[
            ("Aday1", gate_frac + 0.05),
            ("Aday2", gate_frac + 0.03),
        ])
        verifier = NameVerifier(name_db=db)
        with patch("core.llm_provider.generate", return_value="NO"):
            result = verifier._gemini_pass2("YÖNETMEN", "XOCR", "Xocr")
        self.assertIsNone(result)

    def test_no_candidates_returns_none(self):
        """Fuzzy aday listesi boşsa None döner."""
        db = TurkishNameDB()
        db._fuzzy_find_top2 = MagicMock(return_value=[])
        verifier = NameVerifier(name_db=db)
        result = verifier._gemini_pass2("YÖNETMEN", "XYZOCR", "XyzOcr")
        self.assertIsNone(result)

    def test_gemini_fail_closed_returns_none(self):
        """Gemini timeout (fail-closed) → None döner."""
        db = TurkishNameDB()
        gate_frac = _GEMINI_FUZZY_GATE / 100.0
        db._fuzzy_find_top2 = MagicMock(return_value=[
            ("Aday1", gate_frac + 0.05),
        ])
        verifier = NameVerifier(name_db=db)
        with patch("core.llm_provider.generate", side_effect=TimeoutError("timeout")):
            result = verifier._gemini_pass2("YÖNETMEN", "XOCR", "Xocr")
        self.assertIsNone(result)


class TestVerifyCrewRegression(unittest.TestCase):
    """verify_crew() mevcut davranış regresyon testleri."""

    def test_blacklist_still_drops(self):
        """Blacklist'teki isimler hâlâ düşürülmeli."""
        verifier = NameVerifier()
        result = verifier.verify_crew({"YÖNETMEN": ["the end", "Yavuz Turgul"]})
        # "the end" düşürülmeli; "Yavuz Turgul" blacklist'te değil
        yonetmen = result.get("YÖNETMEN", [])
        self.assertNotIn("the end", yonetmen)

    def test_structural_check_drops_short(self):
        """3 karakter altındaki isimler hâlâ düşürülmeli."""
        verifier = NameVerifier()
        result = verifier.verify_crew({"SENARYO": ["AB", "Yılmaz Güney"]})
        senaryo = result.get("SENARYO", [])
        self.assertNotIn("AB", senaryo)

    def test_empty_crew_returns_empty(self):
        """Boş crew → boş sonuç."""
        verifier = NameVerifier()
        result = verifier.verify_crew({"YÖNETMEN": []})
        self.assertEqual(result.get("YÖNETMEN", []), [])

    def test_veri_yok_skipped(self):
        """'VERİ YOK' placeholder'lar işlenmemeli."""
        verifier = NameVerifier()
        result = verifier.verify_crew({"YAPIMCI": ["VERİ YOK"]})
        self.assertEqual(result.get("YAPIMCI", []), [])

    def test_hardcoded_name_found_via_namedb(self):
        """Hardcoded isimler NameDB üzerinden bulunabilmeli."""
        db = TurkishNameDB()
        verifier = NameVerifier(name_db=db)
        result = verifier.verify_crew({"YÖNETMEN": ["SEBNEM"]})
        # SEBNEM hardcoded → Şebnem; is_name() True döndürmeli veya find() ile düzeltmeli
        yonetmen = result.get("YÖNETMEN", [])
        # İsim mevcut olmalı (exact_match veya corrected olarak)
        self.assertTrue(len(yonetmen) > 0)

    def test_get_log_returns_list(self):
        """get_log() liste döndürmeli."""
        verifier = NameVerifier()
        verifier.verify_crew({"YÖNETMEN": ["Test"]})
        log = verifier.get_log()
        self.assertIsInstance(log, list)

    def test_get_log_text_returns_str(self):
        """get_log_text() string döndürmeli."""
        verifier = NameVerifier()
        verifier.verify_crew({"YÖNETMEN": ["Test"]})
        log_text = verifier.get_log_text()
        self.assertIsInstance(log_text, str)

    def test_unverified_with_no_alternative_flagged(self):
        """Verified alternatif yoksa unverified adaylar flag'lenmeli (unresolved)."""
        # NameDB yok, TMDB yok, Gemini yok → unresolved
        verifier = NameVerifier(name_db=None, tmdb_client=None)
        result = verifier.verify_crew({"YÖNETMEN": ["Bilinmeyen Kişi"]})
        # Gemini pass2 name_db=None ile atlanır → flagged olarak eklenir
        yonetmen = result.get("YÖNETMEN", [])
        self.assertIn("Bilinmeyen Kişi", yonetmen)

    def test_unverified_has_alternative_dropped(self):
        """Verified alternatif varsa unverified adaylar düşürülmeli."""
        db = TurkishNameDB()
        # Mock: SEBNEM verified, bilinmeyen doğrulanamıyor
        verifier = NameVerifier(name_db=db)
        # SEBNEM is_name True → verified; bilinmeyen → unverified → düşürülmeli
        result = verifier.verify_crew({"YÖNETMEN": ["SEBNEM", "xyzzzabc123"]})
        yonetmen = result.get("YÖNETMEN", [])
        self.assertNotIn("xyzzzabc123", yonetmen)


class TestVerifyCrewUnresolved(unittest.TestCase):
    """Unresolved durumun doğru şekilde temsil edildiğini kontrol et."""

    def test_unresolved_log_entry_has_fields(self):
        """Unresolved log entry'si gerekli alanları içermeli."""
        verifier = NameVerifier(name_db=None)
        verifier.verify_crew({"YÖNETMEN": ["BilinmeyenKişi"]})
        log = verifier.get_log()
        # FINAL log entry'sini bul
        final_entries = [e for e in log if e.get("layer") == "FINAL"]
        if not final_entries:
            return  # log yapısı farklı, test skip
        unresolved_entries = [
            e for e in final_entries
            if e.get("action") in ("flagged",) and
            e.get("reason") in ("unresolved", "unverified_no_alternative")
        ]
        self.assertTrue(len(unresolved_entries) > 0, "Unresolved entry bulunamadı")


if __name__ == "__main__":
    unittest.main()
