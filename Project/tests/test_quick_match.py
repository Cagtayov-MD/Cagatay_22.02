"""
test_quick_match.py — QuickMatcher sınıfı için birim testler.
"""
import os
import sys

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import unittest
from unittest.mock import MagicMock, patch


class TestQuickMatcherIsFilm(unittest.TestCase):
    """_is_film metodu: film_id 3. bloğuna göre dizi/film ayrımı."""

    def setUp(self):
        from core.quick_match import QuickMatcher
        self.qm = QuickMatcher()

    def test_film_third_block_one_returns_true(self):
        assert self.qm._is_film("1955-0019-1-0000-00-1") is True

    def test_series_third_block_zero_returns_false(self):
        assert self.qm._is_film("1985-0128-0-0004-00-1") is False

    def test_empty_film_id_defaults_to_true(self):
        assert self.qm._is_film("") is True

    def test_short_film_id_defaults_to_true(self):
        assert self.qm._is_film("1955") is True

    def test_third_block_non_zero_non_one_returns_true(self):
        # 3. blok 0 değilse film sayılır
        assert self.qm._is_film("1955-0019-2-0000-00-1") is True


class TestQuickMatcherExtractKeyInfo(unittest.TestCase):
    """_extract_key_info metodu: cdata'dan bilgi çıkarma."""

    def setUp(self):
        from core.quick_match import QuickMatcher
        self.qm = QuickMatcher()

    def test_extracts_film_title(self):
        cdata = {"film_title": "Kül Kedisi", "directors": [], "cast": []}
        title, directors, top_cast = self.qm._extract_key_info(cdata)
        assert title == "Kül Kedisi"

    def test_extracts_director_from_string(self):
        cdata = {"film_title": "", "directors": ["Charles Walters"], "cast": []}
        _, directors, _ = self.qm._extract_key_info(cdata)
        assert directors == ["Charles Walters"]

    def test_extracts_director_from_dict(self):
        cdata = {
            "film_title": "",
            "directors": [{"name": "Charles Walters", "role": "Director"}],
            "cast": [],
        }
        _, directors, _ = self.qm._extract_key_info(cdata)
        assert directors == ["Charles Walters"]

    def test_returns_top_2_cast_by_confidence(self):
        cdata = {
            "film_title": "",
            "directors": [],
            "cast": [
                {"actor_name": "Actor C", "confidence": 0.7},
                {"actor_name": "Actor A", "confidence": 0.99},
                {"actor_name": "Actor B", "confidence": 0.95},
            ],
        }
        _, _, top_cast = self.qm._extract_key_info(cdata)
        assert top_cast == ["Actor A", "Actor B"]
        assert len(top_cast) == 2

    def test_returns_max_2_cast(self):
        cdata = {
            "film_title": "",
            "directors": [],
            "cast": [
                {"actor_name": "Actor A", "confidence": 0.99},
                {"actor_name": "Actor B", "confidence": 0.95},
                {"actor_name": "Actor C", "confidence": 0.90},
                {"actor_name": "Actor D", "confidence": 0.85},
            ],
        }
        _, _, top_cast = self.qm._extract_key_info(cdata)
        assert len(top_cast) == 2

    def test_empty_cast_returns_empty_list(self):
        cdata = {"film_title": "", "directors": [], "cast": []}
        _, _, top_cast = self.qm._extract_key_info(cdata)
        assert top_cast == []

    def test_missing_keys_dont_raise(self):
        cdata = {}
        title, directors, top_cast = self.qm._extract_key_info(cdata)
        assert title == ""
        assert directors == []
        assert top_cast == []


class TestQuickMatcherMatchNoTMDB(unittest.TestCase):
    """match() metodu: TMDB client yokken fallback davranışı."""

    def setUp(self):
        from core.quick_match import QuickMatcher
        self.qm = QuickMatcher(tmdb_client=None, gemini_api_key="")

    def test_no_tmdb_returns_fallback(self):
        cdata = {
            "film_title": "Test Film",
            "directors": ["Test Director"],
            "cast": [{"actor_name": "Test Actor", "confidence": 0.9}],
        }
        result = self.qm.match(cdata, [], "1955-0019-1-0000-00-1")
        assert result["matched"] is False
        assert result["method"] == "fallback_name_verify"
        assert "cdata" in result

    def test_no_tmdb_sets_quick_match_fields(self):
        cdata = {"film_title": "Test Film", "directors": [], "cast": []}
        result = self.qm.match(cdata, [], "1955-0019-1-0000-00-1")
        assert result["cdata"].get("quick_match_method") == "none"
        assert result["cdata"].get("quick_match_skipped_name_verify") is False

    def test_series_no_tmdb_no_gemini_fallback(self):
        """Dizilerde Katman 3 (Gemini) atlanmalı."""
        cdata = {"film_title": "Amanda", "directors": [], "cast": []}
        result = self.qm.match(cdata, [], "1985-0128-0-0004-00-1")
        assert result["matched"] is False
        assert result["method"] == "fallback_name_verify"


class TestQuickMatcherApplyTmdbCredits(unittest.TestCase):
    """_apply_tmdb_credits metodu: TMDB verisiyle cdata güncelleme."""

    def setUp(self):
        from core.quick_match import QuickMatcher
        self.qm = QuickMatcher()

    def test_updates_cast(self):
        cdata = {"film_title": "Old Title", "cast": [], "crew": []}
        entry = {"id": 139799, "title": "The Glass Slipper", "media_type": "movie"}
        credits = {
            "cast": [
                {"name": "Leslie Caron", "character": "Ella", "order": 0},
                {"name": "Michael Wilding", "character": "Prince Charles", "order": 1},
            ],
            "crew": [],
        }
        updated = self.qm._apply_tmdb_credits(cdata, entry, "movie", credits)
        assert len(updated["cast"]) == 2
        assert updated["cast"][0]["actor_name"] == "Leslie Caron"
        assert updated["cast"][0]["is_tmdb_verified"] is True
        assert updated["cast"][0]["frame"] == "tmdb"

    def test_updates_crew(self):
        cdata = {"film_title": "", "cast": [], "crew": []}
        entry = {"id": 139799, "title": "The Glass Slipper"}
        credits = {
            "cast": [],
            "crew": [
                {"name": "Charles Walters", "job": "Director"},
            ],
        }
        updated = self.qm._apply_tmdb_credits(cdata, entry, "movie", credits)
        assert len(updated["crew"]) == 1
        assert updated["crew"][0]["name"] == "Charles Walters"
        assert updated["crew"][0]["is_tmdb_verified"] is True

    def test_updates_film_title(self):
        cdata = {"film_title": "Kül Kedisi", "cast": [], "crew": []}
        entry = {"id": 139799, "title": "The Glass Slipper"}
        credits = {"cast": [], "crew": []}
        updated = self.qm._apply_tmdb_credits(cdata, entry, "movie", credits)
        assert updated["film_title"] == "The Glass Slipper"
        assert updated.get("ocr_title") == "Kül Kedisi"

    def test_sets_verification_status(self):
        cdata = {"film_title": "", "cast": [], "crew": []}
        entry = {"id": 1, "title": "Film"}
        credits = {"cast": [], "crew": []}
        updated = self.qm._apply_tmdb_credits(cdata, entry, "movie", credits)
        assert updated["verification_status"] == "tmdb_verified"

    def test_sets_tmdb_id(self):
        cdata = {"film_title": "", "cast": [], "crew": []}
        entry = {"id": 999, "title": "Film"}
        credits = {"cast": [], "crew": []}
        updated = self.qm._apply_tmdb_credits(cdata, entry, "movie", credits)
        assert updated["tmdb_id"] == 999


class TestQuickMatcherMatchWithMockedTMDB(unittest.TestCase):
    """match() metodu: Mock TMDB ile başarılı eşleşme senaryoları."""

    def _make_mock_tmdb_client(self, enabled=True):
        mock_client = MagicMock()
        mock_client.enabled.return_value = enabled
        mock_client.api_key = "fake_api_key"
        mock_client.bearer = ""
        mock_client.language = "tr-TR"
        return mock_client

    def test_quick_tmdb_match_sets_matched_true(self):
        """_find_tmdb_entry eşleşme bulduğunda matched=True dönmeli."""
        from core.quick_match import QuickMatcher

        mock_entry = {"id": 139799, "title": "The Glass Slipper"}
        mock_credits = {
            "cast": [{"name": "Leslie Caron", "character": "Ella", "order": 0}],
            "crew": [{"name": "Charles Walters", "job": "Director"}],
        }

        qm = QuickMatcher(
            tmdb_client=self._make_mock_tmdb_client(),
            gemini_api_key="",
        )

        with patch("core.quick_match.TMDBVerify") as MockTMDBVerify:
            mock_verifier = MagicMock()
            mock_verifier._find_tmdb_entry.return_value = (mock_entry, "movie", "title")
            mock_verifier._fetch_credits.return_value = mock_credits
            MockTMDBVerify.return_value = mock_verifier

            cdata = {
                "film_title": "Kül Kedisi",
                "directors": ["Charles Walters"],
                "cast": [
                    {"actor_name": "Leslie Caron", "confidence": 0.98},
                    {"actor_name": "Michael Wilding", "confidence": 0.95},
                ],
            }
            result = qm.match(cdata, [], "1955-0019-1-0000-00-1")

        assert result["matched"] is True
        assert result["method"] == "quick_tmdb"
        assert result["cdata"]["quick_match_method"] == "quick_tmdb"
        assert result["cdata"]["quick_match_skipped_name_verify"] is True
        assert result["cdata"]["tmdb_id"] == 139799

    def test_no_tmdb_entry_falls_back(self):
        """_find_tmdb_entry eşleşme bulamazsa fallback dönmeli."""
        from core.quick_match import QuickMatcher

        qm = QuickMatcher(
            tmdb_client=self._make_mock_tmdb_client(),
            gemini_api_key="",
        )

        with patch("core.quick_match.TMDBVerify") as MockTMDBVerify:
            mock_verifier = MagicMock()
            mock_verifier._find_tmdb_entry.return_value = (None, "", "")
            MockTMDBVerify.return_value = mock_verifier

            cdata = {
                "film_title": "Bilinmeyen Film",
                "directors": [],
                "cast": [],
            }
            result = qm.match(cdata, [], "1955-0019-1-0000-00-1")

        assert result["matched"] is False
        assert result["method"] == "fallback_name_verify"

    def test_series_skips_gemini_on_tmdb_miss(self):
        """Dizi için Katman 2 başarısız → Katman 3 (Gemini) atlanmalı."""
        from core.quick_match import QuickMatcher

        qm = QuickMatcher(
            tmdb_client=self._make_mock_tmdb_client(),
            gemini_api_key="fake_gemini_key",
        )

        with patch("core.quick_match.TMDBVerify") as MockTMDBVerify:
            mock_verifier = MagicMock()
            mock_verifier._find_tmdb_entry.return_value = (None, "", "")
            MockTMDBVerify.return_value = mock_verifier

            with patch("core.quick_match.GeminiCastExtractor") as MockGemini:
                cdata = {
                    "film_title": "Amanda",
                    "directors": [],
                    "cast": [],
                }
                result = qm.match(cdata, [], "1985-0128-0-0004-00-1")  # dizi

        # Gemini çağrılmamalı
        MockGemini.assert_not_called()
        assert result["matched"] is False

    def test_gemini_then_tmdb_match(self):
        """Katman 3: Gemini çalıştıktan sonra TMDB eşleşmesi başarılı."""
        from core.quick_match import QuickMatcher

        mock_entry = {"id": 999, "title": "Matched Film"}
        mock_credits = {
            "cast": [{"name": "Some Actor", "character": "", "order": 0}],
            "crew": [],
        }

        qm = QuickMatcher(
            tmdb_client=self._make_mock_tmdb_client(),
            gemini_api_key="fake_gemini_key",
        )

        call_count = [0]

        def mock_find_tmdb(film_title, cast_names, director_names, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (None, "", "")  # İlk deneme başarısız
            return (mock_entry, "movie", "title")  # İkinci deneme başarılı

        with patch("core.quick_match.TMDBVerify") as MockTMDBVerify:
            mock_verifier = MagicMock()
            mock_verifier._find_tmdb_entry.side_effect = mock_find_tmdb
            mock_verifier._fetch_credits.return_value = mock_credits
            MockTMDBVerify.return_value = mock_verifier

            with patch("core.quick_match.GeminiCastExtractor") as MockGemini:
                mock_extractor = MagicMock()
                mock_extractor.extract.return_value = {
                    "cast": [{"actor_name": "Some Actor", "confidence": 0.9}],
                    "crew": [],
                }
                MockGemini.return_value = mock_extractor

                cdata = {
                    "film_title": "Düşük Kalite Film",
                    "directors": [],
                    "cast": [],
                }
                result = qm.match(cdata, [], "1993-0001-1-0000-00-1")  # film

        assert result["matched"] is True
        assert result["method"] == "gemini_then_tmdb"
        assert result["cdata"]["quick_match_method"] == "gemini_then_tmdb"
        assert result["cdata"]["quick_match_skipped_name_verify"] is True


class TestQuickMatcherOcrLinesToText(unittest.TestCase):
    """_ocr_lines_to_text yardımcı fonksiyonu."""

    def test_dict_lines(self):
        from core.quick_match import _ocr_lines_to_text
        lines = [{"text": "Hello"}, {"text": "World"}]
        assert _ocr_lines_to_text(lines) == ["Hello", "World"]

    def test_object_lines(self):
        from core.quick_match import _ocr_lines_to_text

        class FakeLine:
            def __init__(self, text):
                self.text = text

        lines = [FakeLine("Hello"), FakeLine("World")]
        assert _ocr_lines_to_text(lines) == ["Hello", "World"]

    def test_string_lines(self):
        from core.quick_match import _ocr_lines_to_text
        lines = ["Hello", "World"]
        assert _ocr_lines_to_text(lines) == ["Hello", "World"]

    def test_empty_list(self):
        from core.quick_match import _ocr_lines_to_text
        assert _ocr_lines_to_text([]) == []


class TestExportEngineQuickMatchFields(unittest.TestCase):
    """export_engine.generate() quick_match alanlarını JSON'a yazıyor mu."""

    def test_quick_match_fields_present_in_report(self):
        """generate() çıktısındaki raporda quick_match_method ve
        quick_match_skipped_name_verify alanları bulunmalı."""
        import tempfile
        import json
        from core.export_engine import ExportEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExportEngine(tmpdir)
            video_info = {
                "filename": "test_1955-0019-1-0000-00-1-KUL_KEDISI.mp4",
                "filepath": "/tmp/test.mp4",
                "filesize_bytes": 1000000,
                "duration_seconds": 300,
                "duration_human": "5:00",
                "resolution": "720x576",
                "fps": 25,
            }
            credits_data = {
                "film_title": "The Glass Slipper",
                "cast": [],
                "crew": [],
                "directors": [],
                "total_actors": 0,
                "total_crew": 0,
                "total_companies": 0,
                "quick_match_method": "quick_tmdb",
                "quick_match_skipped_name_verify": True,
            }
            stage_stats = {
                "QUICK_MATCH": {
                    "duration_sec": 1.5,
                    "status": "ok",
                    "matched": True,
                    "method": "quick_tmdb",
                }
            }

            jp, *_ = exp.generate(
                video_info=video_info,
                credits_data=credits_data,
                ocr_lines=[],
                stage_stats=stage_stats,
                profile="test",
                scope="video_only",
                first_min=6.0,
                last_min=10.0,
            )

            with open(jp, encoding="utf-8") as f:
                report = json.load(f)

            assert "quick_match_method" in report
            assert report["quick_match_method"] == "quick_tmdb"
            assert "quick_match_skipped_name_verify" in report
            assert report["quick_match_skipped_name_verify"] is True

    def test_quick_match_fields_default_none(self):
        """credits_data'da alan yoksa varsayılan değerler kullanılmalı."""
        import tempfile
        import json
        from core.export_engine import ExportEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExportEngine(tmpdir)
            video_info = {
                "filename": "test.mp4",
                "filepath": "/tmp/test.mp4",
                "filesize_bytes": 1000000,
                "duration_seconds": 300,
                "duration_human": "5:00",
                "resolution": "720x576",
                "fps": 25,
            }
            credits_data = {
                "film_title": "Test Film",
                "cast": [],
                "crew": [],
                "directors": [],
                "total_actors": 0,
                "total_crew": 0,
                "total_companies": 0,
            }

            jp, *_ = exp.generate(
                video_info=video_info,
                credits_data=credits_data,
                ocr_lines=[],
                stage_stats={},
                profile="test",
                scope="video_only",
                first_min=6.0,
                last_min=10.0,
            )

            with open(jp, encoding="utf-8") as f:
                report = json.load(f)

            assert report.get("quick_match_method") == "none"
            assert report.get("quick_match_skipped_name_verify") is False


if __name__ == "__main__":
    unittest.main()
