"""
TranscribeStage._build_initial_prompt() unit testleri.
faster-whisper gerektirmez — sadece _build_initial_prompt fonksiyonunu test eder.
"""
import sys
import os
import unittest

# Proje kokunu path'e ekle
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestBuildInitialPrompt(unittest.TestCase):
    """_build_initial_prompt metodunun dogru calistigini test eder."""

    def _make_stage(self):
        from audio.stages.transcribe import TranscribeStage
        return TranscribeStage(log_cb=lambda *a, **kw: None)

    def test_empty_cast_returns_empty_string(self):
        """Boş cast listesi → boş string döner."""
        stage = self._make_stage()
        self.assertEqual(stage._build_initial_prompt({}), "")
        self.assertEqual(stage._build_initial_prompt({"tmdb_cast": []}), "")
        self.assertEqual(stage._build_initial_prompt({"tmdb_cast": None}), "")

    def test_normal_cast_list_builds_prompt(self):
        """Normal cast listesi → doğru prompt oluşur."""
        stage = self._make_stage()
        cast = [
            {"actor_name": "Michael Keaton", "character_name": "Bruce Wayne"},
            {"actor_name": "Jack Nicholson", "character_name": "Joker"},
        ]
        result = stage._build_initial_prompt({"tmdb_cast": cast})
        self.assertIn("Michael Keaton", result)
        self.assertIn("Bruce Wayne", result)
        self.assertIn("Jack Nicholson", result)
        self.assertIn("Joker", result)
        self.assertTrue(result.startswith("Bu filmde şu isimler geçmektedir:"))
        self.assertTrue(result.endswith("."))

    def test_max_20_names_cap(self):
        """25 girişli liste → max 20 isimle kesilir."""
        stage = self._make_stage()
        # 25 farklı oyuncu ismi oluştur (character yok)
        cast = [{"actor_name": f"Actor{i}"} for i in range(25)]
        result = stage._build_initial_prompt({"tmdb_cast": cast})
        # Prompttaki isim sayısını say: virgülle ayrılmış kısım
        names_part = result.replace("Bu filmde şu isimler geçmektedir: ", "").rstrip(".")
        names = [n.strip() for n in names_part.split(",")]
        self.assertEqual(len(names), 20)

    def test_duplicate_names_are_deduplicated(self):
        """Tekrar eden isimler → deduplicate edilir."""
        stage = self._make_stage()
        cast = [
            {"actor_name": "Jessica Alba", "character_name": "Sue Storm"},
            {"actor_name": "Jessica Alba", "character_name": "Sue Storm"},
            {"actor_name": "Jessica Alba"},
        ]
        result = stage._build_initial_prompt({"tmdb_cast": cast})
        names_part = result.replace("Bu filmde şu isimler geçmektedir: ", "").rstrip(".")
        names = [n.strip() for n in names_part.split(",")]
        self.assertEqual(names.count("Jessica Alba"), 1)
        self.assertEqual(names.count("Sue Storm"), 1)

    def test_alternative_field_names(self):
        """'name' ve 'character' alanları da desteklenir."""
        stage = self._make_stage()
        cast = [
            {"name": "Robert Downey Jr.", "character": "Tony Stark"},
        ]
        result = stage._build_initial_prompt({"tmdb_cast": cast})
        self.assertIn("Robert Downey Jr.", result)
        self.assertIn("Tony Stark", result)

    def test_empty_name_fields_are_skipped(self):
        """Boş isim alanları atlanır."""
        stage = self._make_stage()
        cast = [
            {"actor_name": "", "character_name": ""},
            {"actor_name": "Chris Evans"},
        ]
        result = stage._build_initial_prompt({"tmdb_cast": cast})
        names_part = result.replace("Bu filmde şu isimler geçmektedir: ", "").rstrip(".")
        names = [n.strip() for n in names_part.split(",")]
        self.assertNotIn("", names)
        self.assertIn("Chris Evans", names)

    def test_actor_only_entry(self):
        """Sadece oyuncu ismi olan giriş doğru çalışır."""
        stage = self._make_stage()
        cast = [{"actor_name": "Scarlett Johansson"}]
        result = stage._build_initial_prompt({"tmdb_cast": cast})
        self.assertIn("Scarlett Johansson", result)

    def test_character_only_entry(self):
        """Sadece karakter ismi olan giriş doğru çalışır."""
        stage = self._make_stage()
        cast = [{"character_name": "Black Widow"}]
        result = stage._build_initial_prompt({"tmdb_cast": cast})
        self.assertIn("Black Widow", result)


if __name__ == "__main__":
    unittest.main()
