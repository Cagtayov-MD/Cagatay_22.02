"""test_search_crew_prompt.py

_search_crew_with_gemini — "veri yok" satırları için Gemini 2.5 Flash
arama prompt'u ve min_domain eşiğini doğrular.

Tablo (export_engine.py:959–983):
  Durum              | Prompt şablonu                                     | Min domain
  Film + anchor var  | "{anchor}{yıl} yapımı {başlık} adlı filmin ..."    | 2
  Film + anchor yok  | "{yıl} yılında çekilen {başlık} adlı filmin ..."   | 3
  Dizi + anchor var  | "{anchor}{yıl} yapımı {başlık} adlı dizinin N. bölümünün ..." | 3
  Dizi + anchor yok  | "{yıl} yılında yayımlanan {başlık} adlı dizinin N. bölümünün ..." | 3

Testler:
  SP-01: Film + anchor var  → doğru prompt + min_domains=2
  SP-02: Film + anchor yok  → doğru prompt + min_domains=3
  SP-03: Dizi + anchor var  → doğru prompt + min_domains=3
  SP-04: Dizi + anchor yok  → doğru prompt + min_domains=3
  SP-05: min_domains karşılanmazsa None döner
  SP-06: min_domains karşılanırsa isim döner
  SP-07: rol_sorusu tanımsızsa None döner (erken çıkış)
  SP-08: Anchor seçimi — known_fields'daki ilk dolu rol kullanılır
  SP-09: GeminiClient, gemini-2.5-flash modeliyle kurulur
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# _ROLE_QUESTIONS sabiti
from core.export_engine import _ROLE_QUESTIONS


# ── Prompt + min_domains hesaplama mantığı (export_engine.py:947–983) ─────────

def _build_prompt(
    target_role: str,
    year: str,
    film_title: str,
    known_fields: dict,
    episode_no: str = "YOK",
) -> tuple[str, int]:
    """
    _search_crew_with_gemini içindeki prompt + min_domains bloğunu izole eder.
    Gerçek API çağrısı yapılmaz.
    Returns: (prompt, min_domains)
    """
    rol_sorusu = _ROLE_QUESTIONS.get(target_role)
    assert rol_sorusu, f"Bilinmeyen rol: {target_role}"

    is_series = episode_no not in ("YOK", "0000", "0", "")
    episode_int = (episode_no.lstrip("0") or "0") if is_series else ""

    anchor = ""
    for role, names in known_fields.items():
        if names and role in _ROLE_QUESTIONS:
            first = names[0] if isinstance(names[0], str) else str(names[0])
            if first.strip():
                anchor = f"{_ROLE_QUESTIONS[role].capitalize()} {first.strip()} olan, "
                break

    if is_series:
        içerik = f"dizinin {episode_int}. bölümünün"
        if anchor:
            prompt = (
                f"{anchor}{year} yapımı {film_title} adlı "
                f"{içerik} {rol_sorusu} kimdir? Sadece kişi adını yaz."
            )
        else:
            prompt = (
                f"{year} yılında yayımlanan {film_title} adlı "
                f"{içerik} {rol_sorusu} kimdir? Sadece kişi adını yaz."
            )
        min_domains = 3
    elif anchor:
        prompt = (
            f"{anchor}{year} yapımı {film_title} adlı filmin "
            f"{rol_sorusu} kimdir? Sadece kişi adını yaz."
        )
        min_domains = 2
    else:
        prompt = (
            f"{year} yılında çekilen {film_title} adlı filmin "
            f"{rol_sorusu} kimdir? Sadece kişi adını yaz."
        )
        min_domains = 3

    return prompt, min_domains


# ── SP-01: Film + anchor var ───────────────────────────────────────────────────

class TestSearchCrewPrompt(unittest.TestCase):

    def test_sp01_film_with_anchor(self):
        """Film + anchor var → '{anchor}1991 yapımı X adlı filmin ...' + min_domains=2"""
        prompt, min_domains = _build_prompt(
            target_role="YAPIMCI",
            year="1991",
            film_title="FELANCA FİLM",
            known_fields={"YÖNETMEN": ["Ahmet Yılmaz"]},
            episode_no="YOK",
        )
        self.assertEqual(min_domains, 2)
        self.assertIn("Ahmet Yılmaz olan", prompt)
        self.assertIn("1991 yapımı", prompt)
        self.assertIn("FELANCA FİLM", prompt)
        self.assertIn("adlı filmin", prompt)
        self.assertIn("yapımcısı", prompt)
        self.assertNotIn("bölümünün", prompt)

    def test_sp02_film_no_anchor(self):
        """Film + anchor yok → '1991 yılında çekilen X adlı filmin ...' + min_domains=3"""
        prompt, min_domains = _build_prompt(
            target_role="YAPIMCI",
            year="1991",
            film_title="FELANCA FİLM",
            known_fields={},
            episode_no="YOK",
        )
        self.assertEqual(min_domains, 3)
        self.assertIn("1991 yılında çekilen", prompt)
        self.assertIn("adlı filmin", prompt)
        self.assertIn("yapımcısı", prompt)
        self.assertNotIn("bölümünün", prompt)

    def test_sp03_dizi_with_anchor(self):
        """Dizi + anchor var → '{anchor}1991 yapımı X adlı dizinin N. bölümünün ...' + min_domains=3"""
        prompt, min_domains = _build_prompt(
            target_role="YAPIMCI",
            year="1991",
            film_title="FELANCA DİZİ",
            known_fields={"YÖNETMEN": ["Mehmet Kaya"]},
            episode_no="0042",
        )
        self.assertEqual(min_domains, 3)
        self.assertIn("Mehmet Kaya olan", prompt)
        self.assertIn("1991 yapımı", prompt)
        self.assertIn("adlı dizinin", prompt)
        self.assertIn("42. bölümünün", prompt)
        self.assertIn("yapımcısı", prompt)
        self.assertNotIn("adlı filmin", prompt)

    def test_sp04_dizi_no_anchor(self):
        """Dizi + anchor yok → '1991 yılında yayımlanan X adlı dizinin N. bölümünün ...' + min_domains=3"""
        prompt, min_domains = _build_prompt(
            target_role="YAPIMCI",
            year="1991",
            film_title="FELANCA DİZİ",
            known_fields={},
            episode_no="0042",
        )
        self.assertEqual(min_domains, 3)
        self.assertIn("1991 yılında yayımlanan", prompt)
        self.assertIn("adlı dizinin", prompt)
        self.assertIn("42. bölümünün", prompt)
        self.assertIn("yapımcısı", prompt)
        self.assertNotIn("adlı filmin", prompt)

    def test_sp05_min_domains_not_met_returns_none(self):
        """Yeterli domain yoksa _search_crew_with_gemini None döner."""
        mock_client = MagicMock()
        mock_client.generate_with_search.return_value = ("Bulunan Kişi", ["tek.com"])

        # GeminiClient fonksiyon içinde import edilir → kaynak modülde patch
        with patch("core.gemini_client.GeminiClient", return_value=mock_client), \
             patch("config.runtime_paths.get_gemini_api_key", return_value="fake-key"):
            from core.export_engine import _search_crew_with_gemini
            result = _search_crew_with_gemini(
                target_role="YAPIMCI",
                year="1991",
                film_title="FELANCA FİLM",
                known_fields={},       # anchor yok → min_domains=3
                episode_no="YOK",
            )
        # 1 domain var, min_domains=3 → None dönmeli
        self.assertIsNone(result, "Domain eşiği karşılanmadığında None dönmeli")

    def test_sp06_min_domains_met_returns_name(self):
        """Yeterli domain varsa isim döner."""
        mock_client = MagicMock()
        mock_client.generate_with_search.return_value = (
            "Yapımcının Adı", ["site1.com", "site2.com", "site3.com"]
        )

        with patch("core.gemini_client.GeminiClient", return_value=mock_client), \
             patch("config.runtime_paths.get_gemini_api_key", return_value="fake-key"):
            from core.export_engine import _search_crew_with_gemini
            result = _search_crew_with_gemini(
                target_role="YAPIMCI",
                year="1991",
                film_title="FELANCA FİLM",
                known_fields={},       # min_domains=3, 3 domain var
                episode_no="YOK",
            )
        self.assertEqual(result, "Yapımcının Adı")

    def test_sp07_unknown_role_returns_none(self):
        """Tanımsız rol → erken None."""
        prompt_result = _ROLE_QUESTIONS.get("BİLİNMEYEN_ROL")
        self.assertIsNone(prompt_result, "_ROLE_QUESTIONS'da olmayan rol None dönmeli")

    def test_sp08_anchor_uses_first_known_field(self):
        """Anchor — known_fields'daki ilk dolu rol kullanılır."""
        prompt, _ = _build_prompt(
            target_role="YAPIMCI",
            year="2000",
            film_title="TEST",
            known_fields={
                "YÖNETMEN": ["İlk Yönetmen"],
                "YAPIMCI": ["Yapımcı Adı"],
            },
        )
        self.assertIn("İlk Yönetmen olan", prompt)
        self.assertNotIn("Yapımcı Adı", prompt)

    def test_sp09_uses_gemini_25_flash_model(self):
        """GeminiSearch, GeminiClient'i 2.5 Flash ile kurmalı."""
        mock_client = MagicMock()
        mock_client.generate_with_search.return_value = (
            "Yapımcının Adı", ["site1.com", "site2.com", "site3.com"]
        )

        with patch("core.gemini_client.GeminiClient", return_value=mock_client) as mock_ctor, \
             patch("config.runtime_paths.get_gemini_api_key", return_value="fake-key"):
            from core.export_engine import _search_crew_with_gemini
            result = _search_crew_with_gemini(
                target_role="YAPIMCI",
                year="1991",
                film_title="FELANCA FİLM",
                known_fields={},
                episode_no="YOK",
            )

        self.assertEqual(result, "Yapımcının Adı")
        mock_ctor.assert_called_once_with(model="gemini-2.5-flash", api_key="fake-key")


if __name__ == "__main__":
    unittest.main(verbosity=2)
