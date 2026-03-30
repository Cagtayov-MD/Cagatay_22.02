"""test_structural_check.py

_structural_check ve _is_latin fonksiyonlarının yeni kurallarını test eder.

Senaryolar:
  SC-01: Virgül içeren giriş → sentence_fragment
  SC-02: Slash içeren giriş → slash_reference
  SC-03: Nokta ile biten 3+ kelimeli giriş → sentence_ending
  SC-04: Kiril metin → non_latin (0x0400 sınırı düzeltmesi)
  SC-05: Geçerli isimler hâlâ geçmeli
  SC-06: is_valid_person_name Kiril'i reddeder
"""

import os
import sys

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core.name_verify import _structural_check, _is_latin, is_valid_person_name


# ── SC-01: Virgül içeren girişler ─────────────────────────────────────────────

def test_sc01_comma_in_text_rejected():
    """Virgül içeren giriş cümle parçası olarak reddedilmeli."""
    # "de goturuyor, tipki" → 3 kelime, virgül içeriyor
    passed, reason = _structural_check("goturuyor, tipki de")
    assert not passed
    assert reason == "sentence_fragment"


def test_sc01b_comma_subtitle_rejected():
    """Altyazı cümlesi virgülle reddedilmeli."""
    passed, reason = _structural_check("karimi da goturmek, ama")
    assert not passed
    assert reason == "sentence_fragment"


# ── SC-02: Slash içeren girişler ──────────────────────────────────────────────

def test_sc02_slash_reference_rejected():
    """Slash içeren giriş referans/alıntı olarak reddedilmeli."""
    # 3 kelime + slash → slash_reference (too_many_words'a takılmaz)
    passed, reason = _structural_check("/ Jozsef Attila /")
    assert not passed
    assert reason == "slash_reference"


def test_sc02b_single_slash_rejected():
    """Tek slash da reddedilmeli."""
    passed, reason = _structural_check("Jozsef / Attila")
    assert not passed
    assert reason == "slash_reference"


# ── SC-03: Nokta ile biten + 3+ kelime → cümle ───────────────────────────────

def test_sc03_sentence_ending_period_rejected():
    """Nokta ile biten 3 kelimeli giriş cümle olarak reddedilmeli."""
    passed, reason = _structural_check("Yaninda baska da vardi.")
    assert not passed
    assert reason == "sentence_ending"


def test_sc03b_four_word_sentence_rejected():
    """Nokta ile biten 4 kelimeli giriş reddedilmeli."""
    passed, reason = _structural_check("karimı da goturmek isterim.")
    assert not passed
    assert reason == "sentence_ending"


def test_sc03c_two_word_period_ok():
    """2 kelimeli + nokta → cümle değil, geçmeli (Maria J. gibi)."""
    passed, reason = _structural_check("Maria J.")
    # 2 kelime — sentence_ending koşulu karşılanmaz (len(words) >= 3 değil)
    assert reason != "sentence_ending"


# ── SC-04: Kiril metin → non_latin ───────────────────────────────────────────

def test_sc04_cyrillic_text_rejected_by_structural():
    """Kiril metin _structural_check tarafından non_latin olarak reddedilmeli."""
    passed, reason = _structural_check("Илья Миньковецкий")
    assert not passed
    assert reason == "non_latin"


def test_sc04b_cyrillic_is_not_latin():
    """_is_latin Kiril metni False döndürmeli."""
    assert _is_latin("Илья Миньковецкий") is False
    assert _is_latin("ТОФИГ ТАГЫЗАДА") is False


def test_sc04c_turkish_is_latin():
    """Türkçe karakterler Latin sayılmalı."""
    assert _is_latin("Ömer Seyfettin") is True
    assert _is_latin("Şükrü Güler") is True


# ── SC-05: Geçerli isimler geçmeli ───────────────────────────────────────────

def test_sc05_valid_names_pass():
    """Geçerli person name'ler _structural_check'ten geçmeli."""
    for name in [
        "ASLENIS MARQUEZ",
        "JULIAN PASTOR",
        "Ana Maria Munoz",
        "Federico Teran Gilmore",  # 3 kelime, nokta yok, virgül yok
        "Cesar Rodriguez",
    ]:
        passed, reason = _structural_check(name)
        assert passed, f"'{name}' geçmeli ama reddedildi: {reason}"


# ── SC-06: is_valid_person_name Kiril'i reddeder ─────────────────────────────

def test_sc06_is_valid_person_name_rejects_cyrillic():
    """is_valid_person_name Kiril karakterli girişleri False döndürmeli."""
    assert is_valid_person_name("Илья Миньковецкий") is False
    assert is_valid_person_name("ТОФИГ ТАГЫЗАДА") is False
    assert is_valid_person_name("М. Дабашов") is False


def test_sc06b_is_valid_person_name_accepts_latin():
    """is_valid_person_name geçerli Latin isimlerini True döndürmeli."""
    assert is_valid_person_name("JULIAN PASTOR") is True
    assert is_valid_person_name("Federico Teran Gilmore") is True
