"""Tests for phonetic matching layer and pre-filter improvements."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.turkish_name_db import TurkishNameDB, _phonetic_key
from core.llm_cast_filter import LLMCastFilter


def test_phonetic_key_w_to_n():
    """_phonetic_key: w→n dönüşümü çalışıyor mu?"""
    key = _phonetic_key("Camaw")
    assert "N" in key, f"'w→n' dönüşümü bekleniyor, ancak: {key}"


def test_phonetic_key_dedup():
    """_phonetic_key: tekrarlanan harfler tek yapılıyor mu?"""
    key = _phonetic_key("CAMAAN")
    assert key == _phonetic_key("CAMAN"), f"Tekrarlı harf deduplication bekleniyor: {key}"


def test_phonetic_key_q_to_g():
    """_phonetic_key: q→g dönüşümü çalışıyor mu?"""
    key = _phonetic_key("Qunduz")
    assert key == _phonetic_key("Gunduz"), f"'q→g' dönüşümü bekleniyor: {key}"


def test_find_with_method_returns_tuple():
    """find_with_method() 3-tuple döndürüyor mu?"""
    db = TurkishNameDB()  # DB yokken hardcoded aktif
    result = db.find_with_method("SEBNEM")
    assert len(result) == 3, "3-tuple bekleniyor: (canonical, score, method)"
    canonical, score, method = result
    assert canonical == "Şebnem"
    assert score == 1.0
    assert method == "hardcoded"


def test_find_with_method_empty():
    """find_with_method() boş metin için ('', 0.0, '') döndürüyor."""
    db = TurkishNameDB()
    canonical, score, method = db.find_with_method("")
    assert canonical is None
    assert score == 0.0
    assert method == ""


def test_find_backward_compat():
    """find() eski 2-tuple imzasını koruyor (geriye dönük uyumluluk)."""
    db = TurkishNameDB()
    result = db.find("SEBNEM")
    assert len(result) == 2, "find() 2-tuple döndürmeli"
    canonical, score = result
    assert canonical == "Şebnem"


def test_pre_filter_no_vowels():
    """Sesli harf olmayan metin çöp olarak işaretlenmeli."""
    f = LLMCastFilter(enabled=False)
    assert f._pre_filter_obvious_junk({"actor_name": "bcrdfg"}) is True


def test_pre_filter_low_vowel_ratio():
    """Sesli harf oranı çok düşük metin çöp olarak işaretlenmeli."""
    f = LLMCastFilter(enabled=False)
    assert f._pre_filter_obvious_junk({"actor_name": "bcrdafg"}) is True


def test_pre_filter_special_chars():
    """Özel karakter içeren metin çöp olarak işaretlenmeli."""
    f = LLMCastFilter(enabled=False)
    assert f._pre_filter_obvious_junk({"actor_name": "t..Hacti"}) is True


def test_pre_filter_at_sign():
    """@ içeren metin çöp olarak işaretlenmeli."""
    f = LLMCastFilter(enabled=False)
    assert f._pre_filter_obvious_junk({"actor_name": "user@mail"}) is True


def test_pre_filter_digit_heavy():
    """Rakam ağırlıklı metin çöp olarak işaretlenmeli."""
    f = LLMCastFilter(enabled=False)
    assert f._pre_filter_obvious_junk({"actor_name": "A12345"}) is True


def test_pre_filter_valid_name():
    """Geçerli isim çöp olarak işaretlenmemeli."""
    f = LLMCastFilter(enabled=False)
    assert f._pre_filter_obvious_junk({"actor_name": "Nisa Serezli"}) is False


def test_pre_filter_valid_single_word():
    """Tek kelimeli isim (soyadı olmayan) çöp olarak işaretlenmeli."""
    f = LLMCastFilter(enabled=False)
    assert f._pre_filter_obvious_junk({"actor_name": "Mehmet"}) is True
