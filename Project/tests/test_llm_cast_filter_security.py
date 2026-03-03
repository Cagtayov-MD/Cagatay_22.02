"""Security/quality fix tests for LLMCastFilter.

Covers three bug fixes:
1. Fail-safe when Ollama returns None (no entries approved).
2. Regex false-positive prevention (numbered text not treated as approvals).
3. NameDB Trojan-horse prevention (single-word match no longer protects whole row).
"""

import sys
import os

# Ensure the Project directory is on the path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.llm_cast_filter import LLMCastFilter, _LINE_NUM_RE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(name: str, confidence: float = 0.7) -> dict:
    return {"actor_name": name, "confidence": confidence}


def _make_filter_with_none_response(name_checker=None):
    """Return a LLMCastFilter where _query_ollama always returns None."""
    f = LLMCastFilter(enabled=True, name_checker=name_checker)
    f._query_ollama = lambda prompt: None
    f._check_availability = lambda: True
    return f


# ---------------------------------------------------------------------------
# Fix 1: Fail-safe — no entries approved when Ollama returns None
# ---------------------------------------------------------------------------

def test_fail_safe_none_response_approves_nothing():
    """When Ollama returns None, entries should be preserved (pass-through fail-safe)."""
    f = _make_filter_with_none_response()
    cast = [
        _make_entry("Nisa Serezli"),
        _make_entry("Cihat Tamer"),
    ]
    result = f.filter_cast(cast)
    # fail-safe pass-through: entries must be preserved (not deleted)
    assert len(result) == 2
    # All entries approved via pass-through (is_llm_verified=True)
    assert all(e.get("is_llm_verified") for e in result)


def test_fail_safe_none_response_with_namedb_still_protects_real_names():
    """When Ollama returns None, real names should be preserved via pass-through."""
    f = _make_filter_with_none_response(name_checker=lambda text: True)
    cast = [_make_entry("Nisa Serezli")]
    result = f.filter_cast(cast)
    # With None response, pass-through approves entries (is_llm_verified=True)
    assert len(result) == 1
    assert result[0].get("is_llm_verified") is True


# ---------------------------------------------------------------------------
# Fix 2: Regex — false-positive prevention
# ---------------------------------------------------------------------------

def test_regex_does_not_match_numbered_text():
    """'5. Yonetmen - Isim Degildir' line should not match regex (no false-positive)."""
    line = "5. Yönetmen - İsim Değildir"
    m = _LINE_NUM_RE.match(line.strip())
    assert m is None, f"Regex should not match: {line!r}"


def test_regex_matches_isim_format():
    """'ISIM: 5' format should be parsed correctly."""
    m = _LINE_NUM_RE.match("ISIM: 5")
    assert m is not None
    assert int(m.group(1)) == 5


def test_regex_matches_bare_number():
    """Plain '5' format should be parsed correctly."""
    m = _LINE_NUM_RE.match("5")
    assert m is not None
    assert int(m.group(1)) == 5


def test_regex_matches_number_with_trailing_space():
    """'5 ' (trailing space) should be parsed correctly."""
    m = _LINE_NUM_RE.match("5 ")
    assert m is not None
    assert int(m.group(1)) == 5


def test_parse_response_no_false_positive():
    """A response line '5. Yonetmen - Isim Degildir' must NOT approve entry 5."""
    f = LLMCastFilter(enabled=True)
    response = "5. Yönetmen - İsim Değildir"
    approved = f._parse_response(response, total=10)
    assert 5 not in approved


def test_parse_response_correct_formats():
    """'ISIM: 3' and '7' lines should be parsed and approved."""
    f = LLMCastFilter(enabled=True)
    response = "ISIM: 3\n7"
    approved = f._parse_response(response, total=10)
    assert 3 in approved
    assert 7 in approved


# ---------------------------------------------------------------------------
# Fix 3: NameDB "Trojan Horse" — role+name mix and single-word safeguards
# ---------------------------------------------------------------------------

def _make_reject_filter(name_checker):
    """Return a LLMCastFilter where LLM rejects everything."""
    f = LLMCastFilter(enabled=True, name_checker=name_checker)

    def _reject_all(batch):
        return [
            {**dict(e), "is_llm_verified": False, "confidence": 0.2}
            for e in batch
        ]

    f._filter_batch = _reject_all
    f._check_availability = lambda: True
    return f


def test_namedb_role_plus_name_not_protected():
    """LLM-rejected 'Kamera Asistani Ali' must not be protected even though 'Ali' is in NameDB."""
    # name_checker returns True only for "Ali"
    f = _make_reject_filter(name_checker=lambda text: text.lower() == "ali")
    result = f.filter_cast([_make_entry("Kamera Asistanı Ali")])
    assert len(result) == 0, "Role-containing entry must not be protected via NameDB"


def test_namedb_real_full_name_is_protected():
    """LLM-rejected real full name meeting NameDB criteria should still be protected."""
    # name_checker returns True for every word
    f = _make_reject_filter(name_checker=lambda text: True)
    result = f.filter_cast([_make_entry("Nisa Serezli")])
    assert len(result) == 1
    assert result[0].get("is_name_db_protected") is True


def test_namedb_single_word_not_protected():
    """Single-word name (e.g. 'Ali') must not be NameDB-protected even if known."""
    f = _make_reject_filter(name_checker=lambda text: True)
    result = f.filter_cast([_make_entry("Ali")])
    assert len(result) == 0, "Single-word names must not qualify for NameDB protection"


def test_namedb_low_ratio_not_protected():
    """Entry with only 1 of 3 words in NameDB (33%) must not be protected (< 80% threshold)."""
    # "Zzz Yyy Nisa" — only "Nisa" is in NameDB, ratio = 1/3 < 80%
    known = {"nisa"}
    f = _make_reject_filter(name_checker=lambda text: text.lower() in known)
    result = f.filter_cast([_make_entry("Zzz Yyy Nisa")])
    assert len(result) == 0, "Insufficient NameDB match ratio must not grant protection"


def test_namedb_high_ratio_is_protected():
    """Entry with 100% of words in NameDB must be protected (>= 80% threshold)."""
    known = {"nisa", "serezli"}
    f = _make_reject_filter(name_checker=lambda text: text.lower() in known)
    result = f.filter_cast([_make_entry("Nisa Serezli")])
    assert len(result) == 1
    assert result[0].get("is_name_db_protected") is True

