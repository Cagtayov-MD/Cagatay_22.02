"""Tests for LLMCastFilter NameDB protection logic."""

from core.llm_cast_filter import LLMCastFilter


def _make_entry(name: str, confidence: float = 0.7) -> dict:
    return {"actor_name": name, "confidence": confidence}


def _make_filter(name_checker=None):
    """Return a LLMCastFilter with Ollama stubbed out as unavailable."""
    f = LLMCastFilter(enabled=True, name_checker=name_checker)
    # Stub _filter_batch so LLM rejects everything
    def _reject_all(batch):
        result = []
        for entry in batch:
            e = dict(entry)
            e["is_llm_verified"] = False
            e["confidence"] = 0.2
            result.append(e)
        return result

    f._filter_batch = _reject_all
    f._check_availability = lambda: True
    return f


def test_namedb_known_name_is_protected_when_llm_rejects():
    """LLM reddettiği ama NameDB'de bulunan isim korunuyor."""
    f = _make_filter(name_checker=lambda text: True)
    result = f.filter_cast([_make_entry("Nisa Serezli")])
    assert len(result) == 1
    assert result[0]["actor_name"] == "Nisa Serezli"
    assert result[0].get("is_name_db_protected") is True
    assert result[0].get("is_llm_verified") is False
    # Confidence should NOT be dropped to 0.2
    assert result[0]["confidence"] != 0.2


def test_namedb_unknown_name_is_removed_when_llm_rejects():
    """LLM reddettiği ve NameDB'de olmayan isim silinmeye devam ediyor."""
    f = _make_filter(name_checker=lambda text: False)
    result = f.filter_cast([_make_entry("DCJV alqu")])
    assert len(result) == 0


def test_llm_approved_names_are_always_kept():
    """LLM onayladığı isimler her zamanki gibi korunuyor."""
    f = LLMCastFilter(enabled=True, name_checker=lambda text: False)

    def _approve_all(batch):
        result = []
        for entry in batch:
            e = dict(entry)
            e["is_llm_verified"] = True
            e["confidence"] = round(min(1.0, float(e.get("confidence", 0.6)) + 0.3), 3)
            result.append(e)
        return result

    f._filter_batch = _approve_all
    f._check_availability = lambda: True

    result = f.filter_cast([_make_entry("Cihat Tamer")])
    assert len(result) == 1
    assert result[0].get("is_llm_verified") is True


def test_no_name_checker_keeps_original_behavior():
    """name_checker None olduğunda eski davranış korunuyor (geriye uyumluluk)."""
    f = _make_filter(name_checker=None)
    result = f.filter_cast([_make_entry("Nisa Serezli")])
    # Without name_checker, LLM-rejected entries are still dropped
    assert len(result) == 0
