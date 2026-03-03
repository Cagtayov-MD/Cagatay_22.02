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


# ---------------------------------------------------------------------------
# Fix 1: Fail-safe — Ollama None yanıtında hiçbir giriş onaylanmamalı
# ---------------------------------------------------------------------------

def test_ollama_none_response_no_entries_verified():
    """Ollama yanıtı None olduğunda tüm girişler pass-through olarak onaylanmalı."""
    f = LLMCastFilter(enabled=True)
    f._query_ollama = lambda prompt: None
    f._check_availability = lambda: True

    entries = [_make_entry("Ali Veli"), _make_entry("Mehmet Yılmaz")]
    result = f._filter_batch(entries)

    for e in result:
        assert e.get("is_llm_verified") is True, (
            f"Pass-through ihlali: '{e['actor_name']}' onaylanmadı ama Ollama None döndürdü"
        )


# ---------------------------------------------------------------------------
# Fix 2: Strict regex — format dışı LLM yanıtlarında false-positive olmamalı
# ---------------------------------------------------------------------------

def test_parse_response_no_false_positive_for_annotated_line():
    """'5. Yönetmen - İsim Değildir' formatı 5 numarasını onaylamamalı."""
    f = LLMCastFilter(enabled=True)
    approved = f._parse_response("5. Yönetmen - İsim Değildir", total=10)
    assert 5 not in approved, "False-positive: '5. Yönetmen...' satırından 5 onaylandı"


def test_parse_response_isim_format_accepted():
    """'ISIM: 5' formatı doğru parse edilmeli."""
    f = LLMCastFilter(enabled=True)
    approved = f._parse_response("ISIM: 5", total=10)
    assert 5 in approved, "'ISIM: 5' formatı parse edilemedi"


def test_parse_response_bare_number_accepted():
    """Sadece '5' formatı doğru parse edilmeli."""
    f = LLMCastFilter(enabled=True)
    approved = f._parse_response("5", total=10)
    assert 5 in approved, "Yalnız '5' formatı parse edilemedi"


def test_parse_response_mixed_format():
    """Karışık formatta birden fazla numara doğru parse edilmeli."""
    f = LLMCastFilter(enabled=True)
    response = "ISIM: 1\n3\n5. Sahte Satır\nISIM: 7"
    approved = f._parse_response(response, total=10)
    assert 1 in approved
    assert 3 in approved
    assert 5 not in approved  # "5. Sahte Satır" false-positive olmamalı
    assert 7 in approved


# ---------------------------------------------------------------------------
# Fix 3: NameDB "Trojan Horse" — rol+isim karışımı NameDB tarafından korunmamalı
# ---------------------------------------------------------------------------

def test_namedb_role_keyword_entry_not_protected():
    """Rol kelimesi içeren giriş NameDB'de isim geçse bile korunmamalı."""
    # "kamera" _ROLE_KEYWORDS içinde — "Ali" NameDB'de bulunsa da korunmamalı
    f = _make_filter(name_checker=lambda text: text == "Ali")
    result = f.filter_cast([_make_entry("Kamera Asistanı Ali")])
    assert len(result) == 0, "Rol+isim karışımı NameDB tarafından korundu (Trojan Horse)"


def test_namedb_real_full_name_protected():
    """Gerçek ad-soyad NameDB kriterlerini sağlıyorsa korunmalı."""
    known = {"Nisa", "Serezli"}
    f = _make_filter(name_checker=lambda text: text in known)
    result = f.filter_cast([_make_entry("Nisa Serezli")])
    assert len(result) == 1
    assert result[0].get("is_name_db_protected") is True


def test_namedb_single_word_not_protected():
    """Tek kelimelik isim NameDB'de bulunsa bile korunmamalı."""
    f = _make_filter(name_checker=lambda text: True)  # hepsi NameDB'de
    result = f.filter_cast([_make_entry("Ali")])
    assert len(result) == 0, "Tek kelimelik isim NameDB tarafından korundu"


def test_namedb_partial_match_below_threshold_not_protected():
    """Kelimelerin %80'inden azı NameDB'deyse korunmamalı (rol kw olmasa da)."""
    # "Hakan Sahte": sadece "Hakan" NameDB'de → %50 < %80 → korunmamalı
    known = {"Hakan"}
    f = _make_filter(name_checker=lambda text: text in known)
    result = f.filter_cast([_make_entry("Hakan Sahte")])
    assert len(result) == 0, "Eşleşme oranı düşük isim NameDB tarafından korundu"
