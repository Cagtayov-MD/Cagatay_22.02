"""
test_sanitize_output_roles.py — Değişiklik 5: Aynı rol içi aksan/case normalize dedup.
Roller arası merge yapılmaz.
"""
import os
import sys

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def test_accent_variants_dedupe_same_role():
    """GÜNAY GÜNAYDİN / GÜNAY GÜNAYDIN → tek giriş, İ'li varyant korunur."""
    from core.export_engine import _sanitize_output_roles
    result = _sanitize_output_roles({
        "YÖNETMEN": ["GÜNAY GÜNAYDIN", "GÜNAY GÜNAYDİN"],
    })
    assert len(result["YÖNETMEN"]) == 1
    # Daha zengin varyant (İ içeren) seçilmeli
    assert result["YÖNETMEN"][0] == "GÜNAY GÜNAYDİN"


def test_case_variants_dedupe():
    """'Ali Ozan' ve 'ALI OZAN' aynı isim → tek giriş."""
    from core.export_engine import _sanitize_output_roles
    result = _sanitize_output_roles({
        "YAPIMCI": ["Ali Ozan", "ALI OZAN"],
    })
    assert len(result["YAPIMCI"]) == 1


def test_distinct_names_kept_separate():
    """Ali Ozan ve Ali Ozgun → ayrı kalır."""
    from core.export_engine import _sanitize_output_roles
    result = _sanitize_output_roles({
        "YAPIMCI": ["Ali Ozan", "Ali Ozgun"],
    })
    assert len(result["YAPIMCI"]) == 2


def test_no_cross_role_merge():
    """Aynı isim farklı rollerde → ayrı kalır, birleştirme yapılmaz."""
    from core.export_engine import _sanitize_output_roles
    result = _sanitize_output_roles({
        "YAPIMCI": ["Ali Ozan"],
        "SENARYO": ["Ali Ozan"],
    })
    assert len(result["YAPIMCI"]) == 1
    assert len(result["SENARYO"]) == 1


def test_turkish_char_normalization():
    """Şükrü / Sukru → norm_key eşleşir, zengin varyant korunur."""
    from core.export_engine import _sanitize_output_roles
    result = _sanitize_output_roles({
        "KURGU": ["Sukru Yilmaz", "Şükrü Yılmaz"],
    })
    assert len(result["KURGU"]) == 1
    assert result["KURGU"][0] == "Şükrü Yılmaz"


def test_empty_names_ignored():
    """Boş/None isimler görmezden gelinir."""
    from core.export_engine import _sanitize_output_roles
    result = _sanitize_output_roles({
        "YAPIMCI": ["", None, "  ", "Real Name"],
    })
    assert result["YAPIMCI"] == ["Real Name"]
