"""
test_sanitize_output_roles.py — Yalnızca desteklenen ekip rolleri export'a taşınır.
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


def test_unsupported_roles_are_dropped():
    """Desteklenmeyen ekip rolleri export sözlüğüne hiç alınmaz."""
    from core.export_engine import _sanitize_output_roles
    result = _sanitize_output_roles({
        "YAPIMCI": ["Ali Ozan"],
        "SENARYO": ["Ali Ozan"],
    })
    assert len(result["YAPIMCI"]) == 1
    assert "SENARYO" not in result


def test_turkish_char_normalization_supported_role():
    """Şükrü / Sukru → norm_key eşleşir, zengin varyant korunur."""
    from core.export_engine import _sanitize_output_roles
    result = _sanitize_output_roles({
        "YAPIMCI": ["Sukru Yilmaz", "Şükrü Yılmaz"],
    })
    assert len(result["YAPIMCI"]) == 1
    assert result["YAPIMCI"][0] == "Şükrü Yılmaz"


def test_empty_names_ignored():
    """Boş/None isimler görmezden gelinir."""
    from core.export_engine import _sanitize_output_roles
    result = _sanitize_output_roles({
        "YAPIMCI": ["", None, "  ", "Real Name"],
    })
    assert result["YAPIMCI"] == ["Real Name"]


def test_cyrillic_names_filtered_by_is_non_person():
    """Kiril karakter içeren isimler _is_non_person tarafından reddedilmeli."""
    from core.export_engine import _is_non_person
    assert _is_non_person("Илья Миньковецкий") is True
    assert _is_non_person("ТОФИГ ТАГЫЗАДА") is True
    assert _is_non_person("М. Дабашов") is True


def test_non_export_crew_roles_are_ignored_without_keyerror():
    """SENARYO gibi desteklenmeyen roller sessizce atlanmalı."""
    from core.export_engine import _map_crew_to_roles
    crew = [
        {"name": "Writer Name", "job": "screenplay"},
        {"name": "Producer Name", "job": "producer"},
    ]
    result = _map_crew_to_roles(crew, directors=[])
    assert result["YAPIMCI"] == ["Producer Name"]
    assert "SENARYO" not in result
