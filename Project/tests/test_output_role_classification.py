import os
import sys

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def test_all_seven_output_roles_require_exact_aliases():
    from core.export_engine import _classify_output_role

    assert _classify_output_role("producer") == "YAPIMCI"
    assert _classify_output_role("director") == "YÖNETMEN"
    assert _classify_output_role("assistant director") == "YÖNETMEN YARDIMCISI"
    assert _classify_output_role("director of photography") == "GÖRÜNTÜ YÖNETMENİ"
    assert _classify_output_role("screenplay") == "SENARYO"
    assert _classify_output_role("camera operator") == "KAMERA"
    assert _classify_output_role("edited by") == "KURGU"


def test_director_aliases_are_strict_exact_matches():
    from core.export_engine import _classify_output_role

    assert _classify_output_role("directed by") == "YÖNETMEN"
    assert _classify_output_role("yönetmen:") == "YÖNETMEN"
    assert _classify_output_role("regia di") == "YÖNETMEN"


def test_director_like_but_non_whitelisted_roles_are_rejected():
    from core.export_engine import _classify_output_role

    assert _classify_output_role("Animation Director") is None
    assert _classify_output_role("Co-Director") is None
    assert _classify_output_role("director de") is None


def test_non_exact_producer_and_writing_roles_are_rejected():
    from core.export_engine import _classify_output_role

    assert _classify_output_role("executive producer") is None
    assert _classify_output_role("line producer") is None
    assert _classify_output_role("story by") is None
    assert _classify_output_role("dialogue") is None
    assert _classify_output_role("adaptation") is None


def test_non_exact_camera_and_editing_roles_are_rejected():
    from core.export_engine import _classify_output_role

    assert _classify_output_role("Additional Camera") is None
    assert _classify_output_role("Camera Assistant") is None
    assert _classify_output_role("Steadicam Operator") is None
    assert _classify_output_role("Picture Editor") is None
    assert _classify_output_role("Supervising Editor") is None


def test_exact_multilingual_aliases_still_work():
    from core.export_engine import _classify_output_role

    assert _classify_output_role("producteur") == "YAPIMCI"
    assert _classify_output_role("assistente alla regia") == "YÖNETMEN YARDIMCISI"
    assert _classify_output_role("directeur de la photographie") == "GÖRÜNTÜ YÖNETMENİ"
    assert _classify_output_role("guion") == "SENARYO"
    assert _classify_output_role("operador de camara") == "KAMERA"
    assert _classify_output_role("montaggio") == "KURGU"
