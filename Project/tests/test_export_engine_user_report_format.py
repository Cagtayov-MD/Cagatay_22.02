import os
import sys

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def test_to_upper_tr_converts_ascii_turkish_words():
    from core.export_engine import _to_upper_tr

    text = "kariyerinin kabullenmeyip intihar etmesi"
    assert _to_upper_tr(text) == "KARİYERİNİN KABULLENMEYİP İNTİHAR ETMESİ"


def test_to_upper_tr_keeps_foreign_names_ascii_with_protection():
    from core.export_engine import _to_upper_tr

    text = "chris washington'a gider"
    protected = {"chris", "washington"}
    assert _to_upper_tr(text, protected_words=protected) == "CHRIS WASHINGTON'A GİDER"


def test_collect_summary_name_candidates_detects_foreign_proper_names():
    from core.export_engine import _collect_summary_name_candidates

    summary = "Karen Chris'in teklifini reddedip Washington'a gider."
    result = _collect_summary_name_candidates(summary)
    assert "karen" in result
    assert "chris" in result
    assert "washington" in result
