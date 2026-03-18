"""
test_credits_qa.py — Unit tests for the credits_qa module.

Covers:
  QA-01: Missing actor detected when not in TMDB cast
  QA-02: Actor already in TMDB is NOT flagged
  QA-03: Low-confidence line filtered out (< MIN_CONFIDENCE)
  QA-04: Single-word line filtered out (< MIN_WORDS)
  QA-05: Rare line seen only once and not in opening credits is filtered
  QA-06: Opening-credit line seen once is kept (is_opening_credit path)
  QA-07: tmdb_looks_incomplete flag set correctly
  QA-08: Summary messages follow correct format (0 / 1-3 / 4+)
  QA-09: Sorting — opening credits first, then seen_count descending
  QA-10: to_dict() output has expected keys and types
  QA-11: Works with OCRLine dataclass objects as well as plain dicts
  QA-12: Empty inputs handled gracefully
"""

import sys
import os

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core.credits_qa import (
    check_missing_actors,
    CreditsQA,
    MissingActor,
    MIN_CONFIDENCE,
    MIN_SEEN_FRAMES,
    OPENING_CUTOFF_S,
    MATCH_THRESHOLD,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _line(text, conf=0.92, seen=4, first_sec=10.0):
    """Build a plain-dict OCR line."""
    return {
        "text": text,
        "avg_confidence": conf,
        "seen_count": seen,
        "first_seen": first_sec,
    }


def _tmdb_cast(*names):
    return [{"actor_name": n} for n in names]


# ── QA-01: Missing actor detected ────────────────────────────────────────────

def test_qa01_missing_actor_detected():
    ocr = [_line("MARGUERITE HICKEY")]
    tmdb = _tmdb_cast("Tom Hanks", "Julia Roberts")
    result = check_missing_actors(ocr, tmdb)
    assert len(result.missing_actors) == 1
    assert result.missing_actors[0].name == "MARGUERITE HICKEY"


# ── QA-02: Actor present in TMDB is NOT flagged ───────────────────────────────

def test_qa02_actor_in_tmdb_not_flagged():
    ocr = [_line("Tom Hanks")]
    tmdb = _tmdb_cast("Tom Hanks")
    result = check_missing_actors(ocr, tmdb)
    assert len(result.missing_actors) == 0


# ── QA-03: Low confidence filtered out ───────────────────────────────────────

def test_qa03_low_confidence_filtered():
    ocr = [_line("SOME ACTOR", conf=MIN_CONFIDENCE - 0.01)]
    tmdb = _tmdb_cast("Nobody")
    result = check_missing_actors(ocr, tmdb)
    assert len(result.missing_actors) == 0


# ── QA-04: Single word filtered out ──────────────────────────────────────────

def test_qa04_single_word_filtered():
    ocr = [_line("MADONNA")]
    tmdb = _tmdb_cast("Nobody")
    result = check_missing_actors(ocr, tmdb)
    assert len(result.missing_actors) == 0


# ── QA-05: Rare line (seen=1, not opening) filtered ─────────────────────────

def test_qa05_rare_non_opening_filtered():
    ocr = [_line("JOHN SMITH", seen=1, first_sec=OPENING_CUTOFF_S + 10)]
    tmdb = _tmdb_cast("Nobody")
    result = check_missing_actors(ocr, tmdb)
    assert len(result.missing_actors) == 0


# ── QA-06: Opening credit seen once is kept ──────────────────────────────────

def test_qa06_opening_credit_kept():
    ocr = [_line("JOHN SMITH", seen=1, first_sec=OPENING_CUTOFF_S - 1)]
    tmdb = _tmdb_cast("Nobody")
    result = check_missing_actors(ocr, tmdb)
    assert len(result.missing_actors) == 1
    assert result.missing_actors[0].is_opening_credit is True


# ── QA-07: tmdb_looks_incomplete flag ────────────────────────────────────────

def test_qa07_tmdb_looks_incomplete_true():
    # tmdb < 10 and raw_ocr >= 20
    ocr = [_line(f"ACTOR {i} LASTNAME") for i in range(25)]
    tmdb = _tmdb_cast(*[f"Other {i}" for i in range(5)])
    result = check_missing_actors(ocr, tmdb)
    assert result.tmdb_looks_incomplete is True


def test_qa07_tmdb_looks_incomplete_false_enough_tmdb():
    # tmdb >= 10 → not incomplete
    ocr = [_line(f"ACTOR {i} LASTNAME") for i in range(25)]
    tmdb = _tmdb_cast(*[f"Other Person {i}" for i in range(15)])
    result = check_missing_actors(ocr, tmdb)
    assert result.tmdb_looks_incomplete is False


def test_qa07_tmdb_looks_incomplete_false_few_ocr():
    # ocr < 20 → not incomplete
    ocr = [_line(f"ACTOR {i} LASTNAME") for i in range(5)]
    tmdb = _tmdb_cast(*[f"Other {i}" for i in range(3)])
    result = check_missing_actors(ocr, tmdb)
    assert result.tmdb_looks_incomplete is False


# ── QA-08: Summary messages ───────────────────────────────────────────────────

def test_qa08_summary_zero_findings():
    result = check_missing_actors([], [])
    assert result.summary == ""


def test_qa08_summary_one_to_three():
    ocr = [_line("MARGUERITE HICKEY"), _line("RON FIELD")]
    tmdb = _tmdb_cast("Nobody Here")
    result = check_missing_actors(ocr, tmdb)
    assert len(result.missing_actors) == 2
    assert "eksik görünüyor" in result.summary
    assert "MARGUERITE HICKEY" in result.summary
    assert "RON FIELD" in result.summary


def test_qa08_summary_four_plus():
    ocr = [_line(f"ACTOR {i} LASTNAME") for i in range(5)]
    tmdb = _tmdb_cast("Nobody Here")
    result = check_missing_actors(ocr, tmdb)
    assert len(result.missing_actors) == 5
    assert "muhtemelen eksik doldurulmuş" in result.summary


# ── QA-09: Sorting order ─────────────────────────────────────────────────────

def test_qa09_sorting_opening_first_then_seen_count():
    ocr = [
        _line("ACTOR LATE",    seen=10, first_sec=OPENING_CUTOFF_S + 5),
        _line("ACTOR OPENING", seen=2,  first_sec=OPENING_CUTOFF_S - 5),
        _line("ACTOR MANY",    seen=8,  first_sec=OPENING_CUTOFF_S + 5),
    ]
    tmdb = _tmdb_cast("Nobody")
    result = check_missing_actors(ocr, tmdb)
    names = [a.name for a in result.missing_actors]
    assert names[0] == "ACTOR OPENING"   # opening credit first
    assert names[1] == "ACTOR LATE"      # then highest seen_count
    assert names[2] == "ACTOR MANY"


# ── QA-10: to_dict() output structure ────────────────────────────────────────

def test_qa10_to_dict_structure():
    ocr = [_line("MARGUERITE HICKEY")]
    tmdb = _tmdb_cast("Tom Hanks")
    result = check_missing_actors(ocr, tmdb)
    d = result.to_dict()
    assert "tmdb_cast_count"       in d
    assert "ocr_actor_count"       in d
    assert "tmdb_looks_incomplete" in d
    assert "missing_actor_count"   in d
    assert "missing_actors"        in d
    assert "summary"               in d
    actor = d["missing_actors"][0]
    assert "name"          in actor
    assert "confidence"    in actor
    assert "seen_frames"   in actor
    assert "first_seen_sec" in actor
    assert "opening_credit" in actor
    assert "closest_tmdb"  in actor
    assert "similarity"    in actor


# ── QA-11: Works with OCRLine dataclass objects ───────────────────────────────

def test_qa11_ocr_line_dataclass():
    from dataclasses import dataclass

    @dataclass
    class FakeOCRLine:
        text: str
        avg_confidence: float
        seen_count: int
        first_seen: float

    line = FakeOCRLine(
        text="JOHN WILLIAMS",
        avg_confidence=0.93,
        seen_count=5,
        first_seen=8.0,
    )
    tmdb = _tmdb_cast("Nobody")
    result = check_missing_actors([line], tmdb)
    assert len(result.missing_actors) == 1
    assert result.missing_actors[0].name == "JOHN WILLIAMS"


# ── QA-12: Empty inputs handled gracefully ───────────────────────────────────

def test_qa12_empty_ocr():
    result = check_missing_actors([], _tmdb_cast("Tom Hanks"))
    assert isinstance(result, CreditsQA)
    assert result.missing_actors == []
    assert result.summary == ""


def test_qa12_empty_tmdb():
    ocr = [_line("JOHN SMITH")]
    result = check_missing_actors(ocr, [])
    assert isinstance(result, CreditsQA)
    # No TMDB names → nothing to compare against → 1 missing actor
    assert len(result.missing_actors) == 1


def test_qa12_both_empty():
    result = check_missing_actors([], [])
    assert isinstance(result, CreditsQA)
    assert result.missing_actors == []


# ════════════════════════════════════════════════════════════════════
#  CREW QA testleri (QA-C01 .. QA-C08)
# ════════════════════════════════════════════════════════════════════

from core.credits_qa import check_missing_crew, MissingCrewMember


def _tmdb_crew(*names):
    return [{"name": n} for n in names]


# ── QA-C01: Eksik crew üyesi tespit edilir ────────────────────────
def test_qac01_missing_crew_detected():
    ocr = [_line("IVAN VOLKMAN")]
    tmdb = _tmdb_crew("Ray Milland", "Lionel Lindon")
    result = check_missing_crew(ocr, tmdb)
    assert len(result) == 1
    assert result[0].name == "IVAN VOLKMAN"


# ── QA-C02: TMDB'de olan crew üyesi işaretlenmez ─────────────────
def test_qac02_crew_in_tmdb_not_flagged():
    ocr = [_line("Lionel Lindon")]
    tmdb = _tmdb_crew("Lionel Lindon")
    result = check_missing_crew(ocr, tmdb)
    assert len(result) == 0


# ── QA-C03: Rol başlığı blacklist'i süzer ────────────────────────
def test_qac03_role_title_filtered():
    ocr = [
        _line("Art Director"),
        _line("Assistant Director"),
        _line("Film Editor"),
        _line("Directed By"),
    ]
    result = check_missing_crew(ocr, [])
    assert len(result) == 0


# ── QA-C04: Düşük güven süzülür ──────────────────────────────────
def test_qac04_low_confidence_filtered():
    ocr = [_line("WALTER KELLER", conf=MIN_CONFIDENCE - 0.01)]
    result = check_missing_crew(ocr, [])
    assert len(result) == 0


# ── QA-C05: Tek kelime süzülür ───────────────────────────────────
def test_qac05_single_word_filtered():
    ocr = [_line("KELLER")]
    result = check_missing_crew(ocr, [])
    assert len(result) == 0


# ── QA-C06: Sıralama — açılış jeneriği önce ──────────────────────
def test_qac06_sorting():
    ocr = [
        _line("CREW LATE",    seen=10, first_sec=OPENING_CUTOFF_S + 5),
        _line("CREW OPENING", seen=2,  first_sec=OPENING_CUTOFF_S - 5),
    ]
    result = check_missing_crew(ocr, [])
    assert result[0].name == "CREW OPENING"
    assert result[1].name == "CREW LATE"


# ── QA-C07: MissingCrewMember.to_dict() yapısı ───────────────────
def test_qac07_to_dict_structure():
    ocr = [_line("IVAN VOLKMAN")]
    result = check_missing_crew(ocr, [])
    assert len(result) == 1
    d = result[0].to_dict()
    assert "name"           in d
    assert "job"            in d
    assert "confidence"     in d
    assert "seen_frames"    in d
    assert "first_seen_sec" in d
    assert "opening_credit" in d
    assert "closest_tmdb"   in d
    assert "similarity"     in d


# ── QA-C08: CreditsQA.missing_crew alanı + to_dict ───────────────
def test_qac08_credits_qa_missing_crew_field():
    import dataclasses
    from core.credits_qa import CreditsQA, MissingCrewMember

    crew_member = MissingCrewMember(
        name="IVAN VOLKMAN",
        job="Assistant Director",
        confidence=0.867,
        seen_count=9,
        first_seen_sec=59.0,
        is_opening_credit=False,
        best_tmdb_match="",
        similarity=0.0,
    )
    qa = CreditsQA(
        tmdb_cast_count=11,
        ocr_actor_count=131,
        tmdb_looks_incomplete=False,
        missing_crew=[crew_member],
    )
    d = qa.to_dict()
    assert "missing_crew_count" in d
    assert d["missing_crew_count"] == 1
    assert d["missing_crew"][0]["name"] == "IVAN VOLKMAN"
    # Eski alanlar hala orada
    assert "tmdb_cast_count" in d
    assert "missing_actors"  in d
