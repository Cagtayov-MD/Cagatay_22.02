from types import SimpleNamespace
import sys
import types

# pipeline_runner -> text_filter -> cv2/numpy, test ortamında stub'la
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("numpy", types.SimpleNamespace())

from core.pipeline_runner import PipelineRunner


def test_lock_when_reverse_passed_even_cast_only():
    tmdb_result = SimpleNamespace(
        reverse_passed=True,
        reverse_threshold=4.0,
        reverse_score=4.5,
        rejected=False,
    )
    lock, reason = PipelineRunner._should_lock_tmdb(
        "FilmDizi-Hybrid", tmdb_result, "cast_only"
    )
    assert lock is True
    assert reason == "reverse_validation"


def test_lock_when_title_match():
    tmdb_result = SimpleNamespace(
        reverse_passed=False,
        reverse_threshold=0.0,
        reverse_score=0.0,
        rejected=False,
    )
    lock, reason = PipelineRunner._should_lock_tmdb(
        "FilmDizi-Hybrid", tmdb_result, "title"
    )
    assert lock is True
    assert reason == "title"


def test_no_lock_when_reverse_fails():
    tmdb_result = SimpleNamespace(
        reverse_passed=False,
        reverse_threshold=4.0,
        reverse_score=1.0,
        rejected=False,
    )
    lock, reason = PipelineRunner._should_lock_tmdb(
        "FilmDizi-Hybrid", tmdb_result, "cast_only"
    )
    assert lock is False
    assert reason == ""
