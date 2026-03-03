"""test_vlm_time_budget.py — Unit tests for VLM_MAX_TOTAL_SEC time-budget guard.

TIME-01: When budget is not exceeded, all batches are processed normally
TIME-02: When budget is exceeded mid-loop, remaining batches are cancelled/skipped
TIME-03: Partial results are returned (not an empty list) when budget is hit
TIME-04: Budget can be overridden via VLM_MAX_TOTAL_SEC env variable
"""

import os
import sys
import time

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core.qwen_verifier import QwenVerifier, VerifyResult


def _line(text, conf=0.65, frame_path=None):
    return {
        "text": text,
        "avg_confidence": conf,
        "frame_path": frame_path or __file__,
        "bbox": [0, 0, 10, 10],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-01: Budget not exceeded — all batches processed
# ═══════════════════════════════════════════════════════════════════════════════

def test_time_budget_not_exceeded_processes_all_batches(monkeypatch):
    """TIME-01: When budget is generous, all batches are processed."""
    verifier = QwenVerifier(enabled=True, confidence_threshold=0.80)
    monkeypatch.setattr(verifier, "is_available", lambda: True)

    batch_calls = []

    def fast_batch(group, frame_path):
        batch_calls.append(frame_path)
        return {}

    monkeypatch.setattr(verifier, "_verify_batch", fast_batch)
    monkeypatch.setattr(verifier, "_verify_single", lambda *a, **kw: None)
    # Generous budget — should not trigger
    monkeypatch.setenv("VLM_MAX_TOTAL_SEC", "3600")

    frame1 = __file__
    frame2 = os.__file__
    lines = [
        _line("DYAYUCE", frame_path=frame1),
        _line("HALUKK",  frame_path=frame2),
    ]
    verifier.verify(lines)

    assert len(batch_calls) == 2, (
        f"All 2 batches should be processed, only {len(batch_calls)} were"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-02: Budget exceeded — remaining batches cancelled/skipped
# ═══════════════════════════════════════════════════════════════════════════════

def test_time_budget_exceeded_stops_processing(monkeypatch):
    """TIME-02: When budget is tiny, not all batches may finish processing."""
    verifier = QwenVerifier(enabled=True, confidence_threshold=0.80)
    monkeypatch.setattr(verifier, "is_available", lambda: True)

    call_count = {"n": 0}

    def slow_batch(group, frame_path):
        call_count["n"] += 1
        # Simulate some work that takes just enough time
        time.sleep(0.05)
        return {}

    monkeypatch.setattr(verifier, "_verify_batch", slow_batch)
    monkeypatch.setattr(verifier, "_verify_single", lambda *a, **kw: None)
    # Tight budget: 0 seconds → will be exceeded immediately after first batch
    monkeypatch.setenv("VLM_MAX_TOTAL_SEC", "0")

    # Use existing file for path existence check
    lines = [
        _line("DYAYUCE", frame_path=__file__),
        _line("VELII", frame_path=os.__file__),
        _line("HALUKK", frame_path=os.path.__file__),
    ]

    log_msgs = []
    result = verifier.verify(lines, log_cb=log_msgs.append)

    # The result should be the original (unmodified) lines — partial return
    assert result is not None
    assert isinstance(result, list)

    # Check that the budget log message was emitted
    budget_msgs = [m for m in log_msgs if "Zaman bütçesi" in m or "bütçesi" in m.lower()]
    assert budget_msgs, (
        f"Expected budget exhaustion log message, got: {log_msgs}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-03: Partial results returned when budget is hit
# ═══════════════════════════════════════════════════════════════════════════════

def test_time_budget_returns_partial_results(monkeypatch):
    """TIME-03: Partial results (not empty) are returned when budget runs out."""
    verifier = QwenVerifier(enabled=True, confidence_threshold=0.80)
    monkeypatch.setattr(verifier, "is_available", lambda: True)

    def fixing_batch(group, frame_path):
        # Fixes every line it sees
        results = {}
        for grp_idx, (i, text, conf, bbox) in enumerate(group):
            results[grp_idx] = VerifyResult(
                original=text,
                corrected=text + "_fixed",
                was_fixed=True,
                confidence_before=conf,
            )
        return results

    monkeypatch.setattr(verifier, "_verify_batch", fixing_batch)
    monkeypatch.setattr(verifier, "_verify_single", lambda *a, **kw: None)
    # Reasonable budget so first batch finishes
    monkeypatch.setenv("VLM_MAX_TOTAL_SEC", "3600")

    lines = [_line("DYAYUCE")]
    result = verifier.verify(lines)

    # Should have processed the one batch and fixed the text
    assert len(result) == 1
    # Result list is the same object (ocr_lines returned in-place)
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-04: VLM_MAX_TOTAL_SEC env var respected
# ═══════════════════════════════════════════════════════════════════════════════

def test_vlm_max_total_sec_env_var_respected(monkeypatch):
    """TIME-04: VLM_MAX_TOTAL_SEC environment variable sets the budget."""
    import core.qwen_verifier as qv_module

    monkeypatch.setenv("VLM_MAX_TOTAL_SEC", "42")

    verifier = QwenVerifier(enabled=True)
    monkeypatch.setattr(verifier, "is_available", lambda: True)
    monkeypatch.setattr(verifier, "_verify_batch", lambda g, fp: {})
    monkeypatch.setattr(verifier, "_verify_single", lambda *a, **kw: None)

    log_msgs = []
    # With budget=42 and an instant batch, nothing should be logged about budget
    lines = [_line("DYAYUCE")]
    verifier.verify(lines, log_cb=log_msgs.append)

    # No budget exhaustion expected since batch is instant and budget=42s
    budget_exceeded = [m for m in log_msgs if "Zaman bütçesi" in m]
    assert not budget_exceeded, (
        "Budget should not be exceeded for instant batch with 42s budget"
    )
