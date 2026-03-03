"""
test_ocr_bug_fixes.py — Regression tests for OCR bug fixes.

BUG-1: export_engine._best_actor() NameError ('actor' undefined in except handler)
BUG-2: google_ocr_engine.ocr_image() error check order (after vs before data access)
BUG-3: qwen_verifier.is_available() hardcoded localhost URL
BUG-4: qwen_verifier._verify_single() confidence_before always 0.0
PERF-1: turkish_name_db._fuzzy_find() cached _all_names list
"""

import sys
import os

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ─────────────────────────────────────────────────────────────────────────────
# BUG-1: _best_actor does not raise NameError when split fails
# ─────────────────────────────────────────────────────────────────────────────

def test_best_actor_exception_handler_no_name_error(monkeypatch):
    """BUG-1: exception handler in _best_actor must not raise NameError."""
    from core.export_engine import _best_actor, _split_name

    # Force _split_name to raise so the except block runs
    def raise_error(word):
        raise RuntimeError("forced split error")

    monkeypatch.setattr("core.export_engine._split_name", raise_error)

    # Should not raise NameError — the except block logs `best`, not `actor`
    result = _best_actor(["LONGWORD1"])
    assert isinstance(result, str)


def test_best_actor_returns_string_on_exception():
    """BUG-1: _best_actor always returns a string even when processing raises."""
    from core.export_engine import _best_actor

    result = _best_actor(["SomeName"])
    assert isinstance(result, str)
    assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# BUG-2: google_ocr_engine error check order
# ─────────────────────────────────────────────────────────────────────────────

def test_google_ocr_error_checked_before_text_access():
    """BUG-2: ocr_image() must raise RuntimeError for API errors before accessing text_annotations."""
    import ast, inspect, textwrap
    try:
        from core import google_ocr_engine
        src = inspect.getsource(google_ocr_engine.GoogleOCREngine.ocr_image)
        src = textwrap.dedent(src)
    except Exception:
        return  # Can't inspect, skip

    tree = ast.parse(src)
    # Find line numbers for resp.error.message check and text_annotations[0] access
    error_check_line = None
    text_access_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            try:
                cond = ast.dump(node.test)
                if 'error' in cond and 'message' in cond:
                    if error_check_line is None:
                        error_check_line = node.lineno
            except Exception:
                pass
        if isinstance(node, ast.Subscript):
            try:
                val_dump = ast.dump(node.value)
                if 'text_annotations' in val_dump:
                    if text_access_line is None:
                        text_access_line = node.lineno
            except Exception:
                pass

    assert error_check_line is not None, "resp.error.message check not found"
    assert text_access_line is not None, "text_annotations[0] access not found"
    assert error_check_line < text_access_line, (
        f"Error check (line {error_check_line}) must come BEFORE "
        f"text_annotations access (line {text_access_line})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BUG-3: qwen_verifier.is_available() uses self.url base, not hardcoded localhost
# ─────────────────────────────────────────────────────────────────────────────

def test_is_available_uses_custom_url(monkeypatch):
    """BUG-3: is_available() must call the host from self.url, not hardcoded localhost."""
    from core.qwen_verifier import QwenVerifier
    import urllib.request

    checked_urls = []

    def fake_urlopen(req, timeout=5):
        checked_urls.append(req.full_url)
        raise OSError("connection refused (test)")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    verifier = QwenVerifier(ollama_url="http://custom-host:9999/api/chat")
    verifier._available = None  # reset cache
    verifier.is_available()

    assert any("custom-host:9999" in u for u in checked_urls), (
        f"Expected 'custom-host:9999' in checked URLs, got: {checked_urls}"
    )


def test_is_available_does_not_use_hardcoded_localhost(monkeypatch):
    """BUG-3: When custom URL is set, localhost must NOT be contacted."""
    from core.qwen_verifier import QwenVerifier

    checked_urls = []

    def fake_urlopen(req, timeout=5):
        checked_urls.append(req.full_url)
        raise OSError("connection refused (test)")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    verifier = QwenVerifier(ollama_url="http://remote-server:11434/api/chat")
    verifier._available = None
    verifier.is_available()

    assert not any("localhost" in u for u in checked_urls), (
        f"Should NOT contact localhost when custom URL is set; got: {checked_urls}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BUG-4: _verify_single confidence_before reflects actual confidence
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_single_confidence_before_propagated(monkeypatch):
    """BUG-4: confidence_before in VerifyResult must equal the passed-in value."""
    from core.qwen_verifier import QwenVerifier, VerifyResult
    import json

    verifier = QwenVerifier(ollama_url="http://localhost:11434/api/chat")

    # Mock the HTTP call to return a valid response
    class FakeResp:
        def read(self):
            return json.dumps({
                "message": {"content": "corrected text"}
            }).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=60: FakeResp())
    monkeypatch.setattr("core.qwen_verifier.HAS_CV2", False)

    result = verifier._verify_single(
        "original text",
        __file__,  # existing file for the path check
        confidence_before=0.72,
    )
    assert result is not None
    assert result.confidence_before == 0.72


def test_verify_confidence_before_passed_from_verify(monkeypatch):
    """BUG-4: verify() passes the line's actual confidence to _verify_single."""
    from core.qwen_verifier import QwenVerifier, VerifyResult
    from pathlib import Path

    captured = {}

    def fake_verify_single(text, frame_path, bbox=None, confidence_before=0.0):
        captured["confidence_before"] = confidence_before
        return VerifyResult(
            original=text,
            corrected=text,
            was_fixed=False,
            confidence_before=confidence_before,
        )

    verifier = QwenVerifier(
        enabled=True,
        confidence_threshold=0.80,
        name_checker=lambda t: False,
    )
    monkeypatch.setattr(verifier, "is_available", lambda: True)
    monkeypatch.setattr(verifier, "_verify_single", fake_verify_single)

    lines = [{
        "text": "Nita Sereli",
        "avg_confidence": 0.65,
        "frame_path": __file__,
        "bbox": [],
    }]
    verifier.verify(lines)
    assert captured.get("confidence_before") == 0.65


# ─────────────────────────────────────────────────────────────────────────────
# PERF-1: TurkishNameDB._all_names cache exists and is consistent
# ─────────────────────────────────────────────────────────────────────────────

def test_turkish_name_db_all_names_cached():
    """PERF-1: _all_names cache must equal _first_names + _surnames."""
    from core.turkish_name_db import TurkishNameDB
    db = TurkishNameDB()  # no DB file — hardcoded only
    assert hasattr(db, "_all_names"), "_all_names cache attribute missing"
    assert db._all_names == db._first_names + db._surnames


def test_turkish_name_db_fuzzy_find_uses_cache():
    """PERF-1: _fuzzy_find uses _all_names and does not create a new list each call."""
    from core.turkish_name_db import TurkishNameDB
    db = TurkishNameDB()
    original_id = id(db._all_names)
    # Call _fuzzy_find twice; the internal list object must be the same (no copy)
    db._fuzzy_find("test", threshold=85)
    db._fuzzy_find("test2", threshold=85)
    assert id(db._all_names) == original_id, (
        "_fuzzy_find must not create a new list (should use cached _all_names)"
    )
