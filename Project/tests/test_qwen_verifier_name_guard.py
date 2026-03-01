from core.qwen_verifier import QwenVerifier, VerifyResult


def _line(text: str, conf: float = 1.0):
    return {
        "text": text,
        "avg_confidence": conf,
        "frame_path": __file__,  # existing path for Path.exists()
        "bbox": [0, 0, 10, 10],
    }


def test_high_conf_unknown_name_is_still_verified(monkeypatch):
    verifier = QwenVerifier(
        enabled=True,
        confidence_threshold=0.80,
        name_checker=lambda text: False,
    )

    monkeypatch.setattr(verifier, "is_available", lambda: True)
    monkeypatch.setattr(
        verifier,
        "_verify_single",
        lambda *args, **kwargs: VerifyResult(
            original="Nita Sereli",
            corrected="Nisa Serezli",
            was_fixed=True,
            confidence_before=1.0,
        ),
    )

    lines = [_line("Nita Sereli", 1.0)]
    out = verifier.verify(lines)

    assert out[0]["text"] == "Nisa Serezli"
    assert out[0]["text_qwen_original"] == "Nita Sereli"


def test_high_conf_known_name_is_not_forced(monkeypatch):
    verifier = QwenVerifier(
        enabled=True,
        confidence_threshold=0.80,
        name_checker=lambda text: True,
    )

    called = {"count": 0}
    monkeypatch.setattr(verifier, "is_available", lambda: True)

    def _fake_verify(*args, **kwargs):
        called["count"] += 1
        return None

    monkeypatch.setattr(verifier, "_verify_single", _fake_verify)

    lines = [_line("Nisa Serezli", 1.0)]
    out = verifier.verify(lines)

    assert out[0]["text"] == "Nisa Serezli"
    assert called["count"] == 0
