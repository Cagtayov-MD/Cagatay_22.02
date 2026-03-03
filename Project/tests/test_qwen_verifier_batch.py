"""
test_qwen_verifier_batch.py — Frame-bazlı batch doğrulama testleri.

BATCH-01: Aynı frame'den gelen satırlar tek batch çağrısında gruplanmalı
BATCH-02: Batch parse başarısız olunca _verify_single fallback çalışmalı
BATCH-03: Şüpheli satır yoksa batch çağrısı yapılmamalı
BATCH-04: Tek satırlık batch sonucu metin güncellenmeli
BATCH-05: Batch düzeltmesi sonrası text_qwen_original kaydedilmeli
"""

import os
import sys

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core.qwen_verifier import QwenVerifier, VerifyResult


def _line(text, conf=0.65, frame_path=None, bbox=None):
    return {
        "text": text,
        "avg_confidence": conf,
        "frame_path": frame_path or __file__,
        "bbox": bbox or [0, 0, 10, 10],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH-01: Aynı frame gruplama
# ═══════════════════════════════════════════════════════════════════════════════

def test_batch_groups_same_frame_lines(monkeypatch):
    """BATCH-01: Aynı frame'den gelen satırlar tek batch çağrısında gruplanmalı."""
    verifier = QwenVerifier(enabled=True, confidence_threshold=0.80)
    monkeypatch.setattr(verifier, "is_available", lambda: True)

    batch_calls = []

    def fake_batch(group, frame_path):
        batch_calls.append((frame_path, len(group)))
        return {}

    monkeypatch.setattr(verifier, "_verify_batch", fake_batch)
    monkeypatch.setattr(verifier, "_verify_single", lambda *a, **kw: None)

    # İki gerçek dosya yolu (var olan) kullan
    frame1 = __file__
    frame2 = os.__file__

    lines = [
        _line("DYAYUCE", frame_path=frame1),
        _line("HALUKK", frame_path=frame1),
        _line("VELII", frame_path=frame2),
    ]
    verifier.verify(lines)

    assert len(batch_calls) == 2, (
        f"2 farklı frame → 2 batch çağrısı olmalı, {len(batch_calls)} geldi"
    )
    groups_by_frame = {fp: cnt for fp, cnt in batch_calls}
    assert groups_by_frame[frame1] == 2, "frame1'den 2 satır gruplanmalı"
    assert groups_by_frame[frame2] == 1, "frame2'den 1 satır gruplanmalı"


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH-02: Batch başarısız → _verify_single fallback
# ═══════════════════════════════════════════════════════════════════════════════

def test_batch_parse_failure_falls_back_to_verify_single(monkeypatch):
    """BATCH-02: Batch boş dict döndürünce _verify_single fallback çalışmalı."""
    verifier = QwenVerifier(enabled=True, confidence_threshold=0.80)
    monkeypatch.setattr(verifier, "is_available", lambda: True)

    batch_called = {"count": 0}
    single_called = {"texts": []}

    def fake_batch(group, frame_path):
        batch_called["count"] += 1
        return {}  # boş dict → fallback tetikler

    def fake_single(text, frame_path, bbox=None, confidence_before=0.0):
        single_called["texts"].append(text)
        return VerifyResult(
            original=text,
            corrected=text,
            was_fixed=False,
            confidence_before=confidence_before,
        )

    monkeypatch.setattr(verifier, "_verify_batch", fake_batch)
    monkeypatch.setattr(verifier, "_verify_single", fake_single)

    lines = [
        _line("DYAYUCE"),
        _line("HALUKK"),
    ]
    verifier.verify(lines)

    assert batch_called["count"] == 1, "Aynı frame → 1 batch çağrısı"
    assert len(single_called["texts"]) == 2, "Batch başarısız → 2 single çağrısı"
    assert "DYAYUCE" in single_called["texts"]
    assert "HALUKK" in single_called["texts"]


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH-03: Şüpheli satır yok → batch çağrısı yapılmamalı
# ═══════════════════════════════════════════════════════════════════════════════

def test_empty_suspicious_lines_no_batch_call(monkeypatch):
    """BATCH-03: Şüpheli satır yoksa _verify_batch çağrılmamalı."""
    verifier = QwenVerifier(enabled=True, confidence_threshold=0.80)
    monkeypatch.setattr(verifier, "is_available", lambda: True)

    batch_called = {"count": 0}

    def fake_batch(group, frame_path):
        batch_called["count"] += 1
        return {}

    monkeypatch.setattr(verifier, "_verify_batch", fake_batch)

    # Yüksek confidence + kaliteli metin → şüpheli değil
    lines = [
        _line("Nisa Serezli", conf=1.0),
        _line("Haluk Bilginer", conf=1.0),
    ]
    verifier.verify(lines)

    assert batch_called["count"] == 0, "Şüpheli satır yokken batch çağrılmamalı"


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH-04: Tek satırlık batch
# ═══════════════════════════════════════════════════════════════════════════════

def test_single_line_batch_corrects_text(monkeypatch):
    """BATCH-04: Tek satırlık batch sonucu metin güncellenmeli."""
    verifier = QwenVerifier(enabled=True, confidence_threshold=0.80)
    monkeypatch.setattr(verifier, "is_available", lambda: True)

    def fake_batch(group, frame_path):
        _, original, conf, _ = group[0]
        return {
            0: VerifyResult(
                original=original,
                corrected="Nisa Serezli",
                was_fixed=True,
                confidence_before=conf,
            )
        }

    monkeypatch.setattr(verifier, "_verify_batch", fake_batch)

    lines = [_line("Nita Sereli", conf=0.65)]
    out = verifier.verify(lines)

    assert out[0]["text"] == "Nisa Serezli"


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH-05: text_qwen_original kaydedilmeli
# ═══════════════════════════════════════════════════════════════════════════════

def test_batch_saves_text_qwen_original(monkeypatch):
    """BATCH-05: Batch düzeltmesi sonrası text_qwen_original kaydedilmeli."""
    verifier = QwenVerifier(enabled=True, confidence_threshold=0.80)
    monkeypatch.setattr(verifier, "is_available", lambda: True)

    def fake_batch(group, frame_path):
        _, original, conf, _ = group[0]
        return {
            0: VerifyResult(
                original=original,
                corrected="Düzeltilmiş",
                was_fixed=True,
                confidence_before=conf,
            )
        }

    monkeypatch.setattr(verifier, "_verify_batch", fake_batch)

    lines = [_line("DYAYUCE", conf=0.65)]
    out = verifier.verify(lines)

    assert out[0]["text"] == "Düzeltilmiş"
    assert out[0].get("text_qwen_original") == "DYAYUCE", (
        "text_qwen_original orijinal metin olarak kaydedilmeli"
    )
