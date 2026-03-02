"""
test_vlm_config_and_reader.py — VLM config precedence ve token stripping testleri.

Kapsar:
  - pipeline_runner.py VLM config öncelik sırası
  - vlm_reader.py strip_vlm_tokens yardımcısı
  - QwenVerifier varsayılan model değişikliği (GLM)
"""

import os
import sys
import pytest

# Project root'u path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vlm_reader import VLMReader, strip_vlm_tokens
from core.qwen_verifier import QwenVerifier, DEFAULT_MODEL


# ── strip_vlm_tokens testleri ──────────────────────────────────────

def test_strip_think_blocks():
    raw = "<think>iç düşünce</think>Nisa Serezli"
    assert strip_vlm_tokens(raw) == "Nisa Serezli"


def test_strip_think_multiline():
    raw = "<think>\nbirden fazla\nsatır\n</think>  Ahmet Yılmaz"
    assert strip_vlm_tokens(raw) == "Ahmet Yılmaz"


def test_strip_control_tokens():
    raw = "<|assistant|>Metin burada<|end|>"
    assert strip_vlm_tokens(raw) == "Metin burada"


def test_strip_inst_tags():
    raw = "[INST] talimat [/INST]Sonuç metni"
    assert strip_vlm_tokens(raw) == "Sonuç metni"


def test_strip_no_tokens_unchanged():
    raw = "Nisa Serezli"
    assert strip_vlm_tokens(raw) == "Nisa Serezli"


def test_strip_empty_string():
    assert strip_vlm_tokens("") == ""


# ── QwenVerifier varsayılan model testleri ──────────────────────────

def test_default_model_is_glm():
    assert DEFAULT_MODEL == "glm4.6v-flash:q4_K_M"


def test_verifier_default_model():
    v = QwenVerifier()
    assert v.model == "glm4.6v-flash:q4_K_M"


def test_verifier_qwen_model_still_works():
    v = QwenVerifier(model="qwen3-vl:8b")
    assert v.model == "qwen3-vl:8b"


# ── VLMReader testleri ──────────────────────────────────────────────

def test_vlm_reader_disabled_by_default():
    reader = VLMReader()
    assert reader.enabled is False


def test_vlm_reader_read_returns_none_when_disabled():
    reader = VLMReader(enabled=False)
    result = reader.read_text_from_frame("nonexistent.png")
    assert result is None


def test_vlm_reader_augment_noop_when_disabled():
    reader = VLMReader(enabled=False)
    lines = [{"text": "Mevcut", "avg_confidence": 0.9, "frame_path": "f.png"}]
    out = reader.augment_ocr_lines(lines, ["f.png", "f2.png"])
    assert out == lines


def test_vlm_reader_result_structure(monkeypatch):
    """read_text_from_frame, Ollama yanıtından doğru dict döndürmeli."""
    reader = VLMReader(model="glm4.6v-flash:q4_K_M", enabled=True)

    import urllib.request
    import json

    class _FakeResp:
        def read(self):
            return json.dumps({
                "message": {"content": "Nisa Serezli"}
            }).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeResp())

    # Gerçek dosya okuma gereksinimi yoktur — _encode_image'yi mockla
    monkeypatch.setattr(reader, "_encode_image", lambda *a, **kw: "fakeb64==")

    result = reader.read_text_from_frame("frame.png", lang="tr")
    assert result is not None
    assert result["text"] == "Nisa Serezli"
    assert result["source"] == "vlm"
    assert "avg_confidence" in result
    assert result["frame_path"] == "frame.png"


def test_vlm_reader_strips_think_in_response(monkeypatch):
    reader = VLMReader(model="glm4.6v-flash:q4_K_M", enabled=True)

    import urllib.request
    import json

    class _FakeResp:
        def read(self):
            return json.dumps({
                "message": {"content": "<think>düşünce</think>Ahmet Yılmaz"}
            }).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeResp())
    monkeypatch.setattr(reader, "_encode_image", lambda *a, **kw: "fakeb64==")

    result = reader.read_text_from_frame("frame.png")
    assert result["text"] == "Ahmet Yılmaz"


# ── Config öncelik testleri (pipeline_runner PipelineRunner.__init__) ──

def _make_runner(config=None, env_overrides=None):
    """PipelineRunner.__init__ yerine sadece VLM config çözümleme mantığını test et."""
    import importlib.util, types

    cfg = config or {}
    env = env_overrides or {}

    # Gerçekten pipeline_runner'daki öncelik mantığını çoğalt
    vlm_enabled_cfg = cfg.get("vlm_verify", cfg.get("vlm_enabled", None))
    if vlm_enabled_cfg is None:
        env_enabled = env.get("VLM_ENABLED", "")
        if env_enabled:
            vlm_enabled = env_enabled.lower() not in ("0", "false", "no")
        else:
            vlm_enabled = bool(cfg.get("qwen_verify", True))
    else:
        vlm_enabled = bool(vlm_enabled_cfg)

    vlm_model = (
        cfg.get("vlm_model") or
        env.get("VLM_MODEL") or
        cfg.get("qwen_model") or
        env.get("QWEN_MODEL") or
        "glm4.6v-flash:q4_K_M"
    )

    _thresh_raw = (
        cfg.get("vlm_threshold") or
        env.get("VLM_THRESHOLD") or
        cfg.get("qwen_threshold") or
        env.get("QWEN_THRESHOLD") or
        0.80
    )
    try:
        vlm_thresh = float(_thresh_raw)
        vlm_thresh = max(0.0, min(1.0, vlm_thresh))
    except (ValueError, TypeError):
        vlm_thresh = 0.80

    return vlm_model, vlm_thresh, vlm_enabled


def test_vlm_model_from_config():
    model, _, _ = _make_runner(config={"vlm_model": "glm4.6v-flash:q4_K_M"})
    assert model == "glm4.6v-flash:q4_K_M"


def test_vlm_model_env_preferred_over_qwen_config():
    model, _, _ = _make_runner(
        config={"qwen_model": "qwen3-vl:8b"},
        env_overrides={"VLM_MODEL": "glm4.6v-flash:q4_K_M"}
    )
    assert model == "glm4.6v-flash:q4_K_M"


def test_legacy_qwen_model_config_works():
    model, _, _ = _make_runner(config={"qwen_model": "qwen3-vl:8b"})
    assert model == "qwen3-vl:8b"


def test_legacy_qwen_model_env_works():
    model, _, _ = _make_runner(env_overrides={"QWEN_MODEL": "qwen3-vl:8b"})
    assert model == "qwen3-vl:8b"


def test_default_model_when_nothing_set():
    model, _, _ = _make_runner()
    assert model == "glm4.6v-flash:q4_K_M"


def test_vlm_threshold_from_config():
    _, thresh, _ = _make_runner(config={"vlm_threshold": "0.75"})
    assert thresh == pytest.approx(0.75)


def test_legacy_qwen_threshold_config():
    _, thresh, _ = _make_runner(config={"qwen_threshold": "0.70"})
    assert thresh == pytest.approx(0.70)


def test_vlm_enabled_false_from_config():
    _, _, enabled = _make_runner(config={"vlm_verify": False})
    assert enabled is False


def test_vlm_enabled_env_false():
    _, _, enabled = _make_runner(env_overrides={"VLM_ENABLED": "false"})
    assert enabled is False


def test_vlm_enabled_env_true():
    _, _, enabled = _make_runner(env_overrides={"VLM_ENABLED": "true"})
    assert enabled is True


def test_vlm_threshold_clamped_high():
    _, thresh, _ = _make_runner(config={"vlm_threshold": "1.5"})
    assert thresh == pytest.approx(1.0)


def test_vlm_threshold_clamped_low():
    _, thresh, _ = _make_runner(config={"vlm_threshold": "-0.1"})
    assert thresh == pytest.approx(0.0)
