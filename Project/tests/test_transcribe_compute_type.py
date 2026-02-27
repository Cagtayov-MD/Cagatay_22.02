from core.transcribe import TranscribeStage
import sys
from unittest.mock import patch, MagicMock


def test_resolve_compute_type_defaults_for_none_like_values():
    stage = TranscribeStage()
    for raw in (None, "", "none", "null", "auto", "  AUTO  "):
        assert stage._resolve_compute_type(raw, "cpu") == "int8"
        assert stage._resolve_compute_type(raw, "cuda") == "float16"


def test_resolve_compute_type_cpu_float16_falls_back_to_int8():
    stage = TranscribeStage()
    assert stage._resolve_compute_type("float16", "cpu") == "int8"


def test_resolve_compute_type_keeps_valid_value():
    stage = TranscribeStage()
    assert stage._resolve_compute_type("int8", "cpu") == "int8"
    assert stage._resolve_compute_type(" default ", "cpu") == "default"


def test_cpu_batch_size_defaults_to_1_when_not_specified():
    """CPU'da batch_size belirtilmemişse 1 kullanılmalı."""
    logs = []
    stage = TranscribeStage(log_cb=logs.append)

    # Whisperx yok — ImportError bekleniyor; batch_size'ı log'dan kontrol et
    fake_wx = MagicMock()
    fake_wx.load_model.side_effect = ImportError("whisperx not installed")

    with patch.dict(sys.modules, {"whisperx": fake_wx}):
        with patch("core.vram_manager.VRAMManager.get_device", return_value="cpu"):
            result = stage.run("dummy.wav", batch_size=None)

    assert result["status"] == "error"
    # batch_size=1 log'da gösterilmeli (yükleniyor satırından önce hesaplanmış)
    load_log = next((l for l in logs if "yükleniyor" in l), "")
    assert "(cpu, int8)" in load_log


def test_gpu_batch_size_defaults_to_16_when_not_specified():
    """GPU'da batch_size belirtilmemişse 16 kullanılmalı."""
    logs = []
    stage = TranscribeStage(log_cb=logs.append)

    fake_wx = MagicMock()
    fake_wx.load_model.side_effect = ImportError("whisperx not installed")

    with patch.dict(sys.modules, {"whisperx": fake_wx}):
        with patch("core.vram_manager.VRAMManager.get_device", return_value="cuda"):
            result = stage.run("dummy.wav")

    assert result["status"] == "error"
    load_log = next((l for l in logs if "yükleniyor" in l), "")
    assert "(cuda, float16)" in load_log
