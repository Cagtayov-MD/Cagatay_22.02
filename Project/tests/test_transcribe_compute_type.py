from core.transcribe import TranscribeStage


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
