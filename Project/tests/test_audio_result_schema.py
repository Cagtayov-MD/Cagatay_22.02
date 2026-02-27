import json
from pathlib import Path


def test_audio_result_schema_exists_and_has_required_keys():
    schema_path = Path(__file__).parent.parent / "schemas" / "audio_result.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = set(schema.get("required", []))
    assert {"version", "status", "processing_time_sec", "transcript", "speakers", "stages"}.issubset(required)
