import copy
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core import gemini_film_credit_shadow as shadow


def _write_xml(path: Path, *, original_title: str, turkish_title: str) -> None:
    path.write_text(
        (
            "<ROOT>\n"
            f"  <ORIGINAL_TITLE>{original_title}</ORIGINAL_TITLE>\n"
            f"  <TURKISH_TITLE>{turkish_title}</TURKISH_TITLE>\n"
            "</ROOT>\n"
        ),
        encoding="utf-8",
    )


def _install_pipeline_runner_stubs() -> None:
    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_config(cls, *args, **kwargs):
            return cls()

    stub_defs = {
        "core.frame_extractor": {"FrameExtractor": _Dummy},
        "core.text_filter": {"TextFilter": _Dummy},
        "core.ocr_engine": {"OCREngine": _Dummy},
        "core.qwen_ocr_engine": {"QwenOCREngine": _Dummy},
        "core.credits_parser": {"CreditsParser": _Dummy, "_is_noise": lambda *args, **kwargs: False},
        "core.export_engine": {"ExportEngine": _Dummy, "_map_crew_to_roles": lambda *args, **kwargs: {}},
        "core.qwen_verifier": {"QwenVerifier": _Dummy},
        "core.llm_cast_filter": {"LLMCastFilter": _Dummy},
        "core.vlm_reader": {"VLMReader": _Dummy},
        "core.name_verify": {
            "NameVerifier": _Dummy,
            "_blacklist_check": lambda *args, **kwargs: (False, ""),
            "_structural_check": lambda *args, **kwargs: (True, ""),
        },
        "core.person_verify": {"PersonVerifier": _Dummy},
        "utils.stats_logger": {"StatsLogger": _Dummy},
    }

    for module_name, attrs in stub_defs.items():
        if module_name in sys.modules:
            continue
        module = ModuleType(module_name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[module_name] = module


def _get_pipeline_runner_class():
    if "core.pipeline_runner" not in sys.modules:
        _install_pipeline_runner_stubs()
    module = importlib.import_module("core.pipeline_runner")
    return module.PipelineRunner


def _make_runner():
    PipelineRunner = _get_pipeline_runner_class()
    runner = PipelineRunner.__new__(PipelineRunner)
    runner.config = {
        "database_enabled": True,
        "tmdb_api_key": "",
        "llm_filter_enabled": False,
        "imdb_enabled": False,
    }
    runner._log_messages = []
    runner._log = lambda message: runner._log_messages.append(message)
    return runner


def _ocr_line(text: str, confidence: float = 0.99):
    return SimpleNamespace(text=text, text_original=text, avg_confidence=confidence)


def _sample_credits_data() -> dict:
    return {
        "film_title": "Sirin'in Kalesi",
        "year": 2019,
        "directors": ["Reza Mirkarimi"],
        "crew": [{"name": "Reza Mirkarimi", "job": "Director"}],
        "cast": [
            {"actor_name": "Hamed Behdad", "confidence": 0.99},
            {"actor_name": "Nioosha Alipoor", "confidence": 0.97},
        ],
        "_verification_log": [
            {
                "layer": "NAMEDB",
                "role": "cast",
                "name_in": "Hamed Behdad",
                "name_out": "Hamed Behdad",
                "action": "kept",
                "reason": "exact_db",
            },
            {
                "layer": "NAMEDB",
                "role": "crew",
                "name_in": "Mohsen Gharai",
                "name_out": "Mohsen Gharaei",
                "action": "corrected",
                "reason": "fuzzy_db",
            },
        ],
    }


def _sample_credits_raw() -> dict:
    return {
        "cast": [
            {"actor_name": "Hamed Behdad", "is_verified_name": True},
            {"actor_name": "Nioosha Alipoor", "match_method": "exact_db"},
            {"actor_name": "Noise Value", "match_method": "fuzzy"},
        ]
    }


def test_build_shadow_request_includes_all_hints(tmp_path):
    xml_path = tmp_path / "film.xml"
    _write_xml(
        xml_path,
        original_title="Qasr-e Shirin",
        turkish_title="SIRIN'IN KALESI",
    )

    request = shadow.build_shadow_request(
        video_info={
            "filename": str(
                tmp_path / "evoArcadmin_SIRININ_KALESI_2019-1044-1-0000-56-1.mp4"
            )
        },
        credits_data=_sample_credits_data(),
        credits_raw=_sample_credits_raw(),
        xml_path=str(xml_path),
        filename_title="Sirin'in Kalesi",
    )

    assert request["filename_title"] == "Sirin'in Kalesi"
    assert request["xml_original_title"] == "Qasr-e Shirin"
    assert request["xml_turkish_title"] == "SIRIN'IN KALESI"
    assert request["year_hint"] == 2019
    assert request["director_hints"] == ["Reza Mirkarimi"]
    assert request["actor_hints"] == ["Hamed Behdad", "Nioosha Alipoor"]
    assert request["namedb_names"] == ["Hamed Behdad", "Mohsen Gharaei"]


def test_build_shadow_request_falls_back_to_raw_namedb_entries(tmp_path):
    xml_path = tmp_path / "film.xml"
    _write_xml(
        xml_path,
        original_title="Qasr-e Shirin",
        turkish_title="SIRIN'IN KALESI",
    )

    request = shadow.build_shadow_request(
        video_info={"filename": str(tmp_path / "film_2019.mp4")},
        credits_data={
            "film_title": "Sirin'in Kalesi",
            "year": 2019,
            "directors": [],
            "crew": [],
            "cast": [],
        },
        credits_raw=_sample_credits_raw(),
        xml_path=str(xml_path),
        filename_title="Sirin'in Kalesi",
    )

    assert request["namedb_names"] == ["Hamed Behdad", "Nioosha Alipoor"]


def test_shadow_prompt_has_no_summary_field():
    prompt = shadow.build_shadow_user_prompt(
        {
            "video_filename": "film.mp4",
            "filename_title": "Film",
            "xml_original_title": "Film",
            "xml_turkish_title": "Film",
            "year_hint": 2019,
            "director_hints": ["Director"],
            "actor_hints": ["Actor"],
            "namedb_names": ["Actor"],
        }
    )

    assert "OZET" not in shadow._SYSTEM_PROMPT
    assert "OZET" not in prompt


def test_write_shadow_sidecar_parses_valid_json(tmp_path, monkeypatch):
    xml_path = tmp_path / "film.xml"
    _write_xml(
        xml_path,
        original_title="Qasr-e Shirin",
        turkish_title="SIRIN'IN KALESI",
    )
    monkeypatch.setattr(shadow, "get_gemini_film_credit_api_key", lambda: "shadow-key")
    monkeypatch.setattr(
        shadow,
        "_call_shadow_gemini",
        lambda **kwargs: (
            json.dumps(
                {
                    "FILM_BULUNDU": True,
                    "ESLESME_GUVENI": "high",
                    "ESLESEN_BASLIK_YERLI": "SIRIN'IN KALESI",
                    "ESLESEN_BASLIK_ORJINAL": "GHASR-E SHIRIN",
                    "YIL": 2019,
                    "KAYNAK_DOMAINLER": ["imdb.com", "themoviedb.org"],
                    "KANIT": {
                        "KULLANILAN_XML_BASLIKLARI": ["SIRIN'IN KALESI"],
                        "KULLANILAN_YONETMEN_IPUCLARI": ["Reza Mirkarimi"],
                        "KULLANILAN_OYUNCU_IPUCLARI": ["Hamed Behdad"],
                        "KULLANILAN_NAMEDB_ISIMLERI": ["Hamed Behdad"],
                    },
                    "YONETMEN": ["REZA MIRKARIMI"],
                    "YONETMEN_YARDIMCISI": [],
                    "YAPIMCI": ["REZA MIRKARIMI"],
                    "KAMERA": [],
                    "GORUNTU_YONETMENI": ["MORTEZA HODAEI"],
                    "SENARYO": ["MOHSEN GHARAEI", "MOHAMMAD DAVOODI"],
                    "KURGU": ["REZA MIRKARIMI"],
                    "OYUNCULAR": ["HAMED BEHDAD", "NIOOSHA ALIPOOR"],
                }
            ),
            ["imdb.com", "themoviedb.org"],
            None,
        ),
    )

    out_path = shadow.write_shadow_sidecar(
        db_dir=tmp_path,
        video_info={"filename": str(tmp_path / "film_2019.mp4")},
        credits_data=_sample_credits_data(),
        credits_raw=_sample_credits_raw(),
        xml_path=str(xml_path),
        filename_title="Sirin'in Kalesi",
    )

    data = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["response_json"]["FILM_BULUNDU"] is True
    assert data["meta"]["grounding_domains"] == ["imdb.com", "themoviedb.org"]
    assert data["meta"]["model_reported_domains"] == ["imdb.com", "themoviedb.org"]
    assert data["meta"]["grounding_domain_source"] == "grounding_metadata"
    assert data["request"]["hints"]["xml_original_title"] == "Qasr-e Shirin"


def test_write_shadow_sidecar_canonicalizes_titles_from_hints(tmp_path, monkeypatch):
    xml_path = tmp_path / "film.xml"
    _write_xml(
        xml_path,
        original_title="Qasr-e Shirin",
        turkish_title="Şirin'in Kalesi",
    )
    monkeypatch.setattr(shadow, "get_gemini_film_credit_api_key", lambda: "shadow-key")
    monkeypatch.setattr(
        shadow,
        "_call_shadow_gemini",
        lambda **kwargs: (
            json.dumps(
                {
                    "FILM_BULUNDU": True,
                    "ESLESME_GUVENI": "high",
                    "ESLESEN_BASLIK_YERLI": "Ţirin'in Kalesi",
                    "ESLESEN_BASLIK_ORJINAL": "Qasr-e Shirin",
                    "YIL": 2019,
                    "KAYNAK_DOMAINLER": ["wikipedia.org"],
                    "KANIT": {
                        "KULLANILAN_XML_BASLIKLARI": ["Qasr-e Shirin", "Ţirin'in Kalesi"],
                        "KULLANILAN_YONETMEN_IPUCLARI": [],
                        "KULLANILAN_OYUNCU_IPUCLARI": [],
                        "KULLANILAN_NAMEDB_ISIMLERI": [],
                    },
                    "YONETMEN": ["REZA MIRKARIMI"],
                    "YONETMEN_YARDIMCISI": [],
                    "YAPIMCI": [],
                    "KAMERA": [],
                    "GORUNTU_YONETMENI": [],
                    "SENARYO": [],
                    "KURGU": [],
                    "OYUNCULAR": [],
                }
            ),
            [],
            None,
        ),
    )

    out_path = shadow.write_shadow_sidecar(
        db_dir=tmp_path,
        video_info={"filename": str(tmp_path / "film_2019.mp4")},
        credits_data=_sample_credits_data(),
        credits_raw=_sample_credits_raw(),
        xml_path=str(xml_path),
        filename_title="Şirin'in Kalesi",
    )

    data = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["response_json"]["ESLESEN_BASLIK_YERLI"] == "Şirin'in Kalesi"
    assert data["response_json"]["KANIT"]["KULLANILAN_XML_BASLIKLARI"] == [
        "Qasr-e Shirin",
        "Şirin'in Kalesi",
    ]


def test_write_shadow_sidecar_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "get_gemini_film_credit_api_key", lambda: "shadow-key")
    monkeypatch.setattr(
        shadow,
        "_call_shadow_gemini",
        lambda **kwargs: ('```json\n{"FILM_BULUNDU": true,,}\n```', [], None),
    )

    out_path = shadow.write_shadow_sidecar(
        db_dir=tmp_path,
        video_info={"filename": str(tmp_path / "film_2019.mp4")},
        credits_data=_sample_credits_data(),
        credits_raw=_sample_credits_raw(),
    )

    data = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert data["status"] == "invalid_json"
    assert data["response_json"] is None
    assert "JSON parse error" in data["meta"]["note"]


def test_write_shadow_sidecar_records_error_status(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "get_gemini_film_credit_api_key", lambda: "shadow-key")
    monkeypatch.setattr(
        shadow,
        "_call_shadow_gemini",
        lambda **kwargs: (None, [], "HTTP 500"),
    )

    out_path = shadow.write_shadow_sidecar(
        db_dir=tmp_path,
        video_info={"filename": str(tmp_path / "film_2019.mp4")},
        credits_data=_sample_credits_data(),
        credits_raw=_sample_credits_raw(),
    )

    data = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert data["status"] == "error"
    assert data["meta"]["note"] == "HTTP 500"


def test_write_shadow_sidecar_skips_without_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "get_gemini_film_credit_api_key", lambda: "")

    out_path = shadow.write_shadow_sidecar(
        db_dir=tmp_path,
        video_info={"filename": str(tmp_path / "film_2019.mp4")},
        credits_data=_sample_credits_data(),
        credits_raw=_sample_credits_raw(),
    )

    data = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert data["status"] == "skipped"
    assert data["response_json"] is None
    assert data["meta"]["note"] == "GEMINI_FILM_CREDIT_API_KEY missing"


def test_write_shadow_sidecar_does_not_mutate_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "get_gemini_film_credit_api_key", lambda: "shadow-key")
    monkeypatch.setattr(
        shadow,
        "_call_shadow_gemini",
        lambda **kwargs: ('{"FILM_BULUNDU": false}', [], None),
    )

    video_info = {"filename": str(tmp_path / "film_2019.mp4")}
    credits_data = _sample_credits_data()
    credits_raw = _sample_credits_raw()

    video_before = copy.deepcopy(video_info)
    credits_before = copy.deepcopy(credits_data)
    raw_before = copy.deepcopy(credits_raw)

    shadow.write_shadow_sidecar(
        db_dir=tmp_path,
        video_info=video_info,
        credits_data=credits_data,
        credits_raw=credits_raw,
    )

    assert video_info == video_before
    assert credits_data == credits_before
    assert credits_raw == raw_before


def test_write_shadow_sidecar_keeps_model_reported_domains_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "get_gemini_film_credit_api_key", lambda: "shadow-key")
    monkeypatch.setattr(
        shadow,
        "_call_shadow_gemini",
        lambda **kwargs: (
            json.dumps(
                {
                    "FILM_BULUNDU": True,
                    "KAYNAK_DOMAINLER": ["imdb.com", "themoviedb.org", "imdb.com"],
                    "YONETMEN": ["REZA MIRKARIMI"],
                    "YONETMEN_YARDIMCISI": [],
                    "YAPIMCI": [],
                    "KAMERA": [],
                    "GORUNTU_YONETMENI": [],
                    "SENARYO": [],
                    "KURGU": [],
                    "OYUNCULAR": [],
                }
            ),
            [],
            None,
        ),
    )

    out_path = shadow.write_shadow_sidecar(
        db_dir=tmp_path,
        video_info={"filename": str(tmp_path / "film_2019.mp4")},
        credits_data=_sample_credits_data(),
        credits_raw=_sample_credits_raw(),
    )

    data = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["meta"]["grounding_domains"] == []
    assert data["meta"]["model_reported_domains"] == ["imdb.com", "themoviedb.org"]
    assert data["meta"]["grounding_domain_source"] == "model_reported_only"


def test_call_shadow_gemini_omits_json_mime_with_google_search(monkeypatch):
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": '{"FILM_BULUNDU": true}'}]},
                            "groundingMetadata": {
                                "groundingChunks": [
                                    {"web": {"uri": "https://www.imdb.com/title/tt9806192/"}}
                                ]
                            },
                        }
                    ]
                }
            ).encode("utf-8")

    def _fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(shadow.urllib.request, "urlopen", _fake_urlopen)

    raw_text, domains, error_note = shadow._call_shadow_gemini(
        api_key="shadow-key",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com",
        user_prompt="test prompt",
    )

    assert error_note is None
    assert raw_text == '{"FILM_BULUNDU": true}'
    assert domains == ["www.imdb.com"]
    assert captured["payload"]["tools"] == [{"google_search": {}}]
    assert captured["payload"]["generationConfig"]["temperature"] == 0.1
    assert "responseMimeType" not in captured["payload"]["generationConfig"]


def test_extract_domains_supports_snake_case_and_camel_case():
    camel_candidate = {
        "groundingMetadata": {
            "groundingChunks": [
                {
                    "web": {
                        "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
                        "title": "imdb.com",
                    }
                },
                {"web": {"uri": "https://www.themoviedb.org/movie/298618", "title": "Movie"}},
            ]
        }
    }
    snake_candidate = {
        "grounding_metadata": {
            "grounding_chunks": [
                {"web": {"uri": "https://www.wikipedia.org/wiki/Legend_(2015_film)"}},
            ]
        }
    }

    assert shadow._extract_domains(camel_candidate) == ["imdb.com", "www.themoviedb.org"]
    assert shadow._extract_domains(snake_candidate) == ["www.wikipedia.org"]


def test_extract_domains_includes_citation_metadata_fallback():
    candidate = {
        "citationMetadata": {
            "citationSources": [
                {"uri": "https://www.rottentomatoes.com/m/legend_2015"},
                {"uri": "https://www.imdb.com/title/tt3569230/"},
            ]
        }
    }

    assert shadow._extract_domains(candidate) == ["www.rottentomatoes.com", "www.imdb.com"]


def test_write_database_adds_shadow_sidecar_without_mutating_inputs(tmp_path, monkeypatch):
    runner = _make_runner()
    monkeypatch.setenv("GEMINI_FILM_CREDIT_SHADOW_ENABLED", "1")
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    xml_path = xml_dir / "film.xml"
    _write_xml(
        xml_path,
        original_title="Qasr-e Shirin",
        turkish_title="SIRIN'IN KALESI",
    )

    monkeypatch.setattr(shadow, "get_gemini_film_credit_api_key", lambda: "shadow-key")
    monkeypatch.setattr(
        shadow,
        "_call_shadow_gemini",
        lambda **kwargs: (
            '{"FILM_BULUNDU": true, "YONETMEN": ["REZA MIRKARIMI"], '
            '"YONETMEN_YARDIMCISI": [], "YAPIMCI": [], "KAMERA": [], '
            '"GORUNTU_YONETMENI": [], "SENARYO": [], "KURGU": [], '
            '"OYUNCULAR": ["HAMED BEHDAD"]}',
            ["imdb.com"],
            None,
        ),
    )

    video_info = {
        "filename": str(
            tmp_path / "evoArcadmin_SIRININ_KALESI_2019-1044-1-0000-56-1.mp4"
        )
    }
    credits_data = _sample_credits_data()
    credits_raw = _sample_credits_raw()
    ocr_lines = [_ocr_line("Hamed Behdad")]

    video_before = copy.deepcopy(video_info)
    credits_before = copy.deepcopy(credits_data)
    raw_before = copy.deepcopy(credits_raw)

    runner._write_database(
        video_info=video_info,
        credits_data=credits_data,
        credits_raw=credits_raw,
        ocr_lines=ocr_lines,
        stage_stats={},
        audio_result={"transcript": []},
        work_dir=str(tmp_path),
        content_profile_name="FilmDiziV5.4",
        ts="20260331_202915",
        xml_path=str(xml_path),
    )

    stem = Path(video_info["filename"]).stem
    expected_files = [
        f"{stem}_ocr_scores.json",
        f"{stem}_credits_raw.json",
        f"{stem}_transcript.json",
        f"{stem}_debug.log",
        f"{stem}_ada1.json",
        f"{stem}_GeminiFilmCredit.json",
    ]
    for name in expected_files:
        assert (tmp_path / name).is_file(), f"eksik dosya: {name}"

    sidecar = json.loads(
        (tmp_path / f"{stem}_GeminiFilmCredit.json").read_text(encoding="utf-8")
    )
    assert sidecar["status"] == "ok"
    assert sidecar["request"]["hints"]["filename_title"] == runner._extract_film_title_from_filename(stem)
    assert video_info == video_before
    assert credits_data == credits_before
    assert credits_raw == raw_before


def test_write_database_skips_shadow_sidecar_when_disabled(tmp_path, monkeypatch):
    runner = _make_runner()
    monkeypatch.delenv("GEMINI_FILM_CREDIT_SHADOW_ENABLED", raising=False)
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    xml_path = xml_dir / "film.xml"
    _write_xml(
        xml_path,
        original_title="Qasr-e Shirin",
        turkish_title="SIRIN'IN KALESI",
    )

    video_info = {
        "filename": str(
            tmp_path / "evoArcadmin_SIRININ_KALESI_2019-1044-1-0000-56-1.mp4"
        )
    }
    credits_data = _sample_credits_data()
    credits_raw = _sample_credits_raw()
    ocr_lines = [_ocr_line("Hamed Behdad")]

    runner._write_database(
        video_info=video_info,
        credits_data=credits_data,
        credits_raw=credits_raw,
        ocr_lines=ocr_lines,
        stage_stats={},
        audio_result={"transcript": []},
        work_dir=str(tmp_path),
        content_profile_name="FilmDiziV5.4",
        ts="20260331_202915",
        xml_path=str(xml_path),
    )

    stem = Path(video_info["filename"]).stem
    assert not (tmp_path / f"{stem}_GeminiFilmCredit.json").exists()
    assert any("Shadow sidecar kapalı" in msg for msg in runner._log_messages)
