"""test_ada1_output.py — ADA1 OCR ham çıktı listesi testleri.

Senaryolar:
  ADA1-01: Temel çıktı — tüm alanlar doğru üretiliyor mu?
  ADA1-02: ocr = ham metin (text_original varsa), norm1 = repair sonrası
  ADA1-03: kategori eşleştirme — cast→oyuncu, crew→ekip, company→sirket, unknown→bilinmiyor
  ADA1-04: norm2 — cdata'da bulunan isim norm1'den farklıysa doğru mu alınıyor?
  ADA1-05: Boş confidence / boş satır dayanıklılığı
  ADA1-06: bolum_no regex — dosya adından doğru çıkarılıyor mu?
  ADA1-07: Pipeline'a sıfır etkisi — cdata değişmiyor mu?
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from core.pipeline_runner import PipelineRunner


# ── Yardımcı: minimal PipelineRunner instance (ağ/model yok) ──────────────────

def _make_runner():
    cfg = {
        "tmdb_api_key": "",
        "database_enabled": False,
        "llm_filter_enabled": False,
        "imdb_enabled": False,
    }
    runner = PipelineRunner.__new__(PipelineRunner)
    runner.config = cfg
    runner._log_messages = []
    runner._log = lambda msg: runner._log_messages.append(msg)
    return runner


def _ocr_line(text, text_original=None, confidence=0.95):
    """Dataclass benzeri nesne döndür (SimpleNamespace)."""
    ns = SimpleNamespace()
    ns.text = text
    ns.text_original = text_original or ""
    ns.avg_confidence = confidence
    return ns


def _cdata(cast=None, crew=None, companies=None, film_title="TEST"):
    return {
        "film_title": film_title,
        "cast":       cast or [],
        "crew":       crew or [],
        "companies":  companies or [],
    }


# ── ADA1-01: Temel çıktı yapısı ───────────────────────────────────────────────

def test_ada1_basic_structure():
    runner = _make_runner()
    lines = [_ocr_line("TOLGA SARITAŞ", confidence=0.99)]
    cd = _cdata(cast=[{"actor_name": "Tolga Sarıtaş"}])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        runner._write_ada1(lines, "2021-0016-0-0109-90-1-TESKILAT", cd, path)
        data = json.loads(open(path, encoding="utf-8").read())
    finally:
        os.unlink(path)

    assert data["video_id"]   == "2021-0016-0-0109-90-1-TESKILAT"
    assert data["video_title"] == "TEST"
    assert data["bolum_no"]   == "0109"
    assert "olusturulma" in data
    assert len(data["satirlar"]) == 1

    row = data["satirlar"][0]
    assert "ocr"        in row
    assert "norm1"      in row
    assert "norm2"      in row
    assert "confidence" in row
    assert "kategori"   in row


# ── ADA1-02: ocr = ham metin, norm1 = repair sonrası ─────────────────────────

def test_ada1_ocr_vs_norm1():
    runner = _make_runner()
    # text_original = ham (repair öncesi), text = repair sonrası
    line = _ocr_line(text="Tolga Sarıtaş", text_original="TOLGA SARITAŞ", confidence=0.99)
    cd = _cdata(cast=[{"actor_name": "Tolga Sarıtaş"}])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        runner._write_ada1([line], "2021-0016-0-0109-90-1-TEST", cd, path)
        row = json.loads(open(path, encoding="utf-8").read())["satirlar"][0]
    finally:
        os.unlink(path)

    assert row["ocr"]   == "TOLGA SARITAŞ"   # ham metin
    assert row["norm1"] == "Tolga Sarıtaş"    # repair sonrası


# ── ADA1-03: Kategori eşleştirme ─────────────────────────────────────────────

def test_ada1_kategori_mapping():
    runner = _make_runner()
    lines = [
        _ocr_line("Ali Veli"),
        _ocr_line("Mehmet Yılmaz"),
        _ocr_line("Star Film"),
        _ocr_line("Bilinmeyen Kişi"),
    ]
    cd = _cdata(
        cast      =[{"actor_name": "Ali Veli"}],
        crew      =[{"name": "Mehmet Yılmaz"}],
        companies =[{"name": "Star Film"}],
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        runner._write_ada1(lines, "2021-0016-0-0109-90-1-TEST", cd, path)
        rows = json.loads(open(path, encoding="utf-8").read())["satirlar"]
    finally:
        os.unlink(path)

    kategoriler = {r["norm1"]: r["kategori"] for r in rows}
    assert kategoriler["Ali Veli"]       == "oyuncu"
    assert kategoriler["Mehmet Yılmaz"]  == "ekip"
    assert kategoriler["Star Film"]      == "sirket"
    assert kategoriler["Bilinmeyen Kişi"] == "bilinmiyor"


# ── ADA1-04: norm2 — Gemini/IMDb düzeltmesi varsa alınıyor mu? ───────────────

def test_ada1_norm2_differs_from_norm1():
    runner = _make_runner()
    # OCR "BURAK ARLIEL" okudu, cdata'da düzeltilmiş hali "Burak Arlıel"
    line = _ocr_line(text="BURAK ARLIEL", text_original="BURAK ARLIEL")
    cd = _cdata(crew=[{"name": "Burak Arlıel"}])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        runner._write_ada1([line], "2021-0016-0-0109-90-1-TEST", cd, path)
        row = json.loads(open(path, encoding="utf-8").read())["satirlar"][0]
    finally:
        os.unlink(path)

    assert row["norm1"] == "BURAK ARLIEL"
    assert row["norm2"] == "Burak Arlıel"
    assert row["kategori"] == "ekip"


# ── ADA1-05: Boş confidence ve boş text dayanıklılığı ────────────────────────

def test_ada1_empty_and_zero_confidence():
    runner = _make_runner()
    lines = [
        _ocr_line("Geçerli İsim", confidence=0.0),
        _ocr_line("",              confidence=0.9),   # boş text → atlanmalı
    ]
    cd = _cdata()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        runner._write_ada1(lines, "2021-0016-0-0109-90-1-TEST", cd, path)
        data = json.loads(open(path, encoding="utf-8").read())
    finally:
        os.unlink(path)

    assert len(data["satirlar"]) == 1           # boş text atlandı
    assert data["satirlar"][0]["confidence"] == 0.0
    assert data["satirlar"][0]["norm1"] == "Geçerli İsim"


# ── ADA1-06: bolum_no regex ───────────────────────────────────────────────────

def test_ada1_bolum_no_extraction():
    runner = _make_runner()
    cd = _cdata()

    cases = [
        ("2021-0016-0-0109-90-1-TESKILAT", "0109"),
        ("2023-0042-1-0250-90-1-DIZIAD",   "0250"),
        ("dosya_adi_olmayan_format",        ""),
    ]

    for vname, expected_bolum in cases:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            runner._write_ada1([], vname, cd, path)
            data = json.loads(open(path, encoding="utf-8").read())
        finally:
            os.unlink(path)
        assert data["bolum_no"] == expected_bolum, f"{vname} → beklenen '{expected_bolum}', gelen '{data['bolum_no']}'"


# ── ADA1-07: cdata değişmiyor (pipeline sıfır etki) ──────────────────────────

def test_ada1_does_not_mutate_cdata():
    runner = _make_runner()
    cd = _cdata(
        cast=[{"actor_name": "Ali Veli"}],
        crew=[{"name": "Mehmet Yılmaz"}],
    )
    import copy
    cd_before = copy.deepcopy(cd)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        runner._write_ada1([_ocr_line("Ali Veli")], "test", cd, path)
    finally:
        os.unlink(path)

    assert cd == cd_before, "cdata değişmiş olmamalı"


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_ada1_basic_structure,
        test_ada1_ocr_vs_norm1,
        test_ada1_kategori_mapping,
        test_ada1_norm2_differs_from_norm1,
        test_ada1_empty_and_zero_confidence,
        test_ada1_bolum_no_extraction,
        test_ada1_does_not_mutate_cdata,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} geçti")
