"""
test_xml_sidecar.py — xml_sidecar modülü için birim testleri.

Covers:
  XS-01: find_xml_sidecar mevcut XML dosyasını bulur
  XS-02: find_xml_sidecar XML yoksa None döndürür
  XS-03: resolve_xml_sidecar XML bulunca başarı logu yazar ve yolu döndürür
  XS-04: resolve_xml_sidecar XML yoksa uyarı logu yazar ve None döndürür
  XS-05: resolve_xml_sidecar log_cb=None iken hata fırlatmaz
"""

import sys
import os

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import tempfile
from pathlib import Path


def test_find_xml_sidecar_found():
    """XS-01: Aynı dizinde eşleşen XML varsa tam yolu döndürür."""
    from core.xml_sidecar import find_xml_sidecar

    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "test_film.mp4"
        xml   = Path(tmpdir) / "test_film.xml"
        video.touch()
        xml.touch()

        result = find_xml_sidecar(str(video))
        assert result == str(xml)


def test_find_xml_sidecar_not_found():
    """XS-02: XML yoksa None döndürür."""
    from core.xml_sidecar import find_xml_sidecar

    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "test_film.mp4"
        video.touch()

        result = find_xml_sidecar(str(video))
        assert result is None


def test_resolve_xml_sidecar_found_logs_and_returns():
    """XS-03: XML bulununca başarı mesajı loglanır ve yol döndürülür."""
    from core.xml_sidecar import resolve_xml_sidecar

    logs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "GÜN_BATISI.mp4"
        xml   = Path(tmpdir) / "GÜN_BATISI.xml"
        video.touch()
        xml.touch()

        result = resolve_xml_sidecar(str(video), log_cb=logs.append)

    assert result == str(xml)
    assert len(logs) == 1
    assert "✅" in logs[0]
    assert "XML dosyası başarıyla çekildi" in logs[0]
    assert "GÜN_BATISI.xml" in logs[0]


def test_resolve_xml_sidecar_not_found_logs_warning():
    """XS-04: XML yoksa uyarı mesajı loglanır ve None döndürülür."""
    from core.xml_sidecar import resolve_xml_sidecar

    logs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "GÜN_BATISI.mp4"
        video.touch()

        result = resolve_xml_sidecar(str(video), log_cb=logs.append)

    assert result is None
    assert len(logs) == 1
    assert "⚠️" in logs[0]
    assert "silinmiş/bulunamadı" in logs[0]
    assert "GÜN_BATISI.xml" in logs[0]


def test_resolve_xml_sidecar_no_log_cb():
    """XS-05: log_cb=None iken hata fırlatılmaz."""
    from core.xml_sidecar import resolve_xml_sidecar

    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "film.mp4"
        video.touch()

        # Should not raise
        result = resolve_xml_sidecar(str(video), log_cb=None)
        assert result is None
