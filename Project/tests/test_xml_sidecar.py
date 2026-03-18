"""
test_xml_sidecar.py — xml_sidecar modülü için birim testleri.

Covers:
  XS-01: find_xml_sidecar mevcut XML dosyasını bulur
  XS-02: find_xml_sidecar XML yoksa None döndürür
  XS-03: resolve_xml_sidecar XML bulunca başarı logu yazar ve XmlSidecarInfo döndürür
  XS-04: resolve_xml_sidecar XML yoksa uyarı logu yazar ve None döndürür
  XS-05: resolve_xml_sidecar log_cb=None iken hata fırlatmaz
  XS-06: parse_xml_sidecar XML tag'lerinden başlık çeker
  XS-07: parse_xml_sidecar düz metin satırlarından başlık çeker (tab ayrımlı)
  XS-08: parse_xml_sidecar bozuk XML'de hata fırlatmaz
  XS-09: resolve_xml_sidecar başlık olmayan XML'de None döndürür
  XS-10: resolve_xml_sidecar başlıklı XML'de XmlSidecarInfo döndürür
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
    """XS-03: XML ve başlık bulununca başarı mesajı loglanır ve XmlSidecarInfo döndürülür."""
    from core.xml_sidecar import resolve_xml_sidecar, XmlSidecarInfo

    logs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "GÜN_BATISI.mp4"
        xml   = Path(tmpdir) / "GÜN_BATISI.xml"
        video.touch()
        xml.write_text(
            "<metadata><OriginalTitle>THE SUNDOWNERS</OriginalTitle>"
            "<TurkishTitle>GÜN BATISI</TurkishTitle></metadata>",
            encoding="utf-8",
        )

        result = resolve_xml_sidecar(str(video), log_cb=logs.append)

    assert isinstance(result, XmlSidecarInfo)
    assert result.xml_path == str(xml)
    assert result.original_title == "THE SUNDOWNERS"
    assert result.turkish_title == "GÜN BATISI"
    assert any("✅" in msg for msg in logs)
    assert any("XML dosyası başarıyla çekildi" in msg for msg in logs)
    assert any("GÜN_BATISI.xml" in msg for msg in logs)


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


def test_parse_xml_sidecar_from_tags():
    """XS-06: XML tag'lerinden OriginalTitle ve TurkishTitle çeker."""
    from core.xml_sidecar import parse_xml_sidecar

    with tempfile.TemporaryDirectory() as tmpdir:
        xml = Path(tmpdir) / "film.xml"
        xml.write_text(
            "<metadata><OriginalTitle>THE SUNDOWNERS</OriginalTitle>"
            "<TurkishTitle>GÜN BATISI</TurkishTitle></metadata>",
            encoding="utf-8",
        )
        info = parse_xml_sidecar(str(xml))

    assert info.original_title == "THE SUNDOWNERS"
    assert info.turkish_title == "GÜN BATISI"
    assert info.xml_path == str(xml)


def test_parse_xml_sidecar_from_plain_text_tabs():
    """XS-07: Düz metin tab-ayrımlı formatından başlık çeker."""
    from core.xml_sidecar import parse_xml_sidecar

    with tempfile.TemporaryDirectory() as tmpdir:
        xml = Path(tmpdir) / "film.xml"
        xml.write_text(
            "<info>Orijinal Başlık\tTHE SUNDOWNERS\nTürkçe Başlık\tGÜN BATISI</info>",
            encoding="utf-8",
        )
        info = parse_xml_sidecar(str(xml))

    assert info.original_title == "THE SUNDOWNERS"
    assert info.turkish_title == "GÜN BATISI"


def test_parse_xml_sidecar_broken_xml():
    """XS-08: Bozuk XML dosyasında hata fırlatmaz, boş XmlSidecarInfo döner."""
    from core.xml_sidecar import parse_xml_sidecar

    with tempfile.TemporaryDirectory() as tmpdir:
        xml = Path(tmpdir) / "broken.xml"
        xml.write_text("<this is not valid xml >>>", encoding="utf-8")
        info = parse_xml_sidecar(str(xml))

    assert info.original_title == ""
    assert info.turkish_title == ""


def test_resolve_xml_sidecar_no_titles_returns_none():
    """XS-09: XML bulunup başlık bilgisi yoksa None döndürür."""
    from core.xml_sidecar import resolve_xml_sidecar

    logs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "film.mp4"
        xml   = Path(tmpdir) / "film.xml"
        video.touch()
        xml.write_text("<metadata><other>no titles here</other></metadata>", encoding="utf-8")

        result = resolve_xml_sidecar(str(video), log_cb=logs.append)

    assert result is None
    assert any("⚠️" in msg and "başlık bilgisi bulunamadı" in msg for msg in logs)


def test_resolve_xml_sidecar_only_original_title():
    """XS-10: Sadece orijinal başlık varsa XmlSidecarInfo döner."""
    from core.xml_sidecar import resolve_xml_sidecar, XmlSidecarInfo

    logs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "film.mp4"
        xml   = Path(tmpdir) / "film.xml"
        video.touch()
        xml.write_text("<metadata><OriginalTitle>THE SUNDOWNERS</OriginalTitle></metadata>",
                       encoding="utf-8")

        result = resolve_xml_sidecar(str(video), log_cb=logs.append)

    assert isinstance(result, XmlSidecarInfo)
    assert result.original_title == "THE SUNDOWNERS"
    assert result.turkish_title == ""
