"""xml_sidecar.py — Video yanındaki XML metadata dosyasını oku."""

from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class XmlSidecarInfo:
    """XML sidecar'dan okunan bilgiler."""
    xml_path: str = ""
    original_title: str = ""     # "THE SUNDOWNERS"
    turkish_title: str = ""      # "GÜN BATISI" (XML'de yoksa dosya adından gelir)
    mediaid: str = ""            # "1950-2118-1-0000-90-1"
    repository: str = ""         # "TRT1"


def find_xml_sidecar(video_path: str) -> str | None:
    """Video dosyasının yanında aynı isimle .xml dosyası ara.

    Varsa tam yolunu döndür, yoksa None.
    """
    vp = Path(video_path)
    xml_path = vp.with_suffix(".xml")
    if xml_path.is_file():
        return str(xml_path)
    return None


def _parse_xml_root(xml_path: str) -> Optional[ET.Element]:
    """XML dosyasını parse et; önce UTF-8, başarısız olursa latin-1 dene."""
    try:
        return ET.parse(xml_path).getroot()
    except (ET.ParseError, UnicodeDecodeError):
        pass
    try:
        with open(xml_path, encoding="latin-1", errors="replace") as fh:
            return ET.fromstring(fh.read())
    except Exception:
        return None


def parse_xml_sidecar(xml_path: str) -> XmlSidecarInfo:
    """XML dosyasından orijinal başlık, mediaid ve repository bilgisini çıkar.

    Gerçek format (SOURCE_DATA/WIP_METADATA/ASSET/...):
      <SOURCE_DATA><WIP_METADATA><ASSET>
        <TITLE>THE SUNDOWNERS</TITLE>
        <MEDIAID>1950-2118-1-0000-90-1</MEDIAID>
        <REPOSITORY>TRT1</REPOSITORY>
      </ASSET></WIP_METADATA></SOURCE_DATA>

    Parse başarısız olursa boş XmlSidecarInfo döner (hata fırlatmaz).
    """
    info = XmlSidecarInfo(xml_path=xml_path)

    root = _parse_xml_root(xml_path)
    if root is None:
        return info

    try:
        # Strateji 1: Gerçek format — SOURCE_DATA/WIP_METADATA/ASSET/*
        # root tag'i SOURCE_DATA ise doğrudan alt yolu kullan;
        # root başka bir şeyse (eski test XML'leri gibi) relative path gene çalışır.
        asset = root.find("WIP_METADATA/ASSET")
        if asset is not None:
            for tag, attr in (("TITLE", "original_title"),
                              ("MEDIAID", "mediaid"),
                              ("REPOSITORY", "repository")):
                elem = asset.find(tag)
                if elem is not None and (elem.text or "").strip():
                    setattr(info, attr, elem.text.strip())

        # Strateji 2: Recursive <TITLE> arama (farklı XML yapıları için)
        if not info.original_title:
            for elem in root.iter("TITLE"):
                text = (elem.text or "").strip()
                if text:
                    info.original_title = text
                    break

        # Strateji 3: Genel tag ismi eşleştirmesi (geriye dönük uyumluluk)
        if not info.original_title or not info.turkish_title:
            for elem in root.iter():
                tag_lower = (elem.tag or "").lower().replace("_", "").replace("-", "")
                text = (elem.text or "").strip()
                if not text:
                    continue
                if not info.original_title and (
                    "originaltitle" in tag_lower or "originalname" in tag_lower
                    or "orijinalbaslik" in tag_lower
                ):
                    info.original_title = text
                if not info.turkish_title and (
                    "turkishtitle" in tag_lower or "turkcetitle" in tag_lower
                    or "turkcebaslik" in tag_lower or "localtitle" in tag_lower
                ):
                    info.turkish_title = text
                if info.original_title and info.turkish_title:
                    break

        # Strateji 4: Düz metin satır bazlı arama
        # (format: "Orijinal Başlık\tTHE SUNDOWNERS")
        if not info.original_title:
            all_text = ET.tostring(root, encoding="unicode", method="text") or ""
            marker_orig = "orijinal başlık"
            marker_tr = "türkçe başlık"
            for line in all_text.splitlines():
                line_stripped = line.strip()
                line_lower = line_stripped.lower()

                if not info.original_title and "orijinal" in line_lower and "başlık" in line_lower:
                    parts = line_stripped.split("\t")
                    if len(parts) >= 2:
                        info.original_title = parts[-1].strip()
                    else:
                        idx = line_lower.find(marker_orig)
                        if idx != -1:
                            after = line_stripped[idx + len(marker_orig):].strip()
                            if after:
                                info.original_title = after

                if not info.turkish_title and "türkçe" in line_lower and "başlık" in line_lower:
                    parts = line_stripped.split("\t")
                    if len(parts) >= 2:
                        info.turkish_title = parts[-1].strip()
                    else:
                        idx = line_lower.find(marker_tr)
                        if idx != -1:
                            after = line_stripped[idx + len(marker_tr):].strip()
                            if after:
                                info.turkish_title = after

    except Exception:
        pass  # Herhangi bir hata — sessizce geç

    return info


def resolve_xml_sidecar(video_path: str, log_cb=None) -> XmlSidecarInfo | None:
    """Video dosyasının yanındaki XML'i bul, parse et, logla.

    Log davranışı:
    - Dosya bulunamadı → ⚠️ "XML dosyası silinmiş: ..."
    - Dosya bulundu, original_title YOK → ⚠️ "başlık bilgisi bulunamadı: ..." (✅ yok)
    - Dosya bulundu, original_title VAR → ✅ "başarıyla çekildi" + başlık satırı

    Returns:
        XmlSidecarInfo — XML bulunup original_title dolu ise
        None — XML bulunamazsa veya original_title parse edilemezse
    """
    _log = log_cb or (lambda m: None)
    xml_path = find_xml_sidecar(video_path)

    if not xml_path:
        expected = Path(video_path).with_suffix(".xml").name
        _log(f"  [XML] XML dosyasi silinmis: {expected}")
        return None

    info = parse_xml_sidecar(xml_path)

    if not info.original_title:
        _log(f"  [XML] ⚠️ XML dosyası okundu fakat başlık bilgisi bulunamadı: {Path(xml_path).name}")
        return None

    _log(f"  [XML] ✅ XML dosyası başarıyla çekildi: {Path(xml_path).name}")
    _log(f"  [XML]   Orijinal Başlık: {info.original_title}")

    return info
