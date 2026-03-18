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
    turkish_title: str = ""      # "GÜN BATISI"


def find_xml_sidecar(video_path: str) -> str | None:
    """Video dosyasının yanında aynı isimle .xml dosyası ara.

    Varsa tam yolunu döndür, yoksa None.
    """
    vp = Path(video_path)
    xml_path = vp.with_suffix(".xml")
    if xml_path.is_file():
        return str(xml_path)
    return None


def parse_xml_sidecar(xml_path: str) -> XmlSidecarInfo:
    """XML dosyasından orijinal ve Türkçe başlığı çıkar.

    XML içinde şu formatlardan birini bekler (esnek arama):
    - <OriginalTitle> veya benzer tag'ler
    - veya düz metin içinde "Orijinal Başlık" ve "Türkçe Başlık" satırları

    Parse başarısız olursa boş XmlSidecarInfo döner (hata fırlatmaz).
    """
    info = XmlSidecarInfo(xml_path=xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # XML tag'lerden ara (tag isimleri kesin belli olmadığı için esnek arama)
        for elem in root.iter():
            tag_lower = (elem.tag or "").lower().replace("_", "").replace("-", "")
            text = (elem.text or "").strip()
            if not text:
                continue

            if not info.original_title and (
                "originaltitle" in tag_lower or "originalname" in tag_lower or "orijinalbaslik" in tag_lower
            ):
                info.original_title = text
            elif not info.turkish_title and (
                "turkishtitle" in tag_lower or "turkcetitle" in tag_lower or "turkcebaslik" in tag_lower or "localtitle" in tag_lower
            ):
                info.turkish_title = text

            if info.original_title and info.turkish_title:
                break

        # Eğer XML tag'lerden bulunamadıysa, tüm text node'larda satır bazlı ara
        # (düz metin formatı: "Orijinal Başlık\tTHE SUNDOWNERS")
        if not info.original_title:
            all_text = ET.tostring(root, encoding="unicode", method="text") or ""
            marker_orig = "orijinal başlık"
            marker_tr = "türkçe başlık"
            for line in all_text.splitlines():
                line_stripped = line.strip()
                line_lower = line_stripped.lower()

                if "orijinal" in line_lower and "başlık" in line_lower:
                    # "Orijinal Başlık\tTHE SUNDOWNERS" veya tab/space ayrımlı
                    parts = line_stripped.split("\t")
                    if len(parts) >= 2:
                        info.original_title = parts[-1].strip()
                    else:
                        idx = line_lower.find(marker_orig)
                        if idx != -1:
                            after = line_stripped[idx + len(marker_orig):].strip()
                            if after:
                                info.original_title = after

                if "türkçe" in line_lower and "başlık" in line_lower:
                    parts = line_stripped.split("\t")
                    if len(parts) >= 2:
                        info.turkish_title = parts[-1].strip()
                    else:
                        idx = line_lower.find(marker_tr)
                        if idx != -1:
                            after = line_stripped[idx + len(marker_tr):].strip()
                            if after:
                                info.turkish_title = after

    except ET.ParseError:
        pass  # Bozuk XML — sessizce geç
    except Exception:
        pass  # Herhangi bir hata — sessizce geç

    return info


def resolve_xml_sidecar(video_path: str, log_cb=None) -> XmlSidecarInfo | None:
    """Video dosyasının yanındaki XML'i bul, parse et, logla.

    Returns:
        XmlSidecarInfo — XML bulunup başarıyla parse edildiyse (en az bir başlık var)
        None — XML bulunamazsa veya içinde başlık bilgisi yoksa
    """
    _log = log_cb or (lambda m: None)
    xml_path = find_xml_sidecar(video_path)

    if not xml_path:
        expected = Path(video_path).with_suffix(".xml").name
        _log(f"  [XML] ⚠️ XML dosyası silinmiş/bulunamadı: {expected}")
        return None

    _log(f"  [XML] ✅ XML dosyası başarıyla çekildi: {Path(xml_path).name}")

    info = parse_xml_sidecar(xml_path)

    if info.original_title:
        _log(f"  [XML] Orijinal Başlık: {info.original_title}")
    if info.turkish_title:
        _log(f"  [XML] Türkçe Başlık: {info.turkish_title}")

    if not info.original_title and not info.turkish_title:
        _log(f"  [XML] ⚠️ XML dosyası okundu fakat başlık bilgisi bulunamadı")
        return None

    return info
