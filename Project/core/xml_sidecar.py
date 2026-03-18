"""xml_sidecar.py — Video dosyasına eşlik eden XML sidecar dosyasını bulma."""
from pathlib import Path


def find_xml_sidecar(video_path: str) -> str | None:
    """Video dosyasının yanındaki aynı isimli .xml dosyasını bul.

    Args:
        video_path: Video dosyasının tam yolu.

    Returns:
        XML dosyasının tam yolu (str) veya bulunamazsa None.
    """
    p = Path(video_path)
    xml_path = p.with_suffix(".xml")
    if xml_path.is_file():
        return str(xml_path)
    return None


def resolve_xml_sidecar(video_path: str, log_cb=None) -> str | None:
    """XML sidecar dosyasını bul ve sonucu logla.

    Args:
        video_path: Video dosyasının tam yolu.
        log_cb:     Log mesajı göndermek için callback (opsiyonel).

    Returns:
        XML dosyasının tam yolu (str) veya bulunamazsa None.
    """
    xml_path = find_xml_sidecar(video_path)
    expected = str(Path(video_path).with_suffix(".xml"))

    if log_cb:
        if xml_path:
            log_cb(f"  [XML] ✅ XML dosyası başarıyla çekildi: {xml_path}")
        else:
            log_cb(f"  [XML] ⚠️ XML dosyası silinmiş/bulunamadı: {expected}")

    return xml_path
