"""
google_ocr_engine.py — Optional Google Cloud Vision OCR fallback.

Kullanım:
- GOOGLE_APPLICATION_CREDENTIALS env ile service account json path ver
  veya config["google_credentials_json"] ile path ver.
- config["google_ocr"]["enabled"]=True ise pipeline, düşük kalite durumunda fallback dener.

Not: Google Vision API response bbox/conf her zaman satır bazında gelmeyebilir;
bu engine 'line text' üretmeye odaklanır.
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    from google.cloud import vision
    HAS_GOOGLE = True
except Exception:
    HAS_GOOGLE = False


@dataclass
class GoogleLine:
    text: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    seen_count: int = 1
    avg_confidence: float = 0.90
    bbox: list = field(default_factory=list)
    frame_path: str = ""
    source: str = "google"


class GoogleOCREngine:
    def __init__(self, credentials_json: str = "", log_cb=None, mode: str = "text_detection"):
        if not HAS_GOOGLE:
            raise RuntimeError("google-cloud-vision yüklü değil. Kur: pip install google-cloud-vision")
        self.log_cb = log_cb
        self.mode = mode

        if credentials_json:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_json

        self.client = vision.ImageAnnotatorClient()

    def _log(self, msg: str):
        if self.log_cb:
            self.log_cb(msg)
        else:
            print(msg)

    def ocr_image(self, image_path: str, timecode: float = 0.0) -> List[GoogleLine]:
        p = Path(image_path)
        content = p.read_bytes()
        img = vision.Image(content=content)

        if self.mode == "document_text_detection":
            resp = self.client.document_text_detection(image=img)
            ann = resp.full_text_annotation
            text = ann.text if ann else ""
        else:
            # default: text_detection
            resp = self.client.text_detection(image=img)
            # text_annotations[0].description full text
            text = resp.text_annotations[0].description if resp.text_annotations else ""

        if resp.error.message:
            raise RuntimeError(resp.error.message)

        lines = []
        for ln in (text or "").splitlines():
            t = ln.strip()
            if not t:
                continue
            lines.append(GoogleLine(text=t, first_seen=timecode, last_seen=timecode, frame_path=str(p)))
        return lines

    def process_frames(self, frames: List[str], timecodes: Optional[Dict[str, float]] = None) -> List[GoogleLine]:
        out: List[GoogleLine] = []
        for fp in frames:
            tc = 0.0
            if timecodes and fp in timecodes:
                tc = float(timecodes[fp])
            try:
                out.extend(self.ocr_image(fp, timecode=tc))
            except Exception as e:
                self._log(f"  !! Google OCR hata ({Path(fp).name}): {e}")
        return out
