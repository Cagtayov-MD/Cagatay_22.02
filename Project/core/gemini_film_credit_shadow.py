"""Experimental shadow Gemini film-credit sidecar writer.

This module is intentionally isolated from the main Gemini flow. It only reads
copied pipeline artifacts, calls a secondary Gemini API with Google Search
grounding, and writes an observational sidecar JSON into the DB folder.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import urllib.error
import urllib.request
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from config.runtime_paths import get_gemini_film_credit_api_key
from core.xml_sidecar import parse_xml_sidecar

_DEFAULT_MODEL = "gemini-2.5-flash"
_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
_TIMEOUT_SEC = 60
_MAX_DIRECTOR_HINTS = 6
_MAX_ACTOR_HINTS = 12
_MAX_NAMEDB_HINTS = 20

_SYSTEM_PROMPT = """
ROLUN:
Sen search-grounded calisan profesyonel bir film kimlik dogrulama ve credits cikarma API'sisin.

ANA GOREVIN:
Sana sadece kisitli ipuclari verilebilir:
- dosya adindan cikan baslik
- XML'den gelen yerli ve/veya orijinal baslik
- NameDB'den gelen isimler
- varsa yonetmen ipuclari
- varsa oyuncu ipuclari

Bu ipuclarindan yola cikarak:
1. Once tekil film veya dizi kimligini bul.
2. Sonra bu yapimin credits bilgisini web aramasi ile dogrula.
3. Sadece JSON formatinda cevap ver.

ZORUNLU IS AKISI:
1. Once eslesen yapimi bul.
2. Eslesme guveni dusukse credits uydurma.
3. Eslesme guveni yeterliyse sadece dogrulanmis alanlari doldur.
4. Cevapta sadece JSON olsun. Markdown, aciklama, yorum, code fence yazma.

KURALLAR:
- Web search / grounding kullan.
- Mumkunse en az 2 bagimsiz guvenilir kaynaktan capraz kontrol yap.
- Celiskili sonuclarda en guvenilir ve en tutarli yapimi sec.
- Film kimligi net degilse FILM_BULUNDU=false don.
- Ipuclarindaki isimler OCR kaynakli, eksik, hatali veya karisik olabilir.
- Film bulunmadan credits uydurma.
- Supheli isimleri ekleme.
- Kisi adlarini Basic Latin formatina cevir.
- Rol anahtarlari sadece sunlar olabilir:
  YONETMEN
  YONETMEN_YARDIMCISI
  YAPIMCI
  KAMERA
  GORUNTU_YONETMENI
  SENARYO
  KURGU
  OYUNCULAR
- Bu roller disinda yeni alan uretme.
- Tum credits alanlari liste olmalidir.
- Bir rol icin guvenilir veri yoksa bos liste [] don.
- JSON disinda tek karakter bile yazma.

CIKTI JSON SEMASI:
{
  "FILM_BULUNDU": true,
  "ESLESME_GUVENI": "high",
  "ESLESEN_BASLIK_YERLI": "SIRIN'IN KALESI",
  "ESLESEN_BASLIK_ORJINAL": "GHASR-E SHIRIN",
  "YIL": 2019,
  "KAYNAK_DOMAINLER": ["imdb.com", "themoviedb.org"],
  "KANIT": {
    "KULLANILAN_XML_BASLIKLARI": [],
    "KULLANILAN_YONETMEN_IPUCLARI": [],
    "KULLANILAN_OYUNCU_IPUCLARI": [],
    "KULLANILAN_NAMEDB_ISIMLERI": []
  },
  "YONETMEN": [],
  "YONETMEN_YARDIMCISI": [],
  "YAPIMCI": [],
  "KAMERA": [],
  "GORUNTU_YONETMENI": [],
  "SENARYO": [],
  "KURGU": [],
  "OYUNCULAR": []
}
""".strip()


def _unique(items: list[str], limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if limit and len(out) >= limit:
            break
    return out


def _title_compare_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.casefold()
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _canonicalize_title_with_hints(value: str, hints: list[str], *, min_ratio: float = 0.92) -> str:
    original = str(value or "").strip()
    if not original:
        return original

    hint_values = _unique([str(item or "").strip() for item in hints if str(item or "").strip()])
    if not hint_values:
        return original

    original_key = _title_compare_key(original)
    if not original_key:
        return original

    best_hint = original
    best_score = 0.0
    original_token_count = len(re.findall(r"[a-z0-9]+", original_key))

    for hint in hint_values:
        hint_key = _title_compare_key(hint)
        if not hint_key:
            continue
        if hint_key == original_key:
            return hint

        hint_token_count = len(re.findall(r"[a-z0-9]+", hint_key))
        if original_token_count and hint_token_count and abs(original_token_count - hint_token_count) > 1:
            continue
        if abs(len(hint_key) - len(original_key)) > 4:
            continue

        score = difflib.SequenceMatcher(None, original_key, hint_key).ratio()
        if score > best_score:
            best_score = score
            best_hint = hint

    return best_hint if best_score >= min_ratio else original


def _canonicalize_response_titles(response_json: dict[str, Any], request_data: dict[str, Any]) -> None:
    local_hints = _unique(
        [
            request_data.get("xml_turkish_title", ""),
            request_data.get("filename_title", ""),
        ]
    )
    original_hints = _unique([request_data.get("xml_original_title", "")])
    xml_hints = _unique(original_hints + local_hints)

    local_title = response_json.get("ESLESEN_BASLIK_YERLI")
    if isinstance(local_title, str) and local_hints:
        response_json["ESLESEN_BASLIK_YERLI"] = _canonicalize_title_with_hints(local_title, local_hints)

    original_title = response_json.get("ESLESEN_BASLIK_ORJINAL")
    if isinstance(original_title, str) and original_hints:
        response_json["ESLESEN_BASLIK_ORJINAL"] = _canonicalize_title_with_hints(
            original_title,
            original_hints,
        )

    evidence = response_json.get("KANIT")
    if isinstance(evidence, dict):
        xml_titles = evidence.get("KULLANILAN_XML_BASLIKLARI")
        if isinstance(xml_titles, list) and xml_hints:
            evidence["KULLANILAN_XML_BASLIKLARI"] = _unique(
                [
                    _canonicalize_title_with_hints(str(item or ""), xml_hints)
                    for item in xml_titles
                    if str(item or "").strip()
                ]
            )


def _extract_year_hint(stem: str, credits_data: dict[str, Any]) -> int | None:
    year = credits_data.get("year")
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.strip().isdigit():
        return int(year.strip())
    match = re.search(r"\b(19\d{2}|20\d{2})\b", stem or "")
    if match:
        return int(match.group(1))
    return None


def _extract_director_hints(credits_data: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    for entry in (credits_data.get("directors") or []):
        if isinstance(entry, str):
            hints.append(entry)
        elif isinstance(entry, dict):
            hints.append(entry.get("name", ""))

    for entry in (credits_data.get("crew") or []):
        if not isinstance(entry, dict):
            continue
        job = str(entry.get("job") or entry.get("role") or "").strip().lower()
        if job in {"director", "yonetmen", "yönetmen"}:
            hints.append(entry.get("name", ""))

    return _unique(hints, limit=_MAX_DIRECTOR_HINTS)


def _extract_actor_hints(credits_data: dict[str, Any]) -> list[str]:
    hints = [
        (entry.get("actor_name") or "").strip()
        for entry in (credits_data.get("cast") or [])
        if isinstance(entry, dict)
    ]
    return _unique(hints, limit=_MAX_ACTOR_HINTS)


def _collect_namedb_hints(
    credits_data: dict[str, Any],
    credits_raw: dict[str, Any] | None,
) -> list[str]:
    hints: list[str] = []
    verification_log = credits_data.get("_verification_log") or []
    if isinstance(verification_log, list):
        for entry in verification_log:
            if not isinstance(entry, dict):
                continue
            if entry.get("layer") != "NAMEDB":
                continue
            if entry.get("action") not in {"kept", "corrected"}:
                continue
            hints.append(entry.get("name_out") or entry.get("name_in") or "")

    if hints:
        return _unique(hints, limit=_MAX_NAMEDB_HINTS)

    raw_cast = (credits_raw or {}).get("cast") or []
    for entry in raw_cast:
        if not isinstance(entry, dict):
            continue
        if entry.get("is_verified_name") or entry.get("match_method") == "exact_db":
            hints.append(entry.get("actor_name", ""))
    return _unique(hints, limit=_MAX_NAMEDB_HINTS)


def build_shadow_request(
    *,
    video_info: dict[str, Any],
    credits_data: dict[str, Any],
    credits_raw: dict[str, Any] | None,
    xml_path: str = "",
    filename_title: str = "",
) -> dict[str, Any]:
    filename = str(video_info.get("filename") or "")
    stem = Path(filename or "out").stem
    xml_info = parse_xml_sidecar(xml_path) if xml_path else None

    return {
        "video_filename": filename,
        "video_stem": stem,
        "filename_title": str(filename_title or "").strip(),
        "xml_original_title": (xml_info.original_title if xml_info else "").strip(),
        "xml_turkish_title": (xml_info.turkish_title if xml_info else "").strip(),
        "year_hint": _extract_year_hint(stem, credits_data),
        "director_hints": _extract_director_hints(credits_data),
        "actor_hints": _extract_actor_hints(credits_data),
        "namedb_names": _collect_namedb_hints(credits_data, credits_raw),
    }


def build_shadow_user_prompt(request_data: dict[str, Any]) -> str:
    return (
        "FILM IPUCLARI\n\n"
        f"DOSYA_ADI:\n{request_data.get('video_filename', '')}\n\n"
        f"DOSYA_ADINDAN_CIKAN_BASLIK:\n{request_data.get('filename_title', '')}\n\n"
        f"XML_ORJINAL_BASLIK:\n{request_data.get('xml_original_title', '')}\n\n"
        f"XML_YERLI_BASLIK:\n{request_data.get('xml_turkish_title', '')}\n\n"
        f"YIL_IPUCU:\n{json.dumps(request_data.get('year_hint'), ensure_ascii=False)}\n\n"
        "YONETMEN_IPUCLARI:\n"
        f"{json.dumps(request_data.get('director_hints', []), ensure_ascii=False)}\n\n"
        "OYUNCU_IPUCLARI:\n"
        f"{json.dumps(request_data.get('actor_hints', []), ensure_ascii=False)}\n\n"
        "NAMEDB_ISIMLERI:\n"
        f"{json.dumps(request_data.get('namedb_names', []), ensure_ascii=False)}\n\n"
        "EK NOTLAR:\n"
        "- NameDB isimleri OCR kaynakli olabilir.\n"
        "- XML basliklari dosya adindan gelen basliktan daha guvenilir olabilir.\n"
        "- Oyuncu ve yonetmen ipuclari eksik veya yanlis olabilir.\n"
        "- En olasi dogru yapimi bul ve credits bilgisini sadece guvenliyse doldur.\n"
        "- Sadece JSON cevap ver.\n\n"
        "GOREV:\n"
        "Yukaridaki ipuclarindan yola cikarak filmi veya diziyi bul, kimligini "
        "dogrula, credits bilgisini cikar ve sadece tanimli JSON semasiyla cevap ver."
    )


def _extract_domains(candidate: dict[str, Any]) -> list[str]:
    def _looks_like_domain(text: str) -> bool:
        value = str(text or "").strip().lower()
        if not value or " " in value:
            return False
        return bool(re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", value))

    def _domain_from_web(web: dict[str, Any]) -> str:
        title = str((web or {}).get("title") or "").strip().lower()
        uri = str((web or {}).get("uri") or "").strip()

        if _looks_like_domain(title):
            return title

        if uri:
            parsed = urlparse(uri)
            host = parsed.netloc.strip().lower()
            if host and host != "vertexaisearch.cloud.google.com":
                return host
        return ""

    grounding = (
        (candidate or {}).get("groundingMetadata")
        or (candidate or {}).get("grounding_metadata")
        or {}
    )
    chunks = grounding.get("groundingChunks") or grounding.get("grounding_chunks") or []
    domains: list[str] = []
    for chunk in chunks:
        web = (chunk or {}).get("web") or {}
        domain = _domain_from_web(web)
        if domain:
            domains.append(domain)

    citation_sources = (
        ((candidate or {}).get("citationMetadata") or {}).get("citationSources")
        or ((candidate or {}).get("citation_metadata") or {}).get("citation_sources")
        or []
    )
    for source in citation_sources:
        uri = str((source or {}).get("uri") or "").strip()
        if not uri:
            continue
        host = urlparse(uri).netloc.strip().lower()
        if host:
            domains.append(host)
    return _unique(domains)


def _sanitize_response_text(text: str) -> str:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()
    if clean.startswith("json"):
        clean = clean[4:].strip()
    return clean


def _call_shadow_gemini(
    *,
    api_key: str,
    model: str,
    base_url: str,
    user_prompt: str,
) -> tuple[str | None, list[str], str | None]:
    contents = [
        {"role": "user", "parts": [{"text": _SYSTEM_PROMPT}]},
        {"role": "model", "parts": [{"text": "Understood."}]},
        {"role": "user", "parts": [{"text": user_prompt}]},
    ]
    payload = json.dumps(
        {
            "contents": contents,
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": 0.1,
            },
        }
    ).encode("utf-8")
    url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        candidate = (data.get("candidates") or [{}])[0]
        parts = ((candidate.get("content") or {}).get("parts") or [])
        raw_text = "".join((part or {}).get("text", "") for part in parts).strip()
        return raw_text or None, _extract_domains(candidate), None
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        return None, [], f"HTTP {exc.code}: {exc.reason} — {body}".strip()
    except urllib.error.URLError as exc:
        return None, [], f"URL error: {exc.reason}"
    except TimeoutError:
        return None, [], f"Timeout after {_TIMEOUT_SEC}s"
    except Exception as exc:
        return None, [], f"Unexpected error: {exc}"


def write_shadow_sidecar(
    *,
    db_dir: str | Path,
    video_info: dict[str, Any],
    credits_data: dict[str, Any],
    credits_raw: dict[str, Any] | None,
    xml_path: str = "",
    filename_title: str = "",
    log_cb=None,
) -> str:
    def _log(message: str) -> None:
        if log_cb:
            log_cb(message)

    out_dir = Path(db_dir)
    filename = str(video_info.get("filename") or "out")
    stem = Path(filename).stem
    out_path = out_dir / f"{stem}_GeminiFilmCredit.json"

    request_data = build_shadow_request(
        video_info=video_info,
        credits_data=credits_data,
        credits_raw=credits_raw,
        xml_path=xml_path,
        filename_title=filename_title,
    )
    user_prompt = build_shadow_user_prompt(request_data)

    model = (os.environ.get("GEMINI_FILM_CREDIT_MODEL") or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    base_url = (os.environ.get("GEMINI_FILM_CREDIT_BASE_URL") or _DEFAULT_BASE_URL).strip() or _DEFAULT_BASE_URL
    api_key = get_gemini_film_credit_api_key()

    sidecar: dict[str, Any] = {
        "status": "skipped",
        "request": {
            "hints": request_data,
            "system_prompt": _SYSTEM_PROMPT,
            "user_prompt": user_prompt,
        },
        "response_raw": None,
        "response_json": None,
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "endpoint_type": "gemini_generate_content_google_search",
            "note": "",
            "grounding_domains": [],
            "grounding_domain_source": "none",
            "model_reported_domains": [],
        },
    }

    if not api_key:
        sidecar["meta"]["note"] = "GEMINI_FILM_CREDIT_API_KEY missing"
    else:
        raw_text, domains, error_note = _call_shadow_gemini(
            api_key=api_key,
            model=model,
            base_url=base_url,
            user_prompt=user_prompt,
        )
        sidecar["response_raw"] = raw_text
        sidecar["meta"]["grounding_domains"] = domains

        if error_note:
            sidecar["status"] = "error"
            sidecar["meta"]["note"] = error_note
        elif not raw_text:
            sidecar["status"] = "error"
            sidecar["meta"]["note"] = "Empty response"
        else:
            try:
                sidecar["response_json"] = json.loads(_sanitize_response_text(raw_text))
                if isinstance(sidecar["response_json"], dict):
                    _canonicalize_response_titles(sidecar["response_json"], request_data)
                    reported_domains = sidecar["response_json"].get("KAYNAK_DOMAINLER") or []
                    if isinstance(reported_domains, list):
                        sidecar["meta"]["model_reported_domains"] = _unique(
                            [str(item) for item in reported_domains if str(item or "").strip()]
                        )
                if sidecar["meta"]["grounding_domains"]:
                    sidecar["meta"]["grounding_domain_source"] = "grounding_metadata"
                elif sidecar["meta"]["model_reported_domains"]:
                    sidecar["meta"]["grounding_domain_source"] = "model_reported_only"
                sidecar["status"] = "ok"
                sidecar["meta"]["note"] = "JSON parsed successfully"
            except json.JSONDecodeError as exc:
                sidecar["status"] = "invalid_json"
                sidecar["meta"]["note"] = f"JSON parse error: {exc}"

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, ensure_ascii=False, indent=2)

    _log(f"  [GeminiFilmCredit] Sidecar yazıldı: {out_path.name} ({sidecar['status']})")

    # Expo TXT — shadow JSON → standart kullanıcı çıktı formatı
    try:
        from config.runtime_paths import get_gemini_film_credit_expo_dir
        write_shadow_expo_txt(
            sidecar=sidecar,
            video_info=video_info,
            expo_dir=get_gemini_film_credit_expo_dir(),
            log_cb=log_cb,
        )
    except Exception as exc:
        _log(f"  [GeminiFilmCredit] Expo TXT yazılamadı: {exc}")

    return str(out_path)


def write_shadow_expo_txt(
    *,
    sidecar: dict[str, Any],
    video_info: dict[str, Any],
    expo_dir: str | Path,
    log_cb=None,
) -> str | None:
    """Shadow Gemini JSON çıktısını standart kullanıcı TXT formatına dönüştürür.

    status != 'ok' veya FILM_BULUNDU != true ise dosya yazılmaz.
    Tüm hatalar yutulur — pipeline etkilenmez.
    """
    def _log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    rj = sidecar.get("response_json") or {}
    if sidecar.get("status") != "ok" or not rj.get("FILM_BULUNDU"):
        _log(
            f"  [GeminiFilmCredit] Expo TXT atlandı "
            f"(status={sidecar.get('status')}, FILM_BULUNDU={rj.get('FILM_BULUNDU')})"
        )
        return None

    filename = str(video_info.get("filename") or "out")
    stem = Path(filename).stem

    sep = "=" * 64
    fw = 22
    L: list[str] = []

    # ── FİLM / PROGRAM BİLGİLERİ ──────────────────────────────
    L.append(sep)
    L.append("  FİLM / PROGRAM BİLGİLERİ")
    L.append(sep)
    L.append(f"  {'FİLMİN ADI':<{fw}}:     {rj.get('ESLESEN_BASLIK_YERLI') or 'VERİ YOK'}")
    L.append(f"  {'FİLMİN ORJİNAL ADI':<{fw}}:     {rj.get('ESLESEN_BASLIK_ORJINAL') or 'VERİ YOK'}")
    L.append(f"  {'FİLMİN ID':<{fw}}:     VERİ YOK")
    L.append(f"  {'BÖLÜM':<{fw}}:     YOK")
    resolution = str(video_info.get("resolution") or "").strip()
    L.append(f"  {'ÇÖZÜNÜRLÜK':<{fw}}:     {resolution or 'VERİ YOK'}")
    fps = video_info.get("fps")
    L.append(f"  {'FRAME':<{fw}}:     {f'{fps} FRAME' if fps else 'VERİ YOK'}")
    duration = str(video_info.get("duration_human") or "").strip()
    L.append(f"  {'TOPLAM SÜRE':<{fw}}:     {duration or 'VERİ YOK'}")
    L.append(f"  {'SESLENDİRME DİLİ':<{fw}}:     VERİ YOK")
    guven = str(rj.get("ESLESME_GUVENI") or "").upper()
    L.append(f"  {'KAYNAK':<{fw}}:     GEMİNİ SHADOW ({guven})")

    # ── ÖZET ──────────────────────────────────────────────────
    L.append(sep)
    L.append("  ÖZET")
    L.append(sep)
    L.append("ÖZET OLUŞTURULAMADI.")

    # ── ANAHTAR SÖZCÜKLER ──────────────────────────────────────
    oyuncular: list[str] = [str(a).strip() for a in (rj.get("OYUNCULAR") or []) if str(a).strip()]
    L.append(sep)
    L.append("  ANAHTAR SÖZCÜKLER")
    L.append(sep)
    L.append(" ; ".join(oyuncular[:20]) if oyuncular else "YOK")

    # ── OYUNCULAR ─────────────────────────────────────────────
    L.append(sep)
    L.append("  OYUNCULAR")
    L.append(sep)
    if oyuncular:
        for a in oyuncular[:20]:
            L.append(f"  {a}")
        if len(oyuncular) > 20:
            L.append(f"  ... ve {len(oyuncular) - 20} oyuncu daha")
    else:
        L.append("  VERİ YOK")

    # ── YAPIM EKİBİ ───────────────────────────────────────────
    L.append(sep)
    L.append("  YAPIM EKİBİ")
    L.append(sep)
    _ROLLER = [
        ("YAPIMCI",             "YAPIMCI",           4),
        ("YÖNETMEN",            "YONETMEN",          2),
        ("YÖNETMEN YARDIMCISI", "YONETMEN_YARDIMCISI", 3),
        ("GÖRÜNTÜ YÖNETMENİ",  "GORUNTU_YONETMENI", None),
        ("SENARYO",             "SENARYO",           None),
        ("KAMERA",              "KAMERA",            2),
        ("KURGU",               "KURGU",             2),
    ]
    rw = 22
    for label, key, limit in _ROLLER:
        kisiler: list[str] = [str(k).strip() for k in (rj.get(key) or []) if str(k).strip()]
        if limit:
            kisiler = kisiler[:limit]
        if kisiler:
            L.append(f"  {label:<{rw}}{kisiler[0]}")
            for k in kisiler[1:]:
                L.append(f"  {'':<{rw}}{k}")
        else:
            L.append(f"  {label:<{rw}}VERİ YOK")

    # ── OLUŞTURULMA ───────────────────────────────────────────
    L.append(sep)
    timestamp = sidecar.get("meta", {}).get("timestamp", "")
    L.append(f"OLUŞTURULMA: {timestamp}")

    txt = "\n".join(L) + "\n"

    expo_path = Path(expo_dir) / f"{stem}.txt"
    expo_path.parent.mkdir(parents=True, exist_ok=True)
    with expo_path.open("w", encoding="utf-8-sig") as fh:
        fh.write(txt)

    _log(f"  [GeminiFilmCredit] Expo TXT yazıldı: {expo_path}")
    return str(expo_path)
