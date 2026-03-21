"""
run_lookup_test.py — IMDB + TMDB lookup testi (OCR/UI/ASR olmadan)

Kullanım:
    1. Bu klasöre (lookup_test/) bir cdata.json koy (OCR çıktısı formatında)
    2. python run_lookup_test.py

cdata.json formatı (credits_parser.to_report_dict() çıktısıyla aynı):
    {
        "film_title": "...",
        "year": 2024,
        "cast": [{"actor_name": "...", "confidence": 0.90}, ...],
        "crew": [{"name": "...", "job": "Yönetmen", "role": "Yönetmen"}, ...],
        "technical_crew": [...],
        "directors": [{"name": "..."}],
        "production_companies": [...],
        "production_info": [],
        "verification_status": "ocr_parsed"
    }
"""

import sys
import os
import json
import copy
import pprint

# ── Path ayarı ────────────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # Project/
sys.path.insert(0, _PROJECT_DIR)

# .env yükle (TMDB_API_KEY, TMDB_BEARER_TOKEN, IMDB_DB_PATH)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(_PROJECT_DIR, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        print(f"[ENV] .env yüklendi: {_env_path}")
    else:
        print(f"[ENV] .env bulunamadı ({_env_path}) — ortam değişkenleri kullanılacak")
except ImportError:
    print("[ENV] python-dotenv yok — ortam değişkenleri doğrudan okunuyor")

# ── cdata.json oku ────────────────────────────────────────────────────────────
_CDATA_PATH = os.path.join(_THIS_DIR, "cdata.json")

if not os.path.exists(_CDATA_PATH):
    print(f"[HATA] cdata.json bulunamadı: {_CDATA_PATH}")
    sys.exit(1)

with open(_CDATA_PATH, encoding="utf-8") as f:
    cdata = json.load(f)

print("\n" + "=" * 60)
print("  LOOKUP TESTİ BAŞLIYOR")
print("=" * 60)
print(f"  Başlık  : {cdata.get('film_title', '???')}")
print(f"  Yıl     : {cdata.get('year', '???')}")
print(f"  Oyuncu  : {len(cdata.get('cast', []))} kişi")
print(f"  Crew    : {len(cdata.get('crew', []))} kişi")
print(f"  Director: {[d.get('name') if isinstance(d, dict) else d for d in cdata.get('directors', [])]}")
print("=" * 60)


def _log(msg):
    print(f"  {msg}")


# ── 1) IMDB Lookup ────────────────────────────────────────────────────────────
print("\n── IMDB LOOKUP ──────────────────────────────────────────────")
imdb_result = None
try:
    from core.imdb_lookup import IMDBLookup

    imdb = IMDBLookup(log_cb=_log)
    if not imdb.enabled():
        print("  [IMDB] Devre dışı (DB bulunamadı veya path hatalı)")
        print(f"         IMDB_DB_PATH={os.environ.get('IMDB_DB_PATH', 'ayarlanmamış')}")
    else:
        imdb_cdata = copy.deepcopy(cdata)

        # ── Debug: candidates ──
        from core.imdb_lookup import _norm
        film_title_norm = _norm(cdata.get("film_title", ""))
        print(f"  norm(film_title) = '{film_title_norm}'")
        try:
            import duckdb
            con = duckdb.connect(imdb._db_path, read_only=True)
            candidates = imdb._find_title_candidates(con, film_title_norm)
            print(f"  title candidates ({len(candidates)} adet):")
            for c in candidates[:10]:
                print(f"    {c}")
            if not candidates:
                print("  [!] Başlık DB'de bulunamadı — film_title yazımını kontrol et")
            else:
                # İlk birkaç aday için DB'deki gerçek cast isimlerini göster
                print("\n  [DEBUG] Her aday için DB'deki ilk 5 isim:")
                for cand in candidates[:4]:
                    tc = cand["tconst"]
                    try:
                        rows = con.execute("""
                            SELECT n.primaryName, p.category
                            FROM principals p
                            JOIN names n ON n.nconst = p.nconst
                            WHERE p.tconst = ?
                            ORDER BY p.ordering
                            LIMIT 5
                        """, [tc]).fetchall()
                        names_str = ", ".join(f"{r[0]} ({r[1]})" for r in rows)
                        print(f"    {tc} [{cand['titleType']} {cand['startYear']}]: {names_str or 'veri yok'}")
                    except Exception as e:
                        print(f"    {tc}: hata — {e}")
            con.close()
        except Exception as e:
            print(f"  [DEBUG] candidates alınamadı: {e}")

        imdb_result = imdb.lookup(imdb_cdata)

        print(f"\n  matched : {imdb_result.matched}")
        print(f"  reason  : {imdb_result.reason}")
        if imdb_result.matched:
            print(f"  title   : {imdb_result.title}")
            print(f"  tconst  : {imdb_result.tconst}")
            print(f"  year    : {imdb_result.year}")
            print(f"  directors: {imdb_result.directors}")
            print(f"  cast ({len(imdb_result.cast or [])}): {[c.get('actor_name') for c in (imdb_result.cast or [])[:5]]}")
            print(f"  crew ({len(imdb_result.crew or [])}): {[(c.get('name'), c.get('job')) for c in (imdb_result.crew or [])[:5]]}")
        else:
            print(f"  [IMDB] Eşleşme yok")

except Exception as e:
    print(f"  [IMDB] Hata: {e}")
    import traceback; traceback.print_exc()


# ── 2) TMDB Lookup ────────────────────────────────────────────────────────────
print("\n── TMDB LOOKUP ──────────────────────────────────────────────")
tmdb_result = None
if imdb_result and imdb_result.matched:
    print("  [TMDB] IMDB eşleşti — pipeline'da TMDB atlanır (skipped)")
else:
    try:
        from core.tmdb_verify import TMDBVerify, TMDBVerifyResult
        from config.runtime_paths import get_tmdb_api_key
        api_key = get_tmdb_api_key()
        token   = os.environ.get("TMDB_BEARER_TOKEN", "").strip()

        print(f"  api_key  : {'***' + api_key[-4:] if api_key else 'YOK'}")
        print(f"  token    : {'***' + token[-4:] if token else 'YOK'}")

        if not (api_key or token):
            print("  [TMDB] API key yok")
            print("  Çözüm seçeneklerinden biri:")
            print("    1) Project/config/api_keys.json → {\"tmdb_api_key\": \"...\"}")
            print("    2) .env → TMDB_API_KEY=...")
            print("    3) Ortam değişkeni: set TMDB_API_KEY=...")
        else:
            tmdb_cdata = copy.deepcopy(cdata)
            is_series = cdata.get("_is_series", True)
            verifier = TMDBVerify(
                work_dir=_THIS_DIR,
                api_key=api_key or None,
                bearer_token=token or None,
                language="tr",
                log_cb=_log,
            )
            tmdb_result = verifier.verify_credits(tmdb_cdata, is_series=is_series)

            print(f"\n  updated      : {tmdb_result.updated}")
            print(f"  matched_id   : {tmdb_result.matched_id}")
            print(f"  matched_title: {tmdb_result.matched_title}")
            print(f"  hits         : {tmdb_result.hits}")
            print(f"  misses       : {tmdb_result.misses}")
            print(f"  reason       : {tmdb_result.reason}")

            if tmdb_result.updated:
                directors_after = [
                    d.get("name") if isinstance(d, dict) else d
                    for d in tmdb_cdata.get("directors", [])
                ]
                crew_after = tmdb_cdata.get("crew", [])
                print(f"\n  [cdata sonrası]")
                print(f"  directors: {directors_after}")
                print(f"  crew ({len(crew_after)}): {[(c.get('name'), c.get('job')) for c in crew_after[:5]]}")
            else:
                print("  [TMDB] Eşleşme yok veya güncelleme yapılmadı")

            _tmdb_debug_id = tmdb_result.matched_id
            if not _tmdb_debug_id:
                print("\n  [DEBUG] TMDB eşleşmedi — cast/crew sorgusu atlanıyor")
            else:
                print(f"\n  [DEBUG] TMDB id:{_tmdb_debug_id} için gerçek cast/crew:")
                try:
                    import requests
                    headers = {"Authorization": f"Bearer {token}"} if token else {}
                    params  = {"api_key": api_key} if api_key else {}

                    r = requests.get(
                        f"https://api.themoviedb.org/3/tv/{_tmdb_debug_id}/credits",
                        headers=headers, params={**params, "language": "tr-TR"}, timeout=10
                    )
                    if r.ok:
                        data = r.json()
                        print(f"  [credits] Cast ({len(data.get('cast', []))} toplam, ilk 8):")
                        for p in data.get("cast", [])[:8]:
                            print(f"    {p.get('name')} — {p.get('character', '')}")
                        print(f"  [credits] Crew ({len(data.get('crew', []))} toplam, ilk 8):")
                        for p in data.get("crew", [])[:8]:
                            print(f"    {p.get('name')} — {p.get('job', '')}")
                    else:
                        print(f"  [credits] HTTP {r.status_code}")

                    r2 = requests.get(
                        f"https://api.themoviedb.org/3/tv/{_tmdb_debug_id}/aggregate_credits",
                        headers=headers, params={**params, "language": "tr-TR"}, timeout=10
                    )
                    if r2.ok:
                        crew2 = r2.json().get("crew", [])
                        print(f"\n  [aggregate_credits] Crew ({len(crew2)} toplam):")
                        for p in crew2[:15]:
                            jobs = ", ".join(j.get("job", "") for j in p.get("jobs", []))
                            print(f"    {p.get('name')} — {jobs or p.get('department', '')}")
                    else:
                        print(f"  [aggregate_credits] HTTP {r2.status_code}")
                except Exception as e:
                    print(f"  [DEBUG] TMDB credits alınamadı: {e}")

    except Exception as e:
        print(f"  [TMDB] Hata: {e}")
    import traceback; traceback.print_exc()


# ── Özet ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ÖZET")
print("=" * 60)
imdb_ok = imdb_result and imdb_result.matched
tmdb_ok = tmdb_result and (tmdb_result.updated or tmdb_result.matched_id)
print(f"  IMDB : {'✓ ' + str(getattr(imdb_result, 'tconst', '')) if imdb_ok else '✗ eşleşme yok'}")
print(f"  TMDB : {'✓ ' + str(getattr(tmdb_result, 'matched_title', '')) if tmdb_ok else '✗ eşleşme yok'}")
print("=" * 60 + "\n")

# ── 3) TMDB Enrich testi (IMDB eşleştiyse) ───────────────────────────────────
if imdb_ok:
    print("── TMDB ENRICH (aggregate_credits merge) ────────────────────")
    try:
        from core.tmdb_verify import TMDBClient
        from config.runtime_paths import get_tmdb_api_key as _get_key
        _api_key = _get_key()
        _token   = os.environ.get("TMDB_BEARER_TOKEN", "").strip()
        _is_series = cdata.get("_is_series", True)
        _film_title = imdb_cdata.get("film_title") or cdata.get("film_title", "")

        print(f"  film_title (IMDB sonrası): '{_film_title}'")
        print(f"  is_series: {_is_series}")

        if not (_api_key or _token):
            print("  [SKIP] API key yok")
        else:
            _client = TMDBClient(api_key=_api_key, bearer_token=_token, language="tr-TR")
            _results = _client.search_multi(_film_title)
            _tmdb_id = None
            _kind = None
            for _r in (_results or []):
                _k = _r.get("media_type", "")
                if _is_series and _k == "tv":
                    _tmdb_id = _r["id"]; _kind = "tv"; break
                elif not _is_series and _k == "movie":
                    _tmdb_id = _r["id"]; _kind = "movie"; break

            print(f"  TMDB search sonucu: id={_tmdb_id}, kind={_kind}")

            if _tmdb_id:
                if _kind == "tv":
                    _raw = _client.get_tv_aggregate_credits(_tmdb_id)
                else:
                    _raw = _client.get_movie_credits(_tmdb_id)

                _crew = _raw.get("crew", []) if _raw else []
                print(f"  aggregate_credits crew: {len(_crew)} kişi")

                # Departman özeti
                _depts = {}
                for _p in _crew:
                    _d = _p.get("department", "Unknown")
                    _depts[_d] = _depts.get(_d, 0) + 1
                for _d, _c in sorted(_depts.items(), key=lambda x: -x[1]):
                    print(f"    {_d}: {_c}")

                # IMDB crew öncesi
                _imdb_crew_before = imdb_result.crew or []
                print(f"\n  IMDB crew (önce): {len(_imdb_crew_before)} kişi")
                for _c in _imdb_crew_before[:5]:
                    print(f"    {_c.get('name')} — {_c.get('job')}")

                # ── Gerçek merge: TMDB crew'u imdb_cdata'ya ekle (dedup) ──
                _existing_keys = {
                    (c.get("name") or "").lower().strip()
                    for c in (imdb_cdata.get("crew") or [])
                }
                _added = 0
                for _item in _crew:
                    _name = (_item.get("name") or "").strip()
                    if not _name:
                        continue
                    _dept = _item.get("department", "")
                    _key  = _name.lower()
                    _jobs_list = _item.get("jobs")
                    if _jobs_list:
                        for _je in _jobs_list:
                            _job = _je.get("job", _dept)
                            if _key not in _existing_keys:
                                imdb_cdata.setdefault("crew", []).append({
                                    "name": _name, "job": _job, "role": _job,
                                    "department": _dept,
                                    "episode_count": _je.get("episode_count", 0),
                                    "raw": "tmdb", "is_tmdb_verified": True,
                                })
                                _existing_keys.add(_key)
                                _added += 1
                    else:
                        _job = _item.get("job", _dept)
                        if _key not in _existing_keys:
                            imdb_cdata.setdefault("crew", []).append({
                                "name": _name, "job": _job, "role": _job,
                                "department": _dept,
                                "raw": "tmdb", "is_tmdb_verified": True,
                            })
                            _existing_keys.add(_key)
                            _added += 1
                print(f"  {_added} yeni crew eklendi (TMDB)")

        print(f"\n  BİRLEŞİK SONUÇ:")
        print(f"  Toplam crew: {len(imdb_cdata.get('crew', []))}")
        for c in imdb_cdata.get('crew', []):
            print(f"    {c.get('name')} — {c.get('job')}")

    except Exception as _e:
        print(f"  [HATA] {_e}")
        import traceback; traceback.print_exc()
    print()
