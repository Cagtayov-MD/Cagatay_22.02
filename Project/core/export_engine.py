"""export_engine.py — Report schema uyumlu JSON + okunabilir TXT çıktı."""
import json, os, re
from datetime import datetime
from pathlib import Path

from config.runtime_paths import resolve_name_db_path

# ═══════════════════════════════════════════════════════════════════
# İSİM VERİTABANI — TurkishNameDB (350K+) > ALL_NAMES (9K) fallback
# ═══════════════════════════════════════════════════════════════════
# ISSUE-09 FIX: Global singleton kaldırıldı.
# ExportEngine artık kendi _name_db instance değişkenini kullanır.
# Bu değişiklik thread-safety ve test izolasyonunu sağlar.
_NAME_DB = None  # Geriye dönük uyumluluk (modul-level fonksiyonlar için)

try:
    from core.turkish_name_db import TurkishNameDB
    _HAS_NAME_DB = True
except ImportError:
    _HAS_NAME_DB = False

# Fallback: eski utils.turkish (geriye dönük uyumluluk)
try:
    from utils.turkish import ALL_NAMES, normalize_tr as _ntr
    _HAS_TR = True
except Exception:
    _HAS_TR = False
    ALL_NAMES = set()
    def _ntr(s): return s.lower()


def _is_known_name(name: str) -> bool:
    """İsim veritabanında var mı? NameDB > ALL_NAMES fallback."""
    if not name:
        return False
    if _NAME_DB:
        result, score = _NAME_DB.find(name.strip())
        return result is not None and score >= 0.95
    if _HAS_TR:
        return _ntr(name) in ALL_NAMES
    return False


def _correct_name(name: str) -> str:
    """İsmi veritabanından düzelt. NameDB > OCR substitution fallback."""
    if not name:
        return name
    if _NAME_DB:
        result, score = _NAME_DB.find(name.strip())
        if result and score >= 0.85:
            return result
        return name
    # Fallback: eski _ocr_correct_name mantığı
    return _ocr_correct_name_legacy(name)


def _split_name(word: str) -> str:
    """Birleşik ismi böl. NameDB > split_concatenated_name fallback."""
    if _NAME_DB:
        result = _NAME_DB.correct_line(word)
        if result != word and ' ' in result:
            return result
    if _HAS_TR:
        try:
            from utils.turkish import split_concatenated_name as _scn
            return _scn(word)
        except Exception:
            pass
    return word


def _tr_ascii(s: str) -> str:
    return s.translate(str.maketrans({
        "ç":"c","ğ":"g","ı":"i","ö":"o","ş":"s","ü":"u",
        "Ç":"C","Ğ":"G","İ":"I","Ö":"O","Ş":"S","Ü":"U"
    }))

def _norm_key(s: str) -> str:
    s = _tr_ascii(s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def _noise_score(actor: str) -> float:
    """
    OCR gurultu skoru - dusuk = temiz isim, yuksek = gurultu.
    En temiz varyanti secmek icin kullanilir.
    """
    score = 0.0
    if re.search(r'\.{3,}', actor): score += 10
    if '&' in actor: score += 5
    if re.search(r'\d', actor): score += 3
    words = actor.split()
    if len(words) > 2: score += (len(words) - 2) * 3

    # Türkçe karakter içeriyorsa bonus (OCR doğru okumuş demek)
    _turk = set("çğıöşüÇĞİÖŞÜ")
    if any(c in _turk for c in actor):
        score -= 5

    # Tamamen büyük harf + çok kelime → OCR bozulması ihtimali yüksek
    if len(words) >= 2 and all(w.isupper() for w in words):
        score += 4

    # Boşluksuz 8+ karakter büyük harf → birleşik bozulma
    if ' ' not in actor and actor.isupper() and len(actor) >= 8:
        score += 6
    # Birlesik kelime tespiti
    try:
        for w in words:
            if len(w) >= 8:
                split_result = _split_name(w)
                if split_result != w and ' ' in split_result:
                    score += 5
                elif len(w) >= 10:
                    score += 2
    except Exception:
        for w in words:
            if len(w) >= 10: score += 2
    # Benzer prefixli kelime tekrari: Coldengoney Coldenguney
    nwords = [_norm_key(w) for w in words]
    for i in range(len(nwords)):
        for j in range(i+1, len(nwords)):
            a, b = nwords[i], nwords[j]
            if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
                score += 2
    return score

def _best_actor(actors: list) -> str:
    """
    En temiz actor_name varyantını seç.
    Seçim sonrası split_concatenated_name ile birlesik OCR kelimelerini boz.
    Ornek: Aycamutlugi → Ayca Mutlugi, Volkangirgin → Volkan Girgin
    """
    if not actors: return ""
    scored = sorted(actors, key=_noise_score)
    best = scored[0]
    # Birlesik kelime düzeltmesi
    try:
        words = best.split()
        fixed_words = []
        for w in words:
            if len(w) >= 8:
                split_result = _split_name(w)
                if split_result != w and ' ' in split_result:
                    fixed_words.extend(p.capitalize() for p in split_result.split())
                else:
                    fixed_words.append(w)
            else:
                fixed_words.append(w)
        best = ' '.join(fixed_words)
    except Exception:
        pass
    return best

def _ocr_correct_name_legacy(name: str) -> str:
    """
    [LEGACY FALLBACK] Tek kelime isimde yaygın OCR karakter hatalarını düzelt.
    ALL_NAMES veritabanını kullanarak doğru varyantı bul.
    NameDB yoksa bu fonksiyon kullanılır.
    """
    if not _HAS_TR or not name:
        return name
    if _ntr(name) in ALL_NAMES:
        return name
    ocr_subs = [
        ('v', 'y'), ('v', 'n'), ('u', 'ü'), ('o', 'ö'),
        ('i', 'ı'), ('c', 'ç'), ('s', 'ş'), ('g', 'ğ'),
        ('y', 'v'),
    ]
    best = name
    for wrong, correct in ocr_subs:
        candidate = name.replace(wrong, correct)
        if candidate != name and _ntr(candidate) in ALL_NAMES:
            best = candidate
            break
        candidate2 = name.replace(wrong.upper(), correct.upper())
        if candidate2 != name and _ntr(candidate2) in ALL_NAMES:
            best = candidate2
            break
    return best


def _clean_char(char: str) -> str:
    """Karakter ismindeki OCR gürültüsünü temizle."""
    # & → g
    char = char.replace('&', 'g')
    # Trailing rakam: Satilmis 11 → Satilmis
    char = re.sub(r'\s+\d+\s*$', '', char).strip()
    words = char.split()
    if len(words) == 2:
        n0 = _norm_key(words[0])
        n1 = _norm_key(words[1])
        # Recep Receple → Recep (ikincisi birincinin uzantısı)
        if len(n1) > len(n0) and n1.startswith(n0):
            return words[0]
        # Behive Behiye veya Avten Ayten → isim veritabanından doğruyu seç
        in_db = [w for w in words if _is_known_name(w)]
        not_in_db = [w for w in words if not _is_known_name(w)]
        # Biri doğru diğeri yanlış → doğru olanı al
        if len(in_db) == 1 and len(not_in_db) == 1:
            return in_db[0]
        # İkisi de doğruysa en uzununu al (daha spesifik)
        if len(in_db) == 2:
            return max(words, key=len)
        # Fallback: son kelime (genelde daha doğru OCR)
        if len(n0) >= 4 and n0[:4] == n1[:4]:
            return words[-1]
    # Tek kelime: OCR karakter hatası düzeltmesi (Rüva → Rüya)
    if len(words) == 1:
        return _correct_name(char.strip())
    return char.strip()

def _best_char(chars: list) -> str:
    """Birden fazla karakter ismi varyantından en doğrusunu seç."""
    if not chars: return ""
    cleaned = [_clean_char(c) for c in chars if c]
    if not cleaned: return ""

    def score(c):
        s = 0
        # İsim veritabanından kontrol (NameDB veya ALL_NAMES)
        for w in c.split():
            if _is_known_name(w): s += 5
        # Az çöp karakter
        bad = sum(1 for ch in c if not ch.isalpha() and ch != ' ')
        s -= bad * 2
        # Makul uzunluk (3-15)
        if 3 <= len(c) <= 15: s += 1
        # Rakam yok
        if not re.search(r'\d', c): s += 2
        # Eşit score'da kısa olanı tercih et (tek kelime > iki kelime)
        s -= len(c) * 0.01
        return s

    return max(cleaned, key=score)

def _best_variant(variants: list[str]) -> str:
    """En okunur varyantı seç (geriye dönük uyumluluk için korundu)."""
    if not variants:
        return ""
    def score(t: str) -> float:
        turk = set("çğıöşüÇĞİÖŞÜ")
        td = sum(1 for c in t if c in turk)
        wc = len(t.split())
        bad = sum(1 for c in t if (not c.isalnum()) and c != " ")
        return td*3 + wc*0.6 + len(t)*0.05 - bad*1.0
    return max(variants, key=score)

def _fuzzy_char_key(char: str) -> str:
    """
    Karakter ismi icin fuzzy gruplama key'i.
    'Avten Ayten' ve 'Ayten' ayni bucket'a dusmeli.
    'Recep Receple' ve 'Recep' ayni bucket'a dusmeli.
    'Behive Behiye' ayni bucket'a dusmeli.
    """
    clean = char.strip().replace('&', 'g')
    words = clean.split()
    if len(words) == 1:
        return _norm_key(clean)
    if len(words) == 2:
        n0 = _norm_key(words[0])
        n1 = _norm_key(words[1])
        # Recep Receple: n1 n0 ile basliyor -> n0
        if len(n1) > len(n0) and n1.startswith(n0):
            return n0
        # Behive Behiye / Avten Ayten: ortak prefix
        if len(n0) >= 4 and len(n1) >= 4 and n0[:4] == n1[:4]:
            common = ''
            for a, b in zip(n0, n1):
                if a == b: common += a
                else: break
            return common if len(common) >= 3 else n0[:4]
        # İsim veritabanında olan kelimeyi key olarak kullan
        for w in reversed(words):
            if _is_known_name(w):
                return _norm_key(w)
        return _norm_key(words[-1])
    return _norm_key(clean)


def _canonicalize_cast(cast: list[dict]) -> list[dict]:
    """
    Cast listesini temizle ve tekilleştir.

    İki aşamalı strateji:
    1. Karakter ismi olanlar → karakter bazlı grupla (kapanış jenerik)
       Aynı karakter için birden fazla actor varyantı varsa en temizini seç.
    2. Karakter ismi olmayanlar → actor bazlı grupla (açılış jenerik)

    Her iki grubu birleştirirken aktör tekrarını önle.
    """
    # ── GRUP 1: Karakter ismi olanlar (kapanış jenerik) ──
    char_buckets: dict[str, dict] = {}  # fuzzy_char_key → {actor_variants, char_variants}
    no_char_rows: list[dict] = []

    for row in cast or []:
        a = (row.get("actor_name") or "").strip()
        c = (row.get("character_name") or "").strip()
        if not a and not c:
            continue
        if c:
            key = _fuzzy_char_key(c)
            b = char_buckets.get(key)
            if not b:
                char_buckets[key] = {"actor_variants": [a] if a else [],
                                     "char_variants": [c]}
            else:
                if a: b["actor_variants"].append(a)
                b["char_variants"].append(c)
        else:
            no_char_rows.append({"actor_name": a, "character_name": ""})

    # Karakter bazlı bucket'lardan en iyi oyuncu + karakter seç
    char_based: list[dict] = []
    seen_actors: set[str] = set()
    for key, b in char_buckets.items():
        actor = _best_actor([v for v in b["actor_variants"] if v])
        char  = _best_char([v for v in b["char_variants"] if v])
        if not actor and not char:
            continue
        actor_key = _norm_key(actor)
        if actor_key and actor_key in seen_actors:
            continue
        if actor_key:
            seen_actors.add(actor_key)
        char_based.append({"actor_name": actor, "character_name": char})

    # ── GRUP 2: Karakter ismi olmayanlar (açılış jenerik) ──
    actor_buckets: dict[str, dict] = {}
    for row in no_char_rows:
        a = row["actor_name"]
        key = _norm_key(a)
        if not key: continue
        if key not in actor_buckets:
            actor_buckets[key] = {"actor_variants": [a]}
        else:
            actor_buckets[key]["actor_variants"].append(a)

    no_char_based: list[dict] = []
    for key, b in actor_buckets.items():
        if key in seen_actors:
            continue  # kapanış jenerikle çakışıyorsa atla
        actor = _best_actor([v for v in b["actor_variants"] if v])
        if actor:
            no_char_based.append({"actor_name": actor, "character_name": ""})

    # Birleştir: önce karakter bilgisi olanlar, sonra olmayanlar
    out = char_based + no_char_based
    out.sort(key=lambda r: (_norm_key(r.get("character_name", "")),
                             _norm_key(r.get("actor_name", ""))))
    return out

def _canonicalize_crew(crew: list[dict]) -> list[dict]:
    buckets: dict[tuple[str,str], dict] = {}
    for row in crew or []:
        name = (row.get("name") or "").strip()
        role = (row.get("role") or "").strip()
        if not name and not role:
            continue
        key = (_norm_key(name), _norm_key(role))
        b = buckets.get(key)
        if not b:
            buckets[key] = {"name_variants":[name] if name else [], "role_variants":[role] if role else [], "seen":1}
        else:
            if name: b["name_variants"].append(name)
            if role: b["role_variants"].append(role)
            b["seen"] += 1
    out=[]
    for b in buckets.values():
        name=_best_variant([v for v in b["name_variants"] if v])
        role=_best_variant([v for v in b["role_variants"] if v])
        if name or role:
            out.append({"name":name,"role":role})
    out.sort(key=lambda r: (_norm_key(r.get("role","")), _norm_key(r.get("name",""))))
    return out

class ExportEngine:
    def __init__(self, output_dir, name_db=None):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        # ISSUE-09 FIX: Instance değişken — thread-safe, test izolasyonu sağlar.
        # Global singleton kaldırıldı; her ExportEngine kendi name_db'sine sahip.
        if name_db is not None:
            self._name_db = name_db
        elif _HAS_NAME_DB:
            try:
                db_path = os.environ.get("NAME_DB_PATH", "") or resolve_name_db_path()
                self._name_db = TurkishNameDB(
                    sql_path=db_path if os.path.isfile(db_path) else "")
            except Exception:
                self._name_db = None
        else:
            self._name_db = None

        # Modül-level global'i güncelle (geriye dönük uyumluluk)
        # BUG-K5 FIX: _is_known_name() ve _correct_name() bu global'i kullanıyor.
        # Güncellemezse 356k NameDB yerine ~300 isimlik ALL_NAMES fallback devreye girer.
        global _NAME_DB
        _NAME_DB = self._name_db  # None olsa bile ata — tutarlılık

    def generate(self, video_info, credits_data, ocr_lines, stage_stats,
                 profile, scope, first_min, last_min, keywords=None, logos=None):
        total_sec = sum(s.get("duration_sec", 0) for s in stage_stats.values())
        dur = video_info.get("duration_seconds", 1)

        # Final çıktı temizliği (dedup + en iyi varyant seçimi)
        # ÖNEMLİ: keywords'ten ÖNCE yapılmalı, aksi halde bozuk actor_name'ler keyword'e girer
        try:
            if credits_data.get('cast'):
                credits_data['cast'] = _canonicalize_cast(credits_data['cast'])
            if credits_data.get('crew'):
                credits_data['crew'] = _canonicalize_crew(credits_data['crew'])
        except Exception:
            pass

        # Keywords canonicalize'dan SONRA oluştur — Türkçe karakterler korunmuş olur
        if not keywords:
            keywords = [c["actor_name"] for c in credits_data.get("cast", [])[:20]
                        if c.get("actor_name")]
            keywords += [d for d in self._director_names(credits_data) if d]
        report = {
            "$schema": "arsiv_decode_report_v1",
            "generated_at": datetime.now().isoformat(),
            "profile": profile,
            "file_info": {
                "filename": video_info.get("filename", ""),
                "filepath": video_info.get("filepath", ""),
                "filesize_bytes": video_info.get("filesize_bytes", 0),
                "duration_seconds": dur,
                "duration_human": video_info.get("duration_human", ""),
                "resolution": video_info.get("resolution", ""),
                "fps": video_info.get("fps", 0),
            },
            "processing": {
                "scope": scope,
                "content_type": "film_dizi",
                "ocr_engine": "PaddleOCR (GPU)",
                "first_segment_min": first_min,
                "last_segment_min": last_min,
                "stages": [
                    {"name": k, "duration_sec": v.get("duration_sec", 0),
                     "status": v.get("status", "ok"),
                     "details": {kk: vv for kk, vv in v.items()
                                 if kk not in ("duration_sec", "status")}}
                    for k, v in stage_stats.items()
                ],
                "total_duration_sec": round(total_sec, 2),
                "speed_ratio": round(dur / max(total_sec, 0.1), 2),
            },
            "credits": credits_data,
            "keywords": keywords,
            "logos_detected": logos or [],
            "ocr_results": [
                {"text": (l.text if hasattr(l, "text") else l.get("text", "")),
                 "first_seen": getattr(l, "first_seen", 0),
                 "last_seen": getattr(l, "last_seen", 0),
                 "count": getattr(l, "seen_count", 1),
                 "confidence": getattr(l, "avg_confidence", 0)}
                for l in ocr_lines
            ],
            "errors": [],
        }

        stem = Path(video_info.get("filename", "out")).stem
        jp = self.out / f"{stem}_report.json"
        tp = self.out / f"{stem}_report.txt"

        with open(jp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self._write_txt(report, tp)
        return str(jp), str(tp)

    def _write_txt(self, r, path):
        L = []
        sep = "=" * 65
        L.append(sep)
        L.append("  ARSIV DECODE — ANALIZ RAPORU")
        L.append(sep)
        fi = r["file_info"]
        L.append(f"\n  Dosya      : {fi['filename']}")
        L.append(f"  Sure       : {fi['duration_human']}")
        L.append(f"  Cozunurluk : {fi['resolution']} @ {fi['fps']} FPS")
        L.append(f"  Boyut      : {fi['filesize_bytes']/1024/1024:.1f} MB")
        p = r["processing"]
        L.append(f"\n  Profil     : {r['profile']}")
        L.append(f"  OCR Motor  : {p.get('ocr_engine','?')}")
        L.append(f"  Toplam sure: {p['total_duration_sec']:.1f}s ({p['speed_ratio']}x)")

        L.append(f"\n{'-'*65}")
        L.append("  PIPELINE")
        L.append(f"{'-'*65}")
        for s in p["stages"]:
            ico = "[OK]" if s["status"] in ("ok", "completed") else "[--]" if s["status"] == "skipped" else "[!!]"
            L.append(f"  {ico} {s['name']:20s} {s['duration_sec']:8.1f}s")

        cr = r["credits"]
        L.append(f"\n{'-'*65}")
        L.append(f"  CREDITS [{cr.get('verification_status','unverified')}]")
        L.append(f"{'-'*65}")
        if cr.get("year"):
            L.append(f"\n  YIL: {cr['year']}")
        if cr.get("production_companies"):
            L.append("\n  YAPIM SIRKETLERI:")
            for c in cr["production_companies"]:
                L.append(f"    {c}")
        if cr.get("production_info"):
            L.append("\n  YAPIM BILGISI:")
            for p in cr["production_info"]:
                L.append(f"    {p}")
        if cr.get("directors"):
            L.append("\n  YONETMEN:")
            for d in self._director_names(cr):
                L.append(f"    {d}")
        if cr.get("cast"):
            L.append(f"\n  OYUNCULAR ({cr['total_actors']}):")
            for c in cr["cast"]:
                ch = f" -> {c['character_name']}" if c.get("character_name") else ""
                L.append(f"    {c['actor_name']}{ch}")
        if cr.get("technical_crew"):
            L.append(f"\n  TEKNIK EKIP ({cr['total_crew']}):")
            for t in cr["technical_crew"]:
                L.append(f"    {t['name']} -- {t.get('role_tr','')}")

        if r.get("keywords"):
            L.append(f"\n{'-'*65}")
            L.append("  ANAHTAR KELIMELER")
            L.append(f"{'-'*65}")
            L.append(f"  {' ; '.join(r['keywords'])}")

        ocr = r.get("ocr_results", [])
        if ocr:
            L.append(f"\n{'-'*65}")
            L.append(f"  OCR SONUCLARI ({len(ocr)} benzersiz satir)")
            L.append(f"{'-'*65}")
            for o in ocr:
                tc = o.get("first_seen", 0)
                cf = o.get("confidence", 0)
                cn = o.get("count", 1)
                L.append(f"  [{tc:7.1f}s] conf:{cf:.2f} x{cn}  {o['text']}")

        L.append(f"\n{sep}")
        L.append(f"  Olusturulma: {r['generated_at']}")
        L.append(sep)

        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(L))

    @staticmethod
    def _director_names(credits: dict) -> list[str]:
        """directors alanını (str/dict karışık) güvenli şekilde normalize et."""
        names = []
        for director in credits.get("directors", []):
            if isinstance(director, str):
                name = director.strip()
                if name:
                    names.append(name)
            elif isinstance(director, dict):
                name = str(director.get("name", "")).strip()
                if name:
                    names.append(name)
        return names
