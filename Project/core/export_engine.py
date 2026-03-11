"""export_engine.py — Report schema uyumlu JSON + okunabilir TXT çıktı."""
import json, os, re, unicodedata
from datetime import datetime
from pathlib import Path

from utils.time_utils import fmt_hms as _fmt_hms_shared


def _safe_path(path: Path) -> Path:
    """Dosya çakışması varsa _2, _3 ... ekleyerek güvenli bir yol döndür."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    n = 2
    while True:
        candidate = parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1

from config.runtime_paths import resolve_name_db_path

try:
    from rapidfuzz import fuzz as _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

# Fuzzy matching thresholds
_WORD_FUZZY_THRESHOLD = 70   # fuzz.ratio per-word minimum similarity
_WRATIO_FALLBACK_THRESHOLD = 75  # WRatio fallback when word counts differ

# ═══════════════════════════════════════════════════════════════════
# ÇIKTI DOSYASI YARDIMCı FONKSİYONLARI
# ═══════════════════════════════════════════════════════════════════

# ID pattern: 4 rakam-2-4 rakam-1+ rakam-4 rakam-2-3 rakam-1+ rakam
_ID_PATTERN = re.compile(r'(\d{4}-\d{2,4}-\d+-\d{4}-\d{2,3}-\d+)')


def _extract_output_name(filename: str) -> tuple[str, str, str]:
    """Video dosya adından çıktı adı, film ID ve Türkçe film adı çıkar.

    Örnek:
        'evoArcadmin_TEST2_1949-0039-1-0000-00-1-KADIN_VE_DENİZCİ.mp4'
        → ('1949-0039-1-0000-00-1 KADIN VE DENİZCİ', '1949-0039-1-0000-00-1', 'KADIN VE DENİZCİ')
    """
    stem = Path(filename).stem
    match = _ID_PATTERN.search(stem)

    if not match:
        return (stem, '', stem)

    film_id = match.group(1)
    after_id = stem[match.end():]
    after_id = after_id.lstrip('-_')
    film_name_tr = after_id.replace('_', ' ').replace('-', ' ').strip()

    if film_name_tr:
        output_name = f"{film_id} {film_name_tr}"
    else:
        output_name = film_id

    return (output_name, film_id, film_name_tr)


def _extract_film_id(filename: str) -> str:
    """Dosya adından Film ID kısmını çıkar.

    Örnek:
        "evoArcadmin_TEST2_1949-0039-1-0000-00-1-KADIN_VE_DENİZCİ.mp4"
        → "1949-0039-1-0000-00-1"
    """
    _, film_id, _ = _extract_output_name(filename)
    return film_id

def _extract_episode_from_id(film_id: str) -> str:
    """Film ID'den bölüm bilgisi çıkar.

    Expected format: YYYY-XXXX-B-SSSS-XX-X
      - block[0]: yıl (4 hane)
      - block[1]: seri no
      - block[2]: karar bloğu (1=film, 0=dizi)
      - block[3]: bölüm numarası (block[2]==0 ise geçerli)

    3. blok = 1 → 'YOK' (film)
    3. blok = 0 → 4. blok değeri (dizi bölümü)
    """
    parts = film_id.split('-')
    if len(parts) < 4:
        return 'YOK'
    block3 = parts[2].strip()
    if block3 == '1':
        return 'YOK'
    elif block3 == '0':
        return parts[3].strip()
    return 'YOK'


# Türkçe'ye özgü karakterler (Latince'de bulunmayan)
_TR_ONLY_CHARS = set('çğıöşüÇĞİÖŞÜ')


def _is_turkish_word(word: str) -> bool:
    """Kelimede Türkçe'ye özgü karakter var mı?"""
    return any(c in _TR_ONLY_CHARS for c in word)


def _upper_word(word: str) -> str:
    """Tek kelimeyi büyük harfe çevir (Türkçe veya İngilizce kurala göre)."""
    if _is_turkish_word(word):
        result = word.replace('i', 'İ').replace('ı', 'I')
        result = result.replace('ç', 'Ç').replace('ğ', 'Ğ')
        result = result.replace('ö', 'Ö').replace('ş', 'Ş').replace('ü', 'Ü')
        return result.upper()
    else:
        upper = word.upper()
        cleaned = []
        for ch in upper:
            if ch.isascii() or not ch.isalpha():
                cleaned.append(ch)
            else:
                decomposed = unicodedata.normalize('NFD', ch)
                base = ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn')
                cleaned.append(base if base else ch)
        return ''.join(cleaned)


def _to_upper_tr(text: str) -> str:
    """Tüm metni büyük harfe çevir.

    Varsayılan: Türkçe kuralla i→İ, ı→I dönüşümü (tüm metin için).
    Türkçe'de bulunmayan aksanlı karakterler (é, è, ê vb.) → düz ASCII büyük harfe.
    """
    # Önce Türkçe i/ı dönüşümü (tüm metin için)
    text = text.replace('i', 'İ').replace('ı', 'I')
    # Sonra diğer Türkçe karakterler
    text = text.replace('ç', 'Ç').replace('ğ', 'Ğ')
    text = text.replace('ö', 'Ö').replace('ş', 'Ş').replace('ü', 'Ü')
    # Genel upper() — kalan küçük harfler için
    result = text.upper()
    # Türkçe'de bulunmayan aksanlı karakterleri düz ASCII'ye çevir
    cleaned = []
    for ch in result:
        if ch.isascii() or ch in 'ÇĞİÖŞÜ':
            cleaned.append(ch)
        elif ch.isalpha():
            decomposed = unicodedata.normalize('NFD', ch)
            base = ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn')
            cleaned.append(base if base else ch)
        else:
            cleaned.append(ch)
    return ''.join(cleaned)


def _wrap_max_words(text: str, max_words: int = 20, indent: str = "  ") -> str:
    """Metni satır başına en fazla max_words kelimeyle böl.

    20. kelimeden sonra noktalama işareti/rakam/işaret gelebilir,
    ama 21. kelimenin başlangıcı alt satır olur.
    Satırın başındaki boşluk korunur.
    """
    # Başlangıç boşluğunu koru
    leading = len(text) - len(text.lstrip())
    prefix = text[:leading]

    words = text.split()
    if len(words) <= max_words:
        return text

    lines = []
    i = 0
    while i < len(words):
        line_words = words[i:i + max_words]
        line = ' '.join(line_words)
        # 20. kelimeden sonra noktalama varsa, onu da bu satıra ekle
        next_idx = i + max_words
        while next_idx < len(words) and not words[next_idx][0].isalpha():
            line += ' ' + words[next_idx]
            next_idx += 1
        lines.append(line)
        i = next_idx

    return prefix + ('\n' + indent).join(lines)


# 6 çıktı rolü (sıralı)
_OUTPUT_ROLES = [
    "YÖNETMEN",
    "YAPIMCI",
    "YÖNETMEN YARDIMCISI",
    "KAMERAMAN",
    "KURGU",
    "SENARİST",
]

# Rol adlarını (küçük harf) → çıktı rolü adına map eden sözlük
_ROLE_TO_OUTPUT: dict[str, str] = {
    # Yönetmen
    "yönetmen": "YÖNETMEN", "yonetmen": "YÖNETMEN",
    "yöneten": "YÖNETMEN", "yoneten": "YÖNETMEN",
    "director": "YÖNETMEN", "directed by": "YÖNETMEN",
    "réalisateur": "YÖNETMEN", "réalisatrice": "YÖNETMEN",
    "realisateur": "YÖNETMEN", "réalisation": "YÖNETMEN",
    "regisseur": "YÖNETMEN", "regie": "YÖNETMEN",
    "regia": "YÖNETMEN", "reżyser": "YÖNETMEN",
    "rendező": "YÖNETMEN", "режиссёр": "YÖNETMEN",
    "ein film von": "YÖNETMEN", "a film by": "YÖNETMEN",
    "film by": "YÖNETMEN",
    # Yapımcı
    "yapımcı": "YAPIMCI", "yapimci": "YAPIMCI",
    "yapım yönetmeni": "YAPIMCI",
    "producer": "YAPIMCI", "produced by": "YAPIMCI",
    "executive producer": "YAPIMCI",
    "produzent": "YAPIMCI", "producteur": "YAPIMCI",
    "productrice": "YAPIMCI", "produttore": "YAPIMCI",
    "productor": "YAPIMCI", "продюсер": "YAPIMCI",
    "yürütücü yapımcı": "YAPIMCI", "yurtucu yapimci": "YAPIMCI",
    # Yönetmen Yardımcısı
    "yönetmen yardımcısı": "YÖNETMEN YARDIMCISI",
    "yonetmen yardimcisi": "YÖNETMEN YARDIMCISI",
    "yardımcı yönetmen": "YÖNETMEN YARDIMCISI",
    "yardimci yonetmen": "YÖNETMEN YARDIMCISI",
    "assistant director": "YÖNETMEN YARDIMCISI",
    "first assistant director": "YÖNETMEN YARDIMCISI",
    "second assistant director": "YÖNETMEN YARDIMCISI",
    "1st ad": "YÖNETMEN YARDIMCISI", "2nd ad": "YÖNETMEN YARDIMCISI",
    "regiassistent": "YÖNETMEN YARDIMCISI",
    "assistenzregisseur": "YÖNETMEN YARDIMCISI",
    "reji asistani": "YÖNETMEN YARDIMCISI",
    # Kameraman
    "görüntü yönetmeni": "KAMERAMAN",
    "goruntu yonetmeni": "KAMERAMAN",
    "görüntü": "KAMERAMAN", "goruntu": "KAMERAMAN",
    "cinematographer": "KAMERAMAN",
    "director of photography": "KAMERAMAN",
    "dop": "KAMERAMAN", "dp": "KAMERAMAN",
    "kameramann": "KAMERAMAN", "kameraman": "KAMERAMAN",
    "kamera": "KAMERAMAN", "camera": "KAMERAMAN",
    "camera operator": "KAMERAMAN",
    "directeur de la photographie": "KAMERAMAN",
    "direttore della fotografia": "KAMERAMAN",
    "operatör": "KAMERAMAN", "operatoru": "KAMERAMAN",
    # Kurgu
    "kurgu": "KURGU", "montaj": "KURGU",
    "editor": "KURGU", "film editor": "KURGU",
    "edited by": "KURGU", "editing": "KURGU",
    "monteur": "KURGU", "monteuse": "KURGU",
    "cutter": "KURGU", "schnitt": "KURGU",
    "montaggio": "KURGU", "montage": "KURGU",
    # Senarist
    "senaryo": "SENARİST", "senarist": "SENARİST",
    "screenwriter": "SENARİST", "screenplay": "SENARİST",
    "writer": "SENARİST", "written by": "SENARİST",
    "scénariste": "SENARİST", "scenariste": "SENARİST",
    "drehbuchautor": "SENARİST",
    "sceneggiatore": "SENARİST", "guionista": "SENARİST",
    "сценарист": "SENARİST", "script": "SENARİST",
    "story by": "SENARİST",
}


def _map_crew_to_roles(crew_data: list, directors: list) -> dict[str, list[str]]:
    """Crew verilerini 6 çıktı rolüne dönüştür.

    Returns:
        dict mapping each output role name → list of person names.
    """
    result: dict[str, list[str]] = {role: [] for role in _OUTPUT_ROLES}

    # Directors → YÖNETMEN
    for d in (directors or []):
        if d and d not in result["YÖNETMEN"]:
            result["YÖNETMEN"].append(d)

    # Crew entries → matching output role
    for entry in (crew_data or []):
        name = (entry.get("name") or "").strip()
        if not name:
            continue
        role_raw = (
            entry.get("role_tr") or entry.get("role") or entry.get("job") or ""
        ).strip()
        role_lower = role_raw.lower()
        if not role_lower:
            continue

        output_role = _ROLE_TO_OUTPUT.get(role_lower)
        if output_role and name not in result[output_role]:
            result[output_role].append(name)

    return result


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
        except Exception as e:
            # Log the error but don't fail
            import logging
            logging.debug(f"split_concatenated_name failed for '{word}': {e}")
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
    except Exception as e:
        # Log the error for debugging but return best effort result
        import logging
        logging.debug(f"_ocr_correct_name exception for '{best}': {e}")
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
    # & → g dönüşümü: sadece kelime İÇİNDE ise uygula (OCR Ğ→& bozulması)
    # "Tom & Jerry" → dokunma, "Sa&det" → "Sağdet"
    words = char.split()
    fixed_words = []
    for w in words:
        if w == '&':
            # Bağımsız & → olduğu gibi bırak (Tom & Jerry durumu)
            fixed_words.append(w)
        elif '&' in w:
            # Kelime içi & → g (Sa&det → Sağdet OCR bozulması)
            fixed_words.append(w.replace('&', 'g'))
        else:
            fixed_words.append(w)
    char = ' '.join(fixed_words)
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


def _words_fuzzy_match(a: str, b: str) -> bool:
    """İki ismi kelime bazlı karşılaştır.

    Tüm kelime çiftlerinin benzerliği >= _WORD_FUZZY_THRESHOLD ise True döner.
    Kelime sayıları farklıysa WRatio >= _WRATIO_FALLBACK_THRESHOLD fallback kullanılır.
    rapidfuzz yoksa her zaman False döner.
    """
    if not _HAS_RAPIDFUZZ:
        return False
    words_a = a.split()
    words_b = b.split()
    if len(words_a) != len(words_b):
        return _fuzz.WRatio(a, b) >= _WRATIO_FALLBACK_THRESHOLD
    return all(_fuzz.ratio(wa, wb) >= _WORD_FUZZY_THRESHOLD for wa, wb in zip(words_a, words_b))


def _canonicalize_cast(cast: list[dict]) -> list[dict]:
    """
    Cast listesini temizle ve tekilleştir.

    İki aşamalı strateji:
    1. Karakter ismi olanlar → karakter bazlı grupla (kapanış jenerik)
       Aynı karakter için birden fazla actor varyantı varsa en temizini seç.
    2. Karakter ismi olmayanlar → actor bazlı fuzzy grupla (açılış jenerik)
       OCR varyantlarını (Ali Ozoqwz, Ali Ozogwz vb.) fuzzy clustering ile birleştir.

    Her iki grubu birleştirirken aktör tekrarını önle.
    Son olarak post-merge fuzzy sweep ile kalan tekrarlı girişleri birleştir.
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
                                     "char_variants": [c],
                                     "confidences": [row.get("confidence", 0.6)],
                                     "verified": [row.get("is_verified_name", False) or row.get("is_llm_verified", False)],
                                     "tmdb_orders": [row.get("tmdb_order")]}
            else:
                if a: b["actor_variants"].append(a)
                b["char_variants"].append(c)
                b["confidences"].append(row.get("confidence", 0.6))
                b["verified"].append(row.get("is_verified_name", False) or row.get("is_llm_verified", False))
                b["tmdb_orders"].append(row.get("tmdb_order"))
        else:
            no_char_rows.append({
                "actor_name": a,
                "character_name": "",
                "confidence": row.get("confidence", 0.6),
                "is_verified_name": row.get("is_verified_name", False) or row.get("is_llm_verified", False),
                "seen_count": row.get("seen_count", 1),
            })

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
        best_conf = max(b.get("confidences") or [0.6])
        is_verified = any(b.get("verified") or [])
        tmdb_order_val = min((o for o in b.get("tmdb_orders", []) if o is not None), default=None)
        char_based.append({
            "actor_name": actor,
            "character_name": char,
            "confidence": round(best_conf, 3),
            "is_verified_name": is_verified,
            "is_llm_verified": is_verified,
            "tmdb_order": tmdb_order_val,
        })

    # ── GRUP 2: Karakter ismi olmayanlar — fuzzy clustering ──
    # Exact key yerine RapidFuzz WRatio >= 75 ile kümeleme yapılır.
    # Bu sayede "Ali Ozoqwz", "Ali Ozogwz" vb. OCR varyantları aynı bucket'a düşer.
    clusters: list[dict] = []  # her cluster: {actor_variants, confidences, verified, seen_counts}

    for row in no_char_rows:
        a = row["actor_name"]
        if not a:
            continue
        conf = row.get("confidence", 0.6)
        verified = row.get("is_verified_name", False)
        seen = row.get("seen_count", 1)

        matched = False
        if _HAS_RAPIDFUZZ:
            for cluster in clusters:
                # Mevcut cluster'daki varyantlarla kelime bazlı karşılaştır
                for existing in cluster["actor_variants"]:
                    if _words_fuzzy_match(a, existing):
                        cluster["actor_variants"].append(a)
                        cluster["confidences"].append(conf)
                        cluster["verified"].append(verified)
                        cluster["seen_counts"].append(seen)
                        matched = True
                        break
                if matched:
                    break

        if not matched:
            # Fuzzy yok veya eşleşme bulunamadı — exact key ile dene
            key = _norm_key(a)
            found_exact = False
            for cluster in clusters:
                if cluster.get("exact_key") == key:
                    cluster["actor_variants"].append(a)
                    cluster["confidences"].append(conf)
                    cluster["verified"].append(verified)
                    cluster["seen_counts"].append(seen)
                    found_exact = True
                    break
            if not found_exact:
                clusters.append({
                    "exact_key": key,
                    "actor_variants": [a],
                    "confidences": [conf],
                    "verified": [verified],
                    "seen_counts": [seen],
                })

    no_char_based: list[dict] = []
    for cluster in clusters:
        key = cluster.get("exact_key", "")
        if key in seen_actors:
            continue
        actor = _best_actor([v for v in cluster["actor_variants"] if v])
        if not actor:
            continue
        best_conf = max(cluster.get("confidences") or [0.6])
        is_verified = any(cluster.get("verified") or [])
        total_seen = sum(cluster.get("seen_counts") or [1])

        # Seen count + noise score tabanlı filtreleme:
        # Tek frame'de görülmüş, çok bozuk, düşük confidence → çıkar
        if (total_seen <= 1
                and _noise_score(actor) >= 8
                and best_conf < 0.90):
            continue

        no_char_based.append({
            "actor_name": actor,
            "character_name": "",
            "confidence": round(best_conf, 3),
            "is_verified_name": is_verified,
            "is_llm_verified": is_verified,
        })

    # Birleştir: önce karakter bilgisi olanlar, sonra olmayanlar
    out = char_based + no_char_based

    def _sort_key(r):
        tmdb_order = r.get("tmdb_order")
        if tmdb_order is not None:
            return (0, tmdb_order, _norm_key(r.get("actor_name", "")))
        return (1, 0, _norm_key(r.get("character_name", "")),
                _norm_key(r.get("actor_name", "")))

    out.sort(key=_sort_key)

    # Çöp filtresi: doğrulanmamış ve düşük confidence → çıkar
    final = []
    for row in out:
        verified = (row.get("is_verified_name", False)
                    or row.get("is_llm_verified", False)
                    or row.get("is_name_db_protected", False))
        conf = row.get("confidence", 0.5)
        actor = (row.get("actor_name") or "").strip()
        char = (row.get("character_name") or "").strip()

        # Kısa çöp: actor_name 3 karakterden kısa ve karakter ismi yoksa → çıkar
        if not char and len(actor) < 3:
            continue

        # Tamamen küçük harf tek kelime → OCR gürültüsü (oulon, nibeu, etiol)
        if not char and actor and ' ' not in actor and actor.islower():
            continue

        if verified or conf >= 0.70:
            final.append(row)

    # ── Post-merge fuzzy sweep ──
    # Final listedeki actor_name'leri birbirleriyle karşılaştır.
    # WRatio >= 75 olanları birleştir (en yüksek confidence'lıyı tut).
    if _HAS_RAPIDFUZZ and len(final) > 1:
        merged_flags = [False] * len(final)
        merged_out = []
        for i, row_i in enumerate(final):
            if merged_flags[i]:
                continue
            a_i = row_i.get("actor_name", "")
            for j in range(i + 1, len(final)):
                if merged_flags[j]:
                    continue
                a_j = final[j].get("actor_name", "")
                # Sadece aynı karakter ismine sahip veya ikisi de karakter ismi olmayanları birleştir
                char_i = row_i.get("character_name", "")
                char_j = final[j].get("character_name", "")
                if char_i != char_j:
                    continue
                if a_i and a_j and _words_fuzzy_match(a_i, a_j):
                    # En yüksek confidence'lı olanı tut
                    if final[j].get("confidence", 0) > row_i.get("confidence", 0):
                        final[i] = final[j]
                        row_i = final[i]
                        a_i = row_i.get("actor_name", "")
                    merged_flags[j] = True
                    # verified ise işaretle
                    if final[j].get("is_verified_name") or final[j].get("is_llm_verified"):
                        final[i]["is_verified_name"] = True
                        final[i]["is_llm_verified"] = True
            merged_out.append(final[i])
        final = merged_out

    # ── Çok-kişi birleşim tespiti: 3+ büyük harf kelime → ayır ──
    expanded = []
    for row in final:
        actor = (row.get("actor_name") or "").strip()
        words = actor.split()
        # 3+ kelime, hepsi büyük harfle başlıyor VE toplam uzunluk > 20
        if (len(words) >= 3
                and all(w and w[0].isupper() for w in words)
                and len(actor) > 20):
            mid = len(words) // 2
            name1 = " ".join(words[:mid])
            name2 = " ".join(words[mid:])
            if len(name1) >= 5 and len(name2) >= 5:
                row1 = dict(row)
                row1["actor_name"] = name1
                row2 = dict(row)
                row2["actor_name"] = name2
                expanded.extend([row1, row2])
                continue
        expanded.append(row)
    final = expanded

    return final

def _canonicalize_crew(crew: list[dict]) -> list[dict]:
    # Rol bazlı grupla, sonra her rol grubu içinde isim fuzzy clustering yap
    role_groups: dict[str, list[dict]] = {}  # norm_role_key → entries
    for row in crew or []:
        name = (row.get("name") or "").strip()
        role = (row.get("role") or row.get("job") or "").strip()
        if not name and not role:
            continue
        role_key = _norm_key(role)
        if role_key not in role_groups:
            role_groups[role_key] = []
        role_groups[role_key].append({"name": name, "role": role})

    out = []
    for entries in role_groups.values():
        # Her rol grubunda isimleri fuzzy cluster'la
        clusters: list[dict] = []  # {"name_variants": [...], "role_variants": [...]}
        for entry in entries:
            name = entry["name"]
            role = entry["role"]
            matched = False
            if _HAS_RAPIDFUZZ and name:
                for cluster in clusters:
                    for existing in cluster["name_variants"]:
                        if existing and _words_fuzzy_match(name, existing):
                            cluster["name_variants"].append(name)
                            if role:
                                cluster["role_variants"].append(role)
                            matched = True
                            break
                    if matched:
                        break
            if not matched:
                # Exact key fallback
                name_key = _norm_key(name)
                for cluster in clusters:
                    if cluster.get("exact_key") == name_key:
                        cluster["name_variants"].append(name)
                        if role:
                            cluster["role_variants"].append(role)
                        matched = True
                        break
                if not matched:
                    clusters.append({
                        "exact_key": name_key,
                        "name_variants": [name] if name else [],
                        "role_variants": [role] if role else [],
                    })

        for cluster in clusters:
            name = _best_variant([v for v in cluster["name_variants"] if v])
            role = _best_variant([v for v in cluster["role_variants"] if v])
            if name or role:
                out.append({"name": name, "role": role, "job": role})

    out.sort(key=lambda r: (_norm_key(r.get("role", "")), _norm_key(r.get("name", ""))))
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
                 profile, scope, first_min, last_min, keywords=None, logos=None,
                 content_profile_name: str | None = None,
                 audio_result: dict | None = None,
                 ts: str | None = None):
        total_sec = sum(s.get("duration_sec", 0) for s in stage_stats.values())
        dur = video_info.get("duration_seconds", 1)

        # Final çıktı temizliği (dedup + en iyi varyant seçimi)
        # TMDB kaynaklı veriyi canonicalize'dan geçirme — zaten temiz ve sıralı
        # verification_status veya frame alanı ile TMDB kaynağı kontrol edilir
        try:
            is_tmdb = (
                credits_data.get("verification_status") == "tmdb_verified"
                or all(c.get("frame") == "tmdb" for c in (credits_data.get("cast") or []) if c)
            )
            if not is_tmdb:
                if credits_data.get('cast'):
                    credits_data['cast'] = _canonicalize_cast(credits_data['cast'])
                if credits_data.get('crew'):
                    credits_data['crew'] = _canonicalize_crew(credits_data['crew'])
            # technical_crew her durumda crew ile senkron tutulur
            if credits_data.get('crew') and 'technical_crew' not in credits_data:
                credits_data['technical_crew'] = list(credits_data['crew'])
        except Exception as e:
            # Log canonicalization errors but continue with uncanonicalized data
            import logging
            logging.warning(f"Credits canonicalization failed: {e}")

        if not keywords:
            keywords = [c["actor_name"] for c in credits_data.get("cast", [])[:10]
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
                "content_type": content_profile_name or "FilmDizi",
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
            "film_title": credits_data.get("film_title", ""),
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

        filename = video_info.get("filename", "out")
        output_name, film_id_parsed, film_name_tr_parsed = _extract_output_name(filename)
        if not output_name:
            output_name = Path(filename).stem
        jp = self.out / f"{output_name}_report.json"
        tp = _safe_path(self.out / f"{output_name}_teknik.txt")
        tr_p = _safe_path(self.out / f"{output_name}_tscr.txt")

        with open(jp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self._write_report(report, tp, audio_result=audio_result)
        self._write_transcript(video_info, audio_result, tr_p)
        user_tp = _safe_path(self.out / f"{output_name}.txt")
        self._write_user_report(
            report, user_tp, audio_result=audio_result,
            film_name_tr=film_name_tr_parsed, film_id=film_id_parsed,
        )
        return str(jp), str(tp), str(tr_p), str(user_tp)

    def _write_user_report(self, r, path, audio_result: dict | None = None,
                           film_name_tr: str = "", film_id: str = ""):
        """Kullanıcıya yönelik temiz, sabit formatlı TXT raporu yaz.

        Format:
            ================================================================
              FİLM / PROGRAM BİLGİLERİ
            ================================================================
              FİLMİN ADI            :     KADIN VE DENİZCİ
              ...
            ================================================================
              ÖZET
            ================================================================
              (SPOILER + ÖZET)
            ================================================================
              ANAHTAR SÖZCÜKLER
            ================================================================
              JANE WYMAN, DENNIS MORGAN, ...
            ----
              OYUNCULAR
            ----
              JANE WYMAN
              ...
            ------
              CAST
            ------
              JANE WYMAN — HELEN WRIGHT
              ...
            ================================================================
              OLUŞTURULMA: ...
            ================================================================

        Tüm metin BÜYÜK HARF (Türkçe kurallara göre).
        """
        L = []
        sep = "=" * 64
        sep_short = "-" * 4
        sep_long = "-" * 6
        fi = r["file_info"]
        cr = r["credits"]
        filename = fi.get("filename", "")

        # film_name_tr: dosya adından parse edilen Türkçe isim (generate'den gelir)
        # Fallback: dosya adından yeniden çek
        if not film_name_tr:
            _, _parsed_id, film_name_tr = _extract_output_name(filename)
            if not film_id:
                film_id = _parsed_id

        # original_title: credits/TMDB'den gelen yabancı dil ismi
        original_title = (
            cr.get("tmdb_original_title") or
            cr.get("original_title") or cr.get("original_name") or
            cr.get("film_title") or film_name_tr
        ).strip()

        year = str(cr.get("year") or "")
        resolution = fi.get("resolution", "")
        fps = fi.get("fps", 0)
        fps_str = f"{fps} FRAME" if fps else ""
        duration = fi.get("duration_human", "")

        # ── FİLM / PROGRAM BİLGİLERİ ─────────────────────────────────
        L.append(sep)
        L.append("  FİLM / PROGRAM BİLGİLERİ")
        L.append(sep)

        fw = 22  # field column width
        L.append(f"  {'FİLMİN ADI':<{fw}}:     {film_name_tr}")
        L.append(f"  {'FİLMİN ORJİNAL ADI':<{fw}}:     {original_title}")
        L.append(f"  {'FİLMİN ID':<{fw}}:     {film_id}")
        L.append(f"  {'YAPIM TARİHİ':<{fw}}:     {year}")
        bolum = _extract_episode_from_id(film_id) if film_id else 'YOK'
        L.append(f"  {'BÖLÜM':<{fw}}:     {bolum}")
        L.append(f"  {'ÇÖZÜNÜRLÜK':<{fw}}:     {resolution}")
        L.append(f"  {'FRAME':<{fw}}:     {fps_str}")
        L.append(f"  {'TOPLAM SÜRE':<{fw}}:     {duration}")

        cast_list = cr.get("cast") or []

        # ── ÖZET ─────────────────────────────────────────────────────
        L.append(sep)
        L.append("  ÖZET")
        L.append(sep)
        summary = None
        if audio_result and isinstance(audio_result, dict):
            summary_raw = (
                audio_result.get("summary") or
                audio_result.get("summary_tr") or
                audio_result.get("ollama_summary")
            )
            if isinstance(summary_raw, dict):
                summary = summary_raw.get("en") or summary_raw.get("tr") or ""
            else:
                summary = summary_raw

        # Bölüm numaralı giriş ekle
        episode_no = _extract_episode_from_id(film_id) if film_id else 'YOK'
        if summary and episode_no not in ('YOK', '0000', '0', ''):
            episode_int = episode_no.lstrip('0') or '0'
            episode_prefix = f"{film_name_tr} dizisinin {episode_int}. bölümünde"
            summary = episode_prefix + " " + summary

        if summary:
            L.append(f"  {summary}")
        else:
            L.append("  ÖZET OLUŞTURULAMADI.")

        # ── ANAHTAR SÖZCÜKLER ─────────────────────────────────────────
        L.append(sep)
        L.append("  ANAHTAR SÖZCÜKLER")
        L.append(sep)

        actor_names = [
            (c.get("actor_name") or "").strip()
            for c in cast_list
            if (c.get("actor_name") or "").strip()
        ]
        if actor_names:
            L.append(f"  {' ; '.join(actor_names)}")
        else:
            L.append("  YOK")

        # ── OYUNCULAR ────────────────────────────────────────────────
        L.append(sep_short)
        L.append("  OYUNCULAR")
        L.append(sep_short)

        if cast_list:
            for c in cast_list:
                actor = (c.get("actor_name") or "").strip()
                if actor:
                    L.append(f"  {actor}")
        else:
            L.append("  YOK")

        # ── CAST ─────────────────────────────────────────────────────
        L.append(sep_long)
        L.append("  CAST")
        L.append(sep_long)

        if cast_list:
            for c in cast_list:
                actor = (c.get("actor_name") or "").strip()
                char = (c.get("character_name") or "").strip()
                if actor:
                    if char:
                        L.append(f"  {actor} \u2014 {char}")
                    else:
                        L.append(f"  {actor}")
        else:
            L.append("  YOK")

        L.append(sep)
        L.append(f"  OLUŞTURULMA: {r['generated_at']}")
        L.append(sep)

        # Tüm satırları BÜYÜK HARF'e çevir (Türkçe kurallara göre)
        L = [_to_upper_tr(line) for line in L]

        # Satır başına max 20 kelime
        L_wrapped = []
        for line in L:
            stripped = line.lstrip()
            if stripped.startswith('=') or stripped.startswith('-'):
                L_wrapped.append(line)
            else:
                L_wrapped.append(_wrap_max_words(line, max_words=20, indent="  "))
        L = L_wrapped

        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(L))

    def _write_report(self, r, path, audio_result: dict | None = None):
        L = []
        sep  = "=" * 65
        sep2 = "-" * 65
        fi = r["file_info"]
        p  = r["processing"]
        cr = r["credits"]

        # ═══ BLOK 1 — VİDEO BİLGİLERİ ═══
        L.append(sep)
        L.append("  BLOK 1 — VİDEO BİLGİLERİ")
        L.append(sep)
        L.append(f"\n  Dosya      : {fi['filename']}")
        L.append(f"  Sure       : {fi['duration_human']}")
        L.append(f"  Cozunurluk : {fi['resolution']} @ {fi['fps']} FPS")
        L.append(f"  Boyut      : {fi['filesize_bytes']/1024/1024:.1f} MB")
        L.append(f"  Profil     : {r['profile']}")
        L.append(f"  Icerik     : {p.get('content_type','?')}")
        L.append(f"  OCR Motor  : {p.get('ocr_engine','?')}")

        # ASR motor bilgisi
        asr_model = "—"
        if audio_result and isinstance(audio_result, dict):
            asr_engine = audio_result.get("asr_engine") or "ASR"
            asr_model = audio_result.get("whisper_model") or audio_result.get("model", "—")
            L.append(f"  ASR Motor  : {asr_engine} ({asr_model})")
        else:
            L.append(f"  ASR Motor  : —")

        # Pipeline stages
        L.append(f"\n{sep2}")
        L.append("  PIPELINE")
        L.append(sep2)
        for s in p["stages"]:
            ico = "[OK]" if s["status"] in ("ok", "completed") else "[--]" if s["status"] == "skipped" else "[!!]"
            L.append(f"  {ico} {s['name']:24s} {s['duration_sec']:8.1f}s")

        # Audio stage bilgileri de pipeline'a ekle
        if audio_result and isinstance(audio_result, dict) and audio_result.get("status") == "ok":
            stages_map = audio_result.get("stages") or {}
            if isinstance(stages_map, dict):
                for stage_name, stage_info in stages_map.items():
                    if isinstance(stage_info, dict):
                        st = stage_info.get("status", "ok")
                        dur = stage_info.get("duration_sec", 0)
                        ico = "[OK]" if st in ("ok", "completed") else "[--]" if st == "skipped" else "[!!]"
                        L.append(f"  {ico} {'AUDIO_' + stage_name.upper():24s} {dur:8.1f}s")

        # ═══ BLOK 2 — OYUNCULAR ═══
        L.append(f"\n{sep}")
        L.append("  BLOK 2 — OYUNCULAR")
        L.append(sep)
        if cr.get("cast"):
            # Dinamik hizalama: en uzun oyuncu adına göre sütun genişliği belirle
            max_actor_len = max(
                (len(c.get("actor_name") or "") for c in cr["cast"]),
                default=20
            )
            col_width = max(max_actor_len + 1, 22)  # minimum 22 karakter, +1 for padding
            char_col_width = 22
            L.append(f"\n  {'Oyuncu Adı':<{col_width}}  --  {'Karakter Adı':<{char_col_width}}  [Skor]")
            total_dash = col_width + char_col_width + 12  # 12 = "  " + icon + " " + "  --  " + "  "
            L.append(f"  {'─' * min(total_dash, 63)}")
            for c in cr["cast"]:
                ch = c.get("character_name") or ""
                ac = c.get("actor_name") or ""
                score = c.get("confidence")
                score_str = f"[{score:.2f}]" if score is not None else ""
                icon = self._verification_icon(c)
                if ch:
                    L.append(f"  {icon} {ac:<{col_width}}-- {ch:<{char_col_width}}  {score_str}")
                else:
                    no_char_width = col_width + char_col_width + 4  # combined width
                    L.append(f"  {icon} {ac:<{no_char_width}}  {score_str}")
        else:
            L.append("\n  (Oyuncu verisi yok)")

        # ═══ BLOK 3 — YAPIM EKİBİ ═══
        L.append(f"\n{sep}")
        L.append("  BLOK 3 — YAPIM EKİBİ")
        L.append(sep)

        # Önce yönetmenler
        if cr.get("directors"):
            dir_names = self._director_names(cr)
            if dir_names:
                L.append(f"\n  Yönetmen   : {', '.join(dir_names)}")

        # Teknik ekip
        crew_list = cr.get("technical_crew") or cr.get("crew") or []
        if crew_list:
            L.append(f"\n  {'─' * 63}")  # indented crew separator (63 dashes)
            for t in crew_list:
                role_txt = t.get('role_tr') or t.get('role') or t.get('job') or ''
                L.append(f"  {role_txt:14s}: {t.get('name','')}")

        if not cr.get("directors") and not crew_list:
            L.append("\n  (Ekip verisi yok)")

        # ═══ BLOK 4 — ÖZET ═══
        L.append(f"\n{sep}")
        L.append("  BLOK 4 — ÖZET")
        L.append(sep)

        # Spor profili: Maç Bilgileri BLOK 4 içinde gösterilir
        match_data = r.get("match_data")
        if match_data or p.get("content_type") == "Spor":
            if match_data:
                if match_data.get("spor_turu"):
                    L.append(f"\n  Spor Turu     : {match_data['spor_turu']}")
                if match_data.get("lig"):
                    L.append(f"  Lig           : {match_data['lig']}")
                if match_data.get("sehir"):
                    L.append(f"  Sehir         : {match_data['sehir']}")
                takimlar = match_data.get("takimlar", [])
                if takimlar:
                    L.append("  Takimlar      :")
                    for tk in takimlar:
                        L.append(f"    {tk.get('isim','')} — {tk.get('skor','?')}")
                td = match_data.get("teknik_direktorler", [])
                if td:
                    L.append(f"  Teknik Dir.   : {', '.join(td)}")
                olaylar = match_data.get("olaylar", [])
                if olaylar:
                    L.append("  Olaylar       :")
                    for ol in olaylar:
                        L.append(f"    {ol.get('dakika','')}' {ol.get('olay','')} — {ol.get('oyuncu','')} ({ol.get('takim','')})")
            else:
                L.append("\n  (Mac verisi bulunamadi)")
        else:
            # Gemini özeti varsa göster; yoksa "aktif değil" mesajı
            summary = None
            if audio_result and isinstance(audio_result, dict):
                summary = audio_result.get("summary") or audio_result.get("summary_tr") or audio_result.get("ollama_summary")
            if summary:
                L.append(f"\n  {summary}")
            else:
                L.append("\n  Özet oluşturma aktif değil")

        # Footer
        L.append(f"\n{sep}")
        L.append(f"  Olusturulma: {r['generated_at']}")
        L.append(sep)

        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(L))

    def _write_transcript(self, video_info: dict, audio_result, path):
        """Zaman damgalı transcript dosyası yaz."""
        sep  = "=" * 65
        sep2 = "-" * 65
        L = []
        L.append(sep)
        L.append("  VİTOS — TRANSCRİPT")
        L.append(sep)
        L.append(f"  Dosya: {video_info.get('filename', '')}")
        L.append(f"  Sure : {video_info.get('duration_human', '')}")
        asr_model = "—"
        asr_engine = "ASR"
        if audio_result and isinstance(audio_result, dict):
            asr_engine = audio_result.get("asr_engine") or "ASR"
            asr_model = audio_result.get("whisper_model") or audio_result.get("model", "—")
        L.append(f"  ASR  : {asr_engine} {asr_model}")
        L.append(f"\n{sep2}\n")

        transcript = []
        if audio_result and isinstance(audio_result, dict):
            transcript = audio_result.get("transcript", [])

        if transcript:
            for seg in transcript:
                start = seg.get("start", 0)
                text  = seg.get("text", "").strip()
                speaker = seg.get("speaker", "").strip()
                ts = _fmt_hms_shared(start, with_ms=False)
                if speaker:
                    L.append(f"[{ts}] {speaker}: {text}")
                else:
                    L.append(f"[{ts}] {text}")
        else:
            L.append("  (Transcript verisi yok)")

        L.append(f"\n{sep}")

        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(L))

    # Keep _write_txt as an alias for backward compatibility
    def _write_txt(self, r, path):
        self._write_report(r, path)

    @staticmethod
    def _verification_icon(cast_entry: dict) -> str:
        """Doğrulama kaynağına göre ikon döndür."""
        if cast_entry.get("frame") == "tmdb":
            return "◆"  # TMDB doğrulanmış
        method = cast_entry.get("match_method", "")
        if method == "exact_db":
            return "✓"
        if method in ("fuzzy", "phonetic", "parts", "hardcoded"):
            return "~"
        if cast_entry.get("is_llm_verified"):
            return "L"  # LLM onaylı
        if cast_entry.get("is_verified_name"):
            return "✓"
        return "?"

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
