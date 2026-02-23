"""
turkish.py — Turk isim veritabani + ASCII normalizasyon + akilli isim bolucu.

PaddleOCR dusuk cozunurluklu videolarda bosluklari kaybeder:
  "SEBNEMSONMEZ" → "SEBNEM SONMEZ"
  "HALUKBENER"   → "HALUK BENER"
"""

_TR_MAP = str.maketrans({
    "c": "c", "C": "C", "g": "g", "G": "G",
    "i": "i", "I": "I", "o": "o", "O": "O",
    "s": "s", "S": "S", "u": "u", "U": "U",
    "\u00e7": "c", "\u00c7": "C",  # ç Ç
    "\u011f": "g", "\u011e": "G",  # ğ Ğ
    "\u0131": "i", "\u0130": "I",  # ı İ
    "\u00f6": "o", "\u00d6": "O",  # ö Ö
    "\u015f": "s", "\u015e": "S",  # ş Ş
    "\u00fc": "u", "\u00dc": "U",  # ü Ü
    "\u00e2": "a", "\u00c2": "A",  # â Â
    "\u00ee": "i", "\u00ce": "I",  # î Î
    "\u00fb": "u", "\u00db": "U",  # û Û
})


def normalize_tr(text: str) -> str:
    """Turkce ozel karakterleri ASCII'ye cevir + lowercase + bosluk kaldir."""
    return text.translate(_TR_MAP).lower().replace(" ", "")


def ascii_key(text: str) -> str:
    """Dedup / sozluk anahtari icin ASCII-normalize edilmis dize doner."""
    return (text or "").translate(_TR_MAP).lower().replace(" ", "")


def normalize_tr_keep_spaces(text: str) -> str:
    """Turkce -> ASCII, lowercase, bosluklari koru."""
    return text.translate(_TR_MAP).lower()


# ═══════════════════════════════════════════════════════════════════
# TURK ISIM VERITABANI (~300 yaygin isim, ASCII normalize)
# ═══════════════════════════════════════════════════════════════════
_ERKEK = {
    "ali", "ahmet", "mehmet", "mustafa", "hasan", "huseyin", "ibrahim",
    "ismail", "osman", "yusuf", "murat", "halil", "cemal", "kemal",
    "engin", "okan", "serdar", "haluk", "volkan", "sarp", "senol",
    "cem", "metin", "muharrem", "cetin", "atilla", "fedai", "sezgin",
    "mahmut", "ugur", "deniz", "sadi", "recep", "haydar", "tuncay",
    "cenk", "vahit", "armagan", "satilmis", "selim", "can", "emre",
    "baris", "tolga", "burak", "onur", "sinan", "erkan", "erdem",
    "bulent", "cengiz", "orhan", "faruk", "fikret", "ilhan", "necati",
    "nedim", "nevzat", "nihat", "nuri", "oktay", "omer", "ozcan",
    "ozgur", "remzi", "ridvan", "riza", "sahin", "salih", "sami",
    "sedat", "selcuk", "semih", "sener", "tarik", "temel", "tamer",
    "timur", "tugrul", "turgut", "turhan", "ufuk", "umit", "vedat",
    "veysel", "yakup", "yasar", "yavuz", "yilmaz", "zafer",
    "zeki", "cihan", "dogan", "erhan", "ferhat", "gokhan", "gurkan",
    "hakan", "hamza", "hikmet", "ilker", "irfan", "kadir", "koray",
    "levent", "mert", "mesut", "mithat", "oguz", "polat", "rasim",
    "ruhi", "rustem", "sabri", "savas", "sukru", "suleyman", "tahir",
    "tayfun", "tuncer", "turan", "utku", "yasin", "yigit", "yuksel",
    "alper", "alpay", "adem", "adnan", "akin", "arif", "asim",
    "aydin", "bahadir", "bayram", "bedri", "bekir", "bilal", "cahit",
    "celal", "cuneyt", "devrim", "dincer", "emin", "erdal", "ersin",
    "esat", "eyup", "fatih", "fehmi", "feridun", "fuat", "galip",
    "goksel", "gunduz", "gungor", "habib", "hafiz", "halim", "hamdi",
    "hami", "hayri", "husnu", "idris", "ihsan", "ilyas", "ismet",
    "kaan", "kamil", "kasim", "kazim", "kenan", "kudret", "kursat",
    "latif", "lemi", "lutfi", "macit", "mahir", "mazhar", "muammer",
    "muhsin", "mukadder", "munir", "muzaffer", "naci", "nail", "namik",
    "nasuh", "nazim", "necdet", "necip", "ogun", "orcan", "ozan",
    "ozdemir", "ramazan", "rasit", "rauf", "refik", "reha",
    "resit", "sadik", "sakir", "samim", "sefer",
    "sefik", "selahattin", "serhan", "serif", "sevket", "suat",
    "suphi", "sureyya", "tekin", "tevfik", "tufan", "ulas", "unsal",
    "vural", "yalcin", "yaman", "yunus",
    "efe", "ata", "tan", "han",
}

_KADIN = {
    "sebnem", "tuba", "ayca", "nuray", "gulden", "damla", "sahika",
    "belgin", "seyhan", "meryem", "behiye", "ruya", "cansu", "fidan",
    "ayten", "zeliha", "sedef", "derya", "gonul",
    "asli", "ayse", "fatma", "emine", "hatice", "zeynep", "elif",
    "sultan", "hanife", "merve", "busra", "esra", "kubra", "seda",
    "tugba", "ozlem", "melek", "dilek", "filiz", "sevim", "nursen",
    "nesrin", "nilufer", "nuran", "nurcan", "nurgul", "pelin", "pinar",
    "rabia", "rana", "rengin", "reyhan", "saadet", "sabiha", "selma",
    "sema", "senay", "serpil", "sevda", "sevgi", "sibel", "simge",
    "songul", "suheyla", "sule", "sumeyye", "sureyya", "tanju",
    "turkan", "ulku", "vildan", "yasemin", "yesim", "yildiz",
    "zehra", "zuhal", "zubeyde", "hulya", "idil", "inci", "ipek",
    "jale", "lale", "lamia", "leman", "leyla", "mediha", "mine",
    "muazzez", "munevver", "naciye", "nazan", "nalan", "nese",
    "nevin", "nigar", "nur", "olcay", "oya", "ozge", "perihan",
    "piraye", "saliha", "samiye", "serap",
    "sevil", "sevtap", "sirin", "seval", "banu", "berna",
    "betul", "bilge", "birsen", "burcu", "canan", "ceyda", "cigdem",
    "demet", "duygu", "ebru", "ece", "ela", "elcin", "emel",
    "esen", "evrim", "ferda", "feride", "fulya", "fusun", "gamze",
    "gizem", "gonca", "gozde", "gul", "gulcin", "guler", "gulizar",
    "gulsah", "gulsum", "hande", "hicran", "hilal",
    "gulten", "nebahat", "necla", "nefise", "pervin", "ruhan",
}

ALL_NAMES: set[str] = _ERKEK | _KADIN


# Global TurkishNameDB instance — lazy initialized
_name_db = None


def _get_name_db():
    """
    ISSUE-04 FIX: split_concatenated_name() TurkishNameDB'yi kullanmıyordu.
    ~300 hardcoded ALL_NAMES ile DP yapıyordu, 9699 isimlik DB yoksayılıyordu.
    Lazy init: ilk çağrıda yükler, sonra cache'den döner.
    """
    global _name_db
    if _name_db is None:
        try:
            from core.turkish_name_db import TurkishNameDB
            _name_db = TurkishNameDB()
        except Exception:
            _name_db = False   # Yükleme başarısız — False olarak işaretle
    return _name_db if _name_db is not False else None


def split_concatenated_name(text: str, min_first: int = 2, min_last: int = 3) -> str:
    """
    Birlesik yazilmis isimleri ayir.

    "SEBNEMSONMEZ" -> "SEBNEM SONMEZ"
    "HALUKBENER"   -> "HALUK BENER"
    "Cansu"        -> "Cansu" (sag taraf 2 harf, bolunmez)
    "ENGIN ALKAN"  -> "ENGIN ALKAN" (zaten ayrik)

    ISSUE-04 FIX: TurkishNameDB mevcutsa 9699 isim + DP algoritmasıyla böler.
    Yoksa eski ALL_NAMES (~300 isim) ile devam eder.
    """
    if " " in text.strip() and len(text.strip().split()) >= 2:
        return text

    # TurkishNameDB ile DP-tabanlı bölme (öncelikli)
    db = _get_name_db()
    if db is not None:
        try:
            parts = db.split_concatenated(text)
            if parts and len(parts) > 1:
                return " ".join(parts)
        except Exception:
            pass  # Fallback'e geç

    # Fallback: eski ALL_NAMES (~300) ile
    norm = normalize_tr(text)

    if len(norm) < (min_first + min_last) or len(norm) > 30:
        return text

    best_split = -1
    best_name_len = 0

    for i in range(min_first, len(norm) - min_last + 1):
        left = norm[:i]
        if left in ALL_NAMES and i > best_name_len:
            right = norm[i:]
            if len(right) >= min_last:
                best_split = i
                best_name_len = i

    if best_split > 0:
        original_clean = text.replace(" ", "")
        left_part = original_clean[:best_split]
        right_part = original_clean[best_split:]

        right_norm = normalize_tr(right_part)
        second_split = _try_split(right_part, right_norm, min_first, min_last)
        if second_split:
            return f"{left_part} {second_split}"
        return f"{left_part} {right_part}"

    return text


def _try_split(text, norm, min_first, min_last):
    if len(norm) < min_first + min_last:
        return ""
    for i in range(min_first, len(norm) - min_last + 1):
        left = norm[:i]
        if left in ALL_NAMES:
            right = norm[i:]
            if len(right) >= min_last:
                original_clean = text.replace(" ", "")
                return f"{original_clean[:i]} {original_clean[i:]}"
    return ""


def is_likely_name(text: str) -> bool:
    words = text.strip().split()
    if len(words) < 2 or len(words) > 6:
        return False
    if text.isupper():
        return len(words) >= 2
    caps = sum(1 for w in words if w and w[0].isupper())
    return caps >= len(words) * 0.5


def is_likely_turkish_name(text_normalized: str) -> bool:
    for i in range(2, min(len(text_normalized), 12)):
        if text_normalized[:i] in ALL_NAMES:
            return True
    return False


def normalize_role_text(text: str) -> str:
    return normalize_tr(text)


# ═══════════════════════════════════════════════════════════════════
# ROL PARCASI TESPİTİ — isim olmayanları reddet
# ═══════════════════════════════════════════════════════════════════
ROLE_FRAGMENTS = {
    "yardimcisi", "yardimcilari", "asistani", "asistanlari",
    "sorumlusu", "sefi", "amiri", "miks", "miksaji",
    "koordinatoru", "yonetmeni", "tasarimcisi",
}


def is_role_fragment(text: str) -> bool:
    """Metin bir rol parcasi mi? (isim degil)"""
    norm = normalize_tr(text)
    return norm in ROLE_FRAGMENTS or len(norm) < 3
