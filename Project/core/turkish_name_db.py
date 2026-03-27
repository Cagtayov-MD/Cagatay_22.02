"""
turkish_name_db.py — IMDB-tabanlı isim veritabanı + OCR karakter onarımı.

Kaynak: F:\\IMDB\\db\\imdb.duckdb → names tablosu (15M+ kayıt)

Görevler:
1. imdb.duckdb'den tüm primaryName değerlerini RAM'e yükle
2. OCR ASCII bozulmalarını hardcoded tablo ile düzelt (SEBNEM → Şebnem)
3. Tam eşleşme: normalize(ocr_text) → IMDB canonical ismi
4. Fuzzy eşleşme: RapidFuzz WRatio ile en yakın "isim+soyisim" eşleştir

Kullanım:
    db = TurkishNameDB()
    canonical, score = db.find("Nida Sererli")   # → ("Nisa Serezli", 0.91)
    ok = db.is_name("Nisa Serezli")              # → True
"""

from __future__ import annotations

import unicodedata
from typing import Optional

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded OCR düzeltme tablosu (IMDB eşleşmeden önce uygulanır)
# ─────────────────────────────────────────────────────────────────────────────
_HARDCODED_FIXES: dict[str, str] = {
    'SEBNEM': 'Şebnem',
    'SONMEZ': 'Sönmez',
    'GUNDUZ': 'Gündüz',
    'GUNDUZALP': 'Gündüzalp',
    'CONDOZALP': 'Gündüzalp',
    'CONDOZ': 'Gündüz',
    'CONDOGDU': 'Gündoğdu',
    'CUNDOGDU': 'Gündoğdu',
    'UOLKAN': 'Volkan',
    'DEURIM': 'Devrim',
    'UOLKANGIRGIN': 'Volkan Girgin',
    'DIKMENSEYMEN': 'Dikmen Seymen',
    'AYCAMUTLUGIL': 'Ayça Mutlugil',
    'GOLDENCUNEY': 'Gülden Güney',
    'GOLDENGONEY': 'Gülden Güney',
    'GOLTUGERSOY': 'Göltügersoy',
    'COLTUGERSOY': 'Göltügersoy',
    'SEYHANINCEOGLU': 'Seyhan İnceoğlu',
    'CETINDEURIM': 'Çetin Devrim',
    'FEDAIIPEK': 'Fedai İpek',
    'MAHMUTDEMIR': 'Mahmut Demir',
    'TUBAERDEM': 'Tuba Erdem',
    'ENGINALKAN': 'Engin Alkan',
    'VOLKANGIRGIN': 'Volkan Girgin',
    'DAMLAOZEN': 'Damla Özen',
    'SEDEFPEHLIVANOGLU': 'Sedef Pehlivanoğlu',
    'DERYAYARUC': 'Derya Yaruç',
    'SENOLSENTURK': 'Şenol Şentürk',
    'RUHISARI': 'Ruhi Sarı',
    'OYAYUCE': 'Oya Yüce',
    'AYCA': 'Ayça',
    'GUNEY': 'Güney',
    'GULDEN': 'Gülden',
    'UGUR': 'Uğur',
    'ORGUC': 'Örgüç',
    'TUNALI': 'Tunalı',
    'INCEOGLU': 'İnceoğlu',
    'IPEK': 'İpek',
    'CETIN': 'Çetin',
    'SENOL': 'Şenol',
    'SENTURK': 'Şentürk',
    'SAHIKA': 'Şahika',
    'ERKIRAN': 'Erkıran',
    'PEHLIVANOGLU': 'Pehlivanoğlu',
    'YARUC': 'Yaruç',
    'OZEN': 'Özen',
    'YUCE': 'Yüce',
    'RIZA': 'Rıza',
    'BASARAN': 'Başaran',
    'KALAFATOGLU': 'Kalafatoğlu',
    'SARI': 'Sarı',
    'DEMIR': 'Demir',
    'ONER': 'Öner',
    'NITA': 'Nisa',
    'SERELI': 'Serezli',
    'NITASERELI': 'Nisa Serezli',
    'NISASEREZLI': 'Nisa Serezli',
}


def _normalize_key(text: str) -> str:
    """
    Karşılaştırma anahtarı üret: büyük harf, boşluksuz, Türkçe→ASCII.
    Örnek: 'Nisa Serezli' → 'NISASEREZLI'
    """
    text = text.upper().strip().replace(' ', '')
    tr_map = str.maketrans({
        'Ç': 'C', 'ç': 'C',
        'Ğ': 'G', 'ğ': 'G',
        'İ': 'I', 'ı': 'I',
        'Ö': 'O', 'ö': 'O',
        'Ş': 'S', 'ş': 'S',
        'Ü': 'U', 'ü': 'U',
    })
    text = text.translate(tr_map)
    text = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in text if not unicodedata.combining(c))


def _phonetic_key(text: str) -> str:
    """
    OCR fonetik anahtarı: normalize et + yaygın harf karışıklıklarını düzelt.

    Dönüşümler (büyük harf üzerinde):
      W → N  (W ile N OCR'da karışır)
      Q → G  (Q ile G karışır — Kiril/Türkçe fontlarda)
    Ardından tekrarlanan harfler teke indirilir: CAMAAN → CAMAN
    """
    key = _normalize_key(text)           # büyük harf, boşluksuz, Türkçe→ASCII
    key = key.replace("W", "N")
    key = key.replace("Q", "G")
    # Tekrarlanan ardışık harfleri teke indir
    deduped = []
    for ch in key:
        if not deduped or ch != deduped[-1]:
            deduped.append(ch)
    return "".join(deduped)


class TurkishNameDB:
    """
    IMDB-tabanlı isim veritabanı.

    Tüm 15M+ IMDB ismi RAM'e yüklenir:
      _norm_to_canonical : {normalize(name): primaryName}  — exact match
      _names_list        : [primaryName, ...]               — fuzzy için

    find() akışı: hardcoded → exact → RapidFuzz full-list fuzzy
    """

    def __init__(self, db_path: str = "", log_cb=None, sql_path: str = ""):
        self._log = log_cb or (lambda m: print(m))
        self._norm_to_canonical: dict[str, str] = {}
        self._names_list: list[str] = []
        self._db_load_error: str | None = None

        self._hardcoded: dict[str, str] = {
            _normalize_key(k): v for k, v in _HARDCODED_FIXES.items()
        }

        # PERF-1: _all_names cache — _fuzzy_find her çağrıda yeni liste oluşturmamalı
        self._first_names: list[str] = []
        self._surnames: list[str] = []
        self._all_names: list[str] = []

        if db_path:
            self._load_from_explicit_path(db_path)
        else:
            self._load_imdb()

        # Cache'i yüklemeden sonra güncelle
        self._first_names = self._names_list
        self._all_names = self._first_names + self._surnames

    # ─────────────────────────────────────────────────────────────────
    # Yükleme
    # ─────────────────────────────────────────────────────────────────

    def _load_from_explicit_path(self, path: str) -> None:
        """db_path ile açık olarak verilen yolu işle (test uyumluluğu)."""
        from pathlib import Path
        if path.lower().endswith(".sql"):
            if not Path(path).is_file():
                self._db_load_error = f"SQL seed bulunamadı: {path}"
                self._log(f"  [NameDB] {self._db_load_error}")
        else:
            if not Path(path).is_file():
                self._db_load_error = f"DB bulunamadı: {path}"
                self._log(f"  [NameDB] {self._db_load_error}")

    def _load_imdb(self) -> None:
        """imdb.duckdb'den tüm primaryName değerlerini RAM'e yükle."""
        try:
            import duckdb
        except ImportError:
            self._db_load_error = "duckdb paketi yüklü değil"
            self._log(f"  [NameDB] {self._db_load_error} — hardcoded tablo aktif")
            return

        try:
            from config.runtime_paths import get_imdb_db_path
            imdb_path = get_imdb_db_path()
        except Exception:
            self._db_load_error = "DB yapılandırılmadı"
            self._log("  [NameDB] DB yapılandırılmadı — hardcoded tablo aktif")
            return

        from pathlib import Path
        if not Path(imdb_path).is_file():
            self._db_load_error = f"IMDB DB bulunamadı: {imdb_path}"
            self._log(f"  [NameDB] {self._db_load_error} — hardcoded tablo aktif")
            return

        try:
            self._log(f"  [NameDB] IMDB yükleniyor: {imdb_path}")
            con = duckdb.connect(imdb_path, read_only=True)
            cur = con.execute("SELECT primaryName FROM names")
            loaded = 0
            batch_size = 100_000
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for (name,) in rows:
                    if name:
                        key = _normalize_key(name)
                        if key not in self._norm_to_canonical:
                            self._norm_to_canonical[key] = name
                loaded += len(rows)
                if loaded % 1_000_000 == 0:
                    self._log(f"  [NameDB] {loaded:,} kayıt yüklendi...")
            con.close()

            self._names_list = list(self._norm_to_canonical.values())
            self._log(
                f"  [NameDB] IMDB yüklendi — "
                f"{len(self._names_list):,} benzersiz isim | {imdb_path}"
            )
        except Exception as e:
            self._db_load_error = str(e)
            self._log(f"  [NameDB] IMDB yüklenemedi: {e} — hardcoded tablo aktif")

    # ─────────────────────────────────────────────────────────────────
    # Temel sorgular
    # ─────────────────────────────────────────────────────────────────

    def is_name(self, text: str) -> bool:
        """
        Bu metin IMDB'de kayıtlı bir isim mi? (tam eşleşme)

        Kural: en az 2 kelime (isim+soyisim) zorunlu.
        Tek kelime girişler her zaman False döner.
        """
        if not text:
            return False
        if len(text.strip().split()) < 2:
            return False
        key = _normalize_key(text.replace(' ', ''))
        return key in self._norm_to_canonical or key in self._hardcoded

    # ─────────────────────────────────────────────────────────────────
    # OCR satır onarımı
    # ─────────────────────────────────────────────────────────────────

    def find(
        self, ocr_text: str, fuzzy_threshold: int = 75
    ) -> tuple[Optional[str], float]:
        """
        OCR metninden IMDB canonical isim bul.

        Öncelik:
          1. Hardcoded tam eşleşme
          2. IMDB exact match (normalize edilmiş key)
          3. RapidFuzz token_sort_ratio fuzzy (tüm isimler, score_cutoff ile erken çıkış)
             token_sort_ratio: kelime sırası farkını tolere eder, substring tuzağına düşmez.

        Returns:
            (canonical_name, score 0.0-1.0)
        """
        if not ocr_text or not ocr_text.strip():
            return None, 0.0

        # Kural: en az 2 kelime (isim+soyisim) zorunlu.
        # Tek kelime girişler hardcoded tablosunda olmadıkça hiçbir şeyle eşleşmez.
        stripped = ocr_text.strip()
        key = _normalize_key(stripped.replace(' ', ''))

        # 1. Hardcoded (tek kelime OCR artifact'ları için istisna)
        if key in self._hardcoded:
            return self._hardcoded[key], 1.0

        # 2 kelime minimum kontrolü
        if len(stripped.split()) < 2:
            return None, 0.0

        # 2. Exact match
        if key in self._norm_to_canonical:
            return self._norm_to_canonical[key], 1.0

        # 3. Fuzzy
        return self._fuzzy_find(stripped, fuzzy_threshold)

    def find_with_method(
        self, ocr_text: str, fuzzy_threshold: int = 75
    ) -> tuple[Optional[str], float, str]:
        """
        find() ile aynı, ek olarak eşleşme yöntemini döndürür.

        Returns:
            (canonical_name, score, method)
            method: "hardcoded" | "exact_db" | "fuzzy" | ""
        """
        if not ocr_text or not ocr_text.strip():
            return None, 0.0, ""

        stripped = ocr_text.strip()
        key = _normalize_key(stripped.replace(' ', ''))

        # 1. Hardcoded
        if key in self._hardcoded:
            return self._hardcoded[key], 1.0, "hardcoded"

        # 2 kelime minimum
        if len(stripped.split()) < 2:
            return None, 0.0, ""

        # 2. Exact match
        if key in self._norm_to_canonical:
            return self._norm_to_canonical[key], 1.0, "exact_db"

        canonical, score = self._fuzzy_find(stripped, fuzzy_threshold)
        return canonical, score, "fuzzy" if canonical else ""

    def correct_line(self, line: str) -> str:
        """
        Tek OCR satırını düzelt.
        score >= 0.85 ise düzeltilmiş canonical ismi döndür, altında orijinal.
        """
        stripped = line.strip()
        if not stripped:
            return line
        canonical, score = self.find(stripped)
        if canonical and score >= 0.85:
            return canonical
        return line

    def _fuzzy_find(
        self, text: str, threshold: int
    ) -> tuple[Optional[str], float]:
        """
        RapidFuzz token_sort_ratio ile en yakın IMDB ismini bul.

        token_sort_ratio seçildi çünkü:
        - Kelime sırası farkını tolere eder (Serezli Nisa == Nisa Serezli)
        - WRatio'nun partial_ratio substring tuzağına düşmez
          (WRatio, 'Nida' kısa ismini 'Nida Sererli' sorgusuna ~90 verir)
        - Benzer uzunluktaki isimlerde edit distance oranını doğru yansıtır
        """
        if not HAS_RAPIDFUZZ or not self._names_list:
            return None, 0.0
        result = rf_process.extractOne(
            text, self._names_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result:
            return result[0], round(result[1] / 100.0, 3)
        return None, 0.0

    def _fuzzy_find_top2(
        self, text: str, threshold: int = 0
    ) -> list[tuple[str, float]]:
        """RapidFuzz ile en yakın 2 IMDB ismini döndür."""
        if not HAS_RAPIDFUZZ or not self._names_list:
            return []
        results = rf_process.extract(
            text, self._names_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
            limit=2,
        )
        return [(r[0], round(r[1] / 100.0, 3)) for r in results]

    # ─────────────────────────────────────────────────────────────────
    # Pipeline entegrasyon yardımcıları
    # ─────────────────────────────────────────────────────────────────

    def repair_ocr_lines(self, ocr_lines: list) -> tuple[list, int]:
        """pipeline_runner için drop-in replacement."""
        repaired = 0
        for line in ocr_lines:
            if isinstance(line, dict):
                original = line.get('text', '')
                if not original:
                    continue
                fixed = self.correct_line(original)
                if fixed != original:
                    line['text'] = fixed
                    line['text_original'] = original
                    repaired += 1
            else:
                original = getattr(line, 'text', '')
                if not original:
                    continue
                fixed = self.correct_line(original)
                if fixed != original:
                    line.text = fixed
                    line.text_original = original
                    repaired += 1
        return ocr_lines, repaired

    def repair_layout_pairs(self, layout_pairs: list) -> tuple[list, int]:
        """pipeline_runner için drop-in replacement."""
        repaired = 0
        for pair in layout_pairs:
            original = getattr(pair, 'actor_name', '')
            if not original:
                continue
            fixed = self.correct_line(original)
            if fixed != original:
                pair.actor_name = fixed
                repaired += 1
        return layout_pairs, repaired

    # ─────────────────────────────────────────────────────────────────
    # Geriye dönük uyumluluk stub'ları
    # ─────────────────────────────────────────────────────────────────

    def is_first_name(self, text: str) -> bool:
        """Stub: IMDB ad/soyad ayrımı yapmaz, is_name() olarak davranır."""
        return self.is_name(text)

    def is_surname(self, text: str) -> bool:
        """Stub: IMDB ad/soyad ayrımı yapmaz, is_name() olarak davranır."""
        return self.is_name(text)

    def gender(self, text: str) -> str:
        """Stub: IMDB cinsiyet bilgisi içermez."""
        return '?'

    def check_swap_risk(self, left_token: str, right_token: str) -> bool:
        """Stub: IMDB ad/soyad tipi olmadığından swap tespiti yapılamaz."""
        return False

    def _fix_parts(self, parts: list[str], threshold: int = 85) -> tuple[list[str], bool]:
        """
        Her token için: hardcoded → exact_db → fuzzy.
        Fuzzy yalnızca DB'de bulunamazsa çağrılır (BUG-ANALYZE-02 düzeltmesi).
        Returns (fixed_parts, changed_flag).
        """
        fixed = []
        changed = False
        for part in parts:
            key = _normalize_key(part.replace(" ", ""))

            # 1. Hardcoded
            if key in self._hardcoded:
                canonical = self._hardcoded[key]
                fixed.append(canonical)
                if canonical != part:
                    changed = True
                continue

            # 2. Exact DB match
            if key in self._norm_to_canonical:
                canonical = self._norm_to_canonical[key]
                fixed.append(canonical)
                if canonical != part:
                    changed = True
                continue

            # 3. Fuzzy (sadece DB'de bulunamazsa)
            result, _ = self._fuzzy_find(part, threshold)
            if result and result != part:
                fixed.append(result)
                changed = True
            else:
                fixed.append(part)

        return fixed, changed

    def split_concatenated(self, text: str, max_parts: int = 3) -> list[str]:
        """Birleşik ismi parçalara ayır; hardcoded tablosundan canonical split yapar."""
        if not text:
            return [text]
        stripped = text.strip()
        key = _normalize_key(stripped.replace(" ", ""))
        if key in self._hardcoded:
            canonical = self._hardcoded[key]
            parts = canonical.split()
            if len(parts) >= 2:
                return parts[:max_parts]
        return [stripped]

    # ─────────────────────────────────────────────────────────────────
    # Dunder
    # ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._norm_to_canonical)

    def __repr__(self) -> str:
        return f"TurkishNameDB(imdb={len(self._norm_to_canonical):,}, hardcoded={len(self._hardcoded)})"


# ─────────────────────────────────────────────────────────────────────────────
# Singleton API  — uygulama ömrü boyunca tek örnek, RAM'de kalır
# ─────────────────────────────────────────────────────────────────────────────

import threading as _threading

_SINGLETON: "TurkishNameDB | None" = None
_SINGLETON_LOCK = _threading.Lock()


def get_instance(log_cb=None) -> "TurkishNameDB":
    """Global tekil örneği döndürür; ilk çağrıda yükler (thread-safe)."""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = TurkishNameDB(log_cb=log_cb)
    return _SINGLETON


def start_preload(log_cb=None) -> None:
    """Arka planda yüklemeyi başlatır; birden fazla kez çağrılabilir (idempotent)."""
    if _SINGLETON is not None:
        return  # zaten yüklü
    t = _threading.Thread(
        target=get_instance,
        kwargs={"log_cb": log_cb},
        daemon=True,
        name="NameDB-Preload",
    )
    t.start()
