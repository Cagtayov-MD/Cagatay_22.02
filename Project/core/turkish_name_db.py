"""
turkish_name_db.py — Türkçe isim veritabanı + OCR karakter onarımı.

Kaynak: F:\\Source\\name_db\\compiled\\names.db
        → 356.543 kayıt (166k ad + 190k soyad)
        → Türkçe canonical öncelikli (canonical_tr + soyisimler_tr: 12.726 kayıt)

Görevler:
1. names.db SQLite'dan isimleri yükle (RAM dict — 10-30 isim/video, hız yeterli)
2. OCR ASCII bozulmalarını Türkçe'ye çevir (SEBNEM → Şebnem)
3. Birleşik isim böl (SEBNEMSONMEZ → Şebnem Sönmez)
4. is_name() ile doğrula
5. is_also_surname / is_also_firstname flag'leri → CreditsParser swap tespiti

DB Şeması (names.db):
    CREATE TABLE names (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        normalized TEXT NOT NULL,   -- ASCII key (SEBNEM gibi)
        type TEXT NOT NULL,         -- 'first_male' | 'first_female' | 'surname' | 'international'
        is_also_surname INTEGER DEFAULT 0,   -- Ad ama soyad da olabilir (Ali Kemal → Ali, Kemal)
        is_also_firstname INTEGER DEFAULT 0, -- Soyad ama ad da olabilir
        source TEXT,                -- 'canonical_tr' | 'soyisimler_tr' | 'tally' | ...
        score REAL DEFAULT 1.0
    );
    CREATE INDEX idx_normalized ON names(normalized);
    CREATE INDEX idx_type ON names(type);

Kullanım:
    db = TurkishNameDB("F:/Source/name_db/compiled/names.db")
    canonical, score = db.find("SEBNEM")            # → ("Şebnem", 1.0)
    parts = db.split_concatenated("SEBNEMSONMEZ")   # → ["Şebnem", "Sönmez"]
    ok = db.is_name("Şebnem")                       # → True
    ok = db.is_surname("Sönmez")                    # → True
    swap = db.check_swap_risk("Ali", "Kemal")       # → True (ikisi de ad olabilir)
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Optional

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


# ─────────────────────────────────────────────────────────────────────────────
# Sabit OCR hata düzeltme tablosu
# Büyük öncelik: hardcoded > DB > fuzzy
# Özellikle OCR'ın sistematik hataları buraya girer (V→U, M→N vb.)
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
    # Sık görülen tekli bozulmalar
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
}


def _normalize_key(text: str) -> str:
    """
    Karşılaştırma anahtarı üret: büyük harf, boşluksuz, Türkçe→ASCII.
    Hem DB normalized kolonu hem de OCR girdi bu fonksiyondan geçer.
    Örnek: 'Şebnem Sönmez' → 'SEBNEMSONMEZ'
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


def _normalize_spaced(text: str) -> str:
    """Boşlukları koruyarak normalize et. Çok kelimeli isimler için."""
    parts = text.strip().split()
    return ' '.join(_normalize_key(p) for p in parts)


class NameEntry:
    """DB'den gelen tek isim kaydı."""
    __slots__ = ('name', 'normalized', 'type', 'is_also_surname',
                 'is_also_firstname', 'source', 'score')

    def __init__(self, name, normalized, type_, is_also_surname,
                 is_also_firstname, source, score):
        self.name = name
        self.normalized = normalized
        self.type = type_
        self.is_also_surname = bool(is_also_surname)
        self.is_also_firstname = bool(is_also_firstname)
        self.source = source or ''
        self.score = float(score or 1.0)

    @property
    def is_first_name(self) -> bool:
        return self.type in ('first_male', 'first_female', 'first_neutral')

    @property
    def is_surname(self) -> bool:
        return self.type == 'surname'

    @property
    def gender(self) -> str:
        if self.type == 'first_male':
            return 'E'
        if self.type == 'first_female':
            return 'K'
        return '?'


class TurkishNameDB:
    """
    Türkçe isim veritabanı — names.db SQLite tabanlı.

    RAM dict mimarisi:
      - _db_first  : {normalized_key: NameEntry}  — adlar
      - _db_surname: {normalized_key: NameEntry}  — soyadlar
      - _all_keys  : set — hızlı is_name() için

    Session log kararı: RAM dict, çünkü video başına 10-30 isim işleniyor,
    SQLite'a tekrar tekrar bağlanmak overhead yaratır.
    """

    def __init__(self, db_path: str = "", log_cb=None, sql_path: str = ""):
        self._log = log_cb or (lambda m: print(m))
        self._db_first: dict[str, NameEntry] = {}
        self._db_surname: dict[str, NameEntry] = {}
        # Fuzzy için canonical isim listeleri
        self._first_names: list[str] = []
        self._surnames: list[str] = []

        # Hardcoded düzeltmeleri yükle (ham string → canonical)
        self._hardcoded: dict[str, str] = {
            _normalize_key(k): v for k, v in _HARDCODED_FIXES.items()
        }


        # Backward-compatible alias: some callers use sql_path instead of db_path
        if (not db_path) and sql_path:
            db_path = sql_path

        # Enforce file format contract: runtime expects a SQLite .db.
        # If a .sql seed is provided, auto-materialize a sibling .db once.
        db_path = (db_path or "").strip()
        if db_path.lower().endswith(".sql"):
            sql_seed = Path(db_path)
            db_candidate = sql_seed.with_suffix(".db")
            if not db_candidate.is_file():
                try:
                    self._log(f"  [NameDB] SQL seed bulundu, DB üretiliyor: {db_candidate}")
                    self._materialize_sqlite_from_sql(sql_seed, db_candidate)
                except Exception as e:
                    self._log(f"  [NameDB] SQL→DB dönüşümü başarısız: {e}")
            db_path = str(db_candidate)

        if db_path and Path(db_path).is_file():
            n_first, n_sur = self._load_sqlite(db_path)
            self._log(
                f"  [NameDB] Yüklendi — ad:{n_first:,} soyad:{n_sur:,} "
                f"| {db_path}"
            )
        else:
            self._log(
                "  [NameDB] names.db bulunamadı — sadece hardcoded tablo aktif. "
                f"Beklenen: {db_path}"
            )

        self._first_names = [e.name for e in self._db_first.values()]
        self._surnames = [e.name for e in self._db_surname.values()]
        self._all_keys = set(self._db_first) | set(self._db_surname)


    def _materialize_sqlite_from_sql(self, sql_path: Path, db_path: Path) -> None:
        """Create a SQLite DB from a .sql seed (one-time)."""
        sql_text = sql_path.read_text(encoding="utf-8", errors="ignore")
        # Basic safety: create parent dir if needed
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(db_path))
        try:
            con.executescript(sql_text)
            con.commit()
        finally:
            con.close()

    # ─────────────────────────────────────────────────────────────────────
    # Yükleme
    # ─────────────────────────────────────────────────────────────────────

    def _load_sqlite(self, db_path: str) -> tuple[int, int]:
        """names.db'den tüm kayıtları RAM dict'e yükle."""
        n_first = n_sur = 0
        try:
            con = sqlite3.connect(db_path)
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute("""
                SELECT name, normalized, type, is_also_surname,
                       is_also_firstname, source, score
                FROM names
                ORDER BY score DESC
            """)
            for row in cur.fetchall():
                entry = NameEntry(
                    row['name'], row['normalized'], row['type'],
                    row['is_also_surname'], row['is_also_firstname'],
                    row['source'], row['score']
                )
                key = row['normalized'] or _normalize_key(row['name'])
                if entry.is_first_name:
                    # Yüksek score öncelikli (score DESC ile geldi)
                    if key not in self._db_first:
                        self._db_first[key] = entry
                    n_first += 1
                elif entry.is_surname:
                    if key not in self._db_surname:
                        self._db_surname[key] = entry
                    n_sur += 1
            con.close()
        except Exception as e:
            self._log(f"  [NameDB] SQLite yükleme hatası: {e}")
        return n_first, n_sur

    # ─────────────────────────────────────────────────────────────────────
    # Temel sorgular
    # ─────────────────────────────────────────────────────────────────────

    def is_name(self, text: str) -> bool:
        """Bu metin bir Türkçe isim ya da soyad mı?"""
        key = _normalize_key(text.replace(' ', ''))
        return key in self._all_keys or key in self._hardcoded

    def is_first_name(self, text: str) -> bool:
        key = _normalize_key(text)
        return key in self._db_first

    def is_surname(self, text: str) -> bool:
        key = _normalize_key(text)
        return key in self._db_surname

    def gender(self, text: str) -> str:
        """'E' | 'K' | '?' — ilk kelimeye göre."""
        for word in text.split():
            key = _normalize_key(word)
            if key in self._db_first:
                return self._db_first[key].gender
        return '?'

    # ─────────────────────────────────────────────────────────────────────
    # Swap tespiti (CreditsParser entegrasyonu)
    # ─────────────────────────────────────────────────────────────────────

    def check_swap_risk(self, left_token: str, right_token: str) -> bool:
        """
        Sütun yer değişimi riski var mı?
        Sol sütun token'ı is_also_firstname=1 olan bir soyad ise
        → oyuncu/karakter sütunları karışmış olabilir.

        Örnek:
            Sol sütun: "Kemal"  → hem ad hem soyad olabilir → swap riski
            Sağ sütun: "Ali"    → aynı durum → swap riski
        """
        lk = _normalize_key(left_token)
        rk = _normalize_key(right_token)

        # Sol taraf soyad DB'sinde is_also_firstname=1 ise risk var
        left_entry = self._db_surname.get(lk)
        if left_entry and left_entry.is_also_firstname:
            return True

        # Sağ taraf ad DB'sinde is_also_surname=1 ise risk var
        right_entry = self._db_first.get(rk)
        if right_entry and right_entry.is_also_surname:
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────
    # Birleşik isim bölme
    # ─────────────────────────────────────────────────────────────────────

    def split_concatenated(
        self, text: str, max_parts: int = 3
    ) -> list[str]:
        """
        Birleşik OCR token'ı böl.
        SEBNEMSONMEZ → ["Şebnem", "Sönmez"]
        GOLDENCUNEY  → (hardcoded) ["Gülden", "Güney"]

        Algoritma:
        1. Hardcoded tam eşleşme dene
        2. Tüm bölme noktalarını dene (DP, DB'de eşleşeni seç)
        3. Bulunamazsa orijinali tek parça döndür

        Returns:
            List[str] — canonical parçalar, boş token'lar dahil edilmez
        """
        if not text or not text.strip():
            return [text]

        key = _normalize_key(text)

        # 1. Hardcoded tam eşleşme
        if key in self._hardcoded:
            canonical = self._hardcoded[key]
            return canonical.split()

        # 2. DB'de tek token olarak bulunuyor mu?
        entry = self._db_first.get(key) or self._db_surname.get(key)
        if entry:
            return [entry.name]

        # 3. DP ile bölme: tüm split noktalarını dene
        best = self._dp_split(key, max_parts)
        if best:
            return best

        # 4. Bulunamadı — orijinali döndür
        return [text.strip()]

    def _dp_split(self, normalized_key: str, max_parts: int) -> list[str]:
        """
        Dynamic programming ile birleşik kelimeyi bölme.
        normalized_key: boşluksuz büyük harf ASCII (SEBNEMSONMEZ gibi)

        Her bölme noktasında sol parça DB'de varsa devam eder.
        En fazla max_parts parçaya böler.
        """
        n = len(normalized_key)
        if n < 3:
            return []

        # dp[i] = (canonical_parts, split_score) — i pozisyonuna kadar en iyi bölme
        dp: list[Optional[tuple[list[str], float]]] = [None] * (n + 1)
        dp[0] = ([], 1.0)

        for i in range(1, n + 1):
            for j in range(max(0, i - 15), i):  # max 15 karakter per token
                if dp[j] is None:
                    continue
                prev_parts, prev_score = dp[j]
                if len(prev_parts) >= max_parts:
                    continue
                segment = normalized_key[j:i]
                if len(segment) < 2:
                    continue
                entry = self._db_first.get(segment) or self._db_surname.get(segment)
                if entry:
                    new_parts = prev_parts + [entry.name]
                    # Kaynak önceliği: canonical_tr daha yüksek
                    seg_score = entry.score * (1.2 if entry.source == 'canonical_tr' else 1.0)
                    new_score = prev_score * seg_score
                    if dp[i] is None or new_score > dp[i][1]:
                        dp[i] = (new_parts, new_score)

        result = dp[n]
        # En az 2 parça gerektir (tek parça zaten yukarıda yakalandı)
        if result and len(result[0]) >= 2:
            return result[0]
        return []

    # ─────────────────────────────────────────────────────────────────────
    # OCR satır onarımı
    # ─────────────────────────────────────────────────────────────────────

    def find(
        self, ocr_text: str, fuzzy_threshold: int = 85
    ) -> tuple[Optional[str], float]:
        """
        OCR metninden kanonik Türkçe isim bul.

        Öncelik sırası:
          1. Hardcoded tam eşleşme
          2. DB tam eşleşme (normalized key)
          3. Kelime kelime düzeltme (çok kelimeli)
          4. Fuzzy match (RapidFuzz)

        Returns:
            (canonical_name, score 0.0-1.0)
        """
        if not ocr_text or not ocr_text.strip():
            return None, 0.0

        key = _normalize_key(ocr_text.replace(' ', ''))

        # 1. Hardcoded
        if key in self._hardcoded:
            return self._hardcoded[key], 1.0

        # 2. DB tam eşleşme
        entry = self._db_first.get(key) or self._db_surname.get(key)
        if entry:
            return entry.name, 1.0

        # 3. Çok kelimeli: kelime kelime düzelt
        parts = ocr_text.strip().split()
        if len(parts) > 1:
            fixed, changed = self._fix_parts(parts, fuzzy_threshold)
            if changed:
                return ' '.join(fixed), 0.9

        # 4. Fuzzy
        return self._fuzzy_find(ocr_text, fuzzy_threshold)

    def _fix_parts(
        self, parts: list[str], threshold: int
    ) -> tuple[list[str], bool]:
        """Her kelimeyi ayrı ayrı DB'den düzelt."""
        fixed = []
        changed = False
        for part in parts:
            pk = _normalize_key(part)
            # Hardcoded
            if pk in self._hardcoded:
                fixed.append(self._hardcoded[pk])
                changed = True
                continue
            # DB
            entry = self._db_first.get(pk) or self._db_surname.get(pk)
            if entry and entry.name != part:
                fixed.append(entry.name)
                changed = True
            else:
                # Fuzzy
                candidate, score = self._fuzzy_find(part, threshold)
                if candidate and score >= threshold / 100.0:
                    fixed.append(candidate)
                    changed = True
                else:
                    fixed.append(part)
        return fixed, changed

    def _fuzzy_find(
        self, text: str, threshold: int
    ) -> tuple[Optional[str], float]:
        """RapidFuzz ile en yakın canonical ismi bul."""
        if not HAS_RAPIDFUZZ:
            return None, 0.0
        all_names = self._first_names + self._surnames
        if not all_names:
            return None, 0.0
        result = rf_process.extractOne(
            text, all_names,
            scorer=fuzz.WRatio,
            score_cutoff=threshold
        )
        if result:
            return result[0], round(result[1] / 100.0, 3)
        return None, 0.0

    def correct_line(self, line: str) -> str:
        """
        Tek OCR satırını düzelt.
        Önce birleşik isim mi diye bak, sonra find() ile düzelt.
        score >= 0.85 ise düzeltilmiş, altında orijinal.
        """
        stripped = line.strip()
        if not stripped:
            return line

        # Boşluk yoksa → birleşik isim adayı
        if ' ' not in stripped and len(stripped) > 8:
            parts = self.split_concatenated(stripped)
            if len(parts) > 1:
                return ' '.join(parts)

        canonical, score = self.find(stripped)
        if canonical and score >= 0.85:
            return canonical
        return line

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline entegrasyon yardımcıları
    # ─────────────────────────────────────────────────────────────────────

    def repair_ocr_lines(self, ocr_lines: list) -> tuple[list, int]:
        """
        pipeline_runner._repair_turkish() için drop-in replacement.
        ocr_lines: list of dict veya OCRLine nesneleri
        Returns: (repaired_lines, repair_count)
        """
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
        """
        pipeline_runner._repair_layout_pairs() için drop-in replacement.
        """
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

    def __len__(self) -> int:
        return len(self._db_first) + len(self._db_surname)

    def __repr__(self) -> str:
        return (
            f"TurkishNameDB("
            f"first={len(self._db_first):,}, "
            f"surname={len(self._db_surname):,}, "
            f"hardcoded={len(self._hardcoded)})"
        )
