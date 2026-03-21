"""Gerçek IMDb DuckDB entegrasyon testi — Türkçe dizi arama.

RI-01: DB'de 'Türkan Hanımın Konağı' başlığı (primaryTitle veya akas) var mı?
RI-02: Bulunan tconst için crew.directors sütununda yönetmen nconst'ı var mı?
RI-03: IMDBLookup._strat_b başlık + yönetmen ile doğru diziyi buluyor mu?
"""

import os
import sys
import unittest

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

DB_PATH = os.environ.get("IMDB_DB_PATH", "").strip() or r"F:\IMDB\db\imdb.duckdb"
_DB_EXISTS = os.path.isfile(DB_PATH)

SERIES_TITLE   = "türkan hanımın konağı"   # orijinal Türkçe (OCR çıktısı)
KNOWN_TCONST   = "tt18273934"              # DB'den doğrulanmış
KNOWN_DIRECTOR = "Mehmet Atan"             # crew tablosundan doğrulanmış
LOG_LINES = []


def _log(msg):
    LOG_LINES.append(msg)
    print(msg)


@unittest.skipUnless(_DB_EXISTS, f"IMDb DB bulunamadı: {DB_PATH}")
class TestImdbRealSeries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import duckdb
        cls.con = duckdb.connect(DB_PATH, read_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()

    # ------------------------------------------------------------------
    # RI-01: Başlık var mı?
    # ------------------------------------------------------------------
    def test_RI01_title_exists(self):
        """İlk kelime LIKE + unidecode fuzzy score ile dizi aranıyor."""
        from core.imdb_lookup import _norm_ascii
        first_word = SERIES_TITLE.split()[0]   # "türkan"
        like_first = f"%{first_word}%"
        print(f"\n[RI-01] İlk kelime LIKE: '{like_first}'")
        db_title = "Türkan Hanim'in Konagi"
        print(f"[RI-01] ASCII ocr: '{_norm_ascii(SERIES_TITLE)}' == db: '{_norm_ascii(db_title)}'")
        like_ascii = like_first  # test için alias

        rows = self.con.execute("""
            SELECT t.tconst, t.primaryTitle, t.originalTitle, t.titleType, t.startYear
            FROM titles t
            WHERE lower(t.primaryTitle) LIKE ?
               OR lower(t.originalTitle) LIKE ?
            LIMIT 20
        """, [like_ascii, like_ascii]).fetchall()

        aka_rows = self.con.execute("""
            SELECT DISTINCT t.tconst, t.primaryTitle, t.originalTitle, t.titleType,
                            t.startYear, a.title AS aka_title, a.region
            FROM titles t
            JOIN akas a ON a.tconst = t.tconst
            WHERE lower(a.title) LIKE ?
            LIMIT 20
        """, [like_ascii]).fetchall()

        print(f"[RI-01] primaryTitle eşleşmeleri ({len(rows)}):")
        for r in rows:
            print(f"  tconst={r[0]}  type={r[3]}  year={r[4]}  title='{r[1]}'")
        print(f"[RI-01] akas eşleşmeleri ({len(aka_rows)}):")
        for r in aka_rows:
            print(f"  tconst={r[0]}  type={r[3]}  year={r[4]}  primaryTitle='{r[1]}'  aka='{r[5]}'  region={r[6]}")

        all_found = rows + aka_rows
        self.assertTrue(len(all_found) > 0, f"'{like_ascii}' pattern ile IMDb'de sonuç yok")

        found_tconsts = {r[0] for r in all_found}
        self.assertIn(KNOWN_TCONST, found_tconsts, f"{KNOWN_TCONST} sonuçlar arasında yok")
        TestImdbRealSeries._found_tconst = KNOWN_TCONST
        TestImdbRealSeries._found_rows = all_found
        print(f"\n[RI-01] OK {KNOWN_TCONST} bulundu")

    # ------------------------------------------------------------------
    # RI-02: crew.directors'da yönetmen var mı?
    # ------------------------------------------------------------------
    def test_RI02_director_in_crew(self):
        """Bulunan tconst için crew tablosunda directors nconst listesi çözümleniyor."""
        if not hasattr(TestImdbRealSeries, "_found_rows") or not TestImdbRealSeries._found_rows:
            self.skipTest("RI-01 eşleşme bulamadı")

        tconst = TestImdbRealSeries._found_tconst

        row = self.con.execute(
            "SELECT directors, writers FROM crew WHERE tconst = ?", [tconst]
        ).fetchone()

        print(f"\n[RI-02] crew tablosu ({tconst}): {row}")

        if not row or not row[0]:
            # principals tablosunda director var mı?
            prin_dirs = self.con.execute("""
                SELECT n.primaryName, p.category, p.job
                FROM principals p
                JOIN names n ON n.nconst = p.nconst
                WHERE p.tconst = ? AND (p.category = 'director' OR lower(p.job) LIKE '%director%')
                LIMIT 10
            """, [tconst]).fetchall()
            print(f"[RI-02] principals'daki yönetmenler ({len(prin_dirs)}):")
            for r in prin_dirs:
                print(f"  name='{r[0]}'  category={r[1]}  job={r[2]}")
            self.fail(
                f"crew.directors boş ({tconst}) — yönetmen sadece principals'da. "
                f"_fetch_directors fallback gerekiyor."
            )

        nconsts = [n.strip() for n in str(row[0]).split(",") if n.strip()]
        directors = []
        for nc in nconsts:
            nr = self.con.execute(
                "SELECT primaryName FROM names WHERE nconst = ?", [nc]
            ).fetchone()
            if nr and nr[0]:
                directors.append(nr[0])

        print(f"[RI-02] Yönetmenler: {directors}")
        self.assertTrue(len(directors) > 0, "crew.directors nconst çözümlenemedi")
        TestImdbRealSeries._directors = directors

    # ------------------------------------------------------------------
    # RI-03: IMDBLookup._strat_b ile eşleşme
    # ------------------------------------------------------------------
    def test_RI03_strat_b_match(self):
        """IMDBLookup.lookup() başlık + yönetmen ile diziyi buluyor mu?"""
        from core.imdb_lookup import IMDBLookup

        # DB'den doğrulanmış yönetmen adını kullan
        director_name = KNOWN_DIRECTOR
        print(f"\n[RI-03] Test: title='{SERIES_TITLE}', director='{director_name}'")

        lookup = IMDBLookup(log_cb=_log)
        result = lookup.lookup({
            "film_title": "Türkan Hanımın Konağı",
            "directors": [{"name": director_name}],
            "cast": [],
            "crew": [],
        })

        print(f"[RI-03] matched={result.matched}  via={result.matched_via}")
        print(f"[RI-03] title='{result.title}'  tconst={result.tconst}  year={result.year}")
        print(f"[RI-03] directors={[d['name'] for d in result.directors]}")
        print(f"[RI-03] cast count={len(result.cast)}")

        self.assertTrue(result.matched, f"Eşleşme başarısız: {result.reason}")
        self.assertIn("strat_b", result.matched_via)
        self.assertEqual(result.tconst, TestImdbRealSeries._found_tconst)


if __name__ == "__main__":
    unittest.main(verbosity=2)
