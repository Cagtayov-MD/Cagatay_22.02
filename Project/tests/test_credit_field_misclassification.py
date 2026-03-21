"""test_credit_field_misclassification.py

Kredi Alanı Yanlış Sınıflandırma testleri.

Senaryo: OCR bir kişiyi YÖNETMEN alanına atar; TMDB ise aynı kişinin
'writing' departmanında olduğunu söyler. Pipeline bu uyuşmazlığı tespit
etmeli, TMDB'yi otoriter kabul etmeli ve job'ı düzeltmeli.

Testler:
  CF-01: Cat-1 (YÖNETMEN) — TMDB 'writing' diyor → uyuşmazlık, job düzeltilir
  CF-02: Cat-2 (SENARYO) — TMDB 'directing' diyor → uyuşmazlık, job düzeltilir
  CF-03: Eşleşme yok (TMDB'de yoksa) → veri dokunulmaz
  CF-04: Dept eşleşiyor → uyuşmazlık yok, flag yok
  CF-05: raw="imdb" girişler etkilenmez (yalnızca ocr_verified kontrol edilir)
  CF-06: _tmdb_dept_hints mekanizması — _enrich_cdata_with_tmdb'nin bıraktığı hint kullanılır
  CF-07: Hiçbir veri silinmez (isim + raw + _ocr_job korunur)
  CF-08: Karışık crew: bazı eşleşir, bazı uyuşmaz
"""

import os
import re
import sys
import unicodedata
import unittest

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ── _check_credit_field_misclassification metodunu inline çoğalt ─────────────
# pipeline_runner.py:_check_credit_field_misclassification birebir mantık.

_JOB_TR = {
    "Director": "Yönetmen",
    "Producer": "Yapımcı",
    "Executive Producer": "Baş Yapımcı",
    "Writer": "Yazar",
    "Screenplay": "Senaryo",
    "Editor": "Kurgu",
    "Director of Photography": "Görüntü Yönetmeni",
    "Cinematography": "Görüntü Yönetmeni",
}

_OCR_TO_DEPT = {
    "YONETMEN": "directing",    "YÖNETMEN": "directing",
    "YAPIMCI": "production",
    "OYUNCU": "acting",
    "SENARYO": "writing",
    "GÖRÜNTÜ YÖNETMENİ": "camera", "GORUNTU YONETMENI": "camera",
    "KAMERA": "camera",
    "KURGU": "editing",
    "YONETMEN YARDIMCISI": "directing", "YÖNETMEN YARDIMCISI": "directing",
}

_CAT1 = {"YONETMEN", "YÖNETMEN", "YAPIMCI", "OYUNCU"}


def _norm(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_ = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_.lower())


def _check_credit_field_misclassification(cdata: dict, log_lines: list = None) -> None:
    """pipeline_runner.py _check_credit_field_misclassification birebir."""
    def _log(msg):
        if log_lines is not None:
            log_lines.append(msg)

    crew = cdata.get("crew") or []

    dept_lookup: dict = {}
    for entry in crew:
        if entry.get("raw") == "tmdb" and entry.get("name") and entry.get("department"):
            key = _norm(entry["name"])
            if key not in dept_lookup:
                dept_lookup[key] = {
                    "department": (entry.get("department") or "").lower(),
                    "job": entry.get("job") or "",
                }

    for name_key, hint in (cdata.pop("_tmdb_dept_hints", None) or {}).items():
        key = _norm(name_key)
        if key not in dept_lookup:
            dept_lookup[key] = hint

    if not dept_lookup:
        return

    mismatches = []
    for entry in crew:
        if entry.get("raw") != "ocr_verified":
            continue
        ocr_job = entry.get("job") or ""
        expected_dept = _OCR_TO_DEPT.get(ocr_job)
        if not expected_dept:
            continue

        person_key = _norm(entry.get("name") or "")
        if not person_key:
            continue

        hint = dept_lookup.get(person_key)
        if not hint:
            continue

        tmdb_dept = (hint.get("department") or "").lower()
        tmdb_job = hint.get("job") or ""

        if tmdb_dept == expected_dept:
            continue

        cat = "1" if ocr_job in _CAT1 else "2"
        entry["_field_mismatch"] = True
        entry["_ocr_job"] = ocr_job
        entry["_tmdb_department"] = tmdb_dept
        entry["_tmdb_job"] = tmdb_job

        if tmdb_job:
            entry["job"] = tmdb_job
            entry["role"] = tmdb_job
            entry["role_tr"] = _JOB_TR.get(tmdb_job, tmdb_job)

        mismatches.append({
            "name": entry.get("name"),
            "ocr_job": ocr_job,
            "tmdb_dept": tmdb_dept,
            "tmdb_job": tmdb_job,
            "category": cat,
        })
        _log(
            f"  [KrediAlan] Uyuşmazlık (Cat-{cat}): "
            f"'{entry.get('name')}' OCR={ocr_job} TMDB={tmdb_dept}/{tmdb_job}"
        )

    if mismatches:
        cdata["_credit_field_mismatches"] = mismatches


# ── CF-01 ─────────────────────────────────────────────────────────────────────

class TestCat1YonetmenMismatch(unittest.TestCase):

    def test_cf01_yonetmen_but_tmdb_writing(self):
        """CF-01: OCR=YÖNETMEN, TMDB=writing → Cat-1 uyuşmazlık, job düzeltilir."""
        cdata = {
            "crew": [
                # OCR, bu kişiyi yönetmen olarak atadı
                {"name": "Mehmet Yazar", "job": "YÖNETMEN", "raw": "ocr_verified"},
                # TMDB bu kişiyi 'writing' departmanında biliyor
                {"name": "Mehmet Yazar", "job": "Writer", "department": "writing", "raw": "tmdb"},
            ]
        }
        logs = []
        _check_credit_field_misclassification(cdata, logs)

        ocr_entry = cdata["crew"][0]
        self.assertTrue(ocr_entry.get("_field_mismatch"), "Uyuşmazlık bayrağı set edilmeli")
        self.assertEqual(ocr_entry.get("_ocr_job"), "YÖNETMEN")
        self.assertEqual(ocr_entry.get("_tmdb_department"), "writing")
        self.assertEqual(ocr_entry.get("job"), "Writer", "job TMDB değeriyle güncellenmeli")
        self.assertIn("_credit_field_mismatches", cdata)
        self.assertEqual(cdata["_credit_field_mismatches"][0]["category"], "1")
        self.assertTrue(any("Cat-1" in l for l in logs))


# ── CF-02 ─────────────────────────────────────────────────────────────────────

class TestCat2Senaryo(unittest.TestCase):

    def test_cf02_senaryo_but_tmdb_directing(self):
        """CF-02: OCR=SENARYO, TMDB=directing → Cat-2 uyuşmazlık, job düzeltilir."""
        cdata = {
            "crew": [
                {"name": "Ali Yönetmen", "job": "SENARYO", "raw": "ocr_verified"},
                {"name": "Ali Yönetmen", "job": "Director", "department": "directing", "raw": "tmdb"},
            ]
        }
        _check_credit_field_misclassification(cdata)

        entry = cdata["crew"][0]
        self.assertTrue(entry.get("_field_mismatch"))
        self.assertEqual(entry.get("_ocr_job"), "SENARYO")
        self.assertEqual(entry.get("job"), "Director")
        self.assertEqual(cdata["_credit_field_mismatches"][0]["category"], "2")


# ── CF-03 ─────────────────────────────────────────────────────────────────────

class TestNoTmdbEntry(unittest.TestCase):

    def test_cf03_person_not_in_tmdb_untouched(self):
        """CF-03: Kişi TMDB'de yoksa veri dokunulmaz."""
        cdata = {
            "crew": [
                {"name": "Bilinmeyen Kişi", "job": "YÖNETMEN", "raw": "ocr_verified"},
                # Başka biri TMDB'de var ama bu kişi yok
                {"name": "Başkası", "job": "Director", "department": "directing", "raw": "tmdb"},
            ]
        }
        _check_credit_field_misclassification(cdata)

        entry = cdata["crew"][0]
        self.assertFalse(entry.get("_field_mismatch", False), "TMDB verisi yokken flag olmamalı")
        self.assertEqual(entry.get("job"), "YÖNETMEN", "job değişmemeli")
        self.assertNotIn("_credit_field_mismatches", cdata)


# ── CF-04 ─────────────────────────────────────────────────────────────────────

class TestDeptMatch(unittest.TestCase):

    def test_cf04_dept_matches_no_flag(self):
        """CF-04: OCR=YÖNETMEN, TMDB=directing → eşleşiyor, flag yok."""
        cdata = {
            "crew": [
                {"name": "Burak Yönetmen", "job": "YÖNETMEN", "raw": "ocr_verified"},
                {"name": "Burak Yönetmen", "job": "Director", "department": "directing", "raw": "tmdb"},
            ]
        }
        _check_credit_field_misclassification(cdata)

        entry = cdata["crew"][0]
        self.assertFalse(entry.get("_field_mismatch", False))
        self.assertNotIn("_credit_field_mismatches", cdata)


# ── CF-05 ─────────────────────────────────────────────────────────────────────

class TestImdbEntriesUntouched(unittest.TestCase):

    def test_cf05_imdb_raw_entries_not_touched(self):
        """CF-05: raw='imdb' girişler kontrol dışı — dokunulmaz."""
        cdata = {
            "crew": [
                # IMDb kanonik giriş — hiç değiştirilmemeli
                {"name": "Burak Arliel", "job": "Director", "raw": "imdb"},
                # TMDB referans
                {"name": "Burak Arliel", "job": "Writer", "department": "writing", "raw": "tmdb"},
            ]
        }
        _check_credit_field_misclassification(cdata)

        imdb_entry = cdata["crew"][0]
        self.assertFalse(imdb_entry.get("_field_mismatch", False), "IMDb girişlere dokunulmamalı")
        self.assertEqual(imdb_entry.get("job"), "Director")
        self.assertNotIn("_credit_field_mismatches", cdata)


# ── CF-06 ─────────────────────────────────────────────────────────────────────

class TestTmdbDeptHints(unittest.TestCase):

    def test_cf06_dept_hints_used_when_no_tmdb_crew_entry(self):
        """CF-06: _tmdb_dept_hints kullanılır (TMDB crew entry olmasa bile)."""
        cdata = {
            "crew": [
                # OCR yönetmen atadı; TMDB bu kişiyi crew'a eklemedi (zaten vardı)
                {"name": "Özlem Senarist", "job": "YÖNETMEN", "raw": "ocr_verified"},
                # Başka bir TMDB girişi (Özlem'i değil)
                {"name": "Başkası", "job": "Producer", "department": "production", "raw": "tmdb"},
            ],
            # _enrich_cdata_with_tmdb'nin bıraktığı hint
            "_tmdb_dept_hints": {
                "özlem senarist": {"department": "writing", "job": "Screenplay"},
            },
        }
        _check_credit_field_misclassification(cdata)

        entry = cdata["crew"][0]
        self.assertTrue(entry.get("_field_mismatch"), "Hint'ten gelen uyuşmazlık tespit edilmeli")
        self.assertEqual(entry.get("_ocr_job"), "YÖNETMEN")
        self.assertEqual(entry.get("_tmdb_department"), "writing")
        self.assertEqual(entry.get("job"), "Screenplay")
        # Hint cdata'dan temizlenmeli
        self.assertNotIn("_tmdb_dept_hints", cdata)


# ── CF-07 ─────────────────────────────────────────────────────────────────────

class TestNoDataDeleted(unittest.TestCase):

    def test_cf07_name_and_raw_preserved(self):
        """CF-07: Hiçbir veri silinmez — isim, raw ve _ocr_job korunur."""
        cdata = {
            "crew": [
                {"name": "Ahmet Senarist", "job": "YAPIMCI", "raw": "ocr_verified", "confidence": 0.9},
                {"name": "Ahmet Senarist", "job": "Writer", "department": "writing", "raw": "tmdb"},
            ]
        }
        _check_credit_field_misclassification(cdata)

        entry = cdata["crew"][0]
        # Korunması gerekenler
        self.assertEqual(entry.get("name"), "Ahmet Senarist", "name korunmalı")
        self.assertEqual(entry.get("raw"), "ocr_verified", "raw korunmalı")
        self.assertEqual(entry.get("confidence"), 0.9, "confidence korunmalı")
        self.assertEqual(entry.get("_ocr_job"), "YAPIMCI", "_ocr_job orijinal job'ı saklamalı")
        # Güncellenmiş
        self.assertEqual(entry.get("job"), "Writer", "job TMDB değerine güncellenmeli")


# ── CF-08 ─────────────────────────────────────────────────────────────────────

class TestMixedCrew(unittest.TestCase):

    def test_cf08_mixed_matches_and_mismatches(self):
        """CF-08: Karışık crew — bazısı eşleşir, bazısı uyuşmaz."""
        cdata = {
            "crew": [
                # Doğru: SENARYO + TMDB writing → eşleşiyor
                {"name": "Doğru Senarist", "job": "SENARYO", "raw": "ocr_verified"},
                {"name": "Doğru Senarist", "job": "Screenplay", "department": "writing", "raw": "tmdb"},
                # Yanlış: KURGU + TMDB directing → uyuşmazlık
                {"name": "Yanlış Kurgucu", "job": "KURGU", "raw": "ocr_verified"},
                {"name": "Yanlış Kurgucu", "job": "Director", "department": "directing", "raw": "tmdb"},
                # TMDB'de yok → dokunulmaz
                {"name": "Bilinmeyen", "job": "YÖNETMEN", "raw": "ocr_verified"},
            ]
        }
        _check_credit_field_misclassification(cdata)

        correct_entry = cdata["crew"][0]
        wrong_entry = cdata["crew"][2]
        unknown_entry = cdata["crew"][4]

        self.assertFalse(correct_entry.get("_field_mismatch", False), "Eşleşen giriş flag almamalı")
        self.assertTrue(wrong_entry.get("_field_mismatch"), "Uyuşmazlık flag almalı")
        self.assertFalse(unknown_entry.get("_field_mismatch", False), "TMDB'si olmayan dokunulmaz")

        self.assertIn("_credit_field_mismatches", cdata)
        self.assertEqual(len(cdata["_credit_field_mismatches"]), 1)
        self.assertEqual(cdata["_credit_field_mismatches"][0]["name"], "Yanlış Kurgucu")
        self.assertEqual(cdata["_credit_field_mismatches"][0]["category"], "2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
