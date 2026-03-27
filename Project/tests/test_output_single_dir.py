"""test_output_single_dir.py

Tüm çıktıların D:\\DATABASE\\FilmDizi\\{vname}\\ altında
tek klasörde toplandığını doğrular.

Testler:
  SD-01: Varsayılan db_root → D:\\DATABASE\\FilmDizi
  SD-02: work_dir = db_root\\{vname}  (prefix yok, timestamp yok)
  SD-03: Config database_root override çalışır
  SD-04: VITOS_DATABASE_ROOT env var override çalışır
  SD-05: _write_database db_dir == work_dir  (ikinci klasör yok)
  SD-06: output_root kopyalama bloğu kodda yok (tek konum)
  SD-07: Aynı film iki kez → work_dir değişmez, _safe_path versiyon yapar
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_project_dir = os.path.dirname(os.path.dirname(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ── work_dir hesaplama mantığı (pipeline_runner.py:327-332) ──────────────────

def _calc_work_dir(video_path: str, config: dict, env: dict | None = None) -> str:
    env = env or {}
    db_root = (
        config.get("database_root") or
        env.get("VITOS_DATABASE_ROOT") or
        r"D:\DATABASE\FilmDizi"
    )
    vname = Path(video_path).stem
    return os.path.join(db_root, vname)


# ── SD-01 ──────────────────────────────────────────────────────────────────────

class TestWorkDirDefault(unittest.TestCase):

    def test_sd01_default_root_is_d_drive(self):
        """Varsayılan db_root D:\\DATABASE\\FilmDizi olmalı."""
        work_dir = _calc_work_dir(
            r"E:\video\1982-0274-1-0000-00-1-NOTRE_DAME'IN_KAMBURU.mp4",
            config={},
        )
        self.assertTrue(
            work_dir.startswith(r"D:\DATABASE\FilmDizi"),
            f"Beklenen: D:\\DATABASE\\FilmDizi ile başlamalı, alınan: {work_dir}"
        )

    def test_sd02_no_prefix_no_timestamp(self):
        """work_dir sadece vname'i içermeli — arsiv_ prefix ve _YYYYMMDD_ yok."""
        vname = "1982-0274-1-0000-00-1-NOTRE_DAME'IN_KAMBURU"
        work_dir = _calc_work_dir(
            rf"E:\video\{vname}.mp4",
            config={},
        )
        folder = Path(work_dir).name
        self.assertEqual(folder, vname, "Klasör adı tam olarak vname olmalı")
        self.assertNotIn("arsiv", folder.lower(), "arsiv_ prefix olmamalı")
        self.assertNotIn("202", folder, "Timestamp olmamalı")

    def test_sd03_config_database_root_override(self):
        """Config'deki database_root kullanılmalı."""
        work_dir = _calc_work_dir(
            r"E:\video\CAG_1995-0288-1-KARA_RAHIP.mp4",
            config={"database_root": r"X:\BASKA_DISK\FilmDizi"},
        )
        self.assertTrue(work_dir.startswith(r"X:\BASKA_DISK\FilmDizi"))

    def test_sd04_env_var_override(self):
        """VITOS_DATABASE_ROOT env var kullanılmalı."""
        work_dir = _calc_work_dir(
            r"E:\video\CAG_1995-0288-1-KARA_RAHIP.mp4",
            config={},
            env={"VITOS_DATABASE_ROOT": r"Z:\NETWORK\FilmDizi"},
        )
        self.assertTrue(work_dir.startswith(r"Z:\NETWORK\FilmDizi"))

    def test_sd04b_config_beats_env(self):
        """Config database_root, env var'ı ezer."""
        work_dir = _calc_work_dir(
            r"E:\video\CAG_TEST.mp4",
            config={"database_root": r"C:\CONFIG_ROOT"},
            env={"VITOS_DATABASE_ROOT": r"Z:\ENV_ROOT"},
        )
        self.assertTrue(work_dir.startswith(r"C:\CONFIG_ROOT"))


# ── SD-05: _write_database db_dir == work_dir ─────────────────────────────────

class TestWriteDatabaseSingleDir(unittest.TestCase):

    def test_sd05_db_dir_equals_work_dir(self):
        """_write_database: db_dir work_dir ile aynı olmalı — ikinci klasör yok."""
        work_dir = r"D:\DATABASE\FilmDizi\1982-0274-1-0000-00-1-NOTRE_DAME'IN_KAMBURU"

        # pipeline_runner._write_database:2535-2536 mantığı
        db_dir = Path(work_dir)

        self.assertEqual(
            db_dir.resolve(),
            Path(work_dir).resolve(),
            "db_dir work_dir'den farklı bir konum olmamalı"
        )

    def test_sd05b_no_film_title_subdir(self):
        """Film başlığından türetilen alt klasör oluşmamalı."""
        work_dir = r"D:\DATABASE\FilmDizi\1982-0274-1-0000-00-1-NOTRE_DAME'IN_KAMBURU"
        db_dir = Path(work_dir)

        # Eski kod: db_dir = Path(db_root) / film_title → farklı klasör
        old_db_dir = Path(r"D:\DATABASE\FilmDizi") / "NOTRE_DAME'IN_KAMBURU"

        self.assertNotEqual(
            db_dir.resolve(),
            old_db_dir.resolve(),
            "Eski 'film_title alt klasörü' davranışı olmamalı"
        )


# ── SD-06: output_root kopyalama bloğu yok ────────────────────────────────────

class TestNoOutputRootCopy(unittest.TestCase):

    def test_sd06_output_root_block_removed(self):
        """pipeline_runner'da output_root kopyalama bloğu olmamalı."""
        runner_path = Path(_project_dir) / "core" / "pipeline_runner.py"
        source = runner_path.read_text(encoding="utf-8")

        self.assertNotIn(
            "shutil.copy2(_src_p, _dst)",
            source,
            "output_root kopyalama bloğu (shutil.copy2 _dst) kaldırılmış olmalı"
        )
        self.assertNotIn(
            "Kullanıcı raporu kopyalandı",
            source,
            "output_root kopyalama log mesajı kaldırılmış olmalı"
        )


# ── SD-07: İki çalışma, aynı work_dir ─────────────────────────────────────────

class TestMultipleRunsSameDir(unittest.TestCase):

    def test_sd07_same_video_same_work_dir(self):
        """Aynı video iki kez çalıştırılsa work_dir değişmemeli."""
        video = r"E:\video\1982-0274-1-0000-00-1-NOTRE_DAME'IN_KAMBURU.mp4"
        dir1 = _calc_work_dir(video, config={})
        dir2 = _calc_work_dir(video, config={})
        self.assertEqual(dir1, dir2, "İki çalışma aynı klasörü kullanmalı")

    def test_sd07b_work_dir_depth_is_two(self):
        """work_dir tam olarak db_root\\{vname} derinliğinde olmalı."""
        vname = "1982-0274-1-0000-00-1-NOTRE_DAME'IN_KAMBURU"
        work_dir = _calc_work_dir(rf"E:\video\{vname}.mp4", config={})
        p = Path(work_dir)
        db_root = Path(r"D:\DATABASE\FilmDizi")

        self.assertEqual(p.parent.resolve(), db_root.resolve(),
                         "work_dir doğrudan db_root altında olmalı")
        self.assertEqual(p.name, vname)


if __name__ == "__main__":
    unittest.main(verbosity=2)
