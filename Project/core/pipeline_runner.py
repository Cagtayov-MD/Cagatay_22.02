"""
pipeline_runner.py — DAG execution engine.

v2.0 Değişiklikler:
- TVDB tamamen kaldırıldı (dosya da silinmeli).
- mode (light/medium/heavy) → TextFilter.from_config() ile bağlandı.
- AudioBridge gerçekten entegre edildi — scope="video+audio" artık çalışıyor.
- google_json + logolar_dir parametreleri eklendi (path_resolver sabit yollar).
- BUG-01 (double diarization): transcribe.run() çağrısından hf_token kaldırıldı.
- TMDB: ID girmeden çalışır. film_adı + 1 oyuncu = %100 güven.
"""

import time
import os
from pathlib import Path
from datetime import datetime

from config.runtime_paths import get_tmdb_api_key, resolve_name_db_path

from core.frame_extractor import FrameExtractor
from core.text_filter import TextFilter
from core.ocr_engine import OCREngine
from core.credits_parser import CreditsParser
from core.export_engine import ExportEngine
from core.turkish_name_db import TurkishNameDB
from core.qwen_verifier import QwenVerifier
from utils.stats_logger import StatsLogger


class PipelineRunner:
    """Film/Dizi analiz pipeline."""

    def __init__(self, ffmpeg, ffprobe, tesseract="", config=None,
                 output_root="", google_json="", logolar_dir=""):
        self.config       = config or {}
        self.output_root  = output_root
        self.google_json  = google_json
        self.logolar_dir  = logolar_dir
        self.stats        = StatsLogger(
            os.path.join(output_root, "logs") if output_root else "logs")
        self.stage_stats  = {}
        self._log_cb      = None

        self._ffmpeg    = ffmpeg
        self._ffprobe   = ffprobe
        self._tesseract = tesseract
        self._extractor   = None
        self._text_filter = None
        self._ocr_engine  = None

        # TurkishNameDB
        name_db_path = (
            self.config.get("name_db_path") or
            os.environ.get("NAME_DB_PATH") or
            resolve_name_db_path()
        )
        self._name_db = TurkishNameDB(
            sql_path=name_db_path if os.path.isfile(name_db_path) else "",
        )

        # QwenVerifier
        qwen_enabled = bool(self.config.get("qwen_verify", True))
        qwen_model   = self.config.get("qwen_model", "qwen3-vl:8b")
        qwen_thresh  = float(self.config.get("qwen_threshold", 0.80))
        self._qwen = QwenVerifier(
            model=qwen_model,
            confidence_threshold=qwen_thresh,
            enabled=qwen_enabled,
        )

    def set_log_callback(self, cb):
        self._log_cb = cb
        self._name_db._log = cb

    def _log(self, msg):
        if self._log_cb:
            self._log_cb(msg)
        print(msg)

    # ──────────────────────────────────────────────────────────────
    def run(self, video_path, scope="video_only",
            first_min=4.0, last_min=8.0) -> dict:
        t0 = time.time()
        video_path = str(Path(video_path).resolve())
        vname = Path(video_path).stem
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(
            self.output_root or os.path.dirname(video_path),
            f"arsiv_{vname}_{ts}"
        )
        os.makedirs(work_dir, exist_ok=True)

        self.stats.start_job(video_path, "WORKSTATION", scope, "film_dizi")
        self.stage_stats = {}

        self._log(f"\n{'='*60}")
        self._log(f"  ARSIV DECODE — Pipeline v2.0")
        self._log(f"{'='*60}")
        self._log(f"  Video  : {Path(video_path).name}")
        self._log(f"  Scope  : {scope}")
        self._log(f"  Giriş  : {first_min} dk | Çıkış: {last_min} dk")
        self._log(f"  Mod    : {self.config.get('difficulty','medium')}")
        self._log(f"  NameDB : {len(self._name_db)} isim yüklü")

        try:
            # ── TextFilter mode bağlantısı ──────────────────────
            self._text_filter = TextFilter.from_config(self.config)

            self._extractor = FrameExtractor(
                self._ffmpeg, self._ffprobe, log_cb=self._log)

            self._log(f"\n  PaddleOCR başlatılıyor...")
            self._ocr_engine = OCREngine(
                use_gpu=self.config.get("use_gpu", True),
                lang=self.config.get("ocr_languages", ["en"])[0],
                cfg=self.config,
                log_cb=self._log,
                name_db=self._name_db,
            )

            # ══ [1/6] INGEST ═══════════════════════════════════
            self._log(f"\n[1/6] INGEST")
            t = time.time()
            self.stats.start_stage("INGEST")
            info = self._extractor.probe_video(video_path)
            self._stage("INGEST", time.time() - t,
                        duration=info["duration_human"],
                        resolution=info["resolution"])
            self._log(f"  OK: {info['duration_human']} | "
                      f"{info['resolution']} | {info['fps']} FPS")

            # Audio'yu erkenden başlat (paralel değil; sadece sıra)
            audio_result = None
            if scope in ("audio_only", "video+audio"):
                audio_result = self._run_audio(video_path, work_dir)

            # Varsayılanlar (audio_only için)
            ocr_lines = []
            cdata = self._empty_credits_data(vname)
            tmdb_result = None

            if scope != "audio_only":
                # ══ [2/6] FRAME_EXTRACT ════════════════════════════
                self._log(f"\n[2/6] FRAME_EXTRACT")
                t = time.time()
                self.stats.start_stage("FRAME_EXTRACT")
                fps = self.config.get("ocr_fps", 1.0)
                fd = self._extractor.extract_credits_frames(
                    video_path, work_dir, info, first_min, last_min, fps)
                self._stage("FRAME_EXTRACT", time.time() - t,
                            total=fd["total"],
                            entry=len(fd["entry"]),
                            exit_=len(fd["exit"]))
                self._log(f"  OK: {fd['total']} frame "
                          f"(giriş:{len(fd['entry'])} çıkış:{len(fd['exit'])})")

                # ══ [3/6] TEXT_FILTER ══════════════════════════════
                self._log(f"\n[3/6] TEXT_FILTER "
                          f"[mod={self.config.get('difficulty','medium')}]")
                t = time.time()
                self.stats.start_stage("TEXT_FILTER")
                all_frames = fd["entry"] + fd["exit"]
                candidates = self._text_filter.filter_frames(all_frames)
                if not candidates:
                    self._log(f"  !! Yazı adayı yok — fallback")
                    candidates = self._text_filter.fallback_filter(
                        fd["entry"], fd["exit"], 100)
                rej = 1 - len(candidates) / max(len(all_frames), 1)
                self._stage("TEXT_FILTER", time.time() - t,
                            input=len(all_frames), candidates=len(candidates))
                self._log(f"  OK: {len(all_frames)} -> {len(candidates)} "
                          f"aday (eleme: {rej:.0%})")

                # ══ [4/6] OCR_CREDITS ══════════════════════════════
                self._log(f"\n[4/6] OCR_CREDITS (PaddleOCR GPU)")
                t = time.time()
                self.stats.start_stage("OCR_CREDITS")
                ocr_lines, layout_pairs = self._ocr_engine.process_frames(
                    candidates, log_callback=self._log)
                ocr_time = time.time() - t

                ocr_lines = self._repair_turkish(ocr_lines)
                layout_pairs = self._repair_layout_pairs(layout_pairs)

                qwen_t = time.time()
                ocr_lines = self._qwen.verify(ocr_lines, log_cb=self._log)
                qwen_elapsed = time.time() - qwen_t
                if qwen_elapsed > 0.5:
                    self._log(f"  [Qwen] Doğrulama süresi: {qwen_elapsed:.1f}s")

                ocr_total = ocr_time + qwen_elapsed
                self._stage("OCR_CREDITS", ocr_total,
                            processed=len(candidates),
                            lines=len(ocr_lines),
                            layout_pairs=len(layout_pairs))
                self._log(f"  OK: {len(ocr_lines)} benzersiz satır ({ocr_total:.1f}s)")

                # ══ [5/6] CREDITS_PARSE ════════════════════════════
                self._log(f"\n[5/6] CREDITS_PARSE")
                t = time.time()
                self.stats.start_stage("CREDITS_PARSE")
                parser = CreditsParser(turkish_name_db=self._name_db)
                parsed = parser.parse(ocr_lines, layout_pairs=layout_pairs)
                cdata = parser.to_report_dict(parsed)
                self._stage("CREDITS_PARSE", time.time() - t,
                            actors=cdata["total_actors"],
                            crew=cdata["total_crew"],
                            companies=cdata["total_companies"])
                self._log(f"  OK: Oyuncu:{cdata['total_actors']} "
                          f"Ekip:{cdata['total_crew']} "
                          f"Şirket:{cdata['total_companies']}")

                # ══ [6/6] TMDB_VERIFY ══════════════════════════════
                self._log(f"\n[6/6] TMDB_VERIFY")
                t = time.time()
                self.stats.start_stage("TMDB_VERIFY")
                if not cdata.get("film_title"):
                    cdata["film_title"] = vname
                    self._log(f"  [TMDB] film_title yok, video adı kullanılıyor: {vname}")

                tmdb_result = self._run_tmdb(cdata, work_dir)
                self._stage("TMDB_VERIFY", time.time() - t,
                            status=tmdb_result.reason,
                            hits=tmdb_result.hits,
                            misses=tmdb_result.misses,
                            updated=tmdb_result.updated,
                            matched_title=tmdb_result.matched_title)
                if tmdb_result.updated:
                    self._log(f"  OK: '{tmdb_result.matched_title}' — "
                              f"hits:{tmdb_result.hits} misses:{tmdb_result.misses}")
                    # TMDB-only credits output when verification succeeds
                    cdata = self._apply_tmdb_credits(cdata, tmdb_result)
                else:
                    self._log(f"  --: {tmdb_result.reason}")

            else:
                for stage_name in ("FRAME_EXTRACT", "TEXT_FILTER", "OCR_CREDITS",
                                   "CREDITS_PARSE", "TMDB_VERIFY"):
                    self._stage(stage_name, 0.0, status="skipped",
                                reason="scope=audio_only")

            # ══ EXPORT ════════════════════════════════════════════
            self._log(f"\n[EXPORT]")
            t = time.time()
            self.stats.start_stage("EXPORT")
            exp = ExportEngine(work_dir, name_db=self._name_db)
            jp, tp = exp.generate(
                info, cdata, ocr_lines, self.stage_stats,
                "WORKSTATION", scope, first_min, last_min)
            self._stage("EXPORT", time.time() - t)

            total = time.time() - t0
            self.stats.finish_job(total)

            self._log(f"\n{'='*60}")
            self._log(f"  TAMAMLANDI — {total:.1f}s "
                      f"({info['duration_seconds']/max(total,0.1):.1f}x)")
            self._log(f"  JSON: {jp}")
            self._log(f"  TXT : {tp}")
            self._log(f"{'='*60}")

            return {
                "report_json":  jp,
                "report_txt":   tp,
                "work_dir":     work_dir,
                "video_info":   info,
                "credits":      cdata,
                "ocr_lines":    len(ocr_lines),
                "tmdb_result":  tmdb_result,
                "audio_result": audio_result,
            }

        except Exception as e:
            self.stats.log_error(str(e))
            self.stats.finish_job(time.time() - t0)
            self._log(f"\n!! HATA: {e}")
            import traceback
            traceback.print_exc()
            raise

    # ──────────────────────────────────────────────────────────────
    # SES PİPELİNE — BUG-02 DÜZELTİLDİ
    # ──────────────────────────────────────────────────────────────
    def _run_audio(self, video_path: str, work_dir: str) -> dict:
        """AudioBridge üzerinden ses pipeline'ını çalıştır."""
        try:
            from core.audio_bridge import AudioBridge
        except ImportError:
            self._log("  [Audio] audio_bridge modülü yok — ses atlanıyor")
            return {"status": "skipped", "reason": "audio_bridge not found"}

        self._log(f"\n[AUDIO] Ses analizi başlatılıyor...")

        # Kullanıcı config'de açıkça belirtmişse onu kullan; yoksa profil bazlı seç
        stages = self.config.get("audio_stages", None)
        if stages is None:
            program_type = self.config.get("program_type", "film_dizi")
            if program_type in ("film_dizi", "kisa_haber"):
                # Sadece extract + transcribe (düz deşifre) — denoise/diarize gerekmez
                stages = ["extract", "transcribe"]
            else:
                # mac, muzik_programi vb. tam pipeline gerektirir (ileride genişletilecek)
                stages = ["extract", "denoise", "diarize", "transcribe", "post_process"]

        self._log(f"  [Audio] Stages: {stages}")

        audio_cfg = {
            "program_type": self.config.get("program_type", "film_dizi"),
            "hf_token":     self.config.get("hf_token", ""),
            "ffmpeg":       self._ffmpeg,
            "ollama_url":   self.config.get("ollama_url", "http://localhost:11434"),
            "tmdb_cast":    [],
            "stages": stages,
            "options": {
                "denoise_enabled":  "denoise" in stages,
                "whisper_model":    self.config.get("whisper_model", "large-v3"),
                "whisper_language": self.config.get("whisper_language", "tr"),
                # compute_type verilmezse TranscribeStage cihaz bazlı otomatik seçer
                # (cuda=float16, cpu=int8). CPU'da float16 hatasını önler.
                "compute_type":     self.config.get("compute_type"),
                "max_speakers":     self.config.get("max_speakers", 10),
                "ollama_model":     self.config.get("ollama_model", "llama3.1:8b"),
                "batch_size":       self.config.get("batch_size", 16),
            },
        }

        bridge = AudioBridge(log_cb=self._log, config=self.config)
        result = bridge.run(video_path, work_dir, audio_cfg)

        if result.get("status") == "ok":
            segs = result.get("transcript", [])
            spks = result.get("speakers", {})
            self._log(f"  [Audio] OK — {len(segs)} segment, "
                      f"{len(spks)} konuşmacı")
        else:
            err = result.get("error", "bilinmiyor")
            self._log(f"  [Audio] HATA: {err}")

        return result

    def _apply_tmdb_credits(self, cdata: dict, tmdb_result):
        """TMDB doğrulama başarılıysa rapor içeriğini TMDB kanonik verisiyle sınırla."""
        cast = []
        for item in (tmdb_result.cast or []):
            name = (item.get("name") or "").strip()
            if not name:
                continue
            cast.append({
                "actor_name": name,
                "character_name": (item.get("character") or "").strip(),
                "role": "Cast",
                "role_category": "cast",
                "raw": "tmdb",
                "confidence": 1.0,
                "frame": "tmdb",
            })

        crew = []
        for item in (tmdb_result.crew or []):
            name = (item.get("name") or "").strip()
            if not name:
                continue
            job = (item.get("job") or "").strip()
            crew.append({
                "name": name,
                "job": job,
                "role": job or "Crew",
                "role_category": "crew",
                "raw": "tmdb",
                "confidence": 1.0,
                "frame": "tmdb",
            })

        directors = [
            {"name": c.get("name", "")}
            for c in crew
            if (c.get("job") or "").strip().lower() in ("director", "yonetmen", "yönetmen")
        ]

        cdata["cast"] = cast
        cdata["crew"] = crew
        cdata["technical_crew"] = crew
        cdata["directors"] = directors
        cdata["total_actors"] = len(cast)
        cdata["total_crew"] = len(crew)
        cdata["verification_status"] = "tmdb_verified"
        cdata["keywords_source"] = "tmdb_only"
        return cdata

    # ──────────────────────────────────────────────────────────────
    # TMDB
    # ──────────────────────────────────────────────────────────────
    def _run_tmdb(self, cdata: dict, work_dir: str):
        from core.tmdb_verify import TMDBVerify, TMDBVerifyResult

        api_key = (
            self.config.get("tmdb_api_key") or
            get_tmdb_api_key()
        ).strip()
        token = (
            self.config.get("tmdb_bearer_token") or
            os.environ.get("TMDB_BEARER_TOKEN") or ""
        ).strip()
        lang = (
            self.config.get("tmdb_language") or
            os.environ.get("TMDB_LANGUAGE") or
            "tr-TR"
        ).strip()

        if not (api_key or token):
            self._log(f"  [TMDB] API key yok — .env dosyasına TMDB_API_KEY ekle")
            return TMDBVerifyResult(False, "no tmdb api key")

        verifier = TMDBVerify(
            work_dir=work_dir,
            api_key=api_key,
            bearer_token=token,
            language=lang,
            log_cb=self._log,
        )
        return verifier.verify_credits(cdata)

    # ──────────────────────────────────────────────────────────────
    # YARDIMCI
    # ──────────────────────────────────────────────────────────────
    def _stage(self, name, elapsed, status="ok", **details):
        self.stage_stats[name] = {
            "duration_sec": round(elapsed, 2),
            "status": status,
            **details,
        }
        self.stats.end_stage(name, status)

    @staticmethod
    def _empty_credits_data(film_title: str = "") -> dict:
        """audio_only kapsamı için minimal credits şablonu."""
        return {
            "film_title": film_title,
            "year": None,
            "cast": [],
            "crew": [],
            "technical_crew": [],
            "directors": [],
            "production_companies": [],
            "production_info": [],
            "total_actors": 0,
            "total_crew": 0,
            "total_companies": 0,
            "verification_status": "unverified",
        }

    def _repair_turkish(self, ocr_lines: list) -> list:
        if not ocr_lines:
            return ocr_lines
        repaired = 0
        for line in ocr_lines:
            original = (line.get("text", "") if isinstance(line, dict)
                        else getattr(line, "text", ""))
            if not original:
                continue
            fixed = self._name_db.correct_line(original)
            if fixed != original:
                if isinstance(line, dict):
                    line["text"] = fixed
                    line["text_original"] = original
                else:
                    line.text = fixed
                    line.text_original = original
                repaired += 1
        if repaired:
            self._log(f"  [NameDB] {repaired} satır Türkçe onarımı yapıldı")
        return ocr_lines

    def _repair_layout_pairs(self, layout_pairs: list) -> list:
        """
        Layout pair'lerden gelen bozuk actor isimlerini NameDB ile onar.
        TurkishNameDB.repair_layout_pairs() kullanır (threshold 0.85).
        """
        if not layout_pairs:
            return layout_pairs

        if hasattr(self._name_db, 'repair_layout_pairs'):
            repaired_pairs, diff = self._name_db.repair_layout_pairs(layout_pairs)
            if diff:
                self._log(f"  [NameDB] {diff} layout pair ismi onarıldı")
            return repaired_pairs

        repaired = 0
        for pair in layout_pairs:
            original = getattr(pair, 'actor_name', '')
            if not original:
                continue
            words = original.strip().split()
            fixed_words = []
            changed = False
            for w in words:
                result, score = self._name_db.find(w)
                if result and score >= 0.85 and result != w:
                    fixed_words.append(result)
                    changed = True
                else:
                    fixed_words.append(w)
            if changed:
                pair.actor_name = ' '.join(fixed_words)
                repaired += 1
        if repaired:
            self._log(f"  [NameDB] {repaired} layout pair ismi onarıldı")
        return layout_pairs
