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

import copy
import json
import shutil
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
from core.llm_cast_filter import LLMCastFilter
from core.vlm_reader import VLMReader
from utils.stats_logger import StatsLogger


_BLOK2_FUZZY_THRESHOLD = 85  # BLOK2 dedup: iki satır bu eşiğin üzerindeyse aynı kabul edilir


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
        self._log_messages: list[str] = []

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

        # VLM / QwenVerifier — config precedence:
        # 1. vlm_verify / vlm_enabled (config)
        # 2. VLM_ENABLED (env)
        # 3. qwen_verify (config, legacy)
        # 4. default True
        vlm_enabled_cfg = self.config.get("vlm_verify",
                          self.config.get("vlm_enabled", None))
        if vlm_enabled_cfg is None:
            env_enabled = os.environ.get("VLM_ENABLED", "")
            if env_enabled:
                vlm_enabled = env_enabled.lower() not in ("0", "false", "no")
            else:
                vlm_enabled = bool(self.config.get("qwen_verify", True))
        else:
            vlm_enabled = bool(vlm_enabled_cfg)

        # Model precedence: vlm_model (config) → VLM_MODEL (env) → qwen_model (config) → QWEN_MODEL (env) → default
        vlm_model = (
            self.config.get("vlm_model") or
            os.environ.get("VLM_MODEL") or
            self.config.get("qwen_model") or
            os.environ.get("QWEN_MODEL") or
            "glm4.6v-flash:q4_K_M"
        )

        # Threshold precedence: vlm_threshold (config) → VLM_THRESHOLD (env) → qwen_threshold (config) → QWEN_THRESHOLD (env) → default
        _thresh_raw = (
            self.config.get("vlm_threshold") or
            os.environ.get("VLM_THRESHOLD") or
            self.config.get("qwen_threshold") or
            os.environ.get("QWEN_THRESHOLD") or
            0.80
        )
        # Safe float conversion with validation
        try:
            vlm_thresh = float(_thresh_raw)
            # Clamp to valid range [0.0, 1.0]
            vlm_thresh = max(0.0, min(1.0, vlm_thresh))
        except (ValueError, TypeError) as e:
            self._log(f"  [INIT] Geçersiz vlm_threshold değeri, varsayılan 0.80 kullanılıyor: {e}")
            vlm_thresh = 0.80
        self._qwen = QwenVerifier(
            model=vlm_model,
            confidence_threshold=vlm_thresh,
            enabled=vlm_enabled,
            name_checker=self._name_db.is_name,
        )

        # LLMCastFilter config
        self._llm_filter_enabled = bool(self.config.get("llm_cast_filter", True))

        # VLMReader (VLM-as-OCR) — off by default
        vlm_ocr_enabled = bool(
            self.config.get("vlm_ocr_enabled",
            self.config.get("use_vlm_for_ocr", False))
        )
        self._vlm_reader = VLMReader(
            model=vlm_model,
            enabled=vlm_ocr_enabled,
        )

    def set_log_callback(self, cb):
        self._log_cb = cb
        self._name_db._log = cb

    def _log(self, msg):
        if self._log_cb:
            self._log_cb(msg)
        self._log_messages.append(str(msg))
        print(msg)

    # ──────────────────────────────────────────────────────────────
    def run(self, video_path, scope="video_only",
            first_min=4.0, last_min=8.0, content_profile: dict | None = None) -> dict:
        # İçerik profilinden scope/first_min/last_min değerlerini uygula
        profile_name = "FilmDizi"
        if content_profile:
            profile_name = content_profile.get("_name", "FilmDizi")
            scope      = content_profile.get("scope", scope)
            try:
                first_min = float(content_profile.get("first_segment_minutes", first_min))
            except (ValueError, TypeError):
                print(f"  [Config] Geçersiz first_segment_minutes — varsayılan {first_min} dk kullanılıyor")
            try:
                last_min  = float(content_profile.get("last_segment_minutes", last_min))
            except (ValueError, TypeError):
                print(f"  [Config] Geçersiz last_segment_minutes — varsayılan {last_min} dk kullanılıyor")

        t0 = time.time()
        video_path = str(Path(video_path).resolve())
        vname = Path(video_path).stem
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = os.path.join(
            self.output_root or os.path.dirname(video_path),
            f"arsiv_{vname}_{ts}"
        )
        os.makedirs(work_dir, exist_ok=True)

        self.stats.start_job(video_path, "WORKSTATION", scope, profile_name)
        self.stage_stats = {}
        self._log_messages = []

        self._log(f"\n{'='*60}")
        self._log(f"  VİTOS — Pipeline v7")
        self._log(f"{'='*60}")
        self._log(f"  Video  : {Path(video_path).name}")
        self._log(f"  Profil : {profile_name}")
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
            cdata_raw = None
            tmdb_result = None

            # Profil bazlı OCR kontrolü
            ocr_enabled = bool(content_profile.get("ocr_enabled", True)) if content_profile else True

            if scope != "audio_only" and ocr_enabled:
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

                # Optional VLM-as-OCR fallback (off by default)
                if self._vlm_reader.enabled and self._vlm_reader.is_available():
                    vlm_r_t = time.time()
                    ocr_lines = self._vlm_reader.augment_ocr_lines(
                        ocr_lines, candidates, log_cb=self._log)
                    self._log(f"  [VLM-OCR] Tamamlandı ({time.time()-vlm_r_t:.1f}s)")

                qwen_t = time.time()
                ocr_lines = self._qwen.verify(ocr_lines, log_cb=self._log,
                                              resolution=info.get("resolution", ""))
                qwen_elapsed = time.time() - qwen_t
                if qwen_elapsed > 0.5:
                    self._log(f"  [VLM] Doğrulama süresi: {qwen_elapsed:.1f}s")

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

                # Snapshot before filters (DATABASE credits_raw için)
                cdata_raw = copy.deepcopy(cdata)

                # ══ [6/6] TMDB_VERIFY ══════════════════════════════
                # TMDB önce çalışsın — eşleşirse cast TMDB'den gelecek, LLM gereksiz
                self._log(f"\n[6/6] TMDB_VERIFY")
                t = time.time()
                self.stats.start_stage("TMDB_VERIFY")
                tmdb_matched = False
                if content_profile and content_profile.get("match_parse_enabled"):
                    # Spor profili: TMDB yerine MATCH_PARSE çalıştır (ileride implement edilecek)
                    self._log(f"  [MATCH_PARSE] Profil={profile_name} — match_parse_enabled=True (skipped, ileride implement edilecek)")
                    self._stage("TMDB_VERIFY", 0.0, status="skipped", reason="match_parse_enabled")
                    tmdb_result = None
                else:
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
                        tmdb_matched = True
                    else:
                        self._log(f"  --: {tmdb_result.reason}")

                # ══ BLOK2: TMDB eşleşmedi → VLM ile derin okuma ══
                if not tmdb_matched:
                    if self._vlm_reader.enabled and self._vlm_reader.is_available():
                        self._log("\n[BLOK2] TMDB eşleşmedi — VLM ile derin okuma başlatılıyor...")
                        vlm_t = time.time()
                        vlm_ocr_lines = []
                        for frame_info in candidates:
                            result = self._vlm_reader.read_text_from_frame(
                                frame_info["path"], lang="tr")
                            if result and result.get("text"):
                                for line_text in result["text"].splitlines():
                                    line_text = line_text.strip()
                                    if line_text:
                                        vlm_ocr_lines.append({
                                            "text": line_text,
                                            "avg_confidence": 0.85,
                                            "bbox": [],
                                            "frame_path": frame_info["path"],
                                            "source": "vlm_blok2",
                                        })
                        if vlm_ocr_lines:
                            # VLM sonuçlarına _repair_turkish() uygulanmaz
                            ocr_lines = self._merge_blok2_results(ocr_lines, vlm_ocr_lines)
                            parser = CreditsParser(turkish_name_db=self._name_db)
                            parsed = parser.parse(ocr_lines, layout_pairs=layout_pairs)
                            cdata = parser.to_report_dict(parsed)
                            # BLOK2 sonrası cdata_raw güncelle — DATABASE ham verisi tutarlı olsun
                            cdata_raw = copy.deepcopy(cdata)
                            self._log(
                                f"  [BLOK2] VLM okuma: {len(vlm_ocr_lines)} satır, "
                                f"toplam: {len(ocr_lines)} satır ({time.time()-vlm_t:.1f}s)"
                            )

                # ══ [LLM] LLM_CAST_FILTER ══════════════════════════
                # Sadece TMDB eşleşmezse LLM devreye girsin
                if not tmdb_matched:
                    self._log(f"\n[LLM] Cast Filtreleme (TMDB eşleşmedi)")
                    t = time.time()
                    llm_filter = LLMCastFilter(
                        ollama_url=self.config.get("ollama_url", "http://localhost:11434"),
                        model=self.config.get("llm_filter_model") or self.config.get("ollama_model", "llama3.1:8b"),
                        enabled=self._llm_filter_enabled,
                        log_cb=self._log,
                        name_checker=self._name_db.is_name,
                    )
                    cdata["cast"] = llm_filter.filter_cast(cdata.get("cast", []), log_cb=self._log)
                    cdata["total_actors"] = len(cdata["cast"])
                    llm_elapsed = time.time() - t
                    if llm_elapsed > 0.1:
                        self._log(f"  [LLM] Cast filtreleme: {llm_elapsed:.1f}s")
                else:
                    self._log(f"\n[LLM] TMDB eşleşti — LLM filtresi atlanıyor")

                # ══ [GOOGLE_VI] Akıllı tetik ════════════════════════════
                # TMDB miss + düşük çözünürlük veya non-standard font ise Google VI çağır
                vi_decision = self._decide_google_vi(
                    tmdb_matched=tmdb_matched,
                    resolution=info.get("resolution", ""),
                    ocr_lines=ocr_lines,
                    segment_duration_min=(first_min + last_min),
                    cast_count=cdata.get("total_actors", 0),
                )
                if vi_decision.should_run:
                    self._log(f"\n[GOOGLE_VI] {vi_decision.reason}")
                    for trigger in vi_decision.triggers:
                        self._log(f"  → {trigger}")
                    vi_lines = self._run_google_vi(video_path, info, first_min, last_min)
                    if vi_lines:
                        try:
                            from core.google_video_intelligence import GoogleVITextEngine
                            vi_engine = GoogleVITextEngine(self.config, log_cb=self._log)
                            ocr_lines = vi_engine.merge_with_paddle(ocr_lines, vi_lines)
                        except Exception as _e:
                            self._log(f"  [GOOGLE_VI] Merge hatası: {_e}")
                        # VI sonrası tekrar parse et
                        parser = CreditsParser(turkish_name_db=self._name_db)
                        parsed = parser.parse(ocr_lines, layout_pairs=layout_pairs)
                        cdata = parser.to_report_dict(parsed)
                        self._log(
                            f"  [GOOGLE_VI] Tekrar parse: "
                            f"Oyuncu:{cdata['total_actors']} "
                            f"Ekip:{cdata['total_crew']}"
                        )
                else:
                    self._log(f"\n[GOOGLE_VI] {vi_decision.reason}")

            else:
                if not ocr_enabled:
                    self._log(f"  [OCR] Profil '{profile_name}' — OCR devre dışı")
                for stage_name in ("FRAME_EXTRACT", "TEXT_FILTER", "OCR_CREDITS",
                                   "CREDITS_PARSE", "TMDB_VERIFY"):
                    reason = (f"ocr_enabled=false in {profile_name}"
                              if not ocr_enabled else "scope=audio_only")
                    self._stage(stage_name, 0.0, status="skipped", reason=reason)

            # ══ EXPORT ════════════════════════════════════════════
            self._log(f"\n[EXPORT]")
            t = time.time()
            self.stats.start_stage("EXPORT")
            export_ts = datetime.now().strftime("%d%m%y-%H%M")
            exp = ExportEngine(work_dir, name_db=self._name_db)
            jp, tp, tr_p = exp.generate(
                info, cdata, ocr_lines, self.stage_stats,
                "WORKSTATION", scope, first_min, last_min,
                content_profile_name=profile_name,
                audio_result=audio_result,
                ts=export_ts)
            self._stage("EXPORT", time.time() - t)

            # Kullanıcı çıktı klasörüne (output_root) ana rapor ve transcript TXT'yi kopyala
            if self.output_root:
                out_root = Path(self.output_root)
                out_root.mkdir(parents=True, exist_ok=True)
                for _src in (tp, tr_p):
                    _src_p = Path(_src)
                    if _src_p.is_file():
                        _dst = _safe_path(out_root / _src_p.name)
                        shutil.copy2(_src_p, _dst)
                        self._log(f"  [EXPORT] Kullanıcı klasörüne kopyalandı: {_dst.name}")

            # ══ DATABASE ══════════════════════════════════════════
            try:
                self._write_database(
                    video_info=info,
                    credits_data=cdata,
                    credits_raw=cdata_raw,
                    ocr_lines=ocr_lines,
                    stage_stats=self.stage_stats,
                    audio_result=audio_result,
                    work_dir=work_dir,
                    content_profile_name=profile_name,
                    ts=export_ts,
                )
            except Exception as e:
                self._log(f"  [DATABASE] Yazma hatası (pipeline etkilenmedi): {e}")

            total = time.time() - t0
            self.stats.finish_job(total)

            self._log(f"\n{'='*60}")
            self._log(f"  TAMAMLANDI — {total:.1f}s "
                      f"({info['duration_seconds']/max(total,0.1):.1f}x)")
            self._log(f"  JSON      : {jp}")
            self._log(f"  Rapor     : {tp}")
            self._log(f"  Transcript: {tr_p}")
            self._log(f"{'='*60}")

            return {
                "report_json":       jp,
                "report_txt":        tp,
                "transcript_txt":    tr_p,
                "work_dir":          work_dir,
                "video_info":        info,
                "credits":           cdata,
                "ocr_lines":         len(ocr_lines),
                "tmdb_result":       tmdb_result,
                "audio_result":      audio_result,
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
    # GOOGLE VI
    # ──────────────────────────────────────────────────────────────
    def _decide_google_vi(
        self,
        tmdb_matched: bool,
        resolution: str,
        ocr_lines: list,
        segment_duration_min: float,
        cast_count: int = 0,
    ):
        """GoogleVITextEngine.decide_after_tmdb() çağrısını lazy import ile sarmala."""
        try:
            from core.google_video_intelligence import GoogleVITextEngine, GVIDecision
        except ImportError:
            from dataclasses import dataclass, field

            @dataclass
            class GVIDecision:
                should_run: bool
                reason: str
                triggers: list = field(default_factory=list)

            return GVIDecision(False, "google_vi modülü yok — VI atlanıyor")

        try:
            vi_engine = GoogleVITextEngine(self.config, log_cb=self._log)

            # Font tipini bbox geometrisinden tahmin et
            font_type = "unknown"
            try:
                from core.ocr_engine import OCREngine
                frame_bboxes = []
                for line in ocr_lines:
                    fp = (line.get("frame_path", "") if isinstance(line, dict)
                          else getattr(line, "frame_path", ""))
                    bbox = (line.get("bbox", []) if isinstance(line, dict)
                            else getattr(line, "bbox", []))
                    if fp and bbox:
                        frame_bboxes.append((fp, [bbox]))
                if frame_bboxes:
                    font_type = OCREngine.estimate_font_type(frame_bboxes)
            except Exception as fe:
                self._log(f"  [GOOGLE_VI] Font tahmin hatası: {fe}")

            # Conf ortalaması
            ocr_avg_conf = 0.0
            if ocr_lines:
                confs = []
                for line in ocr_lines:
                    c = (line.get("avg_confidence", line.get("confidence", 0.0))
                         if isinstance(line, dict)
                         else getattr(line, "avg_confidence",
                                      getattr(line, "confidence", 0.0)))
                    confs.append(float(c))
                if confs:
                    ocr_avg_conf = sum(confs) / len(confs)

            return vi_engine.decide_after_tmdb(
                tmdb_matched=tmdb_matched,
                resolution=resolution,
                font_type=font_type,
                segment_duration_min=segment_duration_min,
                ocr_avg_conf=ocr_avg_conf,
                total_ocr_lines=len(ocr_lines),
                cast_count=cast_count,
            )
        except Exception as e:
            self._log(f"  [GOOGLE_VI] Karar hatası: {e}")
            try:
                from core.google_video_intelligence import GVIDecision
                return GVIDecision(False, f"VI karar hatası: {e}")
            except ImportError:
                pass
            from dataclasses import dataclass, field

            @dataclass
            class _GVIDecision:
                should_run: bool
                reason: str
                triggers: list = field(default_factory=list)

            return _GVIDecision(False, f"VI karar hatası: {e}")

    def _run_google_vi(
        self,
        video_path: str,
        info: dict,
        first_min: float,
        last_min: float,
    ) -> list:
        """Google VI'yı entry + exit segmentleriyle çalıştır."""
        try:
            from core.google_video_intelligence import GoogleVITextEngine
        except ImportError:
            self._log("  [GOOGLE_VI] google_video_intelligence modülü yok — atlanıyor")
            return []

        try:
            vi_engine = GoogleVITextEngine(self.config, log_cb=self._log)
            duration = float(info.get("duration_seconds", 0))
            entry_start = 0.0
            entry_end = min(first_min * 60.0, duration)
            exit_end = duration
            exit_start = max(0.0, duration - last_min * 60.0)
            return vi_engine.process_segments(
                video_path, entry_start, entry_end, exit_start, exit_end
            )
        except Exception as e:
            self._log(f"  [GOOGLE_VI] İşlem hatası: {e}")
            return []

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

    def _write_database(self, video_info: dict, credits_data: dict,
                         credits_raw, ocr_lines: list, stage_stats: dict,
                         audio_result, work_dir: str,
                         content_profile_name: str, ts: str) -> None:
        """DATABASE dizinine pipeline çıktılarının bir kopyasını yaz."""
        if not self.config.get("database_enabled", True):
            return

        db_root = (
            self.config.get("database_root") or
            os.environ.get("VITOS_DATABASE_ROOT") or
            r"D:\DATABASE\FilmDizi"
        )

        stem = Path(video_info.get("filename", "out")).stem

        db_dir = Path(db_root) / stem
        db_dir.mkdir(parents=True, exist_ok=True)

        # 1. work_dir içindeki tüm dosyaları DB klasörüne kopyala
        for src in Path(work_dir).iterdir():
            if src.is_file():
                dst = _safe_path(db_dir / src.name)
                shutil.copy2(src, dst)

        # 2. OCR dual-score JSON yaz
        ocr_scores = self._build_ocr_scores(ocr_lines, credits_data)
        with open(_safe_path(db_dir / f"{stem}_{ts}_ocr_scores.json"), "w", encoding="utf-8") as f:
            json.dump(ocr_scores, f, ensure_ascii=False, indent=2)

        # 3. Ham credits JSON yaz (LLM filtre öncesi)
        with open(_safe_path(db_dir / f"{stem}_{ts}_credits_raw.json"), "w", encoding="utf-8") as f:
            json.dump(credits_raw if credits_raw is not None else credits_data,
                      f, ensure_ascii=False, indent=2)

        # 4. Transcript JSON yaz
        transcript = []
        if audio_result and isinstance(audio_result, dict):
            for seg in audio_result.get("transcript", []):
                transcript.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", ""),
                })
        with open(_safe_path(db_dir / f"{stem}_{ts}_transcript.json"), "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)

        # 5. Debug log yaz
        with open(_safe_path(db_dir / f"{stem}_{ts}_debug.log"), "w", encoding="utf-8") as f:
            f.write("\n".join(self._log_messages))

        self._log(f"  [DATABASE] Yazıldı: {db_dir}")

    @staticmethod
    def _build_ocr_scores(ocr_lines: list, credits_data: dict) -> dict:
        """OCR satırları ile pipeline sonuçlarını eşleştirerek ikili skor JSON üret."""
        import re

        def _norm(s: str) -> str:
            s = (s or "").lower()
            s = re.sub(r"[^a-z0-9]", "", s)
            return s

        cast = credits_data.get("cast", [])
        cast_lookup: dict[str, dict] = {}
        for entry in cast:
            for field in ("actor_name", "character_name"):
                name = (entry.get(field) or "").strip()
                if name:
                    cast_lookup[_norm(name)] = entry

        scores = []
        for line in ocr_lines:
            text = (line.text if hasattr(line, "text") else
                    line.get("text", "") if isinstance(line, dict) else "")
            ocr_conf = (getattr(line, "avg_confidence", None)
                        if hasattr(line, "avg_confidence") else
                        line.get("avg_confidence", line.get("confidence", 0))
                        if isinstance(line, dict) else 0)
            seen_count = (getattr(line, "seen_count", 1)
                          if hasattr(line, "seen_count") else
                          line.get("seen_count", 1)
                          if isinstance(line, dict) else 1)

            cast_entry = cast_lookup.get(_norm(text))
            if cast_entry:
                pipeline_conf = cast_entry.get("confidence")
                name_db_match = bool(cast_entry.get("is_verified_name", False))
                llm_verified = bool(cast_entry.get("is_llm_verified", False))
                verdict = "KEEP"
            else:
                pipeline_conf = None
                name_db_match = False
                llm_verified = False
                verdict = "REJECTED"

            scores.append({
                "text": text,
                "ocr_confidence": ocr_conf,
                "pipeline_confidence": pipeline_conf,
                "seen_count": seen_count,
                "name_db_match": name_db_match,
                "llm_verified": llm_verified,
                "verdict": verdict,
            })

        return {"scores": scores}

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

    def _merge_blok2_results(self, paddle_lines: list, vlm_lines: list) -> list:
        """PaddleOCR ve VLM (BLOK2) sonuçlarını fuzzy dedup ile birleştir.

        VLM satırları için _repair_turkish() çağrılmaz (VLM zaten doğru okur).
        PaddleOCR'da zaten bulunan satırlarla fuzzy benzerlik kontrolü yapılır;
        yeterince benzer olanlar eklenmez (tekrar önleme).
        """
        if not vlm_lines:
            return paddle_lines

        try:
            from rapidfuzz.fuzz import ratio as _fuzz_ratio
            _has_rapidfuzz = True
        except ImportError:
            _has_rapidfuzz = False

        paddle_texts = [
            (line.get("text", "") if isinstance(line, dict)
             else getattr(line, "text", "")).strip().lower()
            for line in paddle_lines
        ]

        merged = list(paddle_lines)
        added = 0
        for vlm_line in vlm_lines:
            vtext = (vlm_line.get("text", "")).strip()
            if not vtext:
                continue
            vlow = vtext.lower()
            # Exact match check
            if vlow in paddle_texts:
                continue
            # Fuzzy dedup (rapidfuzz mevcutsa)
            if _has_rapidfuzz:
                is_dup = any(
                    _fuzz_ratio(vlow, pt) >= _BLOK2_FUZZY_THRESHOLD
                    for pt in paddle_texts
                    if pt
                )
                if is_dup:
                    continue
            merged.append(vlm_line)
            paddle_texts.append(vlow)
            added += 1

        if added:
            self._log(f"  [BLOK2] {added} benzersiz VLM satırı eklendi")
        return merged
